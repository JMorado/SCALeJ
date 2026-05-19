"""Train against relative energies and forces for condensed-phase systems."""

from typing import Literal, Optional

import datasets
import descent.train
import descent.utils.loss
import smee
import smee.geometry
import smee.utils
import torch
from tqdm.auto import tqdm

ReferenceMode = Literal["mean", "min", "none", "infinite"]


def _add_v_site_coords(
    topology: smee.TensorTopology | smee.TensorSystem,
    conformer: torch.Tensor,
    force_field: smee.TensorForceField,
) -> torch.Tensor:
    """
    Insert virtual-site coordinates into an atom-only conformer.

    The dataset stores only atomic coordinates.  smee's ``compute_energy``
    expects a conformer that includes vsite positions (appended after the
    atoms of each molecule copy).  This helper reshapes the flat atom-only
    conformer, calls ``smee.geometry.add_v_site_coords`` per topology type,
    and returns the full particle conformer.  All operations are
    differentiable, so ``autograd.grad(energy, atom_coords)`` correctly
    projects vsite forces back onto atoms.

    Parameters
    ----------
    topology
        The topology (single molecule or full periodic system).
    conformer
        Atom-only coordinates with ``shape=(n_atoms_total, 3)``.
    force_field
        The force field (needed to evaluate vsite frame weights).

    Returns
    -------
    torch.Tensor
        Full particle conformer ``shape=(n_particles_total, 3)`` where
        vsites are interleaved immediately after the atoms of each molecule
        copy.
    """
    if isinstance(topology, smee.TensorTopology):
        if topology.n_v_sites == 0:
            return conformer
        return smee.geometry.add_v_site_coords(topology.v_sites, conformer, force_field)

    # TensorSystem: handle each topology type in turn.
    chunks: list[torch.Tensor] = []
    idx_atom = 0
    for mol_top, n_copies in zip(topology.topologies, topology.n_copies, strict=True):
        n_atoms = mol_top.n_atoms
        mol_coords = conformer[idx_atom : idx_atom + n_copies * n_atoms].reshape(
            n_copies, n_atoms, 3
        )
        if mol_top.n_v_sites > 0:
            mol_full = smee.geometry.add_v_site_coords(
                mol_top.v_sites, mol_coords, force_field
            )  # (n_copies, n_atoms + n_vsites, 3)
        else:
            mol_full = mol_coords
        chunks.append(mol_full.reshape(-1, 3))
        idx_atom += n_copies * n_atoms
    return torch.cat(chunks, dim=0)


def _prepare_entry_data(
    entry: dict,
    topology: smee.TensorTopology | smee.TensorSystem,
    reference: ReferenceMode,
    normalize: bool,
    energy_cutoff: Optional[float] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    int,
    torch.Tensor,
    Optional[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Prepare data for a single entry in a condensed-phase dataset.

    Parameters
    ----------
    entry
        A single dataset entry with keys ``"energy"``, ``"coords"``, ``"forces"``,
        and optionally ``"box_vectors"``.
    topology
        The topology of the molecule(s) in the entry.
    reference
        The reference energy mode. See ``_get_reference`` for options.
    normalize
        Whether to apply SCALeJ-style normalization.
    energy_cutoff
        If set, discard conformers whose energy exceeds
        ``min(energy) + energy_cutoff`` (in kcal/mol, before per-molecule
        normalisation).

    Returns
    -------
    tuple
        ``(energy_ref, forces_all, coords_all, box_vectors_all, n_mols,
        delta_energy_ref, ref_idx, weights, energy_var, forces_var,
        ref_coords, ref_box_vectors)``

        ``ref_coords`` and ``ref_box_vectors`` are the coordinates / box vectors
        of the reference conformer from the **original** (pre-filter) arrays.
        They are ``None`` when ``ref_idx`` is ``None``.
    """
    energy_ref = entry["energy"]
    n_confs = len(energy_ref)
    coords_all = entry["coords"].reshape(n_confs, -1, 3)
    box_vectors = entry.get("box_vectors", None)
    box_vectors_all = (
        None
        if (
            box_vectors is None
            or (hasattr(box_vectors, "numel") and box_vectors.numel() == 0)
        )
        else box_vectors.reshape(n_confs, 3, 3).clone().contiguous()
    )
    forces_all = entry["forces"].reshape(n_confs, -1, 3)

    # Per-molecule normalization (energy is extensive; forces are per-atom
    # and are NOT divided by n_mols).
    if normalize and isinstance(topology, smee.TensorSystem):
        n_mols = sum(topology.n_copies)
    else:
        n_mols = 1

    energy_ref = energy_ref / n_mols

    # Save pre-filter arrays. These are needed needed for "min"/"infinite"
    # reference which index the original array, and for saving ref conformer coords.
    energy_ref_full = energy_ref
    coords_full = coords_all
    box_vectors_full = box_vectors_all

    # Filter out high-energy conformers FIRST (matching _loss.py).
    if energy_cutoff:
        mask = (energy_ref - energy_ref.min()) <= energy_cutoff
        valid = mask.nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            raise ValueError("No valid conformers after applying energy cutoff filter.")
        energy_ref = energy_ref[valid]
        coords_all = coords_all[valid]
        forces_all = forces_all[valid]
        if box_vectors_all is not None:
            box_vectors_all = box_vectors_all[valid]
        n_confs = len(valid)

    # Compute reference after filtering.
    if reference == "mean":
        ref_energy = energy_ref.mean()  # mean of VALID conformers
        ref_idx = None
    elif reference == "min":
        ref_idx = int(energy_ref_full.argmin().item())
        ref_energy = energy_ref_full[ref_idx]
    elif reference == "infinite":
        # TODO: This assumes the infinite-separation frame is the last conformer.
        # Think if there's a better way to do this.
        ref_idx = len(energy_ref_full) - 1
        ref_energy = energy_ref_full[ref_idx]
    elif reference == "none":
        ref_energy = torch.zeros(1, device=energy_ref.device, dtype=energy_ref.dtype)
        ref_idx = None
    else:
        raise ValueError(
            f"Invalid reference mode: {reference!r}. Must be one of "
            "'mean', 'min', 'none', or 'infinite'."
        )

    # Save reference coords/box from the pre-filtered arrays so that
    # _compute_reference_prediction can always access them even when the
    # reference conformer is filtered out by the energy cutoff.
    ref_coords = coords_full[ref_idx].clone() if ref_idx is not None else None
    ref_box_vectors = (
        box_vectors_full[ref_idx].clone()
        if (ref_idx is not None and box_vectors_full is not None)
        else None
    )

    ref_energy = ref_energy.detach()

    # Compute the relative reference energies.
    delta_energy_ref = (energy_ref - ref_energy).detach()

    # Uniform weights.
    weights = torch.ones(n_confs, device=energy_ref.device) / n_confs

    # Variance normalization.
    if normalize:
        energy_var = torch.var(delta_energy_ref).detach()
        forces_var = torch.var(forces_all).detach()
        # Guard against zero variance (e.g. single conformer).
        if energy_var == 0:
            energy_var = energy_var.new_ones(1).squeeze()
        if forces_var == 0:
            forces_var = forces_var.new_ones(1).squeeze()
    else:
        energy_var = energy_ref.new_ones(1).squeeze()
        forces_var = energy_ref.new_ones(1).squeeze()

    return (
        energy_ref,
        forces_all,
        coords_all,
        box_vectors_all,
        n_mols,
        delta_energy_ref,
        ref_idx,
        weights,
        energy_var,
        forces_var,
        ref_coords,
        ref_box_vectors,
    )


def _compute_reference_prediction(
    topology: smee.TensorTopology | smee.TensorSystem,
    trainable: descent.train.Trainable,
    params: torch.Tensor,
    ref_coords: torch.Tensor,
    ref_box_vectors: torch.Tensor | None,
    n_mols: int,
    compute_gradient: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute the predicted energy (and its gradient) at the reference conformer.

    Parameters
    ----------
    topology
        The topology of the molecule(s).
    trainable
        The wrapper around trainable parameters.
    params
        The current parameter tensor.
    ref_coords
        Reference conformer coordinates ``(n_atoms, 3)``.
    ref_box_vectors
        Reference conformer box vectors ``(3, 3)``, or ``None``.
    n_mols
        Number of molecules in the system (for per-molecule normalisation).
    compute_gradient
        Whether to compute ``d(e_pred_0)/d(params)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        ``(e_pred_0, grad_e_pred_0)`` — the detached reference energy and its
        parameter gradient (or ``None`` when ``compute_gradient`` is ``False``).
    """
    forcefield = trainable.to_force_field(params.abs())
    coords_pred_0 = smee.utils.tensor_like(
        ref_coords, forcefield.potentials[0].parameters
    ).detach()
    box_vectors_pred_0 = (
        smee.utils.tensor_like(
            ref_box_vectors, forcefield.potentials[0].parameters
        ).detach()
        if ref_box_vectors is not None
        else None
    )
    full_coords_pred_0 = _add_v_site_coords(topology, coords_pred_0, forcefield)
    e_pred_0 = (
        smee.compute_energy(
            topology,
            forcefield,
            full_coords_pred_0,
            box_vectors=box_vectors_pred_0,
        ).squeeze()
        / n_mols
    )

    grad_e_pred_0 = None
    if compute_gradient:
        grad_e_pred_0 = torch.autograd.grad(
            e_pred_0, params, create_graph=False, retain_graph=False
        )[0].detach()

    del forcefield
    return e_pred_0.detach(), grad_e_pred_0


def _compute_batch_loss(
    topology: smee.TensorTopology | smee.TensorSystem,
    trainable: descent.train.Trainable,
    params: torch.Tensor,
    coords_all: torch.Tensor,
    box_vectors_all: torch.Tensor | None,
    forces_all: torch.Tensor,
    batch_slice: slice,
    e_pred_0: torch.Tensor,
    delta_energy_ref: torch.Tensor,
    weights: torch.Tensor,
    energy_var: torch.Tensor,
    forces_var: torch.Tensor,
    ref_idx: Optional[int],
    n_mols: int,
    energy_weight: float,
    force_weight: float,
    normalize: bool,
    compute_gradient: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Compute the loss for one batch of conformers.

    Parameters
    ----------
    topology
        The topology of the molecule(s).
    trainable
        The wrapper around trainable parameters.
    params
        The current parameter tensor.
    coords_all
        All conformer coordinates ``(n_confs, n_atoms, 3)``.
    box_vectors_all
        All conformer box vectors ``(n_confs, 3, 3)``, or ``None``.
    forces_all
        All conformer reference forces ``(n_confs, n_atoms, 3)``.
    batch_slice
        Slice selecting the conformer indices for this batch.
    e_pred_0
        Detached predicted energy at the reference conformer.
    delta_energy_ref
        Relative reference energies ``(n_confs,)``.
    weights
        Per-conformer loss weights ``(n_confs,)``.
    energy_var
        Variance of ``delta_energy_ref`` used for normalisation.
    forces_var
        Variance of reference forces used for normalisation.
    ref_idx
        Index of the reference conformer (or ``None``).
    n_mols
        Number of molecules in the system.
    energy_weight
        Weight applied to the energy MSE term.
    force_weight
        Weight applied to the force MSE term.
    normalize
        Whether per-atom force normalisation is applied.
    compute_gradient
        Whether to compute and return the gradient w.r.t. ``params``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
        ``(batch_loss, dloss_d_e_pred_0_increment, batch_grad, energy_loss, force_loss)`` — all detached.
        ``batch_grad`` is ``None`` when ``compute_gradient`` is ``False``.
    """
    n_confs = len(delta_energy_ref)
    batch_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
    batch_energy_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
    batch_force_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
    dloss_d_e_pred_0 = torch.zeros(1, device=params.device, dtype=params.dtype)

    forcefield = trainable.to_force_field(params.abs())
    ff_dtype = forcefield.potentials[0].parameters.dtype

    # Prepare batch data.
    coords_batch = [
        smee.utils.tensor_like(coords_all[j], forcefield.potentials[0].parameters)
        .detach()
        .requires_grad_(force_weight > 0)
        for j in range(batch_slice.start, batch_slice.stop)
    ]
    box_vectors_batch = [
        None
        if box_vectors_all is None
        else smee.utils.tensor_like(
            box_vectors_all[j], forcefield.potentials[0].parameters
        )
        .detach()
        .clone()
        .contiguous()
        for j in range(batch_slice.start, batch_slice.stop)
    ]

    # Add virtual-site coordinates (atom-only coords from dataset must be extended
    # to full particle coords before calling smee.compute_energy).  The operation is
    # differentiable, so autograd.grad(energy, coords_batch) correctly projects
    # vsite forces back onto atom coordinates.
    full_coords_batch = [
        _add_v_site_coords(topology, c, forcefield) for c in coords_batch
    ]

    # Compute predicted energies for the batch.
    energies_pred_batch = [
        smee.compute_energy(topology, forcefield, c_full, bv).squeeze() / n_mols
        for c_full, bv in zip(full_coords_batch, box_vectors_batch, strict=True)
    ]

    # Compute forces first (before building the energy-loss graph) so that when
    # create_graph=True allocates the mixed-Hessian d^2E/d(coords)d(params) only
    # the raw energy computation graphs are live, minimising peak GPU memory.
    if force_weight > 0:
        energy_sum = torch.stack(energies_pred_batch).sum()
        grads_coords = torch.autograd.grad(
            energy_sum,
            coords_batch,
            create_graph=True,
        )
    else:
        grads_coords = None

    # Compute the energy loss for the batch.
    for k, j in enumerate(range(batch_slice.start, batch_slice.stop)):
        e = energies_pred_batch[k]
        residual = e - e_pred_0 - delta_energy_ref[j].to(ff_dtype)
        residual_val = residual.detach()
        w_j = weights[j].to(ff_dtype)

        energy_loss_j = energy_weight * w_j * residual**2 / energy_var.to(ff_dtype)
        batch_loss = batch_loss + energy_loss_j
        batch_energy_loss = batch_energy_loss + energy_loss_j

        if ref_idx is not None:
            # dL/dE0 = -2 * energy_weight * sum(w_j * residual_j) / var_E
            dloss_d_e_pred_0 = dloss_d_e_pred_0 + (
                -2.0 * energy_weight * w_j * residual_val / energy_var.to(ff_dtype)
            )
    if force_weight > 0:
        for k, j in enumerate(range(batch_slice.start, batch_slice.stop)):
            # grads_coords are d(E/n_mols)/dr = (1/n_mols)*dE/dr, but forces
            # are raw atomic forces (-dE/dr), so multiply grad by n_mols.
            grad_coords_j = grads_coords[k] * n_mols
            f_ref_j = smee.utils.tensor_like(
                forces_all[j], forcefield.potentials[0].parameters
            )
            w_f = 1.0 / (n_confs * f_ref_j.numel()) if normalize else 1.0
            force_loss_j = (
                force_weight
                * ((-grad_coords_j - f_ref_j) ** 2).sum()
                * w_f
                / forces_var.to(ff_dtype)
            )
            batch_loss = batch_loss + force_loss_j
            batch_force_loss = batch_force_loss + force_loss_j

    # Compute gradient w.r.t. params and free the graph immediately so that
    # all intermediates (forcefield, coords, energies, second-derivative
    # buffers) are released before returning.
    batch_grad = None
    if compute_gradient:
        (batch_grad,) = torch.autograd.grad(
            batch_loss,
            params,
            create_graph=False,
            retain_graph=False,
        )
        batch_grad = batch_grad.detach()

    del (
        forcefield,
        coords_batch,
        full_coords_batch,
        box_vectors_batch,
        energies_pred_batch,
    )
    if force_weight > 0:
        del energy_sum, grads_coords

    return (
        batch_loss.detach(),
        dloss_d_e_pred_0.detach(),
        batch_grad,
        batch_energy_loss.detach(),
        batch_force_loss.detach(),
    )


def _process_entry(
    entry: dict,
    topology: smee.TensorTopology | smee.TensorSystem,
    trainable: descent.train.Trainable,
    params: torch.Tensor,
    reference: ReferenceMode,
    energy_weight: float,
    force_weight: float,
    batch_size: int,
    normalize: bool,
    compute_gradient: bool,
    energy_cutoff: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute the loss and gradient contribution for a single dataset entry.

    Parameters
    ----------
    entry
        A single dataset entry with keys ``"energy"``, ``"coords"``, ``"forces"``,
        and optionally ``"box_vectors"``.
    topology
        The topology of the molecule(s) in the entry.
    trainable
        The wrapper around trainable parameters.
    params
        The current parameter tensor (on the target device).
    reference
        The reference energy mode. See ``_get_reference`` for options.
    energy_weight
        Weight applied to the energy MSE term.
    force_weight
        Weight applied to the force MSE term.
    batch_size
        Number of conformers to process per gradient-accumulation step.
    normalize
        Whether to apply SCALeJ-style normalization.
    compute_gradient
        Whether to compute and return the gradient.
    energy_cutoff
        If set, discard conformers whose energy exceeds
        ``min(energy) + energy_cutoff`` (in kcal/mol).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
        The entry loss, accumulated gradient (or ``None`` if ``compute_gradient``
        is ``False``), energy loss component, and force loss component.
    """
    (
        energy_ref,
        forces_all,
        coords_all,
        box_vectors_all,
        n_mols,
        delta_energy_ref,
        ref_idx,
        weights,
        energy_var,
        forces_var,
        ref_coords,
        ref_box_vectors,
    ) = _prepare_entry_data(entry, topology, reference, normalize, energy_cutoff)

    n_confs = len(energy_ref)

    # Compute the gradient of the predicted energy with respect to the parameters.
    # dE_{pred,0}/dparams.
    e_pred_0 = torch.zeros(1, device=params.device, dtype=params.dtype)
    grad_e_pred_0 = None
    if ref_idx is not None:
        e_pred_0, grad_e_pred_0 = _compute_reference_prediction(
            topology,
            trainable,
            params,
            ref_coords,
            ref_box_vectors,
            n_mols,
            compute_gradient,
        )

    # Compute the predicted energies and forces for all conformers in batches.
    entry_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
    entry_energy_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
    entry_force_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
    accum_grad = torch.zeros_like(params) if compute_gradient else None
    dloss_d_e_pred_0 = torch.zeros(1, device=params.device, dtype=params.dtype)
    for i in range(0, n_confs, batch_size):
        batch_slice = slice(i, min(i + batch_size, n_confs))

        batch_loss, d_e0_increment, batch_grad, batch_energy_loss, batch_force_loss = (
            _compute_batch_loss(
                topology,
                trainable,
                params,
                coords_all,
                box_vectors_all,
                forces_all,
                batch_slice,
                e_pred_0,
                delta_energy_ref,
                weights,
                energy_var,
                forces_var,
                ref_idx,
                n_mols,
                energy_weight,
                force_weight,
                normalize,
                compute_gradient,
            )
        )
        dloss_d_e_pred_0 = dloss_d_e_pred_0 + d_e0_increment

        if compute_gradient and batch_grad is not None:
            accum_grad.add_(batch_grad)

        entry_loss = entry_loss + batch_loss
        entry_energy_loss = entry_energy_loss + batch_energy_loss
        entry_force_loss = entry_force_loss + batch_force_loss

    # Chain-rule correction for the reference energy.
    if compute_gradient and grad_e_pred_0 is not None:
        accum_grad.add_(dloss_d_e_pred_0 * grad_e_pred_0)

    return entry_loss, accum_grad, entry_energy_loss, entry_force_loss


def default_closure(
    trainable: descent.train.Trainable,
    topologies: dict[str, smee.TensorTopology | smee.TensorSystem],
    dataset: datasets.Dataset,
    reference: ReferenceMode = "mean",
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    batch_size: int = 1,
    normalize: bool = True,
    energy_cutoff: Optional[float] = None,
) -> descent.utils.loss.ClosureFn:
    """
    Return a closure for training against condensed-phase energies and forces.

    This closure computes energies and forces batch-by-batch to save memory,
    and also manually accumulates gradients to avoid retaining the full
    computation graph of all conformers at once.

    For ``"min"`` and ``"infinite"`` references, ``energy_pred_0`` depends on the
    force field parameters, so a chain-rule correction is applied after the batch
    loop to account for ``d(loss)/d(energy_pred_0) * d(energy_pred_0)/d(params)``.

    The Hessian output is always ``None``, meaning this closure is only compatible
    with first-order optimizers (e.g., Adam optimizer).

    When ``normalize=True`` the loss matches SCALeJ's normalization:

    * Reference energies are divided by the total number of molecules
      in the system (``n_mols = sum(system.n_copies)``).
    * Predicted energies are also divided by ``n_mols``.
    * Forces are per-atom quantities and are NOT divided by ``n_mols``.
    * Conformer weights are uniform and sum to 1 (``w_i = 1/n_confs``).
    * The energy SSE is divided by ``var(E_ref - E_ref_0)``.
    * The force SSE is divided by ``var(F_ref)``.
    * The total loss is averaged over the number of dataset entries.

    Parameters
    ----------
    trainable
        The wrapper around trainable parameters.
    topologies
        The topologies of the molecules in the dataset. Each key should be
        a fully indexed SMILES string or a run ID mapping to a TensorSystem.
    dataset
        The dataset to train against.
    reference
        The reference energy to compute the relative energies with respect to.
        ``"mean"`` uses the mean energy of all conformers; ``"min"`` uses the
        conformer with the lowest reference energy; ``"none"`` uses the raw energies
        without computing relative energies; ``"infinite"`` uses the last conformer
        (e.g., the infinite separation frame).
    energy_weight
        Weight applied to the energy MSE term.
    force_weight
        Weight applied to the force MSE term. Set to 0 to skip force computation
        entirely.
    batch_size
        Number of conformers to process per gradient-accumulation step.
    normalize
        Whether to apply SCALeJ-style normalization (per-molecule, variance,
        per-entry). When ``False``, raw squared errors are summed.
    energy_cutoff
        If set, discard conformers whose energy exceeds
        ``min(energy) + energy_cutoff`` (in kcal/mol) for each entry.
        Note that the filter will be based on the energy per molecule
        or on the total energy depending on whether SCALeJ-style normalization is applied.

    Returns
    -------
    descent.utils.loss.ClosureFn
        The default closure function.
    """
    n_entries = len(dataset)

    def closure_fn(
        params: torch.Tensor,
        compute_gradient: bool = True,
        compute_hessian: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if compute_hessian:
            raise NotImplementedError(
                "Hessian computation is not implemented for condensed-phase training."
            )

        total_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
        accum_grad = torch.zeros_like(params) if compute_gradient else None

        # We process the dataset per entry, and within each entry we process conformers in batches.
        entry_iterator = (
            tqdm(dataset, desc="Evaluating entries", leave=False)
            if n_entries > 1
            else dataset
        )
        total_energy_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
        total_force_loss = torch.zeros(1, device=params.device, dtype=params.dtype)

        for entry in entry_iterator:
            entry_id = entry["id"]
            topology = topologies[entry_id]

            entry_loss, entry_grad, entry_energy_loss, entry_force_loss = (
                _process_entry(
                    entry,
                    topology,
                    trainable,
                    params,
                    reference,
                    energy_weight,
                    force_weight,
                    batch_size,
                    normalize,
                    compute_gradient,
                    energy_cutoff,
                )
            )
            total_loss = total_loss + entry_loss
            total_energy_loss = total_energy_loss + entry_energy_loss
            total_force_loss = total_force_loss + entry_force_loss
            if compute_gradient and entry_grad is not None:
                accum_grad.add_(entry_grad)

        total_loss = total_loss / n_entries
        total_energy_loss = total_energy_loss / n_entries
        total_force_loss = total_force_loss / n_entries

        if compute_gradient and accum_grad is not None:
            accum_grad = accum_grad / n_entries

        closure_fn.last_losses = {
            "energy": total_energy_loss.item(),
            "forces": total_force_loss.item(),
        }

        return total_loss.detach(), accum_grad, None

    return closure_fn
