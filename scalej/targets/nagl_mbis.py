"""Train against condensed-phase data by optimising the NAGL-MBIS alpha mixing parameter.

The NAGL-MBIS polarised charge model computes partial charges as a linear
combination of gas-phase and water-phase NAGL charges:

    q = (1 - alpha) * q_gas + alpha * q_water

This module pre-computes gas and water charges once for every unique
molecule type, then at each training step updates the electrostatic
parameters tensor in-place so that no Interchange or TensorForceField is
ever recreated.
"""

from typing import Optional

import datasets
import descent.train
import descent.utils.loss
import numpy as np
import smee
import smee.utils
import torch
from openff.toolkit.topology import Molecule
from tqdm.auto import tqdm

from .condensed import ReferenceMode, _prepare_entry_data


def _compute_nagl_charges(
    mol: Molecule,
    charge_model,
) -> np.ndarray:
    """Compute NAGL-MBIS charges for a single molecule.

    Parameters
    ----------
    mol
        OpenFF Molecule.
    charge_model
        A loaded NAGL charge model (from ``naglmbis.models.load_charge_model``).

    Returns
    -------
    np.ndarray
        1-D array of partial charges in elementary-charge units, shape
        ``(n_atoms,)``.
    """
    # Compute properties returns a dict with "mbis-charges" key.
    properties = charge_model.compute_properties(mol.to_rdkit())
    charges = properties["mbis-charges"]

    # Convert torch tensor to numpy array.
    if hasattr(charges, "detach"):
        charges = charges.detach().cpu()
    return np.asarray(charges, dtype=float).squeeze()


def _precompute_molecule_charges(
    smiles_list: list[str],
    gas_model,
    water_model,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pre-compute gas and water NAGL charges for every unique SMILES.

    Parameters
    ----------
    smiles_list
        Unique SMILES strings present in the dataset.
    gas_model
        NAGL gas-phase charge model.
    water_model
        NAGL water-phase charge model.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping from SMILES to ``(gas_charges, water_charges)`` arrays,
        each of shape ``(n_atoms,)``.
    """
    charges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for smi in smiles_list:
        mol = Molecule.from_smiles(smi)
        q_gas = _compute_nagl_charges(mol, gas_model)
        q_water = _compute_nagl_charges(mol, water_model)
        charges[smi] = (q_gas, q_water)
    return charges


def _build_charge_tensors(
    topologies: list[smee.TensorTopology],
    smiles_per_topology: list[str],
    molecule_charges: dict[str, tuple[np.ndarray, np.ndarray]],
    n_global_params: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build gas / water charge vectors aligned with the global Electrostatics
    parameter tensor of a shared ``TensorForceField``.

    Because the force field is shared across all dataset entries, the
    Electrostatics parameter tensor may contain parameters for *all*
    unique molecules.  This function places the NAGL charges at the
    correct global parameter indices using each topology's assignment
    matrix.

    Parameters
    ----------
    topologies
        Ordered list of unique topologies for this entry.
    smiles_per_topology
        SMILES string for each topology (same order).
    molecule_charges
        Pre-computed ``(gas, water)`` charge arrays keyed by SMILES.
    n_global_params
        Number of rows in the global Electrostatics parameter tensor.
    dtype
        Target dtype.
    device
        Target device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(q_gas, q_water)`` tensors, each of shape ``(n_global_params, 1)``.
    """
    q_gas = torch.zeros(n_global_params, 1, dtype=dtype, device=device)
    q_water = torch.zeros(n_global_params, 1, dtype=dtype, device=device)

    for topo, smi in zip(topologies, smiles_per_topology, strict=False):
        q_g, q_w = molecule_charges[smi]
        param_map = topo.parameters["Electrostatics"]
        # The assignment matrix is sparse [n_atoms, n_global_params].
        # Each row has one non-zero entry whose column is the global
        # parameter index for that atom.
        sparse_idx = param_map.assignment_matrix.coalesce().indices()
        row_order = sparse_idx[0].argsort()
        param_indices = sparse_idx[1][row_order]

        q_gas[param_indices] = torch.tensor(q_g[:, None], dtype=dtype, device=device)
        q_water[param_indices] = torch.tensor(q_w[:, None], dtype=dtype, device=device)

    return q_gas, q_water


def _interpolate_charges(
    alpha: torch.Tensor,
    q_gas: torch.Tensor,
    q_water: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate between gas and water charges.

    Parameters
    ----------
    alpha
        Scalar mixing parameter (should be in [0, 1]).
    q_gas
        Gas-phase charges, shape ``(n, 1)``.
    q_water
        Water-phase charges, shape ``(n, 1)``.

    Returns
    -------
    torch.Tensor
        Interpolated charges, shape ``(n, 1)``.
    """
    return (1.0 - alpha) * q_gas + alpha * q_water


def _inject_charges(
    force_field: smee.TensorForceField,
    charges: torch.Tensor,
) -> None:
    """Write *charges* into the Electrostatics potential parameters **in-place**.

    Parameters
    ----------
    force_field
        The tensor force field whose electrostatic parameters will be
        overwritten.
    charges
        New charge values, shape matching
        ``force_field.potentials_by_type["Electrostatics"].parameters``.
    """
    e_pot = force_field.potentials_by_type["Electrostatics"]
    # Use data copy so that the computational graph flows through `charges`.
    e_pot.parameters = charges


def _compute_reference_prediction_with_charges(
    topology: smee.TensorTopology | smee.TensorSystem,
    trainable: descent.train.Trainable,
    params: torch.Tensor,
    alpha: torch.Tensor,
    q_gas: torch.Tensor,
    q_water: torch.Tensor,
    ref_coords: torch.Tensor,
    ref_box_vectors: torch.Tensor | None,
    n_mols: int,
) -> torch.Tensor:
    """Compute the predicted energy at the reference conformer.

    This injects the interpolated charges before computing the energy, and
    returns the energy **with gradient attached** to *alpha* (and *params*).
    """
    forcefield = trainable.to_force_field(params.abs())
    charges = _interpolate_charges(alpha, q_gas, q_water)
    _inject_charges(forcefield, charges)

    coords = smee.utils.tensor_like(
        ref_coords, forcefield.potentials[0].parameters
    ).detach()
    box_vectors = (
        smee.utils.tensor_like(
            ref_box_vectors, forcefield.potentials[0].parameters
        ).detach()
        if ref_box_vectors is not None
        else None
    )
    e_pred_0 = (
        smee.compute_energy(
            topology, forcefield, coords, box_vectors=box_vectors
        ).squeeze()
        / n_mols
    )
    return e_pred_0


def _compute_batch_loss_with_charges(
    topology: smee.TensorTopology | smee.TensorSystem,
    trainable: descent.train.Trainable,
    params: torch.Tensor,
    alpha: torch.Tensor,
    q_gas: torch.Tensor,
    q_water: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss for one batch, injecting interpolated charges.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(batch_loss, batch_energy_loss, batch_force_loss)`` — all with
        gradient attached.
    """
    n_confs = len(delta_energy_ref)
    forcefield = trainable.to_force_field(params.abs())
    charges = _interpolate_charges(alpha, q_gas, q_water)
    _inject_charges(forcefield, charges)

    ff_dtype = forcefield.potentials[0].parameters.dtype

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

    energies_pred_batch = [
        smee.compute_energy(topology, forcefield, c, bv).squeeze() / n_mols
        for c, bv in zip(coords_batch, box_vectors_batch, strict=True)
    ]

    if force_weight > 0:
        energy_sum = torch.stack(energies_pred_batch).sum()
        grads_coords = torch.autograd.grad(energy_sum, coords_batch, create_graph=True)
    else:
        grads_coords = None

    batch_loss = torch.zeros(1, device=alpha.device, dtype=ff_dtype)
    batch_energy_loss = torch.zeros(1, device=alpha.device, dtype=ff_dtype)
    batch_force_loss = torch.zeros(1, device=alpha.device, dtype=ff_dtype)

    for k, j in enumerate(range(batch_slice.start, batch_slice.stop)):
        e = energies_pred_batch[k]
        residual = e - e_pred_0 - delta_energy_ref[j].to(ff_dtype)
        w_j = weights[j].to(ff_dtype)
        energy_loss_j = energy_weight * w_j * residual**2 / energy_var.to(ff_dtype)
        batch_loss = batch_loss + energy_loss_j
        batch_energy_loss = batch_energy_loss + energy_loss_j

    if force_weight > 0:
        for k, j in enumerate(range(batch_slice.start, batch_slice.stop)):
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

    return batch_loss, batch_energy_loss, batch_force_loss


def default_closure(
    trainable: descent.train.Trainable,
    topologies: dict[str, smee.TensorTopology | smee.TensorSystem],
    dataset: datasets.Dataset,
    gas_model,
    water_model,
    smiles_per_topology: dict[str, list[str]],
    alpha: torch.Tensor,
    reference: ReferenceMode = "mean",
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    batch_size: int = 1,
    normalize: bool = True,
    energy_cutoff: Optional[float] = None,
) -> descent.utils.loss.ClosureFn:
    """Return a closure that optimises the NAGL-MBIS alpha mixing parameter.

    The closure computes energies and forces exactly like the condensed-phase
    target, but **also** differentiates through the charge interpolation

        q = (1 - alpha) * q_gas + alpha * q_water

    so that the gradient of the loss w.r.t. ``alpha`` is available for
    first-order optimisers.

    Gas-phase and water-phase NAGL charges are computed **once** when this
    function is called.  At each training step the charges are linearly
    combined and injected directly into the ``Electrostatics`` potential
    parameters tensor — no ``Interchange`` is ever recreated.

    Parameters
    ----------
    trainable
        Wrapper around the (non-charge) trainable force-field parameters.
    topologies
        Mapping from dataset entry ID to ``TensorTopology`` or
        ``TensorSystem``.
    dataset
        HuggingFace dataset with entries containing ``"energy"``,
        ``"coords"``, ``"forces"``, and optionally ``"box_vectors"``.
    gas_model
        NAGL gas-phase charge model (from
        ``naglmbis.models.load_charge_model``).
    water_model
        NAGL water-phase charge model.
    smiles_per_topology
        Mapping from dataset entry ID to the ordered list of unique
        SMILES strings for the topologies of that entry.  The order must
        match the topology list inside each ``TensorSystem``.
    alpha
        Scalar ``torch.Tensor`` (requires grad) — the mixing parameter.
    reference
        Reference energy mode (``"mean"``, ``"min"``, ``"none"``, or
        ``"infinite"``).
    energy_weight
        Weight for the energy MSE term.
    force_weight
        Weight for the force MSE term.
    batch_size
        Number of conformers per gradient-accumulation step.
    normalize
        Whether to apply SCALeJ normalisation (per-molecule, variance,
        per-entry).
    energy_cutoff
        Discard conformers above ``min(energy) + energy_cutoff`` (kcal/mol).

    Returns
    -------
    descent.utils.loss.ClosureFn
        A standard closure ``(params, compute_gradient, compute_hessian) ->
        (loss, grad, None)`` that is fully compatible with
        ``descent.utils.loss.combine_closures`` and
        ``scalej.train.run_training_loop``.

    Notes
    -----
    The gradient w.r.t. ``alpha`` is accumulated onto ``alpha.grad`` as a
    **side-effect** of each closure call.  To optimise alpha jointly with
    the force-field parameters, include it in your optimizer::

        optimizer = torch.optim.Adam([
            {"params": [params], "lr": 1e-3},
            {"params": [alpha],  "lr": 1e-2},
        ])

    After each ``optimizer.step()``, clamp alpha to [0, 1]::

        with torch.no_grad():
            alpha.data.clamp_(0.0, 1.0)
    """
    n_entries = len(dataset)

    # ------------------------------------------------------------------
    # Pre-compute gas / water charges for every unique SMILES (once).
    # ------------------------------------------------------------------
    all_smiles: set[str] = set()
    for smi_list in smiles_per_topology.values():
        all_smiles.update(smi_list)

    molecule_charges = _precompute_molecule_charges(
        list(all_smiles), gas_model, water_model
    )

    # Pre-build the per-entry charge vectors aligned with the Electrostatics
    # parameter tensor so we don't redo the concatenation every step.
    #
    # We need to figure out the dtype / device from the force-field once.
    _sample_ff = trainable.to_force_field(trainable.to_values())
    _e_pot = _sample_ff.potentials_by_type["Electrostatics"]
    _ff_dtype = _e_pot.parameters.dtype
    _ff_device = _e_pot.parameters.device
    _n_global_params = _e_pot.parameters.shape[0]
    del _sample_ff, _e_pot

    # For each entry, build (q_gas, q_water) tensors that are shaped
    # identically to the Electrostatics parameter tensor.
    entry_charge_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for entry_id, smi_list in smiles_per_topology.items():
        topology = topologies[entry_id]
        if isinstance(topology, smee.TensorSystem):
            topos = topology.topologies
        else:
            topos = [topology]
        q_gas, q_water = _build_charge_tensors(
            topos,
            smi_list,
            molecule_charges,
            _n_global_params,
            _ff_dtype,
            _ff_device,
        )
        entry_charge_tensors[entry_id] = (q_gas, q_water)

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------

    def closure_fn(
        params: torch.Tensor,
        compute_gradient: bool = True,
        compute_hessian: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if compute_hessian:
            raise NotImplementedError(
                "Hessian computation is not implemented for NAGL-MBIS alpha training."
            )

        total_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
        total_energy_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
        total_force_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
        accum_grad = torch.zeros_like(params) if compute_gradient else None
        alpha_grad_accum = torch.zeros(1, device=alpha.device, dtype=alpha.dtype)

        entry_iterator = (
            tqdm(dataset, desc="Evaluating entries (NAGL-MBIS)", leave=False)
            if n_entries > 1
            else dataset
        )

        print(f"Alpha value: {alpha.item():.4f}")

        for entry in entry_iterator:
            entry_id = entry["id"]
            topology = topologies[entry_id]
            q_gas, q_water = entry_charge_tensors[entry_id]

            # Prepare reference data.
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
            ) = _prepare_entry_data(
                entry, topology, reference, normalize, energy_cutoff
            )

            n_confs = len(energy_ref)

            # Reference prediction.
            e_pred_0 = torch.zeros(1, device=params.device, dtype=params.dtype)
            if ref_idx is not None:
                e_pred_0 = _compute_reference_prediction_with_charges(
                    topology,
                    trainable,
                    params,
                    alpha,
                    q_gas,
                    q_water,
                    ref_coords,
                    ref_box_vectors,
                    n_mols,
                )

            # Batch loop.
            entry_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
            entry_energy_loss = torch.zeros(1, device=params.device, dtype=params.dtype)
            entry_force_loss = torch.zeros(1, device=params.device, dtype=params.dtype)

            for i in range(0, n_confs, batch_size):
                batch_slice = slice(i, min(i + batch_size, n_confs))
                batch_loss, batch_energy_loss, batch_force_loss = (
                    _compute_batch_loss_with_charges(
                        topology,
                        trainable,
                        params,
                        alpha,
                        q_gas,
                        q_water,
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
                    )
                )
                entry_loss = entry_loss + batch_loss
                entry_energy_loss = entry_energy_loss + batch_energy_loss
                entry_force_loss = entry_force_loss + batch_force_loss

            # Accumulate gradients for both params and alpha.
            if compute_gradient:
                grad_targets = [p for p in [params, alpha] if p.requires_grad]
                grads = torch.autograd.grad(
                    entry_loss,
                    grad_targets,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
                grad_idx = 0
                if params.requires_grad:
                    if grads[grad_idx] is not None:
                        accum_grad.add_(grads[grad_idx].detach())
                    grad_idx += 1
                if alpha.requires_grad and grad_idx < len(grads):
                    if grads[grad_idx] is not None:
                        alpha_grad_accum.add_(grads[grad_idx].detach())
                    else:
                        import warnings

                        warnings.warn(
                            f"Alpha gradient is None for entry '{entry_id}'. "
                            "The computational graph may be disconnected — "
                            "smee.compute_energy may not propagate gradients "
                            "through Electrostatics parameters.",
                            stacklevel=1,
                        )

            total_loss = total_loss + entry_loss.detach()
            total_energy_loss = total_energy_loss + entry_energy_loss.detach()
            total_force_loss = total_force_loss + entry_force_loss.detach()

        total_loss = total_loss / n_entries
        total_energy_loss = total_energy_loss / n_entries
        total_force_loss = total_force_loss / n_entries

        if compute_gradient:
            if accum_grad is not None:
                accum_grad = accum_grad / n_entries
            alpha_grad_accum = alpha_grad_accum / n_entries

            # Set alpha.grad as a side-effect so that any optimizer
            # that includes alpha will update it automatically.
            if alpha.requires_grad:
                alpha_grad_val = alpha_grad_accum.detach().squeeze()
                print(
                    f"  alpha={alpha.item():.6f} | "
                    f"alpha_grad={alpha_grad_val.item():.6e}"
                )
                if alpha.grad is None:
                    alpha.grad = alpha_grad_val.clone()
                else:
                    alpha.grad.add_(alpha_grad_val)

        closure_fn.last_losses = {
            "energy": total_energy_loss.item(),
            "forces": total_force_loss.item(),
        }
        closure_fn.alpha = alpha

        return total_loss.detach(), accum_grad, None

    return closure_fn
