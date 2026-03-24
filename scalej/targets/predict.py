"""Prediction functions for energies and forces."""

import logging
from typing import Optional

import datasets
import smee
import torch
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def predict_energies_forces(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    tensor_systems: dict[str, smee.TensorSystem],
    reference: str = "none",
    energy_cutoff: Optional[float] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
    list[str],
]:
    """
    Predict energies and forces for all conformers in a dataset.

    Computes predicted energies and forces using the provided force field,
    with optional filtering by energy cutoff.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with entries containing "smiles", "coords", "forces",
        and optionally "box_vectors".
    force_field : smee.TensorForceField
        Force field to use for predictions (should be on the target device).
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping entry IDs to tensor systems.
    reference : str
        Reference energy mode: "mean", "min", "none", or "infinite".
    energy_cutoff : float, optional
        If set, filter conformers with energy > min_energy + energy_cutoff (kcal/mol).

    Returns
    -------
    tuple
        ``(energy_ref, energy_pred, forces_ref, forces_pred, mask_idxs, entry_ids)``.
        ``forces_ref`` and ``forces_pred`` are lists of tensors (one per entry).
    """
    energy_refs = []
    energy_preds = []
    forces_refs = []
    forces_preds = []
    mask_idxs_all = []
    entry_ids_all = []

    for _, entry in enumerate(tqdm(dataset, desc="Predicting energies/forces", leave=False)):
        entry_id = entry["id"]
        topology = tensor_systems[entry_id]

        # Get reference data
        energy_ref = entry["energy"]
        forces_ref = entry["forces"]
        coords_all = entry["coords"]
        box_vectors_all = entry.get("box_vectors", None)

        n_confs = len(energy_ref)
        coords_all = coords_all.reshape(n_confs, -1, 3)
        forces_ref = forces_ref.reshape(n_confs, -1, 3)

        if box_vectors_all is not None and box_vectors_all.numel() > 0:
            box_vectors_all = (
                box_vectors_all.reshape(n_confs, 3, 3).clone().contiguous()
            )
        else:
            box_vectors_all = None

        # Normalize by number of molecules
        if isinstance(topology, smee.TensorSystem):
            n_mols = sum(topology.n_copies)
        else:
            n_mols = 1

        energy_ref = energy_ref / n_mols
        forces_ref = forces_ref / n_mols

        # Save full (pre-filter) arrays
        energy_ref_full = energy_ref.clone()
        coords_full = coords_all.clone()
        box_vectors_full = (
            box_vectors_all.clone() if box_vectors_all is not None else None
        )

        # Apply energy cutoff filtering
        if energy_cutoff is not None:
            mask = (energy_ref - energy_ref.min()) <= energy_cutoff
            valid_idxs = mask.nonzero(as_tuple=True)[0].tolist()
            if len(valid_idxs) == 0:
                log.warning(
                    f"Entry {entry_id}: no conformers pass energy cutoff, skipping"
                )
                mask_idxs_all.append([])
                entry_ids_all.append(entry_id)
                continue
            energy_ref = energy_ref[valid_idxs]
            forces_ref = forces_ref[valid_idxs]
            coords_all = coords_all[valid_idxs]
            if box_vectors_all is not None:
                box_vectors_all = box_vectors_all[valid_idxs]
            mask_idxs_all.append(valid_idxs)
        else:
            mask_idxs_all.append(list(range(n_confs)))

        # Determine reference conformer index and compute reference prediction energy
        if reference == "mean":
            ref_idx = None  # Will compute mean of all predictions
        elif reference == "min":
            # Use minimum from full (pre-filtered) array
            ref_idx = int(energy_ref_full.argmin().item())
        elif reference == "infinite":
            # Use last conformer from full (pre-filtered) array
            ref_idx = len(energy_ref_full) - 1
        elif reference == "none":
            ref_idx = None
        else:
            raise ValueError(
                f"Invalid reference mode: {reference!r}. Must be one of "
                "'mean', 'min', 'none', or 'infinite'."
            )

        # Compute reference prediction energy if needed (from pre-filter arrays)
        e_pred_0 = None
        if ref_idx is not None:
            ref_coords = coords_full[ref_idx].clone()
            ref_box_vectors = (
                box_vectors_full[ref_idx].clone()
                if box_vectors_full is not None
                else None
            )

            ref_coords = smee.utils.tensor_like(
                ref_coords, force_field.potentials[0].parameters
            )
            if ref_box_vectors is not None:
                ref_box_vectors = smee.utils.tensor_like(
                    ref_box_vectors, force_field.potentials[0].parameters
                )

            e_pred_0 = smee.compute_energy(
                topology, force_field, ref_coords, ref_box_vectors
            )
            e_pred_0 = (e_pred_0.squeeze() / n_mols).detach()

        # Subtract reference data energy from reference energies
        if reference == "mean":
            ref_data_energy = energy_ref.mean()
        elif reference == "none":
            ref_data_energy = torch.tensor(
                0.0, device=energy_ref.device, dtype=energy_ref.dtype
            )
        elif ref_idx is not None:
            ref_data_energy = energy_ref_full[ref_idx]
        else:
            ref_data_energy = torch.tensor(
                0.0, device=energy_ref.device, dtype=energy_ref.dtype
            )

        energy_ref = (energy_ref - ref_data_energy).detach()

        # Compute predicted energies and forces (first pass - no reference subtraction yet)
        energy_preds_entry = []
        forces_preds_entry = []

        conformer_iterator = zip(
            coords_all,
            [None] * len(coords_all)
            if box_vectors_all is None
            else box_vectors_all,
            strict=True,
        )
        if len(coords_all) > 1:
            conformer_iterator = tqdm(
                conformer_iterator,
                total=len(coords_all),
                desc=f"  Entry {entry_id}",
                leave=False,
            )

        for _, (coords, box_vectors) in enumerate(conformer_iterator):
            # Ensure tensors are on the correct device
            coords = smee.utils.tensor_like(
                coords, force_field.potentials[0].parameters
            )
            if box_vectors is not None:
                box_vectors = smee.utils.tensor_like(
                    box_vectors, force_field.potentials[0].parameters
                )

            coords.requires_grad_(True)

            # Compute energy
            energy_pred = smee.compute_energy(
                topology, force_field, coords, box_vectors
            )
            energy_pred = (energy_pred.squeeze() / n_mols).detach()
            energy_preds_entry.append(energy_pred)

            # Compute forces
            coords_grad = coords.detach().requires_grad_(True)
            energy = smee.compute_energy(
                topology, force_field, coords_grad, box_vectors
            )
            grad_energy = torch.autograd.grad(
                energy.sum(), coords_grad, create_graph=False, retain_graph=False
            )[0]
            forces_pred = -grad_energy.detach() / n_mols
            forces_preds_entry.append(forces_pred)

        energy_preds_stack = torch.stack(energy_preds_entry)

        # Now apply reference subtraction based on reference mode
        if reference == "mean":
            # For mean: subtract mean of filtered predictions
            energy_pred_ref = energy_preds_stack.mean()
            energy_preds_stack = energy_preds_stack - energy_pred_ref
        elif reference in ("min", "infinite"):
            # For min/infinite: subtract prediction of reference conformer (from full array)
            if e_pred_0 is not None:
                energy_preds_stack = energy_preds_stack - e_pred_0
        # else: reference == "none", keep absolute energies

        energy_preds.append(energy_preds_stack)
        energy_refs.append(energy_ref)
        forces_refs.append(forces_ref)
        forces_preds.append(torch.stack(forces_preds_entry))
        entry_ids_all.append(entry_id)

    # Concatenate all results
    ff_device = force_field.potentials[0].parameters.device
    energy_ref_all = (
        torch.cat(energy_refs) if energy_refs else torch.tensor([], device=ff_device)
    )
    energy_pred_all = (
        torch.cat(energy_preds) if energy_preds else torch.tensor([], device=ff_device)
    )
    forces_ref_all = forces_refs
    forces_pred_all = forces_preds

    return (
        energy_ref_all,
        energy_pred_all,
        forces_ref_all,
        forces_pred_all,
        mask_idxs_all,
        entry_ids_all,
    )
