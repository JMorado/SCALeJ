"""Evaluation functions for force field assessment."""

import logging
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import smee
import torch
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def compute_metrics(
    energy_ref: torch.Tensor,
    energy_pred: torch.Tensor,
    forces_ref: torch.Tensor | list[torch.Tensor],
    forces_pred: torch.Tensor | list[torch.Tensor],
) -> tuple[float, float, float, float, float, float]:
    """Compute error metrics from prediction results.

    Calculates MAE, RMSE, and R² for both energies and forces.

    Parameters
    ----------
    energy_ref : torch.Tensor
        Reference energies with shape ``(n_total_conformers,)``.
    energy_pred : torch.Tensor
        Predicted energies with shape ``(n_total_conformers,)``.
    forces_ref : torch.Tensor | list[torch.Tensor]
        Reference forces with shape ``(n_total_conformers, n_atoms, 3)``
        or a list of tensors for heterogeneous systems.
    forces_pred : torch.Tensor | list[torch.Tensor]
        Predicted forces with shape ``(n_total_conformers, n_atoms, 3)``
        or a list of tensors for heterogeneous systems.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        ``(energy_mae, energy_rmse, energy_r2, forces_mae, forces_rmse, forces_r2)``.

    Examples
    --------
    >>> energy_ref, energy_pred, forces_ref, forces_pred, *_ = (
    ...     predict_energies_forces(dataset, force_field, systems)
    ... )
    >>> metrics = compute_metrics(energy_ref, energy_pred, forces_ref, forces_pred)
    >>> print(f"Energy MAE: {metrics.energy_mae:.4f} kcal/mol")
    """
    energy_ref = energy_ref.detach().cpu().numpy()
    energy_pred = energy_pred.detach().cpu().numpy()
    if isinstance(forces_ref, list):
        forces_ref = torch.cat([f.flatten() for f in forces_ref])
    else:
        forces_ref = forces_ref.flatten()
    forces_ref = forces_ref.detach().cpu().numpy()

    if isinstance(forces_pred, list):
        forces_pred = torch.cat([f.flatten() for f in forces_pred])
    else:
        forces_pred = forces_pred.flatten()
    forces_pred = forces_pred.detach().cpu().numpy()

    # Energy metrics
    energy_mae = float(np.mean(np.abs(energy_pred - energy_ref)))
    energy_rmse = float(np.sqrt(np.mean((energy_pred - energy_ref) ** 2)))
    energy_r2 = float(
        1
        - np.sum((energy_ref - energy_pred) ** 2)
        / np.sum((energy_ref - np.mean(energy_ref)) ** 2)
    )

    # Force metrics
    forces_mae = float(np.mean(np.abs(forces_pred - forces_ref)))
    forces_rmse = float(np.sqrt(np.mean((forces_pred - forces_ref) ** 2)))
    forces_r2 = float(
        1
        - np.sum((forces_ref - forces_pred) ** 2)
        / np.sum((forces_ref - np.mean(forces_ref)) ** 2)
    )

    return energy_mae, energy_rmse, energy_r2, forces_mae, forces_rmse, forces_r2


def evaluate_force_field(
    force_field: smee.TensorForceField,
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    reference: str = "none",
    energy_cutoff: float | None = None,
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | list[torch.Tensor],
        torch.Tensor | list[torch.Tensor],
        list[list[int]],
        list[str],
    ],
    tuple[float, float, float, float, float, float],
]:
    """Evaluate a force field on a dataset and compute metrics.

    Convenience function that runs prediction and computes metrics in one call.

    Parameters
    ----------
    force_field : smee.TensorForceField
        Force field to evaluate.
    dataset : datasets.Dataset
        Dataset with reference energies and forces.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    reference : str
        Reference energy mode ("mean", "min", or "none").
    energy_cutoff : float, optional
        Energy cutoff for filtering.

    Returns
    -------
    tuple
        ``(prediction, metrics)`` where ``prediction`` is
        ``(energy_ref, energy_pred, forces_ref, forces_pred, mask_idxs, entry_ids)``
        and ``metrics`` is
        ``(energy_mae, energy_rmse, energy_r2, forces_mae, forces_rmse, forces_r2)``.

    Examples
    --------
    >>> prediction, metrics = evaluate_force_field(
    ...     force_field, dataset, systems, reference="mean"
    ... )
    >>> print(f"Energy R²: {metrics.energy_r2:.4f}")
    """
    from ..targets import predict_energies_forces

    prediction = predict_energies_forces(
        dataset,
        force_field,
        tensor_systems=tensor_systems,
        reference=reference,
        energy_cutoff=energy_cutoff,
    )

    energy_ref, energy_pred, forces_ref, forces_pred, _mask, _ids = prediction
    metrics = compute_metrics(energy_ref, energy_pred, forces_ref, forces_pred)

    return prediction, metrics


def save_prediction_parquet(
    prediction: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[list[int]],
        list[str],
    ],
    output_dir: Path | str,
    tag: str,
    scale_factors: dict[str, list[float]] | None = None,
) -> None:
    """
    Save prediction results to a parquet file.

    One row per conformer. Forces are stored as flat lists of length
    ``n_atoms * 3``; reshape with ``.reshape(n_atoms, 3)`` after loading.

    Parameters
    ----------
    prediction : tuple
        ``(energy_ref, energy_pred, forces_ref, forces_pred, mask_idxs, entry_ids)``
        as returned by ``predict_energies_forces`` or ``evaluate_force_field``.
        ``forces_ref`` and ``forces_pred`` may be tensors or lists of tensors.
    output_dir : Path | str
        Directory where the parquet file is written.
    tag : str
        Prefix used to name the output file (e.g. ``"initial"`` or ``"final"``).
    scale_factors : dict[str, list[float]] | None, optional
        Mapping of entry ID to a list of scale factors (one per conformer in
        the dataset, **after** any stride has been applied).  When provided,
        a ``scale_factor`` column is added to the parquet.
    """
    from ..data import save_parquet

    (
        energy_ref_t,
        energy_pred_t,
        forces_ref_t,
        forces_pred_t,
        all_mask_idxs,
        all_entry_ids,
    ) = prediction

    rows = []
    e_ref_np = energy_ref_t.detach().cpu().numpy()
    e_pred_np = energy_pred_t.detach().cpu().numpy()

    if isinstance(forces_ref_t, list):
        f_ref_list = [f.detach().cpu().numpy() for f in forces_ref_t]
        f_pred_list = [f.detach().cpu().numpy() for f in forces_pred_t]
    else:
        f_ref_list = None
        f_pred_list = None
        f_ref_np = forces_ref_t.detach().cpu().numpy()
        f_pred_np = forces_pred_t.detach().cpu().numpy()

    # Group conformers by entry (system/mixture)
    conf_idx = 0
    for entry_idx, (mask_idxs, entry_id) in enumerate(
        tqdm(
            zip(all_mask_idxs, all_entry_ids, strict=True),
            total=len(all_entry_ids),
            desc="Saving predictions",
            leave=False,
        )
    ):
        n_conf_filtered = len(mask_idxs)

        if n_conf_filtered == 0:
            continue

        # Get energies for this entry
        e_ref = e_ref_np[conf_idx : conf_idx + n_conf_filtered]
        e_pred = e_pred_np[conf_idx : conf_idx + n_conf_filtered]

        # Get forces for this entry
        if f_ref_list is not None:
            f_ref_entry = f_ref_list[entry_idx]
            f_pred_entry = f_pred_list[entry_idx]
        else:
            f_ref_entry = f_ref_np[conf_idx : conf_idx + n_conf_filtered]
            f_pred_entry = f_pred_np[conf_idx : conf_idx + n_conf_filtered]

        sf_list = scale_factors.get(entry_id) if scale_factors else None

        for j in range(n_conf_filtered):
            row = {
                "id": entry_id,
                "entry_idx": entry_idx,
                "conformer_idx": int(mask_idxs[j]),
                "energy_ref": float(e_ref[j].item()),
                "energy_pred": float(e_pred[j].item()),
                "forces_ref": f_ref_entry[j].flatten().tolist(),
                "forces_pred": f_pred_entry[j].flatten().tolist(),
            }
            if sf_list is not None:
                row["scale_factor"] = sf_list[int(mask_idxs[j])]
            rows.append(row)

        conf_idx += n_conf_filtered

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{tag}_evaluations.parquet"
    save_parquet(pd.DataFrame(rows), out_path)
    LOGGER.info(f"Saved {tag} predictions -> '{out_path}'")
