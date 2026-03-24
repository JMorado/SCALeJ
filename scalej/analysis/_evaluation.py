"""Evaluation functions for force field assessment."""

import logging
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
import smee
import torch
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


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


def compute_metrics_from_arrays(
    energy_ref: np.ndarray,
    energy_pred: np.ndarray,
    forces_ref: np.ndarray,
    forces_pred: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Compute error metrics from numpy arrays.

    Parameters
    ----------
    energy_ref : np.ndarray
        Reference energies.
    energy_pred : np.ndarray
        Predicted energies.
    forces_ref : np.ndarray
        Reference forces.
    forces_pred : np.ndarray
        Predicted forces.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        ``(energy_mae, energy_rmse, energy_r2, forces_mae, forces_rmse, forces_r2)``.
    """
    # Flatten forces if needed
    forces_ref_flat = forces_ref.flatten()
    forces_pred_flat = forces_pred.flatten()

    # Energy metrics
    energy_mae = float(np.mean(np.abs(energy_pred - energy_ref)))
    energy_rmse = float(np.sqrt(np.mean((energy_pred - energy_ref) ** 2)))

    energy_var = np.var(energy_ref)
    if energy_var > 0:
        energy_r2 = float(
            1
            - np.sum((energy_ref - energy_pred) ** 2)
            / np.sum((energy_ref - np.mean(energy_ref)) ** 2)
        )
    else:
        energy_r2 = 0.0

    # Force metrics
    forces_mae = float(np.mean(np.abs(forces_pred_flat - forces_ref_flat)))
    forces_rmse = float(np.sqrt(np.mean((forces_pred_flat - forces_ref_flat) ** 2)))

    forces_var = np.var(forces_ref_flat)
    if forces_var > 0:
        forces_r2 = float(
            1
            - np.sum((forces_ref_flat - forces_pred_flat) ** 2)
            / np.sum((forces_ref_flat - np.mean(forces_ref_flat)) ** 2)
        )
    else:
        forces_r2 = 0.0

    return energy_mae, energy_rmse, energy_r2, forces_mae, forces_rmse, forces_r2


def run_thermo_benchmark(
    force_field: smee.TensorForceField,
    topologies: dict[str, Any],
    smiles_a: str,
    smiles_b: str | None = None,
    density_ref: float | None = None,
    hvap_ref: float | None = None,
    temperature: float = 298.15,
    pressure: float = 1.0,
    output_dir: Path | str = Path("./predictions"),
    cache_dir: Path | str | None = Path("./cache"),
) -> tuple[
    float | None, float | None, float | None, float | None, float | None, float | None
]:
    """Run thermodynamic benchmark for density and heat of vaporization.

    Uses descent.targets.thermo to compute thermodynamic properties
    and compare against reference values.

    Parameters
    ----------
    force_field : smee.TensorForceField
        Force field to use for predictions.
    topologies : dict
        Dictionary of topologies {smiles: topology}.
    smiles_a : str
        SMILES string for component A.
    smiles_b : str, optional
        SMILES string for component B (for mixtures).
    density_ref : float, optional
        Reference density [g/mL].
    hvap_ref : float, optional
        Reference heat of vaporization [kcal/mol].
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in atm.
    output_dir : Path | str
        Directory for output files.
    cache_dir : Path | str, optional
        Directory for cache files.

    Returns
    -------
    tuple[float | None, float | None, float | None, float | None, float | None, float | None]
        ``(density_ref, density_pred, density_std, hvap_ref, hvap_pred, hvap_std)``.

    Examples
    --------
    >>> result = run_thermo_benchmark(
    ...     force_field, topologies, "CCO",
    ...     density_ref=0.789, hvap_ref=10.1
    ... )
    >>> density_ref, density_pred, density_std, hvap_ref, hvap_pred, hvap_std = result
    """
    import descent.targets.thermo

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    x_a = 1.0
    x_b = None

    entries = []

    # Density entry
    if density_ref is not None:
        entries.append(
            {
                "type": "density",
                "smiles_a": smiles_a,
                "x_a": x_a,
                "smiles_b": smiles_b,
                "x_b": x_b,
                "temperature": temperature,
                "pressure": pressure,
                "value": density_ref,
                "std": 0.0,
                "units": "g/mL",
                "source": "benchmark",
            }
        )

    # Hvap entry
    if hvap_ref is not None:
        entries.append(
            {
                "type": "hvap",
                "smiles_a": smiles_a,
                "x_a": x_a,
                "smiles_b": smiles_b,
                "x_b": x_b,
                "temperature": temperature,
                "pressure": pressure,
                "value": hvap_ref,
                "std": 0.0,
                "units": "kcal/mol",
                "source": "benchmark",
            }
        )

    if not entries:
        return None, None, None, None, None, None

    # Create dataset and run predictions
    dataset = descent.targets.thermo.create_dataset(*entries)

    results = descent.targets.thermo.predict(
        dataset,
        force_field,
        topologies,
        output_dir,
        cached_dir=cache_dir,
        verbose=True,
    )

    # Parse results
    d_ref, d_pred, d_std = None, None, None
    h_ref, h_pred, h_std = None, None, None

    if isinstance(results, tuple) and len(results) == 4:
        y_true, y_true_std, y_pred, y_pred_std = results

        idx = 0
        if density_ref is not None:
            d_ref = float(y_true[idx])
            d_pred = float(y_pred[idx])
            d_std = float(y_pred_std[idx])
            idx += 1

        if hvap_ref is not None:
            h_ref = float(y_true[idx])
            h_pred = float(y_pred[idx])
            h_std = float(y_pred_std[idx])

    return d_ref, d_pred, d_std, h_ref, h_pred, h_std


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
        tqdm(zip(all_mask_idxs, all_entry_ids, strict=True), total=len(all_entry_ids), desc="Saving predictions", leave=False)
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
    log.info(f"Saved {tag} predictions -> '{out_path}'")
