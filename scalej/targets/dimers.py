"""Train against dimer energies."""

from typing import Optional

import datasets
import descent.targets.dimers
import descent.train
import descent.utils.dataset
import descent.utils.loss
import smee
import smee.utils
import torch

from .condensed import ReferenceMode


def _get_reference(
    energy_ref: torch.Tensor, mode: ReferenceMode
) -> tuple[torch.Tensor, Optional[int]]:
    """
    Get the reference energy and index for a given entry.

    Parameters
    ----------
    energy_ref
        The reference energies for all conformers of the entry.
    mode
        The mode to compute the reference energy. See ``ReferenceMode`` for options.

    Returns
    -------
    tuple[torch.Tensor, Optional[int]]
        The reference energy and the index of the reference conformer (if applicable).
    """
    if mode == "mean":
        return energy_ref.mean(), None
    elif mode == "min":
        ref_idx = int(energy_ref.argmin().item())
        return energy_ref.min(), ref_idx
    elif mode == "none":
        return torch.zeros(1, device=energy_ref.device, dtype=energy_ref.dtype), None
    elif mode == "infinite":
        return energy_ref[-1], -1
    else:
        raise ValueError(
            f"Invalid reference mode: {mode!r}. Must be one of "
            "'mean', 'min', 'none', or 'infinite'."
        )


def default_closure(
    trainable: descent.train.Trainable,
    topologies: dict[str, smee.TensorTopology],
    dataset: datasets.Dataset,
    reference: ReferenceMode = "infinite",
    normalize: bool = True,
):
    """
    Return a default closure function for training against dimer energies.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        reference: How to pick the reference energy subtracted **per dimer** before
            computing the loss. ``"infinite"`` (default) uses the last conformer of
            each dimer, which by convention is the infinite-separation geometry.
        normalize: Whether to divide each dimer's squared-error by
            ``var(delta_y_ref)`` for that dimer.

    Returns:
        The default closure function.
    """

    def loss_fn(_x: torch.Tensor) -> torch.Tensor:
        _x = _x.abs()
        force_field = trainable.to_force_field(_x)
        total_loss = torch.zeros(1, dtype=_x.dtype, device=_x.device).squeeze()
        total_energy_loss = torch.zeros(1, dtype=_x.dtype, device=_x.device).squeeze()

        for dimer in descent.utils.dataset.iter_dataset(dataset):
            y_ref, y_pred = descent.targets.dimers._predict(
                dimer, force_field, topologies
            )

            # Per-molecule normalization.
            n_mols = 2 if normalize else 1
            y_ref = y_ref / n_mols
            y_pred = y_pred / n_mols

            # Compute relative energies according to the specified reference mode.
            ref_val, ref_idx = _get_reference(y_ref.detach(), reference)
            if ref_idx is not None:
                pred_ref_val = y_pred[ref_idx]
            else:
                pred_ref_val = y_pred.mean()

            # Pre-compute relative reference energies.
            y_ref_rel = (y_ref - ref_val).detach()
            y_pred_rel = y_pred - pred_ref_val

            # Variance normalization.
            if normalize:
                energy_var = torch.var(y_ref_rel).detach()
                if energy_var == 0:
                    energy_var = energy_var.new_ones(1).squeeze()
            else:
                energy_var = y_ref_rel.new_ones(1).squeeze()

            dimer_loss = ((y_pred_rel - y_ref_rel) ** 2).mean() / energy_var
            total_loss = total_loss + dimer_loss
            total_energy_loss = total_energy_loss + dimer_loss

        return total_loss / len(dataset)

    closure = descent.utils.loss.to_closure(loss_fn)

    return closure
