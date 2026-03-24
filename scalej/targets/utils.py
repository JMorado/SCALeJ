"""Utility functions for targets."""

import descent.utils.loss
import torch


def normalize_closure_weights(
    closures: dict[str, descent.utils.loss.ClosureFn],
    params: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return weights so that every weighted loss term starts at approximately 1.

    Each weight is computed as ``user_weight / initial_loss``, where
    ``initial_loss`` is the value of the closure evaluated at *params* with no
    gradient computation.  This makes the magnitude of every term comparable
    at the start of training regardless of the raw scale of each target.

    Parameters
    ----------
    closures
        Dictionary of named closure functions (same structure as passed to
        ``descent.utils.loss.combine_closures``).
    params
        Current parameter tensor (typically the initial values before training).
    weights
        Optional dictionary of *relative* weights to apply before normalisation.
        If ``None``, all closures are treated equally (relative weight = 1).

    Returns
    -------
    dict[str, float]
        Dictionary of combined weights.
    """
    weights = weights or dict.fromkeys(closures, 1.0)

    normalized_weights = {}

    for name, closure in closures.items():
        with torch.no_grad():
            with torch.enable_grad():
                loss, *_ = closure(
                    params, compute_gradient=False, compute_hessian=False
                )

        initial_loss = float(loss.item())

        if initial_loss > 0.0:
            scale = 1.0 / initial_loss
        else:
            scale = 1.0

        normalized_weights[name] = weights.get(name, 1.0) * scale

    return normalized_weights
