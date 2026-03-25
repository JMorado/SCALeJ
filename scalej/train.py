"""Training loop given a descent-compatible closure."""

import logging

import descent.train
import descent.utils.loss
import torch
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def _collect_losses(closure: descent.utils.loss.ClosureFn) -> dict[str, float]:
    """
    Collect per-component losses from a closure, including sub-closures.

    When ``descent.utils.loss.combine_closures`` wraps multiple closures, the
    combined ``last_losses`` only stores the aggregated loss per target name
    (e.g. ``{"condensed": 0.95}``).  This helper also inspects side-stored
    ``last_losses`` on each *sub*-closure to surface per-component breakdowns
    (e.g. energy / forces).
    """
    top = getattr(closure, "last_losses", {})

    # Try to access sub-closures captured by combine_closures.
    sub_closures: dict | None = None
    for cell in getattr(closure, "__closure__", None) or []:
        obj = cell.cell_contents
        if isinstance(obj, dict) and all(callable(v) for v in obj.values()):
            sub_closures = obj
            break

    if sub_closures is not None:
        result: dict[str, float] = {}
        for name, sub_fn in sub_closures.items():
            sub_losses = getattr(sub_fn, "last_losses", {})
            if sub_losses:
                for k, v in sub_losses.items():
                    result[f"{name}/{k}"] = v
            elif name in top:
                result[name] = top[name]
        # Include any top-level keys not covered by sub-closures.
        for k, v in top.items():
            if k not in sub_closures:
                result[k] = v
        return result

    return dict(top)


def run_training_loop(
    params: torch.Tensor,
    closure: descent.utils.loss.ClosureFn,
    trainable: descent.train.Trainable,
    n_epochs: int,
    lr: float,
    log_every: int = 1,
    clamp: bool = True,
) -> list[float]:
    """
    Run the Adam training loop.

    Parameters
    ----------
    params
        Initial parameter tensor (will be updated in-place via optimizer).
    closure
        A descent-compatible closure: ``(params, compute_gradient, compute_hessian)
        -> (loss, grad | None, hessian | None)``.
    trainable
        Used to clamp parameters after each step.
    n_epochs
        Number of training epochs.
    lr
        Adam learning rate.
    log_every
        Log every this many epochs (and always on the last epoch).
    clamp
        Whether to clamp parameters after each optimizer step.

    Returns
    -------
    list[float]
        Loss value at each epoch.
    """
    optimizer = torch.optim.Adam([params], lr=lr)
    losses = []

    pbar = tqdm(range(n_epochs), desc="Training")
    for epoch in pbar:
        optimizer.zero_grad()
        loss, grad, _ = closure(params, compute_gradient=True, compute_hessian=False)
        if grad is not None:
            params.grad = grad
        optimizer.step()
        if clamp:
            with torch.no_grad():
                params.data = trainable.clamp(params.data)

        loss_val = loss.item()
        losses.append(loss_val)
        pbar.set_postfix({"loss": f"{loss_val:.4e}"})

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            per_target = _collect_losses(closure)
            grad_norms = getattr(closure, "last_grad_norms", {})
            parts = " | ".join(f"{k}={v:.4e}" for k, v in per_target.items())
            gnorm_parts = " | ".join(
                f"{k}_gnorm={v:.4e}" for k, v in grad_norms.items()
            )
            msg = f"Epoch {epoch:4d}/{n_epochs} | loss={loss.item():.4e}"
            if parts:
                msg += f" | {parts}"
            if gnorm_parts:
                msg += f" | {gnorm_parts}"
            LOGGER.info(msg)

    return losses
