"""Data-parallel closure for condensed-phase training across multiple GPUs."""

import io
import threading
from typing import Optional

import datasets
import descent.train
import descent.utils.loss
import smee
import torch
from tqdm.auto import tqdm

from scalej.targets.condensed import ReferenceMode, _process_entry


def _ddp_worker(
    rank: int,
    trainable_bytes: bytes,
    params_snapshot: torch.Tensor,
    dataset: datasets.Dataset,
    topologies: dict,
    entry_indices: list[int],
    reference: ReferenceMode,
    energy_weight: float,
    force_weight: float,
    batch_size: int,
    normalize: bool,
    energy_cutoff: Optional[float],
    results: list,
) -> None:
    """
    Worker thread that processes a subset of dataset entries on ``cuda:{rank}``.

    Parameters
    ----------
    rank
        The CUDA device index to run on.
    trainable_bytes
        Serialized ``Trainable`` object (from ``torch.save``).
    params_snapshot
        CPU snapshot of the current parameter tensor.
    dataset
        The full dataset; entries are accessed by index via ``entry_indices``.
    topologies
        Mapping from SMILES to topology objects.
    entry_indices
        The subset of dataset entry indices assigned to this worker.
    reference
        Reference energy mode. See ``_get_reference`` for options.
    energy_weight
        Weight applied to the energy MSE term.
    force_weight
        Weight applied to the force MSE term.
    batch_size
        Number of conformers per gradient-accumulation step.
    normalize
        Whether to apply SCALeJ-style normalization.
    energy_cutoff
        If set, discard conformers whose energy exceeds
        ``min(energy) + energy_cutoff`` (in kcal/mol).
    results
        Shared list; this worker writes ``(total_loss, n_processed, grad_cpu)``
        to ``results[rank]``.
    """
    import traceback as _tb

    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        local_trainable = torch.load(
            io.BytesIO(trainable_bytes), weights_only=False, map_location=device
        )

        params = params_snapshot.detach().clone().to(device).requires_grad_(True)

        total_loss = torch.zeros(1, device=device)
        total_energy_loss = torch.zeros(1, device=device)
        total_force_loss = torch.zeros(1, device=device)
        accum_grad: torch.Tensor | None = None
        n_processed = 0

        topologies_dev = {
            k: v.to(device) if hasattr(v, "to") else v for k, v in topologies.items()
        }

        entry_iterator = (
            tqdm(entry_indices, position=rank, leave=False)
            if len(entry_indices) > 1
            else entry_indices
        )
        for idx in entry_iterator:
            entry = dataset[idx]
            # Clone immediately. HuggingFace datasets returns views into a shared
            # Arrow buffer. Two threads accessing the same buffer concurrently
            # triggers PyTorch's "lazy wrapper called at most once" error.
            entry = {
                k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                for k, v in entry.items()
            }

            mixture_id = entry["id"]
            if isinstance(entry_iterator, tqdm):
                entry_iterator.set_description(f"mixture:{mixture_id}")
            topology = topologies_dev[mixture_id]

            entry_loss, entry_grad, entry_energy_loss, entry_force_loss = (
                _process_entry(
                    entry,
                    topology,
                    local_trainable,
                    params,
                    reference,
                    energy_weight,
                    force_weight,
                    batch_size,
                    normalize,
                    compute_gradient=True,
                    energy_cutoff=energy_cutoff,
                )
            )
            total_loss = total_loss + entry_loss
            total_energy_loss = total_energy_loss + entry_energy_loss
            total_force_loss = total_force_loss + entry_force_loss
            if entry_grad is not None:
                if accum_grad is None:
                    accum_grad = entry_grad
                else:
                    accum_grad = accum_grad + entry_grad
            n_processed += 1

        grad_cpu = (
            accum_grad.cpu()
            if accum_grad is not None
            else torch.zeros_like(params_snapshot)
        )
        results[rank] = (
            total_loss.item(),
            n_processed,
            grad_cpu,
            total_energy_loss.item(),
            total_force_loss.item(),
        )

    except Exception:
        _tb.print_exc()
        raise


def ddp_closure(
    trainable: descent.train.Trainable,
    topologies: dict[str, smee.TensorTopology | smee.TensorSystem],
    dataset: datasets.Dataset,
    reference: ReferenceMode = "mean",
    energy_weight: float = 1.0,
    force_weight: float = 0.0,
    batch_size: int = 8,
    normalize: bool = True,
    n_gpus: int = 4,
    energy_cutoff: Optional[float] = None,
) -> descent.utils.loss.ClosureFn:
    """
    Return a data-parallel closure that distributes entries across multiple GPUs.

    Dataset entries are distributed evenly (interleaved) across ``n_gpus`` CUDA
    devices using Python threads. Each thread runs the same per-entry gradient
    accumulation as ``default_closure``. Gradients are summed on the CPU and
    returned to the caller.

    The Hessian output is always ``None``. Only first-order optimizers are
    supported (e.g., Adam).

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
        Reference energy mode. See ``_get_reference`` for options.
    energy_weight
        Weight applied to the energy MSE term.
    force_weight
        Weight applied to the force MSE term. Set to 0 to skip force computation
        entirely.
    batch_size
        Number of conformers per gradient-accumulation step.
    normalize
        Whether to apply SCALeJ-style normalization.
    n_gpus
        Number of CUDA GPUs to distribute entries across.
    energy_cutoff
        If set, discard conformers whose energy exceeds
        ``min(energy) + energy_cutoff`` (in kcal/mol) for each entry.

    Returns
    -------
    descent.utils.loss.ClosureFn
        The DDP closure function.
    """

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError(
            "ddp_closure requires at least one CUDA GPU, but none were found."
        )
    if n_gpus > available:
        raise ValueError(
            f"Requested n_gpus={n_gpus} but only {available} CUDA device(s) available."
        )

    # Pre-warm GPUs to avoid first-call latency.
    for _r in range(n_gpus):
        with torch.cuda.device(_r):
            torch.zeros(1, device=f"cuda:{_r}")
            torch.cuda.synchronize(_r)

    n_entries = len(dataset)

    # Serialize trainable once; each worker deserializes its own copy.
    _buf = io.BytesIO()
    torch.save(trainable, _buf)
    trainable_bytes = _buf.getvalue()
    del _buf

    def closure_fn(
        params: torch.Tensor,
        compute_gradient: bool = True,
        compute_hessian: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        if compute_hessian:
            raise NotImplementedError(
                "ddp_closure does not support Hessian computation."
            )
        indices = torch.randperm(n_entries).tolist()
        gpu_indices: list[list[int]] = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(indices):
            gpu_indices[i % n_gpus].append(idx)

        params_cpu = params.detach().clone().cpu()
        results: list = [None] * n_gpus

        threads = [
            threading.Thread(
                target=_ddp_worker,
                args=(
                    rank,
                    trainable_bytes,
                    params_cpu,
                    dataset,
                    topologies,
                    gpu_indices[rank],
                    reference,
                    energy_weight,
                    force_weight,
                    batch_size,
                    normalize,
                    energy_cutoff,
                    results,
                ),
                daemon=True,
            )
            for rank in range(n_gpus)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        failed = [i for i, r in enumerate(results) if r is None]
        if failed:
            raise RuntimeError(f"DDP worker(s) {failed} failed. See traceback above.")

        total_loss_val = sum(r[0] for r in results)
        total_processed = sum(r[1] for r in results)

        denom = max(total_processed, 1)
        avg_loss = torch.tensor(total_loss_val / denom, device=params.device)

        avg_grad = None
        if compute_gradient:
            avg_grad = (
                torch.stack([r[2] for r in results]).sum(dim=0).to(params.device)
                / denom
            )

        closure_fn.last_losses = {
            "energy": sum(r[3] for r in results) / denom,
            "forces": sum(r[4] for r in results) / denom,
        }

        return avg_loss, avg_grad, None

    return closure_fn
