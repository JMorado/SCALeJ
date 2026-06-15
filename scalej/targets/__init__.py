"""Targets for condensed-phase training."""

from .condensed import default_closure
from .condensed_ddp import ddp_closure
from .dimers import default_closure as dimer_closure
from .nagl_mbis import default_closure as nagl_mbis_closure
from .predict import predict_energies_forces
from .utils import normalize_closure_weights

__all__ = [
    "predict_energies_forces",
    "default_closure",
    "ddp_closure",
    "dimer_closure",
    "nagl_mbis_closure",
    "normalize_closure_weights",
]
