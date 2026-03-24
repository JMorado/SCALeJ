"""Analysis module — evaluation metrics, benchmarks, and plots."""

from ._evaluation import (
    compute_metrics,
    compute_metrics_from_arrays,
    evaluate_force_field,
    run_thermo_benchmark,
    save_prediction_parquet,
)

__all__ = [
    "compute_metrics",
    "compute_metrics_from_arrays",
    "evaluate_force_field",
    "run_thermo_benchmark",
]
