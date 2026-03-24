"""Pydantic configuration models for simulation, training, and system setup."""

from typing import Literal

import openmm.unit
import pydantic
from pydantic_units import OpenMMQuantity, quantity_serializer

# Unit type aliases
_KELVIN = openmm.unit.kelvin
_ATMOSPHERE = openmm.unit.atmosphere
_FEMTOSECOND = openmm.unit.femtosecond
_INV_PICOSECOND = 1.0 / openmm.unit.picoseconds


if pydantic.__version__.startswith("1."):

    class BaseModel(pydantic.BaseModel):
        class Config:
            json_encoders = {openmm.unit.Quantity: quantity_serializer}
            arbitrary_types_allowed = True

else:

    class BaseModel(pydantic.BaseModel):
        class Config:
            arbitrary_types_allowed = True


class SimulationConfig(BaseModel):
    """
    Configuration for molecular dynamics simulations.

    Parameters
    ----------
    temperature : openmm.unit.Quantity
        Simulation temperature.
    pressure : openmm.unit.Quantity
        Simulation pressure.
    timestep : openmm.unit.Quantity
        Integration timestep.
    friction_coeff : openmm.unit.Quantity
        Langevin friction coefficient.
    n_minimization_steps : int
        Number of minimization steps (0 uses MinimizationConfig).
    n_nvt_steps : int
        Number of NVT equilibration steps.
    n_npt_equilibration_steps : int
        Number of NPT equilibration steps.
    n_production_steps : int
        Number of production MD steps.
    report_interval : int
        Interval for saving trajectory frames.
    """

    temperature: OpenMMQuantity[_KELVIN] = pydantic.Field(
        300 * _KELVIN,
        description="Simulation temperature with units compatible with kelvin.",
    )
    pressure: OpenMMQuantity[_ATMOSPHERE] = pydantic.Field(
        1.0 * _ATMOSPHERE,
        description="Simulation pressure with units compatible with atmosphere.",
    )
    timestep: OpenMMQuantity[_FEMTOSECOND] = pydantic.Field(
        1.0 * _FEMTOSECOND,
        description="Integration timestep with units compatible with femtosecond.",
    )

    friction_coeff: OpenMMQuantity[_INV_PICOSECOND] = pydantic.Field(
        1.0 * _INV_PICOSECOND,
        description=(
            "Langevin friction coefficient with units compatible with 1/picosecond."
        ),
    )

    n_minimization_steps: int = pydantic.Field(
        0, description="Number of minimization steps (0 uses MinimizationConfig)."
    )
    n_equilibration_nvt_steps: int = pydantic.Field(
        50_000, description="Number of NVT equilibration steps."
    )
    n_equilibration_npt_steps: int = pydantic.Field(
        50_000, description="Number of NPT equilibration steps."
    )
    n_production_steps: int = pydantic.Field(
        1_000_000, description="Number of production MD steps."
    )
    n_mlp_steps: int = pydantic.Field(
        100, description="Number of MLP steps to run after the production MD steps."
    )
    mlp_device: str = pydantic.Field(
        "cpu", description="Device to use for MLP ('cuda' or 'cpu')."
    )
    platform: str = pydantic.Field(
        "CPU", description="Platform to use for OpenMM ('CPU', 'CUDA', etc.)."
    )
    report_interval: int = pydantic.Field(
        1000, description="Interval for saving trajectory frames."
    )


class ScalingConfig(BaseModel):
    """
    Configuration for molecular position scaling.

    Attributes
    ----------
    close_range : tuple[float, float, int]
        (start, end, n_points) for close-range scaling.
    equilibrium_range : tuple[float, float, int]
        (start, end, n_points) for equilibrium-range scaling.
    long_range : tuple[float, float, int]
        (start, end, n_points) for long-range scaling.
    subsample_frequency : int
        Frequency for subsampling trajectory frames.
    n_frames : int
        Number of last frames to load from trajectory for scaling.
    """

    close_range: tuple[float, float, int] = pydantic.Field(
        (0.75, 0.9, 5), description="(start, end, n_points) for close-range scaling."
    )
    equilibrium_range: tuple[float, float, int] = pydantic.Field(
        (0.9, 1.1, 15),
        description="(start, end, n_points) for equilibrium-range scaling.",
    )
    long_range: tuple[float, float, int] = pydantic.Field(
        (1.1, 2.0, 12), description="(start, end, n_points) for long-range scaling."
    )
    subsample_frequency: int = pydantic.Field(
        20, description="Frequency for subsampling trajectory frames."
    )
    n_frames: int = pydantic.Field(
        1, description="Number of last frames to load from trajectory for scaling."
    )


class TrainingConfig(BaseModel):
    """
    Configuration for parameter training.

    Attributes
    ----------
    learning_rate : float
        Learning rate for optimizer.
    n_epochs : int
        Number of training epochs.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : Literal["mean", "min", "none", "infinite"]
        Reference energy mode for relative energies.
    normalize : bool
        Whether to normalize losses by number of conformers/atoms.
    """

    learning_rate: float = pydantic.Field(
        0.01, description="Learning rate for optimizer."
    )
    n_epochs: int = pydantic.Field(100, description="Number of training epochs.")
    energy_weight: float = pydantic.Field(
        1.0, description="Weight for energy loss term."
    )
    force_weight: float = pydantic.Field(1.0, description="Weight for force loss term.")
    reference: Literal["mean", "min", "none", "infinite"] = pydantic.Field(
        "none", description="Reference energy mode for relative energies."
    )
    normalize: bool = pydantic.Field(
        True, description="Whether to normalize losses by number of conformers/atoms."
    )
    energy_cutoff: float | None = pydantic.Field(
        None, description="Energy cutoff in kcal/mol to filter high-energy conformers."
    )
    weighting_method: Literal["uniform", "boltzmann"] = pydantic.Field(
        "uniform", description="Method to weight conformers in loss function."
    )
    weighting_temperature: OpenMMQuantity[_KELVIN] = pydantic.Field(
        300.0 * _KELVIN,
        description="Temperature in Kelvin for Boltzmann weighting.",
    )
    conformer_batch_size: int = pydantic.Field(
        2,
        description=(
            "Number of conformers to process at once within each entry before "
            "accumulating gradients."
        ),
    )
    compute_forces: bool = pydantic.Field(
        True, description="Whether to compute forces."
    )
