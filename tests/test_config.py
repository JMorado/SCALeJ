"""Tests for scalej.config Pydantic models."""

import openmm.unit
import pytest
from pydantic import ValidationError

from scalej.config import (
    MoleculeComponent,
    ScalingConfig,
    SimulationConfig,
    SystemConfig,
    TrainingConfig,
)


class TestSimulationConfig:
    def test_defaults(self):
        config = SimulationConfig()
        assert config.temperature == 300 * openmm.unit.kelvin
        assert config.pressure == 1.0 * openmm.unit.atmosphere
        assert config.timestep == 1.0 * openmm.unit.femtosecond
        assert config.friction_coeff == 1.0 / openmm.unit.picosecond
        assert config.n_minimization_steps == 0
        assert config.n_equilibration_nvt_steps == 50_000
        assert config.n_equilibration_npt_steps == 50_000
        assert config.n_production_steps == 1_000_000
        assert config.n_mlp_steps == 100
        assert config.mlp_device == "cpu"
        assert config.platform == "CPU"
        assert config.report_interval == 1000

    def test_invalid_units(self):
        with pytest.raises(ValidationError):
            SimulationConfig(temperature=300 * openmm.unit.meter)


class TestScalingConfig:
    def test_defaults(self):
        config = ScalingConfig()
        assert config.close_range == (0.75, 0.9, 5)
        assert config.equilibrium_range == (0.9, 1.1, 15)
        assert config.long_range == (1.1, 2.0, 12)
        assert config.subsample_frequency == 20
        assert config.n_frames == 1


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.learning_rate == pytest.approx(0.01)
        assert config.n_epochs == 100
        assert config.device == "cuda"
        assert config.energy_weight == pytest.approx(1.0)
        assert config.force_weight == pytest.approx(1.0)
        assert config.reference == "none"
        assert config.normalize is True
        assert config.energy_cutoff is None
        assert config.weighting_method == "uniform"
        assert config.weighting_temperature == 300.0 * openmm.unit.kelvin

    def test_literal_validation(self):
        with pytest.raises(ValidationError):
            TrainingConfig(reference="invalid")
        with pytest.raises(ValidationError):
            TrainingConfig(weighting_method="invalid")


class TestSystemSetup:
    def test_molecule_component(self):
        comp = MoleculeComponent(smiles="C", nmol=10)
        assert comp.smiles == "C"
        assert comp.nmol == 10

        with pytest.raises(ValidationError):
            MoleculeComponent(smiles="C")  # Missing nmol

    def test_system_config(self):
        comp = MoleculeComponent(smiles="C", nmol=10)
        config = SystemConfig(name="methane", components=[comp])
        assert config.name == "methane"
        assert len(config.components) == 1
        assert config.weight == pytest.approx(1.0)
        assert config.trajectory_path is None