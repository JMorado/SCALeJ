"""Tests for scalej.types dataclasses."""

import pytest

from scalej.types import (
    BenchmarkResult,
    EnergyForceResult,
    EvaluationMetrics,
    ReferenceMode,
    ScalingResult,
    TrajectoryFrames,
    WeightingMethod,
)


class TestScalingResult:
    def test_fields_stored(self):
        coords = [[0.0] * 9]
        box_vectors = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        scale_factors = [0.9, 1.0, 1.1]
        r = ScalingResult(
            coords=coords, box_vectors=box_vectors, scale_factors=scale_factors
        )
        assert r.coords is coords
        assert r.box_vectors is box_vectors
        assert r.scale_factors == pytest.approx([0.9, 1.0, 1.1])


class TestEnergyForceResult:
    def test_fields_stored(self):
        import numpy as np

        energies = np.array([-1.0, -2.0])
        forces = np.zeros((2, 3, 3))
        r = EnergyForceResult(energies=energies, forces=forces)
        assert list(r.energies) == pytest.approx([-1.0, -2.0])
        assert r.forces.shape == (2, 3, 3)
        assert r.forces.sum() == pytest.approx(0.0)


class TestEvaluationMetrics:
    def _make(self):
        return EvaluationMetrics(
            energy_mae=0.1,
            energy_rmse=0.2,
            energy_r2=0.9,
            forces_mae=0.3,
            forces_rmse=0.4,
            forces_r2=0.8,
        )

    def test_fields_stored(self):
        m = self._make()
        assert m.energy_mae == pytest.approx(0.1)
        assert m.energy_rmse == pytest.approx(0.2)
        assert m.energy_r2 == pytest.approx(0.9)
        assert m.forces_mae == pytest.approx(0.3)
        assert m.forces_rmse == pytest.approx(0.4)
        assert m.forces_r2 == pytest.approx(0.8)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        assert set(d.keys()) == {"energy", "forces"}
        assert set(d["energy"].keys()) == {"mae", "rmse", "r2"}
        assert set(d["forces"].keys()) == {"mae", "rmse", "r2"}

    def test_to_dict_values(self):
        d = self._make().to_dict()
        assert d["energy"]["mae"] == pytest.approx(0.1)
        assert d["energy"]["rmse"] == pytest.approx(0.2)
        assert d["energy"]["r2"] == pytest.approx(0.9)
        assert d["forces"]["mae"] == pytest.approx(0.3)
        assert d["forces"]["rmse"] == pytest.approx(0.4)
        assert d["forces"]["r2"] == pytest.approx(0.8)


class TestBenchmarkResult:
    def test_all_none_by_default(self):
        r = BenchmarkResult()
        assert r.density_ref is None
        assert r.density_pred is None
        assert r.density_std is None
        assert r.hvap_ref is None
        assert r.hvap_pred is None
        assert r.hvap_std is None

    def test_partial_fields(self):
        r = BenchmarkResult(density_ref=0.997, density_pred=1.001)
        assert r.density_ref == pytest.approx(0.997)
        assert r.density_pred == pytest.approx(1.001)
        assert r.density_std is None

    def test_all_fields(self):
        r = BenchmarkResult(
            density_ref=0.997,
            density_pred=1.001,
            density_std=0.002,
            hvap_ref=10.5,
            hvap_pred=10.3,
            hvap_std=0.1,
        )
        assert r.hvap_ref == pytest.approx(10.5)
        assert r.hvap_std == pytest.approx(0.1)


class TestTrajectoryFrames:
    def test_fields_stored(self):
        import numpy as np

        coords = np.zeros((5, 6, 3))
        box_vectors = np.stack([np.eye(3)] * 5)
        r = TrajectoryFrames(coords=coords, box_vectors=box_vectors, n_frames=5)
        assert r.n_frames == 5
        assert r.coords.shape == (5, 6, 3)
        assert r.coords.sum() == pytest.approx(0.0)
        assert r.box_vectors.shape == (5, 3, 3)
        assert r.box_vectors[0].diagonal().tolist() == pytest.approx([1.0, 1.0, 1.0])


class TestTypeAliases:
    def test_reference_mode_accepts_valid_values(self):
        valid: list[ReferenceMode] = ["mean", "min", "none", "infinite"]
        assert len(valid) == 4

    def test_weighting_method_accepts_valid_values(self):
        valid: list[WeightingMethod] = ["uniform", "boltzmann", "mixed"]
        assert len(valid) == 3
