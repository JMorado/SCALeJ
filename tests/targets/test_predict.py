"""Tests for scalej.targets.predict."""

import numpy as np
import pytest
import torch

from scalej.targets.predict import predict_energies_forces

# 2 water molecules × 3 atoms each = 6 atoms
N_ATOMS = 6
N_CONFORMERS = 4

# Near-equilibrium geometry (angstroms) for two water molecules.
_BASE_COORDS = np.array(
    [
        [0.000, 0.000, 0.000],   # O1
        [0.960, 0.000, 0.000],   # H1a
        [-0.240, 0.926, 0.000],  # H1b
        [5.000, 5.000, 5.000],   # O2
        [5.960, 5.000, 5.000],   # H2a
        [4.760, 5.926, 5.000],   # H2b
    ]
)

# Four conformers with small perturbations (~0.05 Å) around the base geometry.
COORDS = np.array([
    _BASE_COORDS + np.array([[0.02, -0.01, 0.03], [-0.01, 0.02, -0.01],
                              [0.01, -0.02, 0.02], [0.03, 0.01, -0.02],
                              [-0.02, 0.01, 0.01], [0.01, -0.01, -0.03]]),
    _BASE_COORDS + np.array([[-0.03, 0.02, -0.01], [0.02, -0.01, 0.02],
                              [-0.01, 0.03, -0.01], [-0.02, -0.01, 0.03],
                              [0.01, -0.02, -0.02], [-0.01, 0.02, 0.01]]),
    _BASE_COORDS + np.array([[0.01, 0.01, -0.02], [-0.02, -0.01, 0.01],
                              [0.03, 0.01, 0.01], [0.01, -0.03, 0.01],
                              [-0.01, 0.02, -0.01], [0.02, -0.02, 0.02]]),
    _BASE_COORDS + np.array([[-0.01, 0.03, 0.01], [0.01, -0.02, -0.02],
                              [-0.02, 0.01, 0.03], [0.02, 0.02, -0.01],
                              [0.01, -0.01, 0.02], [-0.03, 0.01, -0.01]]),
])

# 25 Å cubic box (must be > 2 × vdW cutoff of 9 Å).
BOX = np.stack([np.eye(3) * 25.0] * N_CONFORMERS)

# Typical condensed-phase energies (kcal/mol) ramping from -100 to -97.
ENERGIES = np.array([-100.0, -99.0, -98.0, -97.0])

# Small representative forces (kcal/mol/Å) for each conformer.
FORCES = np.array([
    [[0.50, -0.30, 0.10], [0.20, 0.40, -0.60], [-0.70, 0.10, 0.50],
     [0.30, -0.10, 0.20], [0.10, 0.50, -0.30], [-0.40, 0.20, 0.10]],
    [[0.60, -0.20, 0.40], [0.30, 0.30, -0.50], [-0.90, 0.10, 0.10],
     [0.40, -0.40, 0.30], [0.20, 0.60, -0.40], [-0.60, 0.30, 0.30]],
    [[0.40, -0.10, 0.20], [0.10, 0.50, -0.30], [-0.50, 0.20, 0.30],
     [0.20, -0.20, 0.10], [0.30, 0.40, -0.20], [-0.30, 0.10, 0.20]],
    [[0.55, -0.25, 0.15], [0.25, 0.35, -0.55], [-0.80, 0.15, 0.40],
     [0.35, -0.25, 0.25], [0.15, 0.55, -0.35], [-0.50, 0.25, 0.20]],
])


@pytest.fixture(scope="module")
def _water_system():
    from scalej.simulation.systems import create_system_from_smiles

    return create_system_from_smiles(smiles_list=["O"], nmol_list=[2])


@pytest.fixture(scope="module")
def predict_dataset():
    import descent.targets.energy

    entry: descent.targets.energy.Entry = {
        "id": "water",
        "smiles": "water",
        "coords": torch.tensor(COORDS, dtype=torch.float64),
        "energy": torch.tensor(ENERGIES, dtype=torch.float64),
        "forces": torch.tensor(FORCES, dtype=torch.float64),
        "box_vectors": torch.tensor(BOX, dtype=torch.float64),
    }
    ds = descent.targets.energy.create_dataset([entry])
    ds.set_format("torch")
    return ds


@pytest.fixture(scope="module")
def predict_systems(_water_system):
    tensor_system, _, _ = _water_system
    return {"water": tensor_system}


@pytest.fixture(scope="module")
def predict_ff(_water_system):
    _, tensor_forcefield, _ = _water_system
    return tensor_forcefield


class TestPredictEnergiesForces:
    def test_return_shapes(self, predict_dataset, predict_ff, predict_systems):
        e_ref, e_pred, f_ref, f_pred, masks, ids = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="none"
        )
        assert e_ref.shape == (N_CONFORMERS,)
        assert e_pred.shape == (N_CONFORMERS,)
        assert isinstance(f_ref, list)
        assert isinstance(f_pred, list)
        assert f_ref[0].shape[0] == N_CONFORMERS
        assert f_pred[0].shape[0] == N_CONFORMERS
        assert len(masks) == 1
        assert len(ids) == 1

    def test_entry_ids_match(self, predict_dataset, predict_ff, predict_systems):
        *_, ids = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="none"
        )
        assert ids == ["water"]

    def test_mask_indices_no_cutoff(
        self, predict_dataset, predict_ff, predict_systems
    ):
        *_, masks, _ = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="none"
        )
        assert masks[0] == list(range(N_CONFORMERS))

    @pytest.mark.parametrize("reference", ["mean", "min", "none", "infinite"])
    def test_reference_modes(
        self, predict_dataset, predict_ff, predict_systems, reference
    ):
        e_ref, e_pred, f_ref, f_pred, masks, ids = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference=reference
        )
        assert e_ref.shape[0] > 0
        assert e_pred.shape[0] > 0

    def test_relative_energies_mean(
        self, predict_dataset, predict_ff, predict_systems
    ):
        e_ref, e_pred, *_ = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="mean"
        )
        # Relative energies should sum to ~0 for mean reference
        assert e_ref.sum().item() == pytest.approx(0.0, abs=1e-4)

    def test_relative_energies_none(
        self, predict_dataset, predict_ff, predict_systems
    ):
        e_ref, e_pred, *_ = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="none"
        )
        # Absolute energies - no subtraction of reference value
        # Energy sum should NOT be zero (absolute values)
        assert e_ref.abs().sum().item() > 0.0

    def test_energy_cutoff_filters(
        self, predict_dataset, predict_ff, predict_systems
    ):
        # Very tight cutoff: only keep conformers within 0.1 kcal/mol of minimum
        e_ref, e_pred, *_, masks, _ = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems,
            reference="none",
            energy_cutoff=0.1,
        )
        # Should filter out most conformers
        assert len(masks[0]) < N_CONFORMERS

    def test_invalid_reference_raises(
        self, predict_dataset, predict_ff, predict_systems
    ):
        with pytest.raises(ValueError, match="Invalid reference"):
            predict_energies_forces(
                predict_dataset, predict_ff, predict_systems, reference="bad"
            )

    def test_forces_finite(self, predict_dataset, predict_ff, predict_systems):
        _, _, f_ref, f_pred, *_ = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="none"
        )
        assert all(torch.isfinite(f).all() for f in f_ref)
        assert all(torch.isfinite(f).all() for f in f_pred)

    def test_energies_finite(self, predict_dataset, predict_ff, predict_systems):
        e_ref, e_pred, *_ = predict_energies_forces(
            predict_dataset, predict_ff, predict_systems, reference="none"
        )
        assert torch.isfinite(e_ref).all()
        assert torch.isfinite(e_pred).all()
