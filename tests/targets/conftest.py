"""Shared fixtures for targets tests."""

import copy

import numpy as np
import pytest

# 2 water molecules × 3 atoms each = 6 atoms
N_ATOMS = 6
N_CONFORMERS = 4

_RNG = np.random.default_rng(42)

# Near-equilibrium water dimer geometry (angstroms).
_BASE_COORDS = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.960, 0.000, 0.000],
        [-0.240, 0.926, 0.000],
        [5.000, 5.000, 5.000],
        [5.960, 5.000, 5.000],
        [4.760, 5.926, 5.000],
    ]
)

# Slightly perturbed conformers so variance is non-zero.
COORDS = _BASE_COORDS + _RNG.standard_normal((N_CONFORMERS, N_ATOMS, 3)) * 0.1
# Box must be > 2 × vdW cutoff (9 Å) → use 25 Å.
BOX = np.stack([np.eye(3) * 25.0] * N_CONFORMERS)
ENERGIES = np.array([-100.0, -99.0, -98.0, -97.0])
FORCES = _RNG.standard_normal((N_CONFORMERS, N_ATOMS, 3)) * 0.5


@pytest.fixture(scope="module")
def water_system():
    """TensorSystem, TensorForceField, and topologies for a 2-water system."""
    from scalej.simulation.systems import create_system_from_smiles

    tensor_system, tensor_forcefield, topologies = create_system_from_smiles(
        smiles_list=["O"],
        nmol_list=[2],
    )
    return tensor_system, tensor_forcefield, topologies


@pytest.fixture(scope="module")
def condensed_dataset():
    """descent-style energy dataset with one system entry (4 conformers)."""
    import descent.targets.energy
    import torch

    entry: descent.targets.energy.Entry = {
        "id": "water",
        "smiles": "water",
        "coords": torch.tensor(COORDS, dtype=torch.float64),
        "energy": torch.tensor(ENERGIES, dtype=torch.float64),
        "forces": torch.tensor(FORCES, dtype=torch.float64),
        "box_vectors": torch.tensor(BOX, dtype=torch.float64),
    }
    dataset = descent.targets.energy.create_dataset([entry])
    dataset.set_format("torch")
    return dataset


@pytest.fixture(scope="module")
def condensed_topologies(water_system):
    """Topologies dict keyed by the dataset entry ``smiles`` field."""
    tensor_system, _, _ = water_system
    return {"water": tensor_system}


@pytest.fixture
def condensed_trainable(water_system):
    """Fresh trainable (function-scoped to avoid parameter bleed)."""
    import descent.train

    _, tensor_forcefield, _ = water_system
    tf = copy.deepcopy(tensor_forcefield)
    return descent.train.Trainable(
        force_field=tf,
        parameters={"vdW": descent.train.ParameterConfig(cols=["epsilon", "sigma"])},
        attributes={},
    )


@pytest.fixture(scope="module")
def dimer_dataset(water_system):
    """Minimal dimer dataset with 2 conformers of a water–water dimer."""
    import descent.targets.dimers
    import torch

    _, tensor_forcefield, topologies = water_system
    topo = topologies[0]
    n_atoms = topo.n_atoms * 2  # dimer = two copies

    rng = np.random.default_rng(123)
    n_confs = 2

    base_a = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.960, 0.000, 0.000],
            [-0.240, 0.926, 0.000],
        ]
    )
    base_b = np.array(
        [
            [4.000, 4.000, 4.000],
            [4.960, 4.000, 4.000],
            [3.760, 4.926, 4.000],
        ]
    )
    base = np.concatenate([base_a, base_b], axis=0)
    coords = base + rng.standard_normal((n_confs, n_atoms, 3)) * 0.05

    # Compute SMILES the same way descent uses them (mapped).
    smiles = "O"

    entry: descent.targets.dimers.Dimer = {
        "smiles_a": smiles,
        "smiles_b": smiles,
        "coords": torch.tensor(coords, dtype=torch.float64),
        "energy": torch.tensor([-5.0, -4.5], dtype=torch.float64),
        "source": "test",
    }
    dataset = descent.targets.dimers.create_dataset([entry])
    dataset.set_format("torch")
    return dataset


@pytest.fixture(scope="module")
def dimer_topologies(water_system):
    """Topology dict keyed by monomer SMILES for dimer prediction."""
    _, _, topologies = water_system
    return {"O": topologies[0]}


@pytest.fixture
def dimer_trainable(water_system):
    """Fresh trainable for dimer tests."""
    import descent.train

    _, tensor_forcefield, _ = water_system
    tf = copy.deepcopy(tensor_forcefield)
    return descent.train.Trainable(
        force_field=tf,
        parameters={"vdW": descent.train.ParameterConfig(cols=["epsilon", "sigma"])},
        attributes={},
    )
