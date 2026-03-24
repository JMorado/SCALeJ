"""Tests for scalej.targets.condensed_ddp."""

import copy

import numpy as np
import pytest
import torch

# DDP tests require CUDA GPUs. Skip if not available.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

N_ATOMS = 6
N_CONFORMERS = 4

# Near-equilibrium geometry (angstroms) for two water molecules.
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

# Four conformers with small perturbations (~0.05 Å).
COORDS = np.array(
    [
        _BASE_COORDS
        + np.array(
            [
                [0.02, -0.01, 0.03],
                [-0.01, 0.02, -0.01],
                [0.01, -0.02, 0.02],
                [0.03, 0.01, -0.02],
                [-0.02, 0.01, 0.01],
                [0.01, -0.01, -0.03],
            ]
        ),
        _BASE_COORDS
        + np.array(
            [
                [-0.03, 0.02, -0.01],
                [0.02, -0.01, 0.02],
                [-0.01, 0.03, -0.01],
                [-0.02, -0.01, 0.03],
                [0.01, -0.02, -0.02],
                [-0.01, 0.02, 0.01],
            ]
        ),
        _BASE_COORDS
        + np.array(
            [
                [0.01, 0.01, -0.02],
                [-0.02, -0.01, 0.01],
                [0.03, 0.01, 0.01],
                [0.01, -0.03, 0.01],
                [-0.01, 0.02, -0.01],
                [0.02, -0.02, 0.02],
            ]
        ),
        _BASE_COORDS
        + np.array(
            [
                [-0.01, 0.03, 0.01],
                [0.01, -0.02, -0.02],
                [-0.02, 0.01, 0.03],
                [0.02, 0.02, -0.01],
                [0.01, -0.01, 0.02],
                [-0.03, 0.01, -0.01],
            ]
        ),
    ]
)

BOX = np.stack([np.eye(3) * 25.0] * N_CONFORMERS)
ENERGIES = np.array([-100.0, -99.0, -98.0, -97.0])
FORCES = np.array(
    [
        [
            [0.50, -0.30, 0.10],
            [0.20, 0.40, -0.60],
            [-0.70, 0.10, 0.50],
            [0.30, -0.10, 0.20],
            [0.10, 0.50, -0.30],
            [-0.40, 0.20, 0.10],
        ],
        [
            [0.60, -0.20, 0.40],
            [0.30, 0.30, -0.50],
            [-0.90, 0.10, 0.10],
            [0.40, -0.40, 0.30],
            [0.20, 0.60, -0.40],
            [-0.60, 0.30, 0.30],
        ],
        [
            [0.40, -0.10, 0.20],
            [0.10, 0.50, -0.30],
            [-0.50, 0.20, 0.30],
            [0.20, -0.20, 0.10],
            [0.30, 0.40, -0.20],
            [-0.30, 0.10, 0.20],
        ],
        [
            [0.55, -0.25, 0.15],
            [0.25, 0.35, -0.55],
            [-0.80, 0.15, 0.40],
            [0.35, -0.25, 0.25],
            [0.15, 0.55, -0.35],
            [-0.50, 0.25, 0.20],
        ],
    ]
)


@pytest.fixture(scope="module")
def _water_system():
    from scalej.simulation.systems import create_system_from_smiles

    return create_system_from_smiles(smiles_list=["O"], nmol_list=[2])


@pytest.fixture(scope="module")
def ddp_dataset():
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
def ddp_topologies(_water_system):
    tensor_system, _, _ = _water_system
    return {"water": tensor_system}


@pytest.fixture
def ddp_trainable(_water_system):
    import descent.train

    _, tensor_forcefield, _ = _water_system
    tf = copy.deepcopy(tensor_forcefield)
    return descent.train.Trainable(
        force_field=tf,
        parameters={"vdW": descent.train.ParameterConfig(cols=["epsilon", "sigma"])},
        attributes={},
    )


class TestDdpClosure:
    def test_no_cuda_raises(self, mocker, ddp_trainable, ddp_topologies, ddp_dataset):
        from scalej.targets.condensed_ddp import ddp_closure

        mocker.patch("torch.cuda.device_count", return_value=0)
        with pytest.raises(RuntimeError, match="at least one CUDA GPU"):
            ddp_closure(ddp_trainable, ddp_topologies, ddp_dataset, n_gpus=1)

    def test_too_many_gpus_raises(self, ddp_trainable, ddp_topologies, ddp_dataset):
        from scalej.targets.condensed_ddp import ddp_closure

        n_avail = torch.cuda.device_count()
        with pytest.raises(ValueError, match="Requested n_gpus"):
            ddp_closure(ddp_trainable, ddp_topologies, ddp_dataset, n_gpus=n_avail + 1)

    def test_closure_returns_tuple(self, ddp_trainable, ddp_topologies, ddp_dataset):
        from scalej.targets.condensed_ddp import ddp_closure

        closure = ddp_closure(ddp_trainable, ddp_topologies, ddp_dataset, n_gpus=1)
        params = ddp_trainable.to_values().detach().requires_grad_(True)
        result = closure(params, compute_gradient=True, compute_hessian=False)

        assert len(result) == 3
        loss, grad, hess = result
        assert loss.ndim == 0 or (loss.ndim == 1 and loss.shape[0] == 1)
        assert loss.item() >= 0.0
        assert grad is not None
        assert grad.shape == params.shape
        assert hess is None

    def test_hessian_raises(self, ddp_trainable, ddp_topologies, ddp_dataset):
        from scalej.targets.condensed_ddp import ddp_closure

        closure = ddp_closure(ddp_trainable, ddp_topologies, ddp_dataset, n_gpus=1)
        params = ddp_trainable.to_values().detach().requires_grad_(True)

        with pytest.raises(NotImplementedError, match="Hessian"):
            closure(params, compute_gradient=False, compute_hessian=True)

    def test_no_gradient(self, ddp_trainable, ddp_topologies, ddp_dataset):
        from scalej.targets.condensed_ddp import ddp_closure

        closure = ddp_closure(ddp_trainable, ddp_topologies, ddp_dataset, n_gpus=1)
        params = ddp_trainable.to_values().detach().requires_grad_(True)
        loss, grad, _ = closure(params, compute_gradient=False, compute_hessian=False)

        assert loss.item() >= 0.0
        assert grad is None

    def test_loss_finite(self, ddp_trainable, ddp_topologies, ddp_dataset):
        from scalej.targets.condensed_ddp import ddp_closure

        closure = ddp_closure(ddp_trainable, ddp_topologies, ddp_dataset, n_gpus=1)
        params = ddp_trainable.to_values().detach().requires_grad_(True)
        loss, grad, _ = closure(params, compute_gradient=True, compute_hessian=False)
        assert torch.isfinite(loss)
        assert torch.isfinite(grad).all()
