"""Tests for scalej.train.run_training_loop."""

import copy
import logging

import numpy as np
import pytest
import torch

from scalej.train import _collect_losses, run_training_loop

_RNG = np.random.default_rng(42)
_N_ATOMS = 6
_N_CONFS = 4
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
_COORDS = _BASE_COORDS + _RNG.standard_normal((_N_CONFS, _N_ATOMS, 3)) * 0.1
_BOX = np.stack([np.eye(3) * 25.0] * _N_CONFS)
_ENERGIES = np.array([-100.0, -99.0, -98.0, -97.0])
_FORCES = _RNG.standard_normal((_N_CONFS, _N_ATOMS, 3)) * 0.5


@pytest.fixture(scope="module")
def train_dataset():
    import descent.targets.energy

    entry: descent.targets.energy.Entry = {
        "id": "water",
        "smiles": "water",
        "coords": torch.tensor(_COORDS, dtype=torch.float64),
        "energy": torch.tensor(_ENERGIES, dtype=torch.float64),
        "forces": torch.tensor(_FORCES, dtype=torch.float64),
        "box_vectors": torch.tensor(_BOX, dtype=torch.float64),
    }
    ds = descent.targets.energy.create_dataset([entry])
    ds.set_format("torch")
    return ds


@pytest.fixture(scope="module")
def train_topologies(water_system):
    tensor_system, _, _ = water_system
    return {"water": tensor_system}


@pytest.fixture
def train_trainable(water_system):
    import descent.train

    _, tensor_forcefield, _ = water_system
    tf = copy.deepcopy(tensor_forcefield)
    return descent.train.Trainable(
        force_field=tf,
        parameters={"vdW": descent.train.ParameterConfig(cols=["epsilon", "sigma"])},
        attributes={},
    )


@pytest.fixture
def real_closure(train_trainable, train_topologies, train_dataset):
    from scalej.targets.condensed import default_closure

    return default_closure(
        train_trainable,
        train_topologies,
        train_dataset,
        reference="mean",
        energy_weight=1.0,
        force_weight=1.0,
        normalize=True,
    )


def _make_closure(target: float = 0.0):
    def closure(params, compute_gradient, compute_hessian):
        loss = (params[0] - target) ** 2
        grad = None
        if compute_gradient:
            grad = torch.autograd.grad(loss, params, create_graph=False)[0].detach()
        return loss.detach(), grad, None

    return closure


@pytest.fixture()
def mock_trainable(mocker):
    trainable = mocker.Mock()
    trainable.clamp = mocker.Mock(side_effect=lambda data: data)
    return trainable


def test_returns_loss_list_of_correct_length(mock_trainable):
    params = torch.tensor([5.0], requires_grad=True)
    losses = run_training_loop(
        params, _make_closure(), mock_trainable, n_epochs=10, lr=0.1
    )
    assert len(losses) == 10


def test_loss_values_are_finite(mock_trainable):
    params = torch.tensor([5.0], requires_grad=True)
    losses = run_training_loop(
        params, _make_closure(), mock_trainable, n_epochs=20, lr=0.01
    )
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)


def test_clamp_false_skips_clamping(mock_trainable):
    params = torch.tensor([5.0], requires_grad=True)
    run_training_loop(
        params, _make_closure(), mock_trainable, n_epochs=5, lr=0.01, clamp=False
    )
    mock_trainable.clamp.assert_not_called()


def test_clamp_true_calls_clamp(mock_trainable):
    n_epochs = 4
    params = torch.tensor([5.0], requires_grad=True)
    run_training_loop(
        params, _make_closure(), mock_trainable, n_epochs=n_epochs, lr=0.01, clamp=True
    )
    assert mock_trainable.clamp.call_count == n_epochs


def test_log_every_emits_messages(mock_trainable, caplog):
    params = torch.tensor([5.0], requires_grad=True)
    with caplog.at_level(logging.INFO, logger="scalej.train"):
        run_training_loop(
            params, _make_closure(), mock_trainable, n_epochs=10, lr=0.01, log_every=5
        )
    assert len(caplog.records) >= 2


def test_last_losses_attribute_included_in_log(mock_trainable, caplog):
    def closure_with_meta(params, compute_gradient, compute_hessian):
        loss = params[0] ** 2
        grad = None
        if compute_gradient:
            grad = torch.autograd.grad(loss, params, create_graph=False)[0].detach()
        return loss.detach(), grad, None

    closure_with_meta.last_losses = {"energy": 0.5}
    closure_with_meta.last_grad_norms = {"energy": 0.1}

    params = torch.tensor([5.0], requires_grad=True)
    with caplog.at_level(logging.INFO, logger="scalej.train"):
        run_training_loop(
            params, closure_with_meta, mock_trainable, n_epochs=3, lr=0.01, log_every=1
        )
    full_log = " ".join(r.message for r in caplog.records)
    assert "energy" in full_log


def test_loss_decreases_with_real_data(real_closure, train_trainable):
    params = train_trainable.to_values().detach().requires_grad_(True)
    losses = run_training_loop(
        params, real_closure, train_trainable, n_epochs=20, lr=0.01
    )
    assert losses[-1] < losses[0]


def test_params_change_after_training(real_closure, train_trainable):
    params = train_trainable.to_values().detach().requires_grad_(True)
    params_before = params.detach().clone()
    run_training_loop(params, real_closure, train_trainable, n_epochs=5, lr=0.05)
    assert not torch.allclose(params.detach(), params_before)


def test_clamp_is_invoked_with_correct_shape(real_closure, train_trainable):
    params = train_trainable.to_values().detach().requires_grad_(True)
    n_params = params.shape[0]

    clamped_shapes = []
    original_clamp = train_trainable.clamp

    def tracking_clamp(data):
        clamped_shapes.append(data.shape)
        return original_clamp(data)

    train_trainable.clamp = tracking_clamp

    n_epochs = 5
    run_training_loop(
        params, real_closure, train_trainable, n_epochs=n_epochs, lr=0.01, clamp=True
    )
    assert len(clamped_shapes) == n_epochs
    assert all(s == torch.Size([n_params]) for s in clamped_shapes)


def test_all_losses_are_non_negative_with_real_data(real_closure, train_trainable):
    params = train_trainable.to_values().detach().requires_grad_(True)
    losses = run_training_loop(
        params, real_closure, train_trainable, n_epochs=10, lr=0.01
    )
    assert all(v >= 0.0 for v in losses)


def test_collect_losses_plain_closure():
    """A plain closure with last_losses returns them directly."""

    def closure(params, compute_gradient, compute_hessian):
        return torch.tensor(1.0), None, None

    closure.last_losses = {"energy": 0.5, "forces": 0.3}
    result = _collect_losses(closure)
    assert result == {"energy": 0.5, "forces": 0.3}


def test_collect_losses_no_attribute():
    """A closure without last_losses returns an empty dict."""

    def closure(params, compute_gradient, compute_hessian):
        return torch.tensor(1.0), None, None

    result = _collect_losses(closure)
    assert result == {}


def test_collect_losses_combined_closure():
    """_collect_losses surfaces per-sub-closure last_losses via combine_closures."""
    import descent.utils.loss

    def sub_a(params, compute_gradient, compute_hessian):
        loss = params.sum()
        return loss.detach(), None, None

    sub_a.last_losses = {"energy": 0.2, "forces": 0.1}

    def sub_b(params, compute_gradient, compute_hessian):
        loss = params.sum() * 2
        return loss.detach(), None, None

    sub_b.last_losses = {"energy": 0.4, "forces": 0.3}

    combined = descent.utils.loss.combine_closures(
        {"target_a": sub_a, "target_b": sub_b}
    )
    # Trigger one call so combine_closures populates its own last_losses.
    combined(torch.tensor([1.0]), compute_gradient=False, compute_hessian=False)

    result = _collect_losses(combined)
    assert "target_a/energy" in result
    assert "target_a/forces" in result
    assert "target_b/energy" in result
    assert "target_b/forces" in result
    assert result["target_a/energy"] == pytest.approx(0.2)
    assert result["target_b/forces"] == pytest.approx(0.3)


def test_collect_losses_combined_no_sub_losses():
    """When sub-closures have no last_losses, falls back to top-level keys."""
    import descent.utils.loss

    def sub(params, compute_gradient, compute_hessian):
        loss = params.sum()
        return loss.detach(), None, None

    combined = descent.utils.loss.combine_closures({"condensed": sub}, verbose=True)
    combined(torch.tensor([1.0]), compute_gradient=False, compute_hessian=False)

    result = _collect_losses(combined)
    assert "condensed" in result


def test_energy_force_breakdown_logged_with_real_data(
    real_closure, train_trainable, caplog
):
    """The condensed closure populates last_losses with energy and forces."""
    params = train_trainable.to_values().detach().requires_grad_(True)
    with caplog.at_level(logging.INFO, logger="scalej.train"):
        run_training_loop(
            params, real_closure, train_trainable, n_epochs=3, lr=0.01, log_every=1
        )
    full_log = " ".join(r.message for r in caplog.records)
    assert "energy" in full_log
    assert "forces" in full_log
