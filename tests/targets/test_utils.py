"""Tests for scalej.targets.utils."""

import pytest
import torch

from scalej.targets.utils import normalize_closure_weights


def _make_closure(loss_value: float):
    """Return a closure that always reports a fixed scalar loss."""

    def closure(params, compute_gradient, compute_hessian):
        return torch.tensor(loss_value), None, None

    return closure


def test_unit_weights_invert_loss():
    closures = {"a": _make_closure(2.0), "b": _make_closure(5.0)}
    result = normalize_closure_weights(closures, torch.zeros(1))
    assert result["a"] == pytest.approx(0.5)
    assert result["b"] == pytest.approx(0.2)


def test_user_weights_are_scaled_by_loss():
    closures = {"a": _make_closure(2.0), "b": _make_closure(5.0)}
    result = normalize_closure_weights(
        closures, torch.zeros(1), weights={"a": 4.0, "b": 5.0}
    )
    assert result["a"] == pytest.approx(2.0)
    assert result["b"] == pytest.approx(1.0)


def test_zero_initial_loss_gives_user_weight():
    closures = {"x": _make_closure(0.0)}
    result = normalize_closure_weights(closures, torch.zeros(1), weights={"x": 3.0})
    assert result["x"] == pytest.approx(3.0)


def test_missing_user_weight_defaults_to_one():
    closures = {"a": _make_closure(4.0), "b": _make_closure(2.0)}
    result = normalize_closure_weights(closures, torch.zeros(1), weights={"a": 2.0})
    assert result["a"] == pytest.approx(2.0 / 4.0)
    assert result["b"] == pytest.approx(1.0 / 2.0)


def test_returns_all_keys():
    closures = {"energy": _make_closure(1.0), "forces": _make_closure(2.0)}
    result = normalize_closure_weights(closures, torch.zeros(1))
    assert set(result.keys()) == {"energy", "forces"}
