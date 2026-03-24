"""Tests for scalej.targets.dimers."""

import pytest
import torch

from scalej.targets.dimers import _get_reference, default_closure


@pytest.mark.parametrize(
    "mode, expected_val, expected_idx",
    [
        ("mean", -1.625, None),
        ("min", -3.0, 0),
        ("infinite", -0.5, -1),
    ],
)
def test_get_reference_known_modes(mode, expected_val, expected_idx):
    energies = torch.tensor([-3.0, -1.0, -2.0, -0.5], dtype=torch.float64)
    val, idx = _get_reference(energies, mode)
    assert float(val) == pytest.approx(expected_val)
    assert idx == expected_idx


def test_get_reference_none_mode():
    energies = torch.tensor([-3.0, -1.0, -2.0, -0.5], dtype=torch.float64)
    val, idx = _get_reference(energies, "none")
    assert float(val) == pytest.approx(0.0)
    assert idx is None


def test_get_reference_invalid_mode():
    energies = torch.tensor([-3.0, -1.0, -2.0, -0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="Invalid reference mode"):
        _get_reference(energies, "bogus")


def test_get_reference_zero_variance_guard():
    """Constant energies -> var==0 must not produce NaN loss."""
    energies = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    ref_val, _ = _get_reference(energies, "mean")
    y_ref_rel = (energies - ref_val).detach()
    energy_var = torch.var(y_ref_rel)
    if energy_var == 0:
        energy_var = energy_var.new_ones(1).squeeze()
    loss = (y_ref_rel**2).mean() / energy_var
    assert torch.isfinite(loss)


class TestDimerDefaultClosure:
    def test_closure_interface(self, dimer_dataset, dimer_topologies, dimer_trainable):
        params = dimer_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(dimer_trainable, dimer_topologies, dimer_dataset)
        assert callable(closure)

        result = closure(params, compute_gradient=True, compute_hessian=False)
        assert len(result) == 3

    def test_closure_loss_properties(
        self, dimer_dataset, dimer_topologies, dimer_trainable
    ):
        params = dimer_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(dimer_trainable, dimer_topologies, dimer_dataset)
        loss, _, hess = closure(params, compute_gradient=True, compute_hessian=False)

        assert loss.ndim == 0 or (loss.ndim == 1 and loss.shape[0] == 1)
        assert loss.item() >= 0.0
        assert not loss.requires_grad
        assert hess is None

    @pytest.mark.parametrize("compute_gradient", [True, False])
    def test_gradient_behavior(
        self, dimer_dataset, dimer_topologies, dimer_trainable, compute_gradient
    ):
        params = dimer_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(dimer_trainable, dimer_topologies, dimer_dataset)
        _, grad, _ = closure(
            params, compute_gradient=compute_gradient, compute_hessian=False
        )
        if compute_gradient:
            assert grad is not None
            assert grad.shape == params.shape
        else:
            assert grad is None

    @pytest.mark.parametrize("reference", ["mean", "min", "infinite"])
    def test_reference_modes(
        self, dimer_dataset, dimer_topologies, dimer_trainable, reference
    ):
        params = dimer_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            dimer_trainable, dimer_topologies, dimer_dataset, reference=reference
        )
        loss, grad, _ = closure(params, compute_gradient=True, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is not None

    def test_normalize_true_divides_by_variance(
        self, dimer_dataset, dimer_topologies, dimer_trainable
    ):
        params = dimer_trainable.to_values().detach().requires_grad_(True)
        closure_norm = default_closure(
            dimer_trainable, dimer_topologies, dimer_dataset, normalize=True
        )
        closure_raw = default_closure(
            dimer_trainable, dimer_topologies, dimer_dataset, normalize=False
        )
        loss_norm, _, _ = closure_norm(params, False, False)
        loss_raw, _, _ = closure_raw(params, False, False)
        assert loss_norm.item() != pytest.approx(loss_raw.item(), rel=1e-3)

    def test_gradient_changes_with_params(
        self, dimer_dataset, dimer_topologies, dimer_trainable
    ):
        params = dimer_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(dimer_trainable, dimer_topologies, dimer_dataset)
        _, grad_a, _ = closure(params, compute_gradient=True, compute_hessian=False)

        perturbed = params.detach().clone() + 0.1
        perturbed.requires_grad_(True)
        _, grad_b, _ = closure(perturbed, compute_gradient=True, compute_hessian=False)

        assert not torch.allclose(grad_a, grad_b)
