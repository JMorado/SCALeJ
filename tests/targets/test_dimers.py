"""Tests for scalej.targets.dimers."""

import pytest
import torch

from scalej.targets.dimers import default_closure


class TestDimerDefaultClosure:
    @pytest.fixture(autouse=True)
    def _setup(self, dimer_dataset, dimer_topologies, dimer_trainable):
        self.dataset = dimer_dataset
        self.topologies = dimer_topologies
        self.trainable = dimer_trainable
        self.params = dimer_trainable.to_values().detach().requires_grad_(True)

    def test_returns_callable(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        assert callable(closure)

    def test_closure_returns_three_elements(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        result = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert len(result) == 3

    def test_loss_is_scalar_non_negative(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        loss, _, _ = closure(self.params, compute_gradient=False, compute_hessian=False)
        assert loss.ndim == 0 or (loss.ndim == 1 and loss.shape[0] == 1)
        assert loss.item() >= 0.0

    def test_loss_is_detached(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        loss, _, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert not loss.requires_grad

    def test_gradient_shape(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        _, grad, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert grad is not None
        assert grad.shape == self.params.shape

    def test_no_gradient_when_not_requested(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        _, grad, _ = closure(self.params, compute_gradient=False, compute_hessian=False)
        assert grad is None

    def test_hessian_is_none(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        _, _, hess = closure(self.params, compute_gradient=False, compute_hessian=False)
        assert hess is None

    @pytest.mark.parametrize("reference", ["mean", "min", "infinite"])
    def test_reference_modes(self, reference):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset, reference=reference
        )
        loss, grad, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is not None

    def test_normalize_true_divides_by_variance(self):
        """Normalized loss should differ from raw (unless var == 1, which is unlikely)."""
        closure_norm = default_closure(
            self.trainable, self.topologies, self.dataset, normalize=True
        )
        closure_raw = default_closure(
            self.trainable, self.topologies, self.dataset, normalize=False
        )
        loss_norm, _, _ = closure_norm(self.params, False, False)
        loss_raw, _, _ = closure_raw(self.params, False, False)
        assert loss_norm.item() != pytest.approx(loss_raw.item(), rel=1e-3)

    def test_gradient_changes_with_params(self):
        closure = default_closure(self.trainable, self.topologies, self.dataset)
        _, grad_a, _ = closure(self.params, compute_gradient=True, compute_hessian=False)

        perturbed = self.params.detach().clone() + 0.1
        perturbed.requires_grad_(True)
        _, grad_b, _ = closure(perturbed, compute_gradient=True, compute_hessian=False)

        assert not torch.allclose(grad_a, grad_b)
