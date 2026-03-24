"""Tests for scalej.targets.condensed."""

import pytest
import torch

from scalej.targets.condensed import (
    _compute_batch_loss,
    _compute_reference_prediction,
    _prepare_entry_data,
    _process_entry,
    default_closure,
)

# ---------------------------------------------------------------------------
# _prepare_entry_data
# ---------------------------------------------------------------------------

class TestPrepareEntryData:
    @pytest.fixture(autouse=True)
    def _setup(self, condensed_dataset, condensed_topologies):
        self.entry = condensed_dataset[0]
        self.topology = condensed_topologies["water"]

    @pytest.mark.parametrize("reference", ["mean", "min", "none", "infinite"])
    def test_returns_correct_tuple_length(self, reference):
        result = _prepare_entry_data(
            self.entry, self.topology, reference, normalize=True, device=torch.device("cpu")
        )
        assert len(result) == 12

    def test_weights_sum_to_one(self):
        *_, weights, _, _, _, _ = _prepare_entry_data(
            self.entry, self.topology, "mean", normalize=True, device=torch.device("cpu")
        )
        assert weights.sum().item() == pytest.approx(1.0)

    def test_energy_var_positive_when_normalizing(self):
        *_, energy_var, forces_var, _, _ = _prepare_entry_data(
            self.entry, self.topology, "mean", normalize=True, device=torch.device("cpu")
        )
        assert energy_var.item() > 0
        assert forces_var.item() > 0

    def test_variances_one_without_normalization(self):
        *_, energy_var, forces_var, _, _ = _prepare_entry_data(
            self.entry, self.topology, "mean", normalize=False, device=torch.device("cpu")
        )
        assert energy_var.item() == pytest.approx(1.0)
        assert forces_var.item() == pytest.approx(1.0)

    def test_delta_energy_mean_ref_sums_near_zero(self):
        """With reference='mean', the relative energies should sum to ~0."""
        _, _, _, _, _, delta_energy_ref, *_ = _prepare_entry_data(
            self.entry, self.topology, "mean", normalize=True, device=torch.device("cpu")
        )
        assert delta_energy_ref.sum().item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _compute_reference_prediction
# ---------------------------------------------------------------------------

class TestComputeReferencePrediction:
    @pytest.fixture(autouse=True)
    def _setup(self, condensed_dataset, condensed_topologies, condensed_trainable):
        self.entry = condensed_dataset[0]
        self.topology = condensed_topologies["water"]
        self.trainable = condensed_trainable
        self.params = condensed_trainable.to_values().detach().requires_grad_(True)

        # Prepare data to get coords, etc.
        (
            _, _, self.coords_all, self.box_vectors_all,
            self.n_mols, _, self.ref_idx, _, _, _,
            self.ref_coords, self.ref_box_vectors,
        ) = _prepare_entry_data(
            self.entry, self.topology, "min", normalize=True, device=torch.device("cpu")
        )

    def test_returns_detached_energy(self):
        e, _ = _compute_reference_prediction(
            self.topology, self.trainable, self.params,
            self.ref_coords, self.ref_box_vectors,
            self.n_mols, compute_gradient=False,
        )
        assert not e.requires_grad
        assert e.ndim == 0  # scalar

    def test_gradient_computed_when_requested(self):
        e, grad = _compute_reference_prediction(
            self.topology, self.trainable, self.params,
            self.ref_coords, self.ref_box_vectors,
            self.n_mols, compute_gradient=True,
        )
        assert grad is not None
        assert grad.shape == self.params.shape
        assert not grad.requires_grad

    def test_no_gradient_when_not_requested(self):
        _, grad = _compute_reference_prediction(
            self.topology, self.trainable, self.params,
            self.ref_coords, self.ref_box_vectors,
            self.n_mols, compute_gradient=False,
        )
        assert grad is None


# ---------------------------------------------------------------------------
# _compute_batch_loss
# ---------------------------------------------------------------------------

class TestComputeBatchLoss:
    @pytest.fixture(autouse=True)
    def _setup(self, condensed_dataset, condensed_topologies, condensed_trainable):
        self.topology = condensed_topologies["water"]
        self.trainable = condensed_trainable
        self.params = condensed_trainable.to_values().detach().requires_grad_(True)

        (
            _, self.forces_all, self.coords_all, self.box_vectors_all,
            self.n_mols, self.delta_energy_ref, self.ref_idx,
            self.weights, self.energy_var, self.forces_var,
            _, _,
        ) = _prepare_entry_data(
            condensed_dataset[0], self.topology, "min",
            normalize=True, device=torch.device("cpu"),
        )
        self.n_confs = len(self.delta_energy_ref)
        self.e_pred_0 = torch.zeros(1)

    def test_returns_detached_tensors(self):
        loss, d_e0, grad, _, _ = _compute_batch_loss(
            self.topology, self.trainable, self.params,
            self.coords_all, self.box_vectors_all, self.forces_all,
            slice(0, 2), self.e_pred_0, self.delta_energy_ref, self.weights,
            self.energy_var, self.forces_var, self.ref_idx, self.n_mols,
            energy_weight=1.0, force_weight=1.0, normalize=True,
            compute_gradient=True,
        )
        assert not loss.requires_grad
        assert not d_e0.requires_grad
        assert grad is not None
        assert not grad.requires_grad

    def test_loss_non_negative(self):
        loss, _, _, _, _ = _compute_batch_loss(
            self.topology, self.trainable, self.params,
            self.coords_all, self.box_vectors_all, self.forces_all,
            slice(0, 2), self.e_pred_0, self.delta_energy_ref, self.weights,
            self.energy_var, self.forces_var, self.ref_idx, self.n_mols,
            energy_weight=1.0, force_weight=1.0, normalize=True,
            compute_gradient=False,
        )
        assert loss.item() >= 0.0

    def test_grad_none_when_not_requested(self):
        _, _, grad, _, _ = _compute_batch_loss(
            self.topology, self.trainable, self.params,
            self.coords_all, self.box_vectors_all, self.forces_all,
            slice(0, 2), self.e_pred_0, self.delta_energy_ref, self.weights,
            self.energy_var, self.forces_var, self.ref_idx, self.n_mols,
            energy_weight=1.0, force_weight=0.0, normalize=True,
            compute_gradient=False,
        )
        assert grad is None

    def test_grad_shape_matches_params(self):
        _, _, grad, _, _ = _compute_batch_loss(
            self.topology, self.trainable, self.params,
            self.coords_all, self.box_vectors_all, self.forces_all,
            slice(0, 2), self.e_pred_0, self.delta_energy_ref, self.weights,
            self.energy_var, self.forces_var, self.ref_idx, self.n_mols,
            energy_weight=1.0, force_weight=1.0, normalize=True,
            compute_gradient=True,
        )
        assert grad.shape == self.params.shape

    def test_energy_only_loss(self):
        """force_weight=0 should skip force computation entirely."""
        loss, _, grad, _, _ = _compute_batch_loss(
            self.topology, self.trainable, self.params,
            self.coords_all, self.box_vectors_all, self.forces_all,
            slice(0, 2), self.e_pred_0, self.delta_energy_ref, self.weights,
            self.energy_var, self.forces_var, self.ref_idx, self.n_mols,
            energy_weight=1.0, force_weight=0.0, normalize=True,
            compute_gradient=True,
        )
        assert loss.item() >= 0.0
        assert grad is not None


# ---------------------------------------------------------------------------
# _process_entry
# ---------------------------------------------------------------------------

class TestProcessEntry:
    @pytest.fixture(autouse=True)
    def _setup(self, condensed_dataset, condensed_topologies, condensed_trainable):
        self.entry = condensed_dataset[0]
        self.topology = condensed_topologies["water"]
        self.trainable = condensed_trainable
        self.params = condensed_trainable.to_values().detach().requires_grad_(True)

    @pytest.mark.parametrize("reference", ["mean", "min", "infinite"])
    def test_loss_non_negative(self, reference):
        loss, _, _, _ = _process_entry(
            self.entry, self.topology, self.trainable, self.params,
            reference=reference, energy_weight=1.0, force_weight=1.0,
            batch_size=2, normalize=True, compute_gradient=False,
        )
        assert loss.item() >= 0.0

    def test_gradient_shape(self):
        _, grad, _, _ = _process_entry(
            self.entry, self.topology, self.trainable, self.params,
            reference="mean", energy_weight=1.0, force_weight=1.0,
            batch_size=2, normalize=True, compute_gradient=True,
        )
        assert grad.shape == self.params.shape

    def test_no_gradient_when_not_requested(self):
        _, grad, _, _ = _process_entry(
            self.entry, self.topology, self.trainable, self.params,
            reference="mean", energy_weight=1.0, force_weight=1.0,
            batch_size=2, normalize=True, compute_gradient=False,
        )
        assert grad is None

    def test_batch_size_does_not_change_loss(self):
        """Different batch sizes should produce the same loss (up to float tol)."""
        loss_bs1, _, _, _ = _process_entry(
            self.entry, self.topology, self.trainable, self.params,
            reference="mean", energy_weight=1.0, force_weight=1.0,
            batch_size=1, normalize=True, compute_gradient=False,
        )
        loss_bs4, _, _, _ = _process_entry(
            self.entry, self.topology, self.trainable, self.params,
            reference="mean", energy_weight=1.0, force_weight=1.0,
            batch_size=4, normalize=True, compute_gradient=False,
        )
        assert loss_bs1.item() == pytest.approx(loss_bs4.item(), rel=1e-4)


# ---------------------------------------------------------------------------
# default_closure
# ---------------------------------------------------------------------------

class TestDefaultClosure:
    @pytest.fixture(autouse=True)
    def _setup(self, condensed_dataset, condensed_topologies, condensed_trainable):
        self.dataset = condensed_dataset
        self.topologies = condensed_topologies
        self.trainable = condensed_trainable
        self.params = condensed_trainable.to_values().detach().requires_grad_(True)

    def test_returns_callable(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        assert callable(closure)

    def test_closure_returns_three_elements(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        result = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert len(result) == 3

    def test_closure_loss_is_scalar(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        loss, grad, hess = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert loss.ndim == 0 or (loss.ndim == 1 and loss.shape[0] == 1)
        assert loss.item() >= 0.0
        assert hess is None

    def test_closure_gradient_shape(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        _, grad, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert grad.shape == self.params.shape

    def test_closure_no_gradient(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        _, grad, _ = closure(self.params, compute_gradient=False, compute_hessian=False)
        assert grad is None

    def test_hessian_raises(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        with pytest.raises(NotImplementedError):
            closure(self.params, compute_gradient=False, compute_hessian=True)

    @pytest.mark.parametrize("reference", ["mean", "min", "infinite"])
    def test_reference_modes(self, reference):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
            reference=reference,
        )
        loss, grad, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is not None

    def test_energy_only(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
            force_weight=0.0,
        )
        loss, grad, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is not None

    def test_loss_detached(self):
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        loss, _, _ = closure(self.params, compute_gradient=True, compute_hessian=False)
        assert not loss.requires_grad

    def test_gradient_changes_with_params(self):
        """Perturbed params should produce a different gradient."""
        closure = default_closure(
            self.trainable, self.topologies, self.dataset,
        )
        _, grad_a, _ = closure(self.params, compute_gradient=True, compute_hessian=False)

        perturbed = self.params.detach().clone()
        perturbed += 0.1
        perturbed.requires_grad_(True)
        _, grad_b, _ = closure(perturbed, compute_gradient=True, compute_hessian=False)

        assert not torch.allclose(grad_a, grad_b)
