"""Tests for scalej.targets.condensed."""

import descent.targets.energy
import pytest
import torch

from scalej.targets.condensed import (
    _compute_batch_loss,
    _compute_reference_prediction,
    _prepare_entry_data,
    _process_entry,
    default_closure,
)


class TestPrepareEntryData:
    @pytest.mark.parametrize("reference", ["mean", "min", "none", "infinite"])
    def test_returns_correct_tuple_length(
        self, condensed_dataset, condensed_topologies, reference
    ):
        result = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            reference,
            normalize=True,
        )
        assert len(result) == 12

    def test_weights_sum_to_one(self, condensed_dataset, condensed_topologies):
        *_, weights, _, _, _, _ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        assert weights.sum().item() == pytest.approx(1.0)

    def test_energy_var_positive_when_normalizing(
        self, condensed_dataset, condensed_topologies
    ):
        *_, energy_var, forces_var, _, _ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        assert energy_var.item() > 0
        assert forces_var.item() > 0

    def test_variances_one_without_normalization(
        self, condensed_dataset, condensed_topologies
    ):
        *_, energy_var, forces_var, _, _ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=False,
        )
        assert energy_var.item() == pytest.approx(1.0)
        assert forces_var.item() == pytest.approx(1.0)

    def test_delta_energy_mean_ref_sums_near_zero(
        self, condensed_dataset, condensed_topologies
    ):
        _, _, _, _, _, delta_energy_ref, *_ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        assert delta_energy_ref.sum().item() == pytest.approx(0.0, abs=1e-5)

    def test_per_molecule_energy_normalization(
        self, condensed_dataset, condensed_topologies
    ):
        energy_ref, *_ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "none",
            normalize=True,
        )
        expected = torch.tensor([-50.0, -49.5, -49.0, -48.5], dtype=energy_ref.dtype)
        assert torch.allclose(energy_ref, expected, atol=1e-6)

    def test_uniform_weights_value(self, condensed_dataset, condensed_topologies):
        *_, weights, _, _, _, _ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        expected = torch.full((4,), 0.25)
        assert torch.allclose(weights, expected, atol=1e-7)

    @pytest.mark.parametrize(
        "reference, expected_delta, expected_ref_idx",
        [
            ("mean", [-0.75, -0.25, 0.25, 0.75], None),
            ("min", [0.0, 0.5, 1.0, 1.5], 0),
            ("infinite", [-1.5, -1.0, -0.5, 0.0], 3),
        ],
    )
    def test_delta_energy_ref_values(
        self,
        condensed_dataset,
        condensed_topologies,
        reference,
        expected_delta,
        expected_ref_idx,
    ):
        _, _, _, _, _, delta_energy_ref, ref_idx, *_ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            reference,
            normalize=True,
        )
        expected = torch.tensor(expected_delta, dtype=delta_energy_ref.dtype)
        assert torch.allclose(delta_energy_ref, expected, atol=1e-6)
        assert ref_idx == expected_ref_idx

    def test_energy_variance_value(self, condensed_dataset, condensed_topologies):
        *_, energy_var, _, _, _ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        assert energy_var.item() == pytest.approx(5 / 12, rel=1e-5)

    def test_forces_variance_value(self, condensed_dataset, condensed_topologies):
        *_, forces_var, _, _ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        assert forces_var.item() == pytest.approx(0.05368128988823037, rel=1e-5)

    def test_energy_cutoff_filters_high_energy_conformers(
        self, condensed_dataset, condensed_topologies
    ):
        # Per-mol energies [-50, -49.5, -49, -48.5], min=-50.
        # cutoff=0.25 -> only conformer 0 (delta=0.0) survives.
        _, _, _, _, _, delta_energy_ref, *_ = _prepare_entry_data(
            condensed_dataset[0],
            condensed_topologies["water"],
            "mean",
            normalize=True,
            energy_cutoff=0.25,
        )
        assert len(delta_energy_ref) == 1
        assert delta_energy_ref[0].item() == pytest.approx(0.0)

    def test_energy_cutoff_with_large_gap(self, condensed_topologies):
        # Two conformers with a 100 kcal/mol gap -> only the lowest survives.
        n_atoms = 6
        entry: descent.targets.energy.Entry = {
            "smiles": "water",
            "coords": torch.zeros((2, n_atoms, 3), dtype=torch.float64),
            "energy": torch.tensor([-100.0, 0.0], dtype=torch.float64),
            "forces": torch.zeros((2, n_atoms, 3), dtype=torch.float64),
            "box_vectors": torch.eye(3, dtype=torch.float64)
            .unsqueeze(0)
            .expand(2, -1, -1)
            * 25.0,
        }
        _, _, _, _, _, delta, *_ = _prepare_entry_data(
            entry,
            condensed_topologies["water"],
            "mean",
            normalize=True,
            energy_cutoff=1.0,
        )
        assert len(delta) == 1

    def test_identical_conformers_zero_variance_guard(self, condensed_topologies):
        # Two identical conformers -> var==0, guard clamps to 1.
        n_atoms = 6
        coords = torch.zeros((2, n_atoms, 3), dtype=torch.float64)
        entry: descent.targets.energy.Entry = {
            "smiles": "water",
            "coords": coords,
            "energy": torch.tensor([-100.0, -100.0], dtype=torch.float64),
            "forces": torch.zeros((2, n_atoms, 3), dtype=torch.float64),
            "box_vectors": torch.eye(3, dtype=torch.float64)
            .unsqueeze(0)
            .expand(2, -1, -1)
            * 25.0,
        }
        *_, energy_var, forces_var, _, _ = _prepare_entry_data(
            entry,
            condensed_topologies["water"],
            "mean",
            normalize=True,
        )
        assert energy_var.item() == pytest.approx(1.0)
        assert forces_var.item() == pytest.approx(1.0)


class TestComputeReferencePrediction:
    @pytest.fixture()
    def ref_data(self, condensed_dataset, condensed_topologies, condensed_trainable):
        topology = condensed_topologies["water"]
        params = condensed_trainable.to_values().detach().requires_grad_(True)

        (
            _,
            _,
            coords_all,
            box_vectors_all,
            n_mols,
            _,
            ref_idx,
            _,
            _,
            _,
            ref_coords,
            ref_box_vectors,
        ) = _prepare_entry_data(condensed_dataset[0], topology, "min", normalize=True)
        return {
            "topology": topology,
            "trainable": condensed_trainable,
            "params": params,
            "n_mols": n_mols,
            "ref_coords": ref_coords,
            "ref_box_vectors": ref_box_vectors,
        }

    def test_returns_detached_energy(self, ref_data):
        e, _ = _compute_reference_prediction(
            ref_data["topology"],
            ref_data["trainable"],
            ref_data["params"],
            ref_data["ref_coords"],
            ref_data["ref_box_vectors"],
            ref_data["n_mols"],
            compute_gradient=False,
        )
        assert not e.requires_grad
        assert e.ndim == 0

    @pytest.mark.parametrize("compute_gradient", [True, False])
    def test_gradient_behavior(self, ref_data, compute_gradient):
        _, grad = _compute_reference_prediction(
            ref_data["topology"],
            ref_data["trainable"],
            ref_data["params"],
            ref_data["ref_coords"],
            ref_data["ref_box_vectors"],
            ref_data["n_mols"],
            compute_gradient=compute_gradient,
        )
        if compute_gradient:
            assert grad is not None
            assert grad.shape == ref_data["params"].shape
            assert not grad.requires_grad
        else:
            assert grad is None


class TestComputeBatchLoss:
    @pytest.fixture()
    def batch_data(self, condensed_dataset, condensed_topologies, condensed_trainable):
        topology = condensed_topologies["water"]
        params = condensed_trainable.to_values().detach().requires_grad_(True)

        (
            _,
            forces_all,
            coords_all,
            box_vectors_all,
            n_mols,
            delta_energy_ref,
            ref_idx,
            weights,
            energy_var,
            forces_var,
            _,
            _,
        ) = _prepare_entry_data(condensed_dataset[0], topology, "min", normalize=True)
        return {
            "topology": topology,
            "trainable": condensed_trainable,
            "params": params,
            "coords_all": coords_all,
            "box_vectors_all": box_vectors_all,
            "forces_all": forces_all,
            "delta_energy_ref": delta_energy_ref,
            "ref_idx": ref_idx,
            "weights": weights,
            "energy_var": energy_var,
            "forces_var": forces_var,
            "n_mols": n_mols,
        }

    def _call(self, batch_data, energy_weight=1.0, force_weight=1.0, gradient=True):
        return _compute_batch_loss(
            batch_data["topology"],
            batch_data["trainable"],
            batch_data["params"],
            batch_data["coords_all"],
            batch_data["box_vectors_all"],
            batch_data["forces_all"],
            slice(0, 2),
            torch.zeros(1),
            batch_data["delta_energy_ref"],
            batch_data["weights"],
            batch_data["energy_var"],
            batch_data["forces_var"],
            batch_data["ref_idx"],
            batch_data["n_mols"],
            energy_weight=energy_weight,
            force_weight=force_weight,
            normalize=True,
            compute_gradient=gradient,
        )

    def test_with_gradient(self, batch_data):
        loss, d_e0, grad, _, _ = self._call(batch_data, gradient=True)
        assert not loss.requires_grad
        assert not d_e0.requires_grad
        assert grad is not None
        assert not grad.requires_grad
        assert grad.shape == batch_data["params"].shape

    def test_without_gradient(self, batch_data):
        loss, _, grad, _, _ = self._call(batch_data, gradient=False)
        assert loss.item() >= 0.0
        assert grad is None

    def test_energy_only_loss(self, batch_data):
        loss, _, grad, _, _ = self._call(batch_data, force_weight=0.0, gradient=True)
        assert loss.item() >= 0.0
        assert grad is not None


class TestProcessEntry:
    def _call(self, entry, topology, trainable, params, **kwargs):
        defaults = {
            "reference": "mean",
            "energy_weight": 1.0,
            "force_weight": 1.0,
            "batch_size": 2,
            "normalize": True,
            "compute_gradient": False,
        }
        defaults.update(kwargs)
        return _process_entry(entry, topology, trainable, params, **defaults)

    @pytest.mark.parametrize("reference", ["mean", "min", "infinite"])
    def test_loss_non_negative(
        self, condensed_dataset, condensed_topologies, condensed_trainable, reference
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        loss, _, _, _ = self._call(
            condensed_dataset[0],
            condensed_topologies["water"],
            condensed_trainable,
            params,
            reference=reference,
        )
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("compute_gradient", [True, False])
    def test_gradient_behavior(
        self,
        condensed_dataset,
        condensed_topologies,
        condensed_trainable,
        compute_gradient,
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        _, grad, _, _ = self._call(
            condensed_dataset[0],
            condensed_topologies["water"],
            condensed_trainable,
            params,
            compute_gradient=compute_gradient,
        )
        if compute_gradient:
            assert grad is not None
            assert grad.shape == params.shape
        else:
            assert grad is None

    def test_batch_size_does_not_change_loss(
        self, condensed_dataset, condensed_topologies, condensed_trainable
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        entry = condensed_dataset[0]
        topology = condensed_topologies["water"]

        loss_bs1, _, _, _ = self._call(
            entry, topology, condensed_trainable, params, batch_size=1
        )
        loss_bs4, _, _, _ = self._call(
            entry, topology, condensed_trainable, params, batch_size=4
        )
        assert loss_bs1.item() == pytest.approx(loss_bs4.item(), rel=1e-4)


class TestDefaultClosure:
    def test_closure_interface(
        self, condensed_dataset, condensed_topologies, condensed_trainable
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable, condensed_topologies, condensed_dataset
        )
        assert callable(closure)

        result = closure(params, compute_gradient=True, compute_hessian=False)
        assert len(result) == 3

    def test_closure_loss_properties(
        self, condensed_dataset, condensed_topologies, condensed_trainable
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable, condensed_topologies, condensed_dataset
        )
        loss, _, hess = closure(params, compute_gradient=True, compute_hessian=False)

        assert loss.ndim == 0 or (loss.ndim == 1 and loss.shape[0] == 1)
        assert loss.item() >= 0.0
        assert not loss.requires_grad
        assert hess is None

    @pytest.mark.parametrize("compute_gradient", [True, False])
    def test_closure_gradient_behavior(
        self,
        condensed_dataset,
        condensed_topologies,
        condensed_trainable,
        compute_gradient,
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable, condensed_topologies, condensed_dataset
        )
        _, grad, _ = closure(
            params, compute_gradient=compute_gradient, compute_hessian=False
        )

        if compute_gradient:
            assert grad is not None
            assert grad.shape == params.shape
        else:
            assert grad is None

    def test_hessian_raises(
        self, condensed_dataset, condensed_topologies, condensed_trainable
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable, condensed_topologies, condensed_dataset
        )
        with pytest.raises(NotImplementedError):
            closure(params, compute_gradient=False, compute_hessian=True)

    @pytest.mark.parametrize("reference", ["mean", "min", "infinite"])
    def test_reference_modes(
        self, condensed_dataset, condensed_topologies, condensed_trainable, reference
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable,
            condensed_topologies,
            condensed_dataset,
            reference=reference,
        )
        loss, grad, _ = closure(params, compute_gradient=True, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is not None

    def test_energy_only(
        self, condensed_dataset, condensed_topologies, condensed_trainable
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable,
            condensed_topologies,
            condensed_dataset,
            force_weight=0.0,
        )
        loss, grad, _ = closure(params, compute_gradient=True, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is not None

    def test_gradient_changes_with_params(
        self, condensed_dataset, condensed_topologies, condensed_trainable
    ):
        params = condensed_trainable.to_values().detach().requires_grad_(True)
        closure = default_closure(
            condensed_trainable, condensed_topologies, condensed_dataset
        )
        _, grad_a, _ = closure(params, compute_gradient=True, compute_hessian=False)

        perturbed = params.detach().clone() + 0.1
        perturbed.requires_grad_(True)
        _, grad_b, _ = closure(perturbed, compute_gradient=True, compute_hessian=False)

        assert not torch.allclose(grad_a, grad_b)
