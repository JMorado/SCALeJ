"""Tests for scalej.analysis._evaluation."""

import numpy as np
import pytest
import torch

from scalej.analysis._evaluation import (
    compute_metrics,
    save_prediction_parquet,
)

# Typical condensed-phase energies (kcal/mol) and forces (kcal/mol/Å)
# for a 2-water system with 4 conformers.
_ENERGY_REF = torch.tensor([-100.0, -99.0, -98.0, -97.0], dtype=torch.float64)
_ENERGY_PRED = torch.tensor([-100.2, -98.8, -98.1, -96.5], dtype=torch.float64)
_FORCES_REF = torch.tensor(
    [
        [[0.5, -0.3, 0.1], [0.2, 0.4, -0.6], [-0.7, 0.1, 0.5]],
        [[0.3, -0.1, 0.2], [0.1, 0.5, -0.3], [-0.4, 0.2, 0.1]],
        [[0.6, -0.2, 0.4], [0.3, 0.3, -0.5], [-0.9, 0.1, 0.1]],
        [[0.4, -0.4, 0.3], [0.2, 0.6, -0.4], [-0.6, 0.3, 0.3]],
    ],
    dtype=torch.float64,
)
_FORCES_PRED = torch.tensor(
    [
        [[0.4, -0.2, 0.2], [0.3, 0.3, -0.5], [-0.8, 0.2, 0.4]],
        [[0.2, -0.2, 0.3], [0.0, 0.6, -0.4], [-0.3, 0.1, 0.2]],
        [[0.7, -0.1, 0.3], [0.2, 0.4, -0.6], [-1.0, 0.0, 0.2]],
        [[0.3, -0.3, 0.4], [0.1, 0.5, -0.3], [-0.5, 0.4, 0.2]],
    ],
    dtype=torch.float64,
)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        e_mae, e_rmse, e_r2, f_mae, f_rmse, f_r2 = compute_metrics(
            _ENERGY_REF, _ENERGY_REF.clone(), _FORCES_REF, _FORCES_REF.clone()
        )
        assert e_mae == pytest.approx(0.0, abs=1e-10)
        assert e_rmse == pytest.approx(0.0, abs=1e-10)
        assert e_r2 == pytest.approx(1.0, abs=1e-10)
        assert f_mae == pytest.approx(0.0, abs=1e-10)
        assert f_rmse == pytest.approx(0.0, abs=1e-10)
        assert f_r2 == pytest.approx(1.0, abs=1e-10)

    def test_returns_six_floats(self):
        result = compute_metrics(_ENERGY_REF, _ENERGY_PRED, _FORCES_REF, _FORCES_PRED)
        assert len(result) == 6
        assert all(isinstance(v, float) for v in result)

    def test_mae_is_non_negative(self):
        e_mae, e_rmse, _, f_mae, f_rmse, _ = compute_metrics(
            _ENERGY_REF, _ENERGY_PRED, _FORCES_REF, _FORCES_PRED
        )
        assert e_mae >= 0.0
        assert e_rmse >= 0.0
        assert f_mae >= 0.0
        assert f_rmse >= 0.0

    def test_rmse_geq_mae(self):
        e_mae, e_rmse, _, f_mae, f_rmse, _ = compute_metrics(
            _ENERGY_REF, _ENERGY_PRED, _FORCES_REF, _FORCES_PRED
        )
        assert e_rmse >= e_mae - 1e-10
        assert f_rmse >= f_mae - 1e-10

    def test_known_values(self):
        # ref=[1,2,3], pred=[1.1,2.2,2.7] → errors=[0.1,0.2,0.3]
        energy_ref = torch.tensor([1.0, 2.0, 3.0])
        energy_pred = torch.tensor([1.1, 2.2, 2.7])
        forces_ref = torch.tensor([[[1.0, 0.0, 0.0]]])
        forces_pred = torch.tensor([[[1.0, 0.0, 0.0]]])

        e_mae, e_rmse, _, f_mae, f_rmse, f_r2 = compute_metrics(
            energy_ref, energy_pred, forces_ref, forces_pred
        )
        # MAE = mean(|0.1, 0.2, 0.3|) = 0.2
        assert e_mae == pytest.approx(0.2, abs=1e-6)
        # RMSE = sqrt(mean(0.01, 0.04, 0.09)) = sqrt(0.04667) ≈ 0.2160
        assert e_rmse == pytest.approx(0.21602, abs=1e-4)
        assert f_mae == pytest.approx(0.0, abs=1e-10)

    def test_list_forces(self):
        """Cover the isinstance(forces, list) branches (lines 56 & 62)."""
        f_ref = [
            torch.tensor([[[0.5, -0.3, 0.1], [0.2, 0.4, -0.6]]]),
            torch.tensor([[[0.3, -0.1, 0.2], [0.1, 0.5, -0.3]]]),
        ]
        f_pred = [
            torch.tensor([[[0.4, -0.2, 0.2], [0.3, 0.3, -0.5]]]),
            torch.tensor([[[0.2, -0.2, 0.3], [0.0, 0.6, -0.4]]]),
        ]
        e_ref = torch.tensor([-100.0, -99.0])
        e_pred = torch.tensor([-100.2, -98.8])

        result = compute_metrics(e_ref, e_pred, f_ref, f_pred)
        assert len(result) == 6
        assert all(isinstance(v, float) for v in result)


class TestSavePredictionParquet:
    def test_basic_save(self, tmp_path):
        prediction = (
            torch.tensor([1.0, 2.0, 3.0]),  # energy_ref
            torch.tensor([1.1, 2.1, 3.1]),  # energy_pred
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            [[0, 1, 2]],  # mask_idxs
            ["water"],  # entry_ids
        )
        save_prediction_parquet(prediction, tmp_path, "test")
        out = tmp_path / "test_evaluations.parquet"
        assert out.exists()

        import pandas as pd

        df = pd.read_parquet(out)
        assert len(df) == 3
        assert "energy_ref" in df.columns
        assert "energy_pred" in df.columns
        assert "id" in df.columns

    def test_with_scale_factors(self, tmp_path):
        prediction = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.1, 2.1]),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            [[0, 1]],
            ["sys_a"],
        )
        sf = {"sys_a": [0.8, 1.2]}
        save_prediction_parquet(prediction, tmp_path, "sf_test", scale_factors=sf)
        out = tmp_path / "sf_test_evaluations.parquet"
        assert out.exists()

        import pandas as pd

        df = pd.read_parquet(out)
        assert "scale_factor" in df.columns
        assert df["scale_factor"].tolist() == [0.8, 1.2]

    def test_creates_parent_dirs(self, tmp_path):
        prediction = (
            torch.tensor([1.0]),
            torch.tensor([1.1]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            [[0]],
            ["w"],
        )
        out_dir = tmp_path / "nested" / "deep"
        save_prediction_parquet(prediction, out_dir, "tag")
        assert (out_dir / "tag_evaluations.parquet").exists()

    def test_multiple_entries(self, tmp_path):
        prediction = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([1.1, 2.1, 3.1, 4.1]),
            torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
            ),
            torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
            ),
            [[0, 1], [0, 1]],
            ["water", "methane"],
        )
        save_prediction_parquet(prediction, tmp_path, "multi")
        import pandas as pd

        df = pd.read_parquet(tmp_path / "multi_evaluations.parquet")
        assert len(df) == 4
        assert set(df["id"].unique()) == {"water", "methane"}

    def test_empty_mask_skipped(self, tmp_path):
        prediction = (
            torch.tensor([1.0]),
            torch.tensor([1.1]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.1, 0.2, 0.3]]),
            [[], [0]],
            ["empty_sys", "good_sys"],
        )
        save_prediction_parquet(prediction, tmp_path, "skip")
        import pandas as pd

        df = pd.read_parquet(tmp_path / "skip_evaluations.parquet")
        assert len(df) == 1
        assert df["id"].iloc[0] == "good_sys"

    def test_heterogeneous_systems(self, tmp_path):
        # Entry 1: 2 conformers, 2 atoms each -> (2, 2, 3)
        # Entry 2: 1 conformer, 3 atoms each -> (1, 3, 3)
        prediction = (
            torch.tensor([1.0, 1.1, 2.0]), # energies
            torch.tensor([1.05, 1.15, 2.05]),
            [
                torch.zeros((2, 2, 3)),
                torch.ones((1, 3, 3))
            ], # forces_ref
            [
                torch.zeros((2, 2, 3)),
                torch.ones((1, 3, 3))
            ], # forces_pred
            [[0, 1], [0]], # masks
            ["mol2", "mol3"], # ids
        )
        save_prediction_parquet(prediction, tmp_path, "hete")
        import pandas as pd
        df = pd.read_parquet(tmp_path / "hete_evaluations.parquet")
        assert len(df) == 3
        assert len(df[df["id"] == "mol2"]) == 2
        assert len(df[df["id"] == "mol3"]) == 1
        # Check force shapes (mol2 should be 2*3=6 floats, mol3 should be 3*3=9 floats)
        assert len(df[df["id"] == "mol2"].iloc[0]["forces_ref"]) == 6
        assert len(df[df["id"] == "mol3"].iloc[0]["forces_ref"]) == 9


class TestEvaluateForceField:
    def test_returns_prediction_and_metrics(self, mocker):
        from scalej.analysis._evaluation import evaluate_force_field

        fake_prediction = (
            torch.tensor([-100.0, -99.0]),
            torch.tensor([-100.2, -98.8]),
            torch.tensor([[[0.5, -0.3, 0.1]], [[-0.2, 0.4, 0.3]]]),
            torch.tensor([[[0.4, -0.2, 0.2]], [[-0.3, 0.3, 0.4]]]),
            [[0, 1]],
            ["water"],
        )
        mocker.patch(
            "scalej.targets.predict_energies_forces",
            return_value=fake_prediction,
        )

        prediction, metrics = evaluate_force_field(
            force_field=mocker.MagicMock(),
            dataset=mocker.MagicMock(),
            tensor_systems={"water": mocker.MagicMock()},
            reference="mean",
        )

        assert prediction is fake_prediction
        assert len(metrics) == 6
        assert all(isinstance(v, float) for v in metrics)


