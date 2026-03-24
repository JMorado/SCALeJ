"""Tests for MD simulation functions."""

import numpy as np
import pytest

from scalej.simulation.simulation import (
    load_trajectory_frames_smee,
    run_simulation_omm,
    run_simulation_smee,
)


class TestRunSimulationOmm:
    def test_returns_arrays(self, ani2x_simulation, initial_coords_box):
        import openmm.unit

        coords_q, box_q = initial_coords_box
        coords = coords_q.value_in_unit(openmm.unit.angstrom)
        box = box_q.value_in_unit(openmm.unit.angstrom)

        final_coords, final_box = run_simulation_omm(
            ani2x_simulation,
            coords * openmm.unit.angstrom,
            box * openmm.unit.angstrom,
            n_steps=10,
        )
        assert hasattr(final_coords, "value_in_unit")
        assert hasattr(final_box, "value_in_unit")

        coords_np = final_coords.value_in_unit(openmm.unit.angstrom)
        box_np = final_box.value_in_unit(openmm.unit.angstrom)
        assert coords_np.shape == coords.shape
        assert box_np.shape == (3, 3)


class TestLoadTrajectoryFrames:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_trajectory_frames_smee(tmp_path / "missing.dcd")

    def test_too_many_frames_raises(self, tmp_path, mocker):
        import smee.mm
        import torch

        fake_coord = torch.zeros(6, 3)
        fake_box = torch.eye(3) * 10.0

        mocker.patch.object(
            smee.mm._reporters,
            "unpack_frames",
            side_effect=lambda f: iter([(fake_coord, fake_box, None, None)]),
        )

        dummy_path = tmp_path / "dummy.dcd"
        dummy_path.write_bytes(b"")

        with pytest.raises(ValueError, match="frames"):
            load_trajectory_frames_smee(dummy_path, n_frames=5)

    @pytest.mark.parametrize(
        "n_frames, from_end, expected_coords_shape, expected_box_shape",
        [
            (1, True, (6, 3), (3, 3)),
            (3, True, (3, 6, 3), (3, 3, 3)),
            (2, False, (2, 6, 3), (2, 3, 3)),
        ],
    )
    def test_frames_shape(
        self,
        tmp_path,
        mocker,
        n_frames,
        from_end,
        expected_coords_shape,
        expected_box_shape,
    ):
        import smee.mm
        import torch

        total_frames = 4
        fake_coords = [torch.full((6, 3), float(i)) for i in range(total_frames)]
        fake_box = torch.eye(3) * 10.0

        mocker.patch.object(
            smee.mm._reporters,
            "unpack_frames",
            side_effect=lambda f: ((c, fake_box, None, None) for c in fake_coords),
        )

        dummy_path = tmp_path / "dummy.dcd"
        dummy_path.write_bytes(b"")

        coords, box_vectors, n_frames_out = load_trajectory_frames_smee(
            dummy_path, n_frames=n_frames, from_end=from_end
        )

        assert isinstance(coords, np.ndarray)
        assert n_frames_out == n_frames
        assert coords.shape == expected_coords_shape
        assert box_vectors.shape == expected_box_shape


class TestRunSimulationSmee:
    @pytest.mark.parametrize("save_pdb", [False, True])
    def test_returns_coords_and_box(self, water_system, tmp_path, mocker, save_pdb):
        import openmm.app
        import openmm.unit
        import smee.mm

        tensor_system, tensor_forcefield, _ = water_system
        n_atoms = sum(
            t.n_atoms * n
            for t, n in zip(
                tensor_system.topologies, tensor_system.n_copies, strict=True
            )
        )
        mock_coords = np.zeros((n_atoms, 3)) * openmm.unit.nanometer
        mock_box = np.eye(3) * 2.0 * openmm.unit.nanometer

        mocker.patch.object(
            smee.mm, "generate_system_coords", return_value=(mock_coords, mock_box)
        )

        fake_reporter = mocker.MagicMock()
        fake_ctx = mocker.MagicMock()
        fake_ctx.__enter__ = mocker.Mock(return_value=fake_reporter)
        fake_ctx.__exit__ = mocker.Mock(return_value=False)

        mocker.patch.object(smee.mm, "tensor_reporter", return_value=fake_ctx)
        mocker.patch.object(smee.mm, "simulate")
        mocker.patch.object(openmm.app, "PDBReporter", return_value=mocker.MagicMock())

        from scalej.config import SimulationConfig

        result_coords, result_box = run_simulation_smee(
            tensor_system,
            tensor_forcefield,
            tmp_path / "traj.dcd",
            config=SimulationConfig(
                n_equilibration_nvt_steps=0,
                n_equilibration_npt_steps=0,
                n_production_steps=0,
            ),
            save_pdb=save_pdb,
        )

        assert result_coords is mock_coords
        assert result_box is mock_box
