"""Tests for scalej.data._datasets."""

import numpy as np
import pytest
import torch


class TestCreateFromScalej:
    @pytest.fixture()
    def arrow_file(self, tmp_path):
        """Write a minimal Arrow IPC file mimicking convert_npz_to_parquet.py output."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        n_frames = 6
        n_atoms = 3  # single water molecule

        # Near-equilibrium water coordinates (angstroms).
        base_coords = np.array(
            [[0.000, 0.000, 0.000], [0.960, 0.000, 0.000], [-0.240, 0.926, 0.000]]
        )
        coords = np.stack(
            [base_coords + i * np.array([0.01, -0.005, 0.003]) for i in range(n_frames)]
        )
        # Box: 20 Å cubic.
        box = np.stack([np.eye(3) * 20.0] * n_frames)
        # Energies (kcal/mol) decreasing slightly per frame.
        energies = np.array([-50.0, -49.8, -49.5, -49.2, -49.0, -48.7])
        # Small forces (kcal/mol/Å).
        forces = np.array(
            [
                [[0.10, -0.05, 0.02], [0.03, 0.04, -0.06], [-0.13, 0.01, 0.04]],
                [[0.08, -0.04, 0.03], [0.02, 0.05, -0.05], [-0.10, 0.02, 0.02]],
                [[0.12, -0.06, 0.01], [0.04, 0.03, -0.07], [-0.16, 0.03, 0.06]],
                [[0.09, -0.03, 0.04], [0.01, 0.06, -0.04], [-0.10, 0.01, 0.00]],
                [[0.11, -0.05, 0.02], [0.03, 0.04, -0.06], [-0.14, 0.01, 0.04]],
                [[0.07, -0.02, 0.05], [0.00, 0.07, -0.03], [-0.07, 0.02, -0.02]],
            ]
        )

        row = {
            "id": "water_run_001",
            "smiles": "[OH2]",
            "energy": energies.tolist(),
            "coords": coords.flatten().tolist(),
            "forces": forces.flatten().tolist(),
            "box_vectors": box.flatten().tolist(),
        }

        table = pa.table({k: [v] for k, v in row.items()})
        arrow_path = tmp_path / "energies_forces.arrow"
        with ipc.new_file(str(arrow_path), table.schema) as writer:
            writer.write_table(table)
        return arrow_path, n_frames, n_atoms

    def test_loads_all_frames(self, arrow_file):
        from scalej.data._datasets import create_from_scalej

        path, n_frames, n_atoms = arrow_file
        entry = create_from_scalej(path)

        assert entry["smiles"] == "[OH2]"
        assert entry["id"] == "water_run_001"
        assert entry["energy"].shape == (n_frames,)
        assert entry["coords"].shape == (n_frames, n_atoms, 3)
        assert entry["forces"].shape == (n_frames, n_atoms, 3)
        assert entry["box_vectors"].shape == (n_frames, 3, 3)

    def test_stride(self, arrow_file):
        from scalej.data._datasets import create_from_scalej

        path, n_frames, n_atoms = arrow_file
        entry = create_from_scalej(path, stride=2)

        expected_frames = len(range(0, n_frames, 2))  # 0, 2, 4 → 3 frames
        assert entry["energy"].shape == (expected_frames,)
        assert entry["coords"].shape == (expected_frames, n_atoms, 3)
        assert entry["forces"].shape == (expected_frames, n_atoms, 3)

    def test_stride_one_returns_all(self, arrow_file):
        from scalej.data._datasets import create_from_scalej

        path, n_frames, _ = arrow_file
        entry = create_from_scalej(path, stride=1)
        assert entry["energy"].shape == (n_frames,)

    def test_energy_values(self, arrow_file):
        from scalej.data._datasets import create_from_scalej

        path, _, _ = arrow_file
        entry = create_from_scalej(path)
        expected = torch.tensor(
            [-50.0, -49.8, -49.5, -49.2, -49.0, -48.7],
            dtype=entry["energy"].dtype,
        )
        assert torch.allclose(entry["energy"], expected, atol=1e-6)

    def test_missing_file_raises(self, tmp_path):
        from scalej.data._datasets import create_from_scalej

        with pytest.raises(FileNotFoundError, match="Arrow file not found"):
            create_from_scalej(tmp_path / "nonexistent.arrow")

    def test_dtype_is_float64(self, arrow_file):
        from scalej.data._datasets import create_from_scalej

        path, _, _ = arrow_file
        entry = create_from_scalej(path)
        assert entry["energy"].dtype == torch.float64
        assert entry["coords"].dtype == torch.float64
        assert entry["forces"].dtype == torch.float64
        assert entry["box_vectors"].dtype == torch.float64
