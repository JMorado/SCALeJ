"""Dataset creation and utility functions for condensed-phase energy/force data."""

from pathlib import Path

import descent.targets.energy
import numpy as np
import torch


def create_from_scalej(
    arrow_path: Path | str,
    stride: int = 1,
) -> descent.targets.energy.Entry:
    """
    Load a per-run Arrow IPC file and return an energy target Entry.

    The file is expected to have been written by ``convert_npz_to_parquet.py``
    (one row per system) with columns ``smiles``, ``coords``, ``box_vectors``,
    ``energy``, and ``forces`` stored as flat ``float64`` lists.

    Parameters
    ----------
    arrow_path
        Path to an ``energies_forces.arrow`` file.
    stride
        Keep every *stride*-th frame (default 1 = all frames).

    Returns
    -------
    descent.targets.energy.Entry
        Entry dict with keys ``smiles``, ``coords``, ``energy``, ``forces``,
        and ``box_vectors`` (optional), suitable for passing to
        ``descent.targets.energy.create_dataset``.
    """
    import pyarrow.ipc as _ipc

    arrow_path = Path(arrow_path)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")

    with _ipc.open_file(str(arrow_path)) as reader:
        table = reader.read_all()

    row = {col: table.column(col)[0].as_py() for col in table.column_names}

    energy = np.array(row["energy"], dtype=np.float64)
    n_frames = len(energy)
    coords_flat = np.array(row["coords"], dtype=np.float64)
    forces_flat = np.array(row["forces"], dtype=np.float64)
    box_flat = row.get("box_vectors")

    n_dof = len(coords_flat) // n_frames
    coords = coords_flat.reshape(n_frames, n_dof // 3, 3)
    forces = forces_flat.reshape(n_frames, n_dof // 3, 3)
    box = (
        np.array(box_flat, dtype=np.float64).reshape(n_frames, 3, 3)
        if box_flat is not None
        else None
    )

    if stride > 1:
        idxs = np.arange(0, n_frames, stride)
        energy = energy[idxs]
        coords = coords[idxs]
        forces = forces[idxs]
        if box is not None:
            box = box[idxs]

    return {
        "id": row.get("id"),
        "smiles": row["smiles"],
        "coords": torch.tensor(coords, dtype=torch.float64),
        "energy": torch.tensor(energy, dtype=torch.float64),
        "forces": torch.tensor(forces, dtype=torch.float64),
        "box_vectors": torch.tensor(box, dtype=torch.float64)
        if box is not None
        else None,
    }
