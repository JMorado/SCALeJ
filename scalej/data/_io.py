"""File I/O utilities."""

import json
import pickle
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
import torch


def load_object(file_path: Path | str) -> Any:
    """
    Load an object serialised with save_object().

    Parameters
    ----------
    file_path : Path | str
        Path to the serialised .pt file.

    Returns
    -------
    Any
        The loaded object.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    return torch.load(file, weights_only=False, map_location="cpu")


def save_object(obj: Any, file_path: Path | str) -> None:
    """
    Save an arbitrary object using torch.save().

    Parameters
    ----------
    obj : Any
        The object to save.
    file_path : Path | str
        Path for the output .pt file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, file_path)


def load_pickle(file_path: Path | str) -> Any:
    """
    Load an object from a standard pickle file.

    Parameters
    ----------
    file_path : Path | str
        Path to the .pkl file.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, file_path: Path | str) -> None:
    """
    Save an object to a standard pickle file.

    Parameters
    ----------
    obj : Any
        The object to save.
    file_path : Path | str
        Path for the output .pkl file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def save_dataset(dataset: datasets.Dataset, path: Path | str) -> None:
    """
    Save a HuggingFace Dataset to disk as Arrow IPC files.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to persist.
    path : Path | str
        Directory where the Arrow files will be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def load_dataset(path: Path | str) -> datasets.Dataset:
    """
    Load a HuggingFace Dataset previously saved with :func:`save_dataset`.

    Parameters
    ----------
    path : Path | str
        Directory containing the Arrow IPC files.

    Returns
    -------
    datasets.Dataset
    """
    import datasets as _datasets

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    dataset = _datasets.load_from_disk(str(path))
    dataset.set_format("torch")
    return dataset


def save_parquet(df: pd.DataFrame, file_path: Path | str) -> None:
    """
    Write a pandas DataFrame to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    file_path : Path | str
        Output ``.parquet`` path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def load_parquet(file_path: Path | str) -> pd.DataFrame:
    """
    Read a Parquet file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path | str
        Path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
    """
    import pandas as _pd

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return _pd.read_parquet(file_path)


def save_json(obj: Any, file_path: Path | str) -> None:
    """
    Write a JSON-serialisable object to a file.

    Parameters
    ----------
    obj : Any
        Must be JSON-serialisable.
    file_path : Path | str
        Output ``.json`` path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(file_path: Path | str) -> Any:
    """
    Read a JSON file.

    Parameters
    ----------
    file_path : Path | str
        Path to the ``.json`` file.

    Returns
    -------
    Any
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path) as f:
        return json.load(f)
