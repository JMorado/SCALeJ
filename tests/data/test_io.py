"""Tests for I/O utility functions."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from scalej.data._io import (
    load_arrow,
    load_dataset,
    load_json,
    load_object,
    load_parquet,
    load_pickle,
    save_arrow,
    save_dataset,
    save_json,
    save_object,
    save_parquet,
    save_pickle,
)


class TestSaveLoadObject:
    def test_roundtrip(self, tmp_path):
        obj = {"a": 1, "b": [1, 2, 3]}
        path = tmp_path / "obj.pt"
        save_object(obj, path)
        assert load_object(path) == obj

        tensor = torch.tensor([1.0, 2.0, 3.0])
        path = tmp_path / "tensor.pt"
        save_object(tensor, path)
        assert torch.allclose(load_object(path), tensor)

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "arr.pt"
        save_object(arr, path)
        assert np.array_equal(load_object(path), arr)

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "obj.pt"
        save_object({"x": 42}, path)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path):
        path = str(tmp_path / "obj.pt")
        save_object({"x": 1}, path)
        assert load_object(path) == {"x": 1}

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_object(tmp_path / "missing.pt")


class TestSaveLoadPickle:
    def test_roundtrip(self, tmp_path):
        obj = {"key": "value", "nums": [1, 2, 3]}
        path = tmp_path / "obj.pkl"
        save_pickle(obj, path)
        assert load_pickle(path) == obj

        lst = [1, "hello", 3.14, None]
        path = tmp_path / "list.pkl"
        save_pickle(lst, path)
        assert load_pickle(path) == lst

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "arr.pkl"
        save_pickle(arr, path)
        assert np.array_equal(load_pickle(path), arr)

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "obj.pkl"
        save_pickle({"x": 1}, path)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path):
        path = str(tmp_path / "obj.pkl")
        save_pickle({"x": 99}, path)
        assert load_pickle(path) == {"x": 99}

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "missing.pkl")


class TestSaveLoadDataset:
    @pytest.fixture()
    def simple_dataset(self, water_dimer_coords, water_dimer_box):
        import descent.targets.energy

        entry = {
            "smiles": "O.O",
            "coords": torch.tensor(water_dimer_coords).unsqueeze(0),
            "box_vectors": torch.tensor(water_dimer_box).unsqueeze(0),
            "energy": torch.tensor([-9.42], dtype=torch.float64),
            "forces": torch.zeros((1, 6, 3)),
        }
        return descent.targets.energy.create_dataset([entry])

    def test_roundtrip(self, simple_dataset, tmp_path):
        path = tmp_path / "ds"
        save_dataset(simple_dataset, path)
        loaded = load_dataset(path)
        loaded.set_format("torch")

        assert len(loaded) == len(simple_dataset)
        assert set(loaded.column_names) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }
        assert torch.allclose(
            simple_dataset[0]["coords"].float(),
            loaded[0]["coords"].float(),
        )
        assert torch.allclose(
            simple_dataset[0]["energy"].float(),
            loaded[0]["energy"].float(),
        )

    def test_creates_parent_dirs(self, simple_dataset, tmp_path):
        path = tmp_path / "nested" / "ds"
        save_dataset(simple_dataset, path)
        assert path.exists()

    def test_accepts_string_path(self, simple_dataset, tmp_path):
        path = str(tmp_path / "ds")
        save_dataset(simple_dataset, path)
        assert len(load_dataset(path)) == 1

    def test_load_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "missing_ds")


class TestSaveLoadParquet:
    @pytest.fixture()
    def simple_df(self):
        return pd.DataFrame(
            {"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": ["x", "y", "z"]}
        )

    def test_roundtrip(self, simple_df, tmp_path):
        path = tmp_path / "data.parquet"
        save_parquet(simple_df, path)
        loaded = load_parquet(path)

        pd.testing.assert_frame_equal(loaded, simple_df)
        assert loaded.shape == simple_df.shape
        assert list(loaded.columns) == ["a", "b", "c"]

    def test_creates_parent_dirs(self, simple_df, tmp_path):
        path = tmp_path / "sub" / "data.parquet"
        save_parquet(simple_df, path)
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "missing.parquet")


class TestSaveLoadJson:
    def test_roundtrip(self, tmp_path):
        obj = {"name": "water", "n_atoms": 3, "masses": [15.999, 1.008, 1.008]}
        path = tmp_path / "data.json"
        save_json(obj, path)
        assert load_json(path) == obj

        lst = [1, 2.5, "hello", True, None]
        path = tmp_path / "list.json"
        save_json(lst, path)
        assert load_json(path) == lst

        with open(path) as f:
            assert json.load(f) == lst

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "data.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path):
        path = str(tmp_path / "data.json")
        save_json({"x": 42}, path)
        assert load_json(path) == {"x": 42}

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "missing.json")


class TestSaveLoadArrow:
    @pytest.fixture()
    def simple_df(self):
        return pd.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [-0.5, 0.0, 0.5], "label": ["a", "b", "c"]}
        )

    def test_roundtrip(self, simple_df, tmp_path):
        path = tmp_path / "data.arrow"
        save_arrow(simple_df, path)
        loaded = load_arrow(path)

        pd.testing.assert_frame_equal(loaded, simple_df)
        assert loaded.shape == simple_df.shape
        assert list(loaded.columns) == ["x", "y", "label"]

    def test_creates_parent_dirs(self, simple_df, tmp_path):
        path = tmp_path / "nested" / "deep" / "data.arrow"
        save_arrow(simple_df, path)
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_arrow(tmp_path / "missing.arrow")

    def test_numeric_precision(self, tmp_path):
        df = pd.DataFrame({"val": [1.23456789012345, -9.87654321098765]})
        path = tmp_path / "precise.arrow"
        save_arrow(df, path)
        loaded = load_arrow(path)
        pd.testing.assert_frame_equal(loaded, df)
