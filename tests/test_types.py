"""Tests for scalej.types type aliases."""

import pytest

from scalej.types import (
    ReferenceMode,
    WeightingMethod,
)


class TestTypeAliases:
    def test_reference_mode_accepts_valid_values(self):
        valid: list[ReferenceMode] = ["mean", "min", "none", "infinite"]
        assert len(valid) == 4

    def test_weighting_method_accepts_valid_values(self):
        valid: list[WeightingMethod] = ["uniform", "boltzmann", "mixed"]
        assert len(valid) == 3
