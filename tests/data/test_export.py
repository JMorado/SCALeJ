"""Tests for exporting force fields to OFFXML format."""

import pytest

from scalej.data._export import export_forcefield_to_offxml


class TestExportForcefieldToOffxml:
    @pytest.fixture(scope="class")
    def base_ff(self):
        from openff.toolkit import ForceField

        return ForceField("openff-2.0.0.offxml", load_plugins=True)

    def test_roundtrip(self, water_system, base_ff, tmp_path):
        """Output file is created, loadable, and the return value is a ForceField."""
        from openff.toolkit import ForceField

        _, tensor_ff, _ = water_system
        out = tmp_path / "out.offxml"
        result = export_forcefield_to_offxml(base_ff, tensor_ff, out)
        assert isinstance(result, ForceField)
        assert out.exists()
        loaded = ForceField(str(out), load_plugins=True)
        assert "vdW" in loaded.registered_parameter_handlers

    def test_creates_parent_dirs(self, water_system, base_ff, tmp_path):
        _, tensor_ff, _ = water_system
        out = tmp_path / "nested" / "ff.offxml"
        export_forcefield_to_offxml(base_ff, tensor_ff, out)
        assert out.exists()

    def test_does_not_mutate_base_forcefield(self, water_system, base_ff, tmp_path):
        _, tensor_ff, _ = water_system
        original_params = [
            (p.smirks, p.epsilon, p.rmin_half)
            for p in base_ff.get_parameter_handler("vdW").parameters
        ]
        export_forcefield_to_offxml(base_ff, tensor_ff, tmp_path / "out.offxml")
        after_params = [
            (p.smirks, p.epsilon, p.rmin_half)
            for p in base_ff.get_parameter_handler("vdW").parameters
        ]
        assert original_params == after_params

    def test_ep_tag_none_skips_virtual_site_handler(
        self, water_system, base_ff, tmp_path
    ):
        """When no EP handler is registered the EP branch is not entered.

        openff-2.0.0.offxml has no VirtualSites handler, so ep_tag should be
        None and the export should still succeed without touching any EP handler.
        """
        _, tensor_ff, _ = water_system
        out = tmp_path / "no_ep.offxml"
        result = export_forcefield_to_offxml(base_ff, tensor_ff, out)
        # Neither DoubleExponentialVirtualSites nor VirtualSites should be registered.
        assert "VirtualSites" not in result.registered_parameter_handlers
        assert (
            "DoubleExponentialVirtualSites" not in result.registered_parameter_handlers
        )
        assert out.exists()
