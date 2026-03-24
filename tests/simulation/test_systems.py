"""Tests for system creation functions."""

import smee
from openff.toolkit import ForceField


class TestCreateSystemFromSmiles:
    def test_return_types(self, water_system):
        tensor_system, tensor_forcefield, topologies = water_system
        assert isinstance(tensor_system, smee.TensorSystem)
        assert isinstance(tensor_forcefield, smee.TensorForceField)
        assert isinstance(topologies, list)
        assert all(isinstance(t, smee.TensorTopology) for t in topologies)

    def test_single_component(self, water_system):
        tensor_system, _, topologies = water_system
        assert len(topologies) == 1
        assert topologies[0].n_atoms == 3
        assert tensor_system.n_copies == [2]
        assert tensor_system.is_periodic is True

    def test_multi_component(self, water_methane_system):
        tensor_system, _, topologies = water_methane_system
        assert len(topologies) == 2
        assert topologies[0].n_atoms == 3  # water
        assert topologies[1].n_atoms == 5  # methane
        assert tensor_system.n_copies == [2, 3]

    def test_forcefield_has_standard_potentials(self, water_system):
        _, tensor_forcefield, _ = water_system
        potentials = tensor_forcefield.potentials_by_type
        assert "Bonds" in potentials
        assert "Angles" in potentials
        assert "vdW" in potentials
        assert "Electrostatics" in potentials

    def test_charge_assignment_callback_is_called(self):
        """Callback is invoked once per SMILES and its charges propagate to the FF.

        We assign custom charges O=-0.8 e, H=+0.4 e, H=+0.4 e and verify that
        the resulting TensorForceField Electrostatics parameters match exactly.
        """
        import torch
        from openff.units import unit as off_unit

        from scalej.simulation.systems import create_system_from_smiles

        custom_q = [-0.8, 0.4, 0.4]  # e

        called_on = []

        def callback(mol):
            called_on.append(mol)
            mol.assign_partial_charges("zeros")
            mol.partial_charges = custom_q * off_unit.elementary_charge

        _, tensor_forcefield, _ = create_system_from_smiles(
            smiles_list=["O"],
            nmol_list=[1],
            charge_assignment_callback=callback,
        )

        assert len(called_on) == 1  # one SMILES -> callback called exactly once

        charges = tensor_forcefield.potentials_by_type["Electrostatics"].parameters
        expected = torch.tensor([[q] for q in custom_q], dtype=charges.dtype)
        assert torch.allclose(charges, expected, atol=1e-6), (
            f"Expected charges {custom_q}, got {charges.squeeze().tolist()}"
        )


class TestCreateCompositeSystem:
    def test_return_types(self, composite_system):
        ctf, cts, ctops, systems, off_ff = composite_system
        assert isinstance(ctf, smee.TensorForceField)
        assert isinstance(cts, smee.TensorSystem)
        assert isinstance(ctops, list)
        assert isinstance(systems, dict)
        assert isinstance(off_ff, ForceField)

    def test_composite_topologies(self, composite_system):
        _, _, ctops, _, _ = composite_system
        assert len(ctops) == 2
        assert ctops[0].n_atoms == 3  # water
        assert ctops[1].n_atoms == 5  # methane

    def test_individual_systems(self, composite_system):
        _, _, _, systems, _ = composite_system
        assert set(systems.keys()) == {"water", "methane"}
        assert systems["water"].n_copies == [2]
        assert systems["methane"].n_copies == [3]
        assert systems["water"].is_periodic is True
        assert systems["methane"].is_periodic is True

    def test_shared_forcefield(self, composite_system):
        ctf, _, _, _, _ = composite_system
        assert "Bonds" in ctf.potentials_by_type
        assert "Angles" in ctf.potentials_by_type
        assert "vdW" in ctf.potentials_by_type
        assert "Electrostatics" in ctf.potentials_by_type
