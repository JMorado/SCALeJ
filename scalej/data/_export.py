"""Export force field parameters to OpenFF XML format."""

import copy
from pathlib import Path

import smee
from openff.toolkit import ForceField


def export_forcefield_to_offxml(
    base_forcefield: ForceField,
    tensor_forcefield: smee.TensorForceField,
    output_path: Path | str,
) -> ForceField:
    """
    Export tensor force field parameters to OpenFF XML format.

    Notes
    -----
    This function ONLY updates vdW parameters (and virtual site parameters if present)
    in the provided base force field.

    Parameters
    ----------
    base_forcefield : ForceField
        The original OpenFF force field to update.
    tensor_forcefield : smee.TensorForceField
        The tensor-based force field containing new parameters.
    output_path : Path | str
        Path for the output OFFXML file.

    Returns
    -------
    ForceField
        The updated OpenFF force field.
    """
    from openff.units import unit as offunit

    forcefield = copy.deepcopy(base_forcefield)

    # Determine which handler to use.
    tag = (
        "vdW"
        if "vdW" in forcefield.registered_parameter_handlers
        else "DoubleExponential"
    )

    potential_vdw = tensor_forcefield.potentials_by_type["vdW"]
    off_potential_vdw = forcefield.get_parameter_handler(tag)

    # Determine the virtual-site (EP) handler if one is registered
    _EP_TAGS = ("DoubleExponentialVirtualSites", "VirtualSites")
    ep_tag = next(
        (t for t in _EP_TAGS if t in forcefield.registered_parameter_handlers), None
    )
    off_potential_ep = forcefield.get_parameter_handler(ep_tag) if ep_tag else None

    for i in range(potential_vdw.parameters.shape[1]):
        col = potential_vdw.parameter_cols[i]
        for j in range(potential_vdw.parameters.shape[0]):
            smirk_id = potential_vdw.parameter_keys[j].id
            val = potential_vdw.parameters[j, i]
            param_unit = (
                offunit.kilocalories_per_mole if col == "epsilon" else offunit.angstrom
            )
            if "EP" in smirk_id:
                # Virtual-site key id format: "[smirks] [name] [match]"
                if off_potential_ep is None:
                    continue
                parts = smirk_id.split(" ")
                ep_smirks, ep_name = parts[0], parts[1]
                candidates = off_potential_ep.get_parameter({"smirks": ep_smirks})
                candidates = [p for p in candidates if p.name == ep_name] or candidates
                if not candidates:
                    continue
                setattr(candidates[0], col, val.item() * param_unit)
            else:
                param = off_potential_vdw.get_parameter({"smirks": smirk_id})[0]
                setattr(param, col, val.item() * param_unit)

    # Write back handler-level attributes (e.g. alpha, beta for DoubleExponential)
    if (
        potential_vdw.attributes is not None
        and potential_vdw.attribute_cols is not None
    ):
        for k, col in enumerate(potential_vdw.attribute_cols):
            if not hasattr(off_potential_vdw, col):
                continue
            val = potential_vdw.attributes[k].item()
            attr_unit = (
                potential_vdw.attribute_units[k]
                if potential_vdw.attribute_units is not None
                else offunit.dimensionless
            )
            setattr(off_potential_vdw, col, val * attr_unit)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forcefield.to_file(str(output_path))

    return forcefield
