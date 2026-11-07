from pathlib import Path

import numpy as np
import pytest

from fr_gtm import GTMConfig, Mode
from fr_gtm.config import SensorSingleConfig, WindConfig
from fr_gtm.resources import load_aero_tables


@pytest.mark.parametrize("is_fullscale", [True, False])
@pytest.mark.parametrize("mode", list(Mode))
def test_all_reference_configurations_load(
    is_fullscale: bool,
    mode: Mode,
) -> None:
    cfg = (GTMConfig.fullscale if is_fullscale else GTMConfig.subscale)(mode=mode)
    assert cfg.is_fullscale is is_fullscale
    assert cfg.aero.mode is mode
    assert len(cfg.aircraft.inertia0) == 6
    assert cfg.aero.tables.basic_airframe.cx_cz.shape == (32, 27, 10, 2)
    assert cfg.aero.tables.dynamic_derivatives.w.shape == (24, 9, 21, 6)


def test_scale_model_is_geometrically_smaller() -> None:
    full = GTMConfig.fullscale()
    scale = GTMConfig.subscale()
    assert full.aircraft.S > scale.aircraft.S
    assert full.aircraft.b > scale.aircraft.b
    assert full.aircraft.w0 > scale.aircraft.w0
    full_ground = GTMConfig.fullscale(enable_ground=True).ground
    scale_ground = GTMConfig.subscale(enable_ground=True).ground
    assert full_ground.enabled
    assert scale_ground.enabled
    assert np.linalg.norm(full_ground.contacts[0].point_body) > np.linalg.norm(
        scale_ground.contacts[0].point_body
    )
    assert (
        full_ground.contacts[0].normal_stiffness
        > scale_ground.contacts[0].normal_stiffness
    )


def test_external_npz_path_is_supported() -> None:
    package_table = (
        Path(__file__).parents[1] / "src" / "fr_gtm" / "data" / "aero_tables.npz"
    )
    cfg = GTMConfig.fullscale(data_path=package_table)
    assert cfg.aero.tables.reynolds.values.shape == (40, 40)


def test_aero_resource_rejects_invalid_shape_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "src" / "fr_gtm" / "data" / "aero_tables.npz"
    with np.load(source, allow_pickle=False) as archive:
        tables = {name: archive[name] for name in archive.files}

    malformed = dict(tables)
    malformed["C2_bas_Re_dep__data"] = tables["C2_bas_Re_dep__data"][:-1]
    bad_shape = tmp_path / "bad-shape.npz"
    np.savez(bad_shape, **malformed)
    with pytest.raises(ValueError, match="C2_bas_Re_dep__data.*shape"):
        load_aero_tables(bad_shape)

    malformed = dict(tables)
    values = tables["reynolds__reynolds"].copy()
    values[0, 0] = np.nan
    malformed["reynolds__reynolds"] = values
    nonfinite = tmp_path / "nonfinite.npz"
    np.savez(nonfinite, **malformed)
    with pytest.raises(ValueError, match="reynolds__reynolds.*finite"):
        load_aero_tables(nonfinite)

    malformed = dict(tables)
    axis = tables["reynolds__mach"].copy()
    axis[1] = axis[0]
    malformed["reynolds__mach"] = axis
    invalid_axis = tmp_path / "invalid-axis.npz"
    np.savez(invalid_axis, **malformed)
    with pytest.raises(ValueError, match="reynolds__mach.*strictly increasing"):
        load_aero_tables(invalid_axis)


@pytest.mark.parametrize("fullscale", [True, False])
def test_factories_inject_component_geometry(fullscale: bool) -> None:
    factory = GTMConfig.fullscale if fullscale else GTMConfig.subscale
    cfg = factory()
    aircraft = cfg.aircraft
    assert cfg.aero.S == cfg.engine.loads.S == aircraft.S
    assert cfg.aero.b == cfg.wind.wing_span == aircraft.b
    assert cfg.aero.Cbar == aircraft.Cbar
    assert cfg.aero.cg_pos == cfg.engine.loads.cg_pos == aircraft.cg_pos0
    assert cfg.aero.ref_cg == aircraft.ref_cg


def test_configuration_preserves_supported_validation() -> None:
    with pytest.raises(ValueError):
        SensorSingleConfig(noise_std=-1.0)
    with pytest.raises(ValueError):
        WindConfig(scale_length_high=0.0)


def test_modes_are_validated_and_tables_excluded_from_dump() -> None:
    cfg = GTMConfig.fullscale(mode="1_flap_gear_down")
    assert cfg.aero.mode is Mode.FULL_FLAP_GEAR_DOWN
    assert "tables" not in cfg.aero.model_dump()
    with pytest.raises(ValueError):
        GTMConfig.subscale(mode="unsupported")


def test_factories_create_independent_component_configs() -> None:
    first = GTMConfig.fullscale()
    second = GTMConfig.fullscale()
    first.aero.interpolation_method = "cubic"
    first.actuator.gain[0] = 1.0
    assert second.aero.interpolation_method == "linear"
    assert second.actuator.gain[0] != 1.0
