from pathlib import Path

import numpy as np
import pytest

from fr_gtm import GTMConfig, Mode
from fr_gtm.config import ActuatorConfig, AircraftConfig, SensorSingleConfig, WindConfig
from fr_gtm.resources import load_aero_tables


@pytest.mark.parametrize("is_fullscale", [True, False])
@pytest.mark.parametrize("mode", list(Mode))
def test_all_reference_configurations_load(
    is_fullscale: bool,
    mode: Mode,
) -> None:
    cfg = GTMConfig.default(is_fullscale=is_fullscale, mode=mode)
    assert cfg.is_fullscale is is_fullscale
    assert cfg.mode is mode
    assert len(cfg.aircraft.inertia0) == 6
    assert cfg.aero.tables.basic_airframe.cx_cz.shape == (32, 27, 10, 2)
    assert cfg.aero.tables.dynamic_derivatives.w.shape == (24, 9, 21, 6)


def test_scale_model_is_geometrically_smaller() -> None:
    full = GTMConfig.default(True)
    scale = GTMConfig.default(False)
    assert full.aircraft.S > scale.aircraft.S
    assert full.aircraft.b > scale.aircraft.b
    assert full.aircraft.w0 > scale.aircraft.w0
    full_ground = GTMConfig.default(True, enable_ground=True).ground
    scale_ground = GTMConfig.default(False, enable_ground=True).ground
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
    cfg = GTMConfig.default(data_path=package_table)
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


def test_sparse_toml_override_preserves_loaded_aero_resources(tmp_path: Path) -> None:
    config_path = tmp_path / "gtm.toml"
    config_path.write_text(
        "[wind]\nwind_speed_low = 12.5\n\n[aero.force_moment]\narm_x = 1.25\n",
        encoding="utf-8",
    )

    cfg = GTMConfig.from_toml(config_path)

    assert cfg.wind.wind_speed_low == 12.5
    assert cfg.aero.force_moment.arm_x == 1.25
    assert cfg.aero.force_moment.Sref == GTMConfig.default().aero.force_moment.Sref
    assert cfg.aero.tables.basic_airframe.cx_cz.shape == (32, 27, 10, 2)


def test_aircraft_toml_override_rederives_dependent_configuration(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gtm.toml"
    config_path.write_text(
        "[aircraft]\nw0 = 1234.5\nS = 42.0\nengl_pos = [1.0, 2.0, 3.0]\n",
        encoding="utf-8",
    )

    cfg = GTMConfig.from_toml(config_path)

    assert cfg.aircraft.w0 == 1234.5
    assert cfg.uncertainty.w0 == 1234.5
    assert cfg.engine.engl_pos == [1.0, 2.0, 3.0]
    assert cfg.aero.force_moment.Sref == 42.0


def test_configuration_rejects_invalid_physical_shapes_and_scales() -> None:
    with pytest.raises(ValueError, match="6 finite"):
        AircraftConfig(inertia0=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="lower limits"):
        ActuatorConfig(position_lower=[1.0] * 4, position_upper=[0.0] * 4)
    with pytest.raises(ValueError):
        SensorSingleConfig(noise_std=-1.0)
    with pytest.raises(ValueError):
        WindConfig(scale_length_high=0.0)
