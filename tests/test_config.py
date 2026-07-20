from pathlib import Path

import numpy as np
import pytest

from fr_gtm import GTMConfig, Mode
from fr_gtm.config import ActuatorConfig, AircraftConfig, SensorSingleConfig, WindConfig


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
    assert np.asarray(cfg.aero.basic_airframe.data_cx_cz).shape == (32, 27, 10, 2)
    assert np.asarray(cfg.aero.dynamic_derivatives.data_w).shape == (24, 9, 21, 6)


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
    assert np.asarray(cfg.aero.reynolds.reynolds_values).shape == (40, 40)


def test_configuration_rejects_invalid_physical_shapes_and_scales() -> None:
    with pytest.raises(ValueError, match="6 finite"):
        AircraftConfig(inertia0=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="lower limits"):
        ActuatorConfig(position_lower=[1.0] * 4, position_upper=[0.0] * 4)
    with pytest.raises(ValueError):
        SensorSingleConfig(noise_std=-1.0)
    with pytest.raises(ValueError):
        WindConfig(scale_length_high=0.0)
