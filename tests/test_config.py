from pathlib import Path

import numpy as np
import pytest

from fr_gtm import GTMConfig, Mode


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


def test_external_npz_path_is_supported() -> None:
    package_table = (
        Path(__file__).parents[1] / "src" / "fr_gtm" / "data" / "aero_tables.npz"
    )
    cfg = GTMConfig.default(data_path=package_table)
    assert np.asarray(cfg.aero.reynolds.reynolds_values).shape == (40, 40)
