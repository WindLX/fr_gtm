"""无硬件验证完整快照到四通道 command 的映射。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fly_stick import DeviceDescription, JoystickState

from fr_gtm import GTMConfig
from fr_gtm.interface.stick import StickConfig, to_command


@dataclass
class Snapshot:
    axes: dict[str, float]

    def get_alias_axes(self, desc: DeviceDescription) -> dict[str, float]:
        return self.axes


@pytest.fixture
def description() -> DeviceDescription:
    return DeviceDescription.from_toml(
        str(Path(__file__).parents[1] / "examples/fly_stick_profiles/t16000m.toml")
    )


def test_neutral_and_optional_throttle(description: DeviceDescription) -> None:
    frame = Snapshot({"ABS_X": 0, "ABS_Y": 0, "ABS_RZ": 0})
    np.testing.assert_array_equal(
        to_command(frame, description, GTMConfig.fullscale()), [0, 0, 0, 40]
    )
    # 真实空设备快照也遵守必需轴合同，不能悄悄当作回中。
    with pytest.raises(KeyError):
        to_command(JoystickState(), description, GTMConfig.fullscale())


@pytest.mark.parametrize("reverse", [False, True])
def test_asymmetric_limits_and_directions(
    description: DeviceDescription, reverse: bool
) -> None:
    cfg = GTMConfig.fullscale()
    cfg.actuator.position_lower = [-10, -20, -30]
    cfg.actuator.position_upper = [15, 25, 35]
    cfg.engine = cfg.engine.model_copy(
        update={
            "lag": cfg.engine.lag.model_copy(
                update={"command_lower": 10, "command_upper": 80}
            )
        }
    )
    config = StickConfig(
        control_scale=1,
        invert_pitch=reverse,
        invert_roll=reverse,
        invert_yaw=reverse,
        invert_throttle=reverse,
    )
    for raw in (-2, 2):
        frame = Snapshot(
            dict.fromkeys(["ABS_X", "ABS_Y", "ABS_RZ", "ABS_THROTTLE"], raw)
        )
        positive = (raw > 0) == reverse
        expected = [15, 25, 35, 80] if positive else [-10, -20, -30, 10]
        np.testing.assert_array_equal(
            to_command(frame, description, cfg, config=config), expected
        )


def test_deadzone_aliases_and_nominal_clipping(description: DeviceDescription) -> None:
    cfg = GTMConfig.fullscale()
    config = StickConfig(
        pitch_axis="pitch",
        roll_axis="roll",
        yaw_axis="yaw",
        throttle_axis=None,
        axis_deadband=0.1,
        control_scale=0.5,
        nominal_throttle=120,
        invert_pitch=True,
    )
    frame = Snapshot({"pitch": 0.55, "roll": 0.1, "yaw": -0.05})
    result = to_command(frame, description, cfg, config=config)
    assert result[0] == pytest.approx(cfg.actuator.position_upper[0] * 0.25)
    np.testing.assert_array_equal(result[1:], [0, 0, 100])
    frame.axes["pitch"] = float("nan")
    with pytest.raises(ValueError):
        to_command(frame, description, cfg, config=config)
    with pytest.raises(KeyError):
        to_command(Snapshot({}), description, cfg)
    with pytest.raises(ValueError):
        to_command(
            Snapshot(
                {"ABS_X": 0, "ABS_Y": 0, "ABS_RZ": 0, "ABS_THROTTLE": float("inf")}
            ),
            description,
            cfg,
        )


@pytest.mark.parametrize(
    "config",
    [
        {"control_scale": -1.0},
        {"control_scale": float("nan")},
        {"axis_deadband": 1.0},
        {"axis_deadband": -0.1},
        {"nominal_throttle": float("inf")},
    ],
)
def test_invalid_mapping(config: dict[str, float]) -> None:
    # 各字段都为 float；显式分支保持静态类型，不通过无类型 kwargs 扩散。
    with pytest.raises(ValueError):
        if "control_scale" in config:
            StickConfig(control_scale=config["control_scale"])
        elif "axis_deadband" in config:
            StickConfig(axis_deadband=config["axis_deadband"])
        else:
            StickConfig(nominal_throttle=config["nominal_throttle"])
