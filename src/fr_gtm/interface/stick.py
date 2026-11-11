"""把 fly_stick 的设备快照转换为 GTM 四通道操纵命令。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from fly_ruler_core.runtime.backend import NumpyArray
from fly_stick import DeviceDescription

from ..config import GTMConfig


class _StickSnapshot(Protocol):
    """fly_stick 完整快照所提供的别名读取接口。"""

    def get_alias_axes(self, desc: DeviceDescription) -> dict[str, float]: ...


@dataclass(frozen=True)
class StickConfig:
    """轴别名、方向与操纵幅度；设备标定与连接由 fly_stick 管理。

    Args:
        roll_axis: 滚转轴别名。
        pitch_axis: 俯仰轴别名。
        yaw_axis: 偏航轴别名。
        throttle_axis: 可选油门轴别名，None 表示不用油门轴。
        invert_roll: 相对默认负向映射反转滚转轴。
        invert_pitch: 相对默认负向映射反转俯仰轴。
        invert_yaw: 相对默认负向映射反转偏航轴。
        invert_throttle: 相对默认负向映射反转油门轴。
        control_scale: 舵面正负行程比例，范围 [0,1]。
        nominal_throttle: 没有油门轴时的油门指令，%。
        axis_deadband: 舵面中心死区，范围 [0,1)，其余行程线性重标定。
    """

    roll_axis: str = "ABS_X"
    """滚转轴别名。"""
    pitch_axis: str = "ABS_Y"
    """俯仰轴别名。"""
    yaw_axis: str = "ABS_RZ"
    """偏航轴别名。"""
    throttle_axis: str | None = "ABS_THROTTLE"
    """可选油门轴别名；None 禁用。"""
    invert_roll: bool = False
    """相对默认负向映射反转滚转轴。"""
    invert_pitch: bool = False
    """相对默认负向映射反转俯仰轴。"""
    invert_yaw: bool = False
    """相对默认负向映射反转偏航轴。"""
    invert_throttle: bool = False
    """相对默认负向映射反转油门轴。"""
    control_scale: float = 0.6
    """舵面行程比例 [0,1]。"""
    nominal_throttle: float = 40.0
    """无油门轴时的固定油门指令，%。"""
    axis_deadband: float = 0.03
    """舵面死区 [0,1)。"""

    def __post_init__(self) -> None:
        if not 0.0 <= self.control_scale <= 1.0:
            raise ValueError("control_scale must be in [0,1]")
        if not 0.0 <= self.axis_deadband < 1.0:
            raise ValueError("axis_deadband must be in [0,1)")
        if not np.isfinite(self.nominal_throttle):
            raise ValueError("nominal_throttle must be finite")


_DEFAULT_CONFIG = StickConfig()


def to_command(
    state: _StickSnapshot,
    description: DeviceDescription,
    cfg: GTMConfig,
    *,
    config: StickConfig = _DEFAULT_CONFIG,
) -> NumpyArray:
    """读取必需操纵轴并映射为 [升降舵deg,副翼deg,方向舵deg,油门%]。

    Args:
        state: fly_stick 完整设备快照，不是事件差分。
        description: 设备 profile，定义硬件轴到别名的映射。
        cfg: 提供舵面与发动机指令边界的整机配置。
        config: 轴映射与手感设置。

    Returns:
        形状 (4,) 的 float64 命令，按配置限幅；不构造模型 u 向量。

    Raises:
        KeyError: 必需的俯仰、滚转或偏航轴别名缺失。
        ValueError: 轴值不是有限值。
    """
    axes = state.get_alias_axes(description)
    # 模型顺序为 elevator/aileron/rudder，不是设备的 roll/pitch/yaw 顺序。
    raw = np.array(
        [axes[config.pitch_axis], axes[config.roll_axis], axes[config.yaw_axis]]
    )
    if not np.isfinite(raw).all():
        raise ValueError("stick axes must be finite")
    signs = np.array([config.invert_pitch, config.invert_roll, config.invert_yaw])
    normalized = np.clip(raw, -1.0, 1.0) * np.where(signs, 1.0, -1.0)
    normalized = (
        np.sign(normalized)
        * np.maximum(np.abs(normalized) - config.axis_deadband, 0.0)
        / (1.0 - config.axis_deadband)
    )
    lower = np.asarray(cfg.actuator.position_lower)
    upper = np.asarray(cfg.actuator.position_upper)
    # 正负方向分别使用对应行程，保留不对称舵面边界。
    surfaces = normalized * np.where(normalized >= 0.0, upper, -lower)
    surfaces *= config.control_scale
    lag = cfg.engine.lag
    throttle = config.nominal_throttle
    if config.throttle_axis is not None and config.throttle_axis in axes:
        raw_throttle = axes[config.throttle_axis]
        if not np.isfinite(raw_throttle):
            raise ValueError("throttle axis must be finite")
        axis = np.clip(raw_throttle, -1.0, 1.0)
        if not config.invert_throttle:
            axis = -axis
        throttle = lag.command_lower + (axis + 1.0) * 0.5 * (
            lag.command_upper - lag.command_lower
        )
    return np.array(
        [
            *np.clip(surfaces, lower, upper),
            np.clip(throttle, lag.command_lower, lag.command_upper),
        ],
        dtype=np.float64,
    )


__all__ = ["StickConfig", "to_command"]
