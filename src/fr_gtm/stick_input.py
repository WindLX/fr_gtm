"""手柄轴输入归一化与主动侧杆/T16000 合并助手。"""

from dataclasses import dataclass
from typing import Protocol

STICK_VISUAL_MODEL_ID = "boeing_737_800"
"""三维视图中使用的机型标识，与可视化资产目录同名。"""


@dataclass(frozen=True)
class StickAxes:
    """归一化手柄轴输入。

    Attributes:
        roll: 滚转轴，范围 [-1, 1]。
        pitch: 俯仰轴，范围 [-1, 1]。
        yaw: 偏航轴，范围 [-1, 1]。
        throttle: 油门轴；为 `None` 时表示该轴未提供。
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float | None = None


class ActiveAxisTelemetry(Protocol):
    """主动侧杆单个轴的遥测读数。"""

    @property
    def position_rad(self) -> float:
        """轴位置，单位 rad。"""
        ...


class ActiveStickTelemetry(Protocol):
    """主动侧杆单根手柄的滚转与俯仰遥测。"""

    @property
    def roll(self) -> ActiveAxisTelemetry:
        """滚转轴遥测。"""
        ...

    @property
    def pitch(self) -> ActiveAxisTelemetry:
        """俯仰轴遥测。"""
        ...


class ActiveSidestickState(Protocol):
    """主动侧杆设备的连接状态与两根手柄遥测。"""

    @property
    def stick_1(self) -> ActiveStickTelemetry:
        """1 号手柄遥测。"""
        ...

    @property
    def stick_2(self) -> ActiveStickTelemetry:
        """2 号手柄遥测。"""
        ...

    @property
    def connected(self) -> bool:
        """设备是否已连接。"""
        ...

    @property
    def stale(self) -> bool:
        """遥测数据是否已过期。"""
        ...


def merge_active_sidestick_axes(
    t16000_axes: StickAxes,
    active_state: ActiveSidestickState | None,
    *,
    stick_index: int,
    roll_limit_rad: float,
    pitch_limit_rad: float,
) -> StickAxes:
    """用主动侧杆的滚转/俯仰，同时保留 T16000 的偏航/油门。

    Args:
        t16000_axes: T16000 手柄的归一化轴输入，提供偏航与油门。
        active_state: 主动侧杆状态；为 `None`、未连接或数据过期时只回退到
            T16000 的偏航与油门。
        stick_index: 选择主动侧杆 1 或 2，取值只能是 1 或 2。
        roll_limit_rad: 滚转轴满量程对应的杆位移，单位 rad，用于归一化。
        pitch_limit_rad: 俯仰轴满量程对应的杆位移，单位 rad，用于归一化。

    Returns:
        StickAxes: 滚转/俯仰来自主动侧杆并限幅到 [-1, 1]，偏航/油门来自
            T16000。

    Note:
        仅当 `active_state` 已连接且数据未过期时读取主动侧杆；此时忽略
        `t16000_axes.roll` 与 `t16000_axes.pitch`。
    """

    if active_state is None or not active_state.connected or active_state.stale:
        return StickAxes(yaw=t16000_axes.yaw, throttle=t16000_axes.throttle)
    stick = active_state.stick_1 if stick_index == 1 else active_state.stick_2
    return StickAxes(
        roll=max(-1.0, min(1.0, stick.roll.position_rad / roll_limit_rad)),
        pitch=max(-1.0, min(1.0, stick.pitch.position_rad / pitch_limit_rad)),
        yaw=t16000_axes.yaw,
        throttle=t16000_axes.throttle,
    )
