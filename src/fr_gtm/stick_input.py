from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StickAxes:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float | None = None


class ActiveAxisTelemetry(Protocol):
    @property
    def position_rad(self) -> float: ...


class ActiveStickTelemetry(Protocol):
    @property
    def roll(self) -> ActiveAxisTelemetry: ...

    @property
    def pitch(self) -> ActiveAxisTelemetry: ...


class ActiveSidestickState(Protocol):
    @property
    def stick_1(self) -> ActiveStickTelemetry: ...

    @property
    def stick_2(self) -> ActiveStickTelemetry: ...

    @property
    def connected(self) -> bool: ...

    @property
    def stale(self) -> bool: ...


def merge_active_sidestick_axes(
    t16000_axes: StickAxes,
    active_state: ActiveSidestickState | None,
    *,
    stick_index: int,
    roll_limit_rad: float,
    pitch_limit_rad: float,
) -> StickAxes:
    """Use active-stick roll/pitch while preserving T16000 rudder/throttle."""

    if active_state is None or not active_state.connected or active_state.stale:
        return StickAxes(yaw=t16000_axes.yaw, throttle=t16000_axes.throttle)
    stick = active_state.stick_1 if stick_index == 1 else active_state.stick_2
    return StickAxes(
        roll=max(-1.0, min(1.0, stick.roll.position_rad / roll_limit_rad)),
        pitch=max(-1.0, min(1.0, stick.pitch.position_rad / pitch_limit_rad)),
        yaw=t16000_axes.yaw,
        throttle=t16000_axes.throttle,
    )
