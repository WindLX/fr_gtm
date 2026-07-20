from dataclasses import dataclass

import pytest

from fr_gtm.stick_input import (
    STICK_VISUAL_MODEL_ID,
    StickAxes,
    merge_active_sidestick_axes,
)


@dataclass(frozen=True)
class AxisTelemetry:
    position_rad: float


@dataclass(frozen=True)
class StickTelemetry:
    roll: AxisTelemetry
    pitch: AxisTelemetry


@dataclass(frozen=True)
class ActiveState:
    stick_1: StickTelemetry
    stick_2: StickTelemetry
    connected: bool = True
    stale: bool = False


def test_stick_examples_select_the_737_visual_profile() -> None:
    assert STICK_VISUAL_MODEL_ID == "boeing_737_800"


def test_active_sidestick_replaces_roll_pitch_only() -> None:
    t16000 = StickAxes(roll=0.9, pitch=0.8, yaw=0.25, throttle=-0.5)
    state = ActiveState(
        stick_1=StickTelemetry(AxisTelemetry(0.1), AxisTelemetry(-0.15)),
        stick_2=StickTelemetry(AxisTelemetry(-0.2), AxisTelemetry(0.3)),
    )

    axes = merge_active_sidestick_axes(
        t16000,
        state,
        stick_index=1,
        roll_limit_rad=0.2,
        pitch_limit_rad=0.3,
    )

    assert axes.roll == pytest.approx(0.5)
    assert axes.pitch == pytest.approx(-0.5)
    assert axes.yaw == t16000.yaw
    assert axes.throttle == t16000.throttle


def test_stale_active_sidestick_neutralizes_roll_pitch() -> None:
    t16000 = StickAxes(roll=0.9, pitch=0.8, yaw=-0.2, throttle=0.4)
    telemetry = StickTelemetry(AxisTelemetry(0.1), AxisTelemetry(0.1))
    state = ActiveState(
        stick_1=telemetry,
        stick_2=telemetry,
        stale=True,
    )

    axes = merge_active_sidestick_axes(
        t16000,
        state,
        stick_index=1,
        roll_limit_rad=0.2,
        pitch_limit_rad=0.2,
    )

    assert axes == StickAxes(yaw=-0.2, throttle=0.4)
