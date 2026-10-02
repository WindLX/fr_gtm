"""GTM 双发推力与力矩 Block。"""

import fly_ruler_core as fr
import numpy as np
from fly_ruler_core import Signal

from ..config import GTMConfig


def _engine_direction(angles_deg: list[float]) -> np.ndarray:
    angles = np.radians(angles_deg)
    return np.array(
        [
            np.cos(angles[1]) * np.cos(angles[2]),
            np.sin(angles[2]),
            -np.sin(angles[1]) * np.cos(angles[2]),
        ]
    )


def _engine_loads(
    cfg: GTMConfig,
    throttle: Signal,
    omega_body: Signal,
    pressure_ratio: Signal,
    qbar: Signal,
) -> tuple[Signal, Signal]:
    """按 gtm_ad 多项式顺序在 FRD 轴中计算两台发动机的载荷。

    Args:
        cfg: GTM 配置，提供发动机多项式、安装位置/方向与参考面积。
        throttle: 油门百分比，单位 %，范围 [0, 100]。
        omega_body: 机体系角速率，单位 rad/s，用于转子陀螺力矩。
        pressure_ratio: 当地静压与海平面静压之比，无量纲。
        qbar: 动压，单位 Pa，用于冲压阻力。

    Returns:
        tuple[Signal, Signal]: 依次为机体系合力（N）与合力矩（N·m）。
    """

    engine = cfg.engine
    aircraft = cfg.aircraft
    left_direction = fr.math.vector(_engine_direction(aircraft.engl_ang).tolist())
    right_direction = fr.math.vector(_engine_direction(aircraft.engr_ang).tolist())
    left_arm = fr.math.vector(
        (np.asarray(aircraft.engl_pos) - np.asarray(engine.cg_pos)).tolist()
    )
    right_arm = fr.math.vector(
        (np.asarray(aircraft.engr_pos) - np.asarray(engine.cg_pos)).tolist()
    )

    # Coefficients are highest-order first, matching fr.math.polyval and gtm_ad.
    rpm_reference = fr.math.polyval(engine.throttle_coefficients, throttle)
    ram_drag = engine.ram_drag_coeff * qbar * aircraft.S
    thrust = (
        fr.math.polyval(engine.thrust_coefficients, rpm_reference)
        * engine.T_gain
        * pressure_ratio
        - ram_drag
    )
    thrust = fr.math.max(thrust, 0.0)

    shaft_momentum = (
        np.pi
        * engine.shaft_diameter**4
        / 32.0
        * engine.shaft_length
        * engine.density
        * (2.0 * np.pi / 60.0)
    )
    momentum = (
        engine.RPMref_gain
        * rpm_reference
        * (123_000.0 / 100.0)
        * shaft_momentum
        * engine.h_gain
    )
    left_thrust = fr.math.vector(
        [
            thrust * left_direction[0],
            thrust * left_direction[1],
            thrust * left_direction[2],
        ]
    )
    right_thrust = fr.math.vector(
        [
            thrust * right_direction[0],
            thrust * right_direction[1],
            thrust * right_direction[2],
        ]
    )
    force = fr.math.vector(
        [
            left_thrust[0] + right_thrust[0],
            left_thrust[1] + right_thrust[1],
            left_thrust[2] + right_thrust[2],
        ]
    )
    angular_momentum = fr.math.vector(
        [
            momentum * (left_direction[0] + right_direction[0]),
            momentum * (left_direction[1] + right_direction[1]),
            momentum * (left_direction[2] + right_direction[2]),
        ]
    )
    # Moments combine body-rate/shaft-momentum coupling with r x F from each
    # engine's position relative to the center of mass.
    moment_raw = (
        fr.math.cross(omega_body, angular_momentum)
        + fr.math.cross(left_arm, left_thrust)
        + fr.math.cross(right_arm, right_thrust)
    )
    moment = fr.math.vector([moment_raw[0], moment_raw[1], moment_raw[2]])
    return force, moment


@fr.block(config=GTMConfig)
def Engine(cfg: GTMConfig) -> None:
    """由油门、空速与机体角速率计算双发 FRD 载荷。

    Args:
        cfg: GTM 配置，提供发动机拉力/参考 RPM 多项式、安装几何与参考面积。

    Note:
        作动器第 4 通道（零基索引 3）是油门百分比。左右发动机共用同一套
        参考 RPM/推力多项式，但保留各自的安装方向与位置向量，用于力方向
        与偏置力矩。
    """
    actuators = fr.input("actuators", (4,), description="Actuator positions")
    omega_body = fr.input("omega_body", (3,), unit="rad/s")
    pressure_ratio = fr.input("pressure_ratio", (1,))
    qbar = fr.input("qbar", (1,), unit="Pa")

    force, moment = _engine_loads(
        cfg, actuators[3], omega_body, pressure_ratio[0], qbar[0]
    )

    fr.output("force", force, dim=(3,), unit="N")
    fr.output("moment", moment, dim=(3,), unit="N*m")
