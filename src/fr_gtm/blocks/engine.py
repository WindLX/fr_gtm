"""GTM 双发推力与力矩 Block。"""

import fly_ruler_core as fr
import numpy as np
from fly_ruler_core import Signal

from ..config import FirstOrderEngineLagConfig, GTMConfig


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
    effective_throttle: Signal,
    omega_body: Signal,
    pressure_ratio: Signal,
    qbar: Signal,
) -> tuple[Signal, Signal]:
    """按 gtm_ad 多项式顺序在 FRD 轴中计算两台发动机的载荷。

    Args:
        cfg: GTM 配置，提供发动机多项式、安装位置/方向与参考面积。
        effective_throttle: 滞后后的有效油门百分比，单位 %；瞬态可超出
            指令的 [0, 100] 范围，不在此处裁剪。
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
    rpm_reference = fr.math.polyval(engine.throttle_coefficients, effective_throttle)
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
    """由油门指令、空速与机体角速率计算双发 FRD 载荷。

    油门指令先限幅，再经过配置选择的滞后模型得到有效油门，最后由静态
    多项式映射到参考转速与推力。默认一阶模型为 `δ̇ = (δ_c − δ) / τ`，
    原二阶模型为 `ẋ_e = (A x_e + B δ_c) / c`、`δ = C x_e`。

    Args:
        cfg: GTM 配置，提供滞后、推力/转速多项式及安装几何。

    Note:
        使用 `cfg.engine.lag.trim_state(throttle)` 初始化 `lag_state`。
        一阶模型含一个有效油门状态，原二阶模型含两个抽象内部状态。
    """
    throttle_command = fr.input("throttle_command", (1,), unit="%")
    omega_body = fr.input("omega_body", (3,), unit="rad/s")
    pressure_ratio = fr.input("pressure_ratio", (1,))
    qbar = fr.input("qbar", (1,), unit="Pa")

    lag = cfg.engine.lag
    command = fr.math.clip(throttle_command[0], lag.command_lower, lag.command_upper)
    if isinstance(lag, FirstOrderEngineLagConfig):
        lag_state = fr.state("lag_state", (1,), default=[0.0])
        effective_throttle = lag_state[0]
        lag_rate = fr.math.vector(
            [(command - effective_throttle) / lag.time_constant_s]
        )
    else:
        a11, a12, a21, a22 = lag.state_matrix
        b1, b2 = lag.input_matrix
        c1, c2 = lag.output_matrix
        lag_state = fr.state("lag_state", (2,), default=[0.0, 0.0])
        rate_gain = 1.0 / lag.time_scale
        lag_rate = fr.math.vector(
            [
                rate_gain * (a11 * lag_state[0] + a12 * lag_state[1] + b1 * command),
                rate_gain * (a21 * lag_state[0] + a22 * lag_state[1] + b2 * command),
            ]
        )
        effective_throttle = c1 * lag_state[0] + c2 * lag_state[1]
    fr.dynamics({lag_state: lag_rate})

    force, moment = _engine_loads(
        cfg, effective_throttle, omega_body, pressure_ratio[0], qbar[0]
    )

    fr.output(
        "effective_throttle",
        effective_throttle,
        dim=(1,),
        unit="%",
        description="Effective throttle after the configured engine lag",
    )
    fr.output("force", force, dim=(3,), unit="N")
    fr.output("moment", moment, dim=(3,), unit="N*m")
