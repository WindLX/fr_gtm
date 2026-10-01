"""由外部噪声驱动的 GTM 因果物理反馈。"""

import fly_ruler_core as fr

from ..config import SensorConfig
from ..constants import FEEDBACK_YDIM

_DEFAULT_CONFIG = SensorConfig()


@fr.block
def Sensor(cfg: SensorConfig = _DEFAULT_CONFIG) -> None:
    """组装 17 个物理反馈通道并施加传感器误差。

    Args:
        cfg: 各传感器通道的偏置、比例与白噪声标准差，顺序与反馈向量一致。

    Note:
        历史 GTM 观测会在末尾追加四个作动器指令遥测值，这些值由 `fr_gtm.system`
        在本 Block 之外拼接。把指令输入排除在物理传感器之外，才使
        `feedback_y` 成为适合内部闭环控制的因果信号。气动角、欧拉姿态与
        机体角速率按冻结的 gtm_ad 布局以度输出，而动力学内部使用弧度。
    """

    gamma = fr.input("gamma", (1,), unit="deg", description="Flight path angle")
    alpha = fr.input("alpha", (1,), unit="deg", description="Angle of attack")
    beta = fr.input("beta", (1,), unit="deg", description="Sideslip angle")
    V = fr.input("V", (1,), unit="m/s", description="True airspeed")
    altitude = fr.input("altitude", (1,), unit="m", description="Altitude")
    euler = fr.input("euler", (3,), unit="rad", description="Euler angles")
    omega_body = fr.input("omega_body", (3,), unit="rad/s", description="Body rates")
    nz = fr.input("nz", (1,), description="Normal load factor")
    qbar = fr.input("qbar", (1,), unit="Pa", description="Dynamic pressure")
    act_pos = fr.input("act_pos", (4,), description="Actual actuator positions")
    noise = fr.input(
        "noise", (FEEDBACK_YDIM,), description="Standardized physical sensor noise"
    )

    clean_feedback = fr.math.concatenate(
        [
            gamma,
            alpha,
            beta,
            V,
            altitude,
            fr.math.degrees(euler),
            fr.math.degrees(omega_body),
            nz,
            qbar,
            act_pos,
        ]
    )
    bias = fr.math.vector(
        [
            cfg.gamma.bias,
            cfg.alpha.bias,
            cfg.beta.bias,
            cfg.V.bias,
            cfg.altitude.bias,
            cfg.phi.bias,
            cfg.theta.bias,
            cfg.psi.bias,
            cfg.p.bias,
            cfg.q.bias,
            cfg.r.bias,
            cfg.nz.bias,
            cfg.qbar.bias,
            cfg.elevator.bias,
            cfg.aileron.bias,
            cfg.rudder.bias,
            0.0,
        ]
    )
    scale = fr.math.vector(
        [
            cfg.gamma.scale,
            cfg.alpha.scale,
            cfg.beta.scale,
            cfg.V.scale,
            cfg.altitude.scale,
            cfg.phi.scale,
            cfg.theta.scale,
            cfg.psi.scale,
            cfg.p.scale,
            cfg.q.scale,
            cfg.r.scale,
            cfg.nz.scale,
            cfg.qbar.scale,
            cfg.elevator.scale,
            cfg.aileron.scale,
            cfg.rudder.scale,
            1.0,
        ]
    )
    noise_std = fr.math.vector(
        [
            cfg.gamma.noise_std,
            cfg.alpha.noise_std,
            cfg.beta.noise_std,
            cfg.V.noise_std,
            cfg.altitude.noise_std,
            cfg.phi.noise_std,
            cfg.theta.noise_std,
            cfg.psi.noise_std,
            cfg.p.noise_std,
            cfg.q.noise_std,
            cfg.r.noise_std,
            cfg.nz.noise_std,
            cfg.qbar.noise_std,
            cfg.elevator.noise_std,
            cfg.aileron.noise_std,
            cfg.rudder.noise_std,
            0.0,
        ]
    )
    feedback = bias + scale * clean_feedback + noise * noise_std
    fr.output("clean_feedback", clean_feedback, dim=(FEEDBACK_YDIM,))
    fr.output("feedback", feedback, dim=(FEEDBACK_YDIM,))
