"""由外部噪声驱动的 GTM 因果物理反馈。"""

from __future__ import annotations

from functools import partial

from fly_ruler_core import api as fr
from fly_ruler_std.libs.units import Length, Pressure, Speed
from pydantic import BaseModel, ConfigDict, Field

from ..layout import Y_DIM


class SensorSingleConfig(BaseModel):
    """单个传感器通道的偏置、比例与白噪声标准差。

    Attributes:
        bias: 加性偏置，单位与对应物理量一致。
        scale: 乘性比例系数。
        noise_std: 白噪声标准差，单位与对应物理量一致，必须非负。

    Note:
        `bias`、`scale` 与 `noise_std` 都必须是有限值。
    """

    bias: float = Field(default=0.0, allow_inf_nan=False)
    scale: float = Field(default=1.0, allow_inf_nan=False)
    noise_std: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class SensorConfig(BaseModel):
    """传感器输出向量的 GTM 传感器误差配置。

    Attributes:
        gamma: 航迹角通道误差，单位 deg。
        alpha: 迎角通道误差，单位 deg。
        beta: 侧滑角通道误差，单位 deg。
        V: 真空速通道误差，单位 m/s。
        altitude: 高度通道误差，单位 m。
        phi: 滚转角通道误差，单位 deg。
        theta: 俯仰角通道误差，单位 deg。
        psi: 偏航角通道误差，单位 deg。
        p: 滚转角速率通道误差，单位 deg/s。
        q: 俯仰角速率通道误差，单位 deg/s。
        r: 偏航角速率通道误差，单位 deg/s。
        nz: 法向过载通道误差，无量纲。
        qbar: 动压通道误差，单位 Pa。
        elevator: 升降舵位置通道误差，单位 deg。
        aileron: 副翼位置通道误差，单位 deg。
        rudder: 方向舵位置通道误差，单位 deg。
        throttle: 油门位置通道误差，无量纲。

    Note:
        各通道误差按 `bias + scale * clean + noise * noise_std` 施加，其中
        `noise` 是外部输入的标准化噪声。
    """

    model_config = ConfigDict(frozen=True)
    """传感器通道集合构造后不可替换。"""

    gamma: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.125)
    )
    alpha: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.125)
    )
    beta: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.125)
    )
    V: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.5)
    )
    altitude: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=3.0)
    )
    phi: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.125)
    )
    theta: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.125)
    )
    psi: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.125)
    )
    p: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.5)
    )
    q: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.5)
    )
    r: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.5)
    )
    nz: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.015)
    )
    qbar: SensorSingleConfig = Field(
        default_factory=partial(
            SensorSingleConfig, noise_std=Pressure.psi_to_pa(0.00028)
        )
    )
    elevator: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.01)
    )
    aileron: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.01)
    )
    rudder: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=0.01)
    )
    throttle: SensorSingleConfig = Field(
        default_factory=partial(SensorSingleConfig, noise_std=1.0)
    )

    @classmethod
    def fullscale(cls) -> SensorConfig:
        """返回全尺寸传感器标定。

        Returns:
            使用全尺寸默认误差的配置。
        """
        return cls()

    @classmethod
    def subscale(cls) -> SensorConfig:
        """返回缩比传感器标定。

        Returns:
            使用缩比通道误差的配置。
        """
        return cls(
            gamma=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.033),
            alpha=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.031),
            beta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.033),
            V=SensorSingleConfig(
                bias=Speed.kts_to_mps(0.07),
                scale=1.0,
                noise_std=Speed.kts_to_mps(0.06),
            ),
            altitude=SensorSingleConfig(
                bias=0.0,
                scale=1.0,
                noise_std=Length.ft_to_m(2.36),
            ),
            phi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.02),
            theta=SensorSingleConfig(bias=-0.09, scale=1.0, noise_std=0.02),
            psi=SensorSingleConfig(bias=-0.04, scale=1.0, noise_std=0.02),
            p=SensorSingleConfig(bias=-0.0057, scale=1.0, noise_std=0.5350),
            q=SensorSingleConfig(bias=-0.0115, scale=1.0, noise_std=0.4737),
            r=SensorSingleConfig(bias=-0.0286, scale=1.0, noise_std=0.3397),
            nz=SensorSingleConfig(bias=0.00086, scale=1.0, noise_std=0.0082),
        )


_DEFAULT_CONFIG = SensorConfig()


@fr.block(config=SensorConfig)
def Sensor(cfg: SensorConfig = _DEFAULT_CONFIG) -> None:
    """组装 17 个物理反馈通道并施加传感器误差。

    Args:
        cfg: 各传感器通道的偏置、比例与白噪声标准差，顺序与反馈向量一致。

    Note:
        本块与整机输出都只保留 17 个物理反馈通道，不包含命令遥测。
        指令留在根输入 u 中；output 可直接用于内部闭环控制。
        气动角、欧拉姿态与
        机体角速率以度输出，而动力学内部使用弧度。
    """

    gamma = fr.input("gamma", (1,), unit="deg", description="Flight path angle")
    alpha = fr.input("alpha", (1,), unit="deg", description="Angle of attack")
    beta = fr.input("beta", (1,), unit="deg", description="Sideslip angle")
    V = fr.input("V", (1,), unit="m/s", description="True airspeed")
    altitude = fr.input("altitude", (1,), unit="m", description="Altitude")
    euler = fr.input(
        "euler",
        (3,),
        unit="rad",
        description="Body-to-NED Euler attitude [roll, pitch, yaw]",
    )
    omega_body = fr.input(
        "omega_body",
        (3,),
        unit="rad/s",
        description="Body-FRD angular velocity [p, q, r]",
    )
    nz = fr.input("nz", (1,), description="Normal load factor")
    qbar = fr.input("qbar", (1,), unit="Pa", description="Dynamic pressure")
    act_pos = fr.input(
        "act_pos",
        (4,),
        description=(
            "Actual [elevator, aileron, rudder] positions in degrees and "
            "effective throttle in percent"
        ),
    )
    noise = fr.input(
        "noise",
        (Y_DIM,),
        description="17 standardized sensor-noise channels in OUTPUT_INDEX order",
    )

    clean_output = fr.math.concatenate(
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
            cfg.throttle.bias,
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
            cfg.throttle.scale,
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
            cfg.throttle.noise_std,
        ]
    )
    output = bias + scale * clean_output + noise * noise_std
    fr.output(
        "clean_output",
        clean_output,
        dim=(Y_DIM,),
        description=(
            "Noise-free physical feedback in OUTPUT_INDEX order; angles and "
            "rates in degrees, no command telemetry"
        ),
    )
    fr.output(
        "output",
        output,
        dim=(Y_DIM,),
        description=(
            "Physical feedback with bias, scale and noise in OUTPUT_INDEX "
            "order; no command telemetry"
        ),
    )


__all__ = ["SensorConfig", "SensorSingleConfig", "Sensor"]
