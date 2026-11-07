"""由标准风 Block 组合而成的 GTM 风子系统。"""

import fly_ruler_core as fr
from fly_ruler_std.blocks.wind import (
    DrydenConfig,
    DrydenTurbulence,
    SteadyWind,
    SteadyWindConfig,
    WindShear,
    WindShearConfig,
)
from pydantic import BaseModel, Field


class WindConfig(BaseModel):
    """编译期风切变与 Dryden 标定值。

    Attributes:
        runway_altitude: 风切变使用的跑道高度，单位 m。
        wind_speed_low: 低空风强度，单位 m/s，必须为正。
        scale_length_high: 高空尺度长度，单位 m，必须为正。
        noise_sample_time: 噪声采样时间，单位 s，必须为正。
        wing_span: Dryden 使用的翼展，单位 m，必须为正。

    Note:
        平均风速、风向与湍流强度仍是运行时参数；风向在公开模型中统一使用
        弧度。
    """

    runway_altitude: float = 0.0  # Runway altitude used by wind shear (m).
    wind_speed_low: float = Field(default=15.0, gt=0.0)  # Low-altitude intensity [m/s].
    scale_length_high: float = Field(default=533.4, gt=0.0)  # High scale [m].
    noise_sample_time: float = Field(default=0.1, gt=0.0)  # Noise sample time [s].
    wing_span: float = Field(default=10.0, gt=0.0)  # Wing span [m].


@fr.block
def WindParams() -> None:
    """稳态风与湍流 Block 共享的运行时风参数。

    Note:
        风向为从 NED 北向顺时针的弧度。`SteadyWind` 与 `DrydenTurbulence`
        现在消费同一套 SI 角度单位，因此无需模型侧换算或兼容分支。
    """
    wind_speed = fr.param(
        "wind_speed",
        (1,),
        default=[0.0],
        unit="m/s",
        description="Mean wind speed at reference altitude before wind shear",
    )
    wind_direction = fr.param(
        "wind_direction",
        (1,),
        default=[0.0],
        unit="rad",
        description="Wind-from azimuth clockwise from NED north",
    )
    turbulence = fr.param(
        "turbulence_level",
        (1,),
        default=[0.0],
        description=(
            "Dryden probability/intensity setting; values at or below 1e-3 "
            "disable gust excitation and output"
        ),
    )

    fr.output(
        "wind_speed_out",
        wind_speed,
        dim=(1,),
        unit="m/s",
        description="Shared reference-altitude mean wind speed",
    )
    fr.output(
        "wind_direction_out",
        wind_direction,
        dim=(1,),
        unit="rad",
        description="Shared wind-from azimuth clockwise from NED north",
    )
    fr.output(
        "turbulence_level_out",
        turbulence,
        dim=(1,),
        description=(
            "Shared Dryden probability/intensity setting and gust enable control"
        ),
    )


@fr.system(config=WindConfig)
def Wind(cfg: WindConfig) -> None:
    """组合经风切变修正的稳态风与带门控的 Dryden 湍流。

    Args:
        cfg: GTM 配置，提供跑道高度、低空风强度、高空尺度长度与翼展。

    Note:
        Dryden 滤波器时间常数由机体系速度模值驱动，刻意不使用经风修正的
        真空速，否则会经气动模型形成 Block 级代数环。湍流参数为零时同时
        门控激励与输出，但保留标准 Dryden 状态布局不变。
    """
    altitude = fr.input(
        "altitude",
        (1,),
        unit="m",
        description=(
            "Altitude above mean sea level; shear accounts for configured "
            "runway altitude"
        ),
    )
    velocity_body = fr.input(
        "velocity_body",
        (3,),
        unit="m/s",
        description="Body FRD velocity vector for Dryden time constants",
    )
    DCM = fr.input(
        "DCM",
        (3, 3),
        description="Direction cosine matrix rotating body FRD vectors to NED",
    )
    noise = fr.input(
        "noise",
        (4,),
        description=(
            "Four standardized Dryden white-noise channels; gated off when "
            "turbulence is disabled"
        ),
    )

    params = WindParams("params")
    shear = WindShear(
        "shear",
        WindShearConfig(runway_altitude=cfg.runway_altitude, expose_params=False),
    )
    steady = SteadyWind("steady", SteadyWindConfig(enable_wind_shear=True))
    dryden = DrydenTurbulence(
        "dryden",
        DrydenConfig(
            wind_speed_low=cfg.wind_speed_low,
            scale_length_high=cfg.scale_length_high,
            wing_span=cfg.wing_span,
            expose_wing_span=False,
            minimum_airspeed=0.1,
            minimum_scale_length=0.1,
        ),
    )

    turbulence_on = params.turbulence_level_out[0] > 1e-3
    dryden_noise = fr.math.if_else(turbulence_on, noise, fr.math.zeros(4))

    altitude >> shear.altitude
    DCM >> steady.DCM
    params.wind_speed_out >> steady.wind_speed
    params.wind_direction_out >> steady.wind_direction
    shear.shear_factor >> steady.wind_shear_factor

    dryden_noise >> dryden.noise
    fr.math.smooth_norm(velocity_body, epsilon=1e-3) >> dryden.airspeed
    altitude >> dryden.altitude
    params.wind_direction_out >> dryden.wind_direction
    params.turbulence_level_out >> dryden.turb_prob
    DCM >> dryden.DCM

    gust = fr.math.if_else(turbulence_on, dryden.gust_vel_body, fr.math.zeros(3))
    omega = fr.math.if_else(turbulence_on, dryden.gust_omega_body, fr.math.zeros(3))

    fr.export(
        uvw=steady.steady_wind + gust,
        omega=omega,
    )
    fr.effects(
        steady=steady.steady_wind,
        gust=gust,
    )


__all__ = ["WindConfig", "WindParams", "Wind"]
