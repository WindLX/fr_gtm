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

from ..config import GTMConfig


@fr.block
def WindParams() -> None:
    """稳态风与湍流 Block 共享的运行时风参数。

    Note:
        风向为从 NED 北向顺时针的弧度。`SteadyWind` 与 `DrydenTurbulence`
        现在消费同一套 SI 角度单位，因此无需模型侧换算或兼容分支。
    """
    wind_speed = fr.param("wind_speed", (1,), default=[0.0], unit="m/s")
    wind_direction = fr.param("wind_direction", (1,), default=[0.0], unit="rad")
    turbulence = fr.param("turbulence_level", (1,), default=[0.0])

    fr.output("wind_speed_out", wind_speed, dim=(1,), unit="m/s")
    fr.output("wind_direction_out", wind_direction, dim=(1,), unit="rad")
    fr.output("turbulence_level_out", turbulence, dim=(1,))


@fr.system(config=GTMConfig)
def Wind(cfg: GTMConfig) -> None:
    """组合经风切变修正的稳态风与带门控的 Dryden 湍流。

    Args:
        cfg: GTM 配置，提供跑道高度、低空风强度、高空尺度长度与翼展。

    Note:
        Dryden 滤波器时间常数由机体系速度模值驱动，刻意不使用经风修正的
        真空速，否则会经气动模型形成 Block 级代数环。湍流参数为零时同时
        门控激励与输出，但保留标准 Dryden 状态布局不变。
    """
    altitude = fr.input("altitude", (1,), unit="m")
    velocity = fr.input("velocity", (3,), unit="m/s")
    DCM = fr.input("DCM", (3, 3))
    noise = fr.input("noise", (4,))

    params = WindParams("params")
    shear = WindShear(
        "shear",
        WindShearConfig(runway_altitude=cfg.wind.runway_altitude, expose_params=False),
    )
    steady = SteadyWind("steady", SteadyWindConfig(enable_wind_shear=True))
    dryden = DrydenTurbulence(
        "dryden",
        DrydenConfig(
            wind_speed_low=cfg.wind.wind_speed_low,
            scale_length_high=cfg.wind.scale_length_high,
            wing_span=cfg.aircraft.b,
            expose_wing_span=False,
            minimum_airspeed=0.1,
            minimum_scale_length=0.1,
        ),
    )

    airspeed = fr.math.vector([fr.math.smooth_norm(velocity, epsilon=1.0e-3)])
    turbulence_on = params.turbulence_level_out[0] > 1e-3
    dryden_noise = fr.math.if_else(turbulence_on, noise, fr.math.vector([0.0] * 4))

    altitude >> shear.altitude
    DCM >> steady.DCM
    params.wind_speed_out >> steady.wind_speed
    params.wind_direction_out >> steady.wind_direction
    shear.shear_factor >> steady.wind_shear_factor

    dryden_noise >> dryden.noise
    airspeed >> dryden.airspeed
    altitude >> dryden.altitude
    params.wind_direction_out >> dryden.wind_direction
    params.turbulence_level_out >> dryden.turb_prob
    DCM >> dryden.DCM

    gust = fr.math.if_else(
        turbulence_on, dryden.gust_vel_body, fr.math.vector([0.0] * 3)
    )
    omega = fr.math.if_else(
        turbulence_on, dryden.gust_omega_body, fr.math.vector([0.0] * 3)
    )

    fr.export(
        uvw=steady.steady_wind + gust,
        omega=omega,
        steady=steady.steady_wind,
        gust=gust,
    )
