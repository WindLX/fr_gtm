"""GTM wind subsystem composed from standard wind blocks."""

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
    """Runtime wind parameters shared by steady and turbulence blocks.

    Direction is radians clockwise from NED North. SteadyWind and Dryden now
    consume the same SI angular unit, so no model-specific conversion or
    compatibility branch is required.
    """
    wind_speed = fr.param("wind_speed", (1,), default=[0.0], unit="m/s")
    wind_direction = fr.param("wind_direction", (1,), default=[0.0], unit="rad")
    turbulence = fr.param("turbulence_level", (1,), default=[0.0])

    fr.output("wind_speed_out", wind_speed, dim=(1,), unit="m/s")
    fr.output("wind_direction_out", wind_direction, dim=(1,), unit="rad")
    fr.output("turbulence_level_out", turbulence, dim=(1,))


@fr.system
def Wind(cfg: GTMConfig) -> None:
    """Compose shear-corrected steady wind and gated Dryden turbulence.

    Body-velocity magnitude drives Dryden filter time constants. It deliberately
    does not use wind-corrected TAS, which would create a block-level algebraic
    loop through the aerodynamic model. A zero turbulence parameter gates both
    excitation and outputs while retaining the standard Dryden state layout.
    """
    altitude = fr.input("altitude", (1,), unit="m")
    velocity = fr.input("velocity", (3,), unit="m/s")
    DCM = fr.input("DCM", (3, 3))
    noise = fr.input("noise", (4,))

    params = WindParams(name="params")
    shear = WindShear(
        WindShearConfig(runway_altitude=cfg.wind.runway_altitude, expose_params=False),
        name="shear",
    )
    steady = SteadyWind(SteadyWindConfig(enable_wind_shear=True), name="steady")
    dryden = DrydenTurbulence(
        DrydenConfig(
            wind_speed_low=cfg.wind.wind_speed_low,
            scale_length_high=cfg.wind.scale_length_high,
            wing_span=cfg.aircraft.b,
            expose_wing_span=False,
            minimum_airspeed=0.1,
            minimum_scale_length=0.1,
        ),
        name="dryden",
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
