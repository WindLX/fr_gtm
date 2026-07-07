"""GTM atmosphere, air-data, and aerodynamic load block."""

import fly_ruler_core as fr
from fly_ruler_std.libs.atmosphere import atmos76
from fly_ruler_std.libs.rotation import Quaternion

from ..config import GTMConfig
from ..model import aerodynamic_loads


@fr.block
def Aero(cfg: GTMConfig) -> None:
    """Compute GTM air-data and aerodynamic force/moment."""
    actuators = fr.input("actuators", (4,), description="Actuator positions")
    position = fr.input("position", (3,), unit="m")
    velocity = fr.input("velocity", (3,), unit="m/s")
    quat = fr.input("quat", (4,))
    omega_body = fr.input("omega_body", (3,), unit="rad/s")
    wind_uvw = fr.input("wind_uvw", (3,), unit="m/s")
    wind_omega = fr.input("wind_omega", (3,), unit="rad/s")

    c_static_unc = fr.param("c_static_unc", (6,), default=[0.0] * 6)
    c_e_unc = fr.param("c_e_unc", (3,), default=[0.0] * 3)
    c_a_unc = fr.param("c_a_unc", (6,), default=[0.0] * 6)
    c_r_unc = fr.param("c_r_unc", (6,), default=[0.0] * 6)

    dcm_body_to_ned = Quaternion.to_dcm(quat)
    vel_ned = dcm_body_to_ned @ velocity
    altitude = -position[2]
    ground_speed = fr.math.sqrt(vel_ned[0] ** 2 + vel_ned[1] ** 2 + 1e-8)
    gamma = fr.math.degrees(fr.math.atan2(-vel_ned[2], ground_speed))

    air_velocity = velocity - wind_uvw
    air_omega = omega_body - wind_omega
    tas = fr.math.sqrt(fr.math.sum(air_velocity**2) + 1e-8)
    alpha = fr.math.degrees(fr.math.atan2(air_velocity[2], air_velocity[0]))
    beta = fr.math.degrees(
        fr.math.atan2(
            air_velocity[1],
            fr.math.sqrt(air_velocity[0] ** 2 + air_velocity[2] ** 2 + 1e-8),
        )
    )
    atmosphere = atmos76(altitude)
    qbar = 0.5 * atmosphere.density * tas**2
    mach = tas / atmosphere.speed_of_sound
    equivalent_airspeed = tas * fr.math.sqrt(atmosphere.density / 1.225 + 1e-8)

    aero = aerodynamic_loads(
        cfg,
        altitude=altitude,
        mach=mach,
        qbar=qbar,
        alpha=alpha,
        beta=beta,
        tas=tas,
        pqr=air_omega,
        actuators=actuators,
        c_static_unc=c_static_unc,
        c_e_unc=c_e_unc,
        c_a_unc=c_a_unc,
        c_r_unc=c_r_unc,
    )

    fr.output("force", aero.force, dim=(3,), unit="N")
    fr.output("moment", aero.moment, dim=(3,), unit="N*m")
    fr.output("coefficients", aero.coefficients, dim=(6,))
    fr.output("reynolds", fr.math.vector([aero.reynolds]), dim=(1,))
    fr.output("vel_ned", vel_ned, dim=(3,), unit="m/s")
    fr.output("gamma", fr.math.vector([gamma]), dim=(1,), unit="deg")
    fr.output("altitude", fr.math.vector([altitude]), dim=(1,), unit="m")
    fr.output("V", fr.math.vector([tas]), dim=(1,), unit="m/s")
    fr.output("alpha", fr.math.vector([alpha]), dim=(1,), unit="deg")
    fr.output("beta", fr.math.vector([beta]), dim=(1,), unit="deg")
    fr.output("qbar", fr.math.vector([qbar]), dim=(1,), unit="Pa")
    fr.output("mach", fr.math.vector([mach]), dim=(1,))
    fr.output("VE", fr.math.vector([equivalent_airspeed]), dim=(1,), unit="m/s")
    fr.output("pressure_ratio", fr.math.vector([atmosphere.pressure_ratio]), dim=(1,))
