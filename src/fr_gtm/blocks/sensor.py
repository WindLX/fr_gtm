"""GTM Sensor block — assembles the output vector and applies sensor errors."""

import fly_ruler_core as fr

from ..config import SensorConfig
from ..constants import YDIM

_DEFAULT_CONFIG = SensorConfig()


@fr.block
def Sensor(cfg: SensorConfig = _DEFAULT_CONFIG) -> None:
    """Assemble clean output vector and apply bias/scale/noise.

    Inputs (individual signals):
        gamma(1,), alpha(1,), beta(1,), V(1,), altitude(1,),
        euler(3,), omega_body(3,), nz(1,), qbar(1,),
        act_pos(4,), act_cmd(4,), noise(21,)

    Outputs:
        clean_y(21,) — clean GTM output vector
        measurement(21,) — noisy sensor measurement
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
    act_pos = fr.input("act_pos", (4,), description="Actuator positions")
    act_cmd = fr.input("act_cmd", (4,), description="Actuator commands")
    noise = fr.input("noise", (YDIM,), description="Sensor noise vector")

    # Convert euler from rad→deg and omega from rad/s→deg/s
    euler_deg = fr.math.degrees(euler)
    omega_deg = fr.math.degrees(omega_body)

    # Assemble clean_y: [gamma, alpha, beta, V, altitude, euler(3), omega(3),
    #                    nz, qbar, act_pos(4), act_cmd(4)] → 21 elements
    clean_y = fr.math.concatenate(
        [
            gamma,  # 1
            alpha,  # 2
            beta,  # 3
            V,  # 4
            altitude,  # 5
            euler_deg,  # 6,7,8
            omega_deg,  # 9,10,11
            nz,  # 12
            qbar,  # 13
            act_pos,  # 14,15,16,17
            act_cmd,  # 18,19,20,21
        ]
    )

    # Extract bias, scale, noise_std from config
    bias_vec = fr.math.vector(
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
            0.0,  # thr_act bias
            0.0,  # elev_cmd bias
            0.0,  # ail_cmd bias
            0.0,  # rud_cmd bias
            0.0,  # thr_cmd bias
        ]
    )
    scale_vec = fr.math.vector(
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
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    noise_std_vec = fr.math.vector(
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
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    measurement = bias_vec + scale_vec * clean_y + noise * noise_std_vec

    fr.output("clean_y", clean_y, dim=(YDIM,), description="Clean GTM output vector")
    fr.output(
        "measurement",
        measurement,
        dim=(YDIM,),
        description="Noisy sensor measurement",
    )
