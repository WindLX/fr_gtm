"""Public GTM system composed from semantic FlyRuler blocks."""

from __future__ import annotations

from typing import Any, cast

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.blocks.actuator import SimpleActuator, SimpleActuatorConfig
from fly_ruler_std.blocks.physics import RigidBody6DOF, RigidBody6DOFConfig
from fly_ruler_std.libs.rotation import Quaternion

from .blocks import Aero, Engine, Mass, Sensor, Wind
from .config import GTMConfig

_DEFAULT_CONFIG = GTMConfig.default()


@fr.system
def GTMSystem(cfg: GTMConfig = _DEFAULT_CONFIG) -> None:
    """Build the GTM model with explicit physical blocks."""
    u = fr.input("u", (4,), description="Actuator commands")
    d = fr.input("d", (4,), description="Dryden white-noise inputs")
    n = fr.input("n", (21,), description="Sensor white-noise inputs")

    actuator = SimpleActuator(
        SimpleActuatorConfig(
            adim=4,
            gain=cfg.actuator.gain,
            rate_limit=cfg.actuator.rate_limit,
            position_lower=cfg.actuator.position_lower,
            position_upper=cfg.actuator.position_upper,
            expose_params=False,
        ),
        name="actuator",
    )
    wind = cast(Any, Wind(cfg, name="wind"))
    mass = Mass(cfg, name="mass")
    aero = Aero(cfg, name="aero")
    engine = Engine(cfg, name="engine")
    eom = RigidBody6DOF(
        RigidBody6DOFConfig(
            enable_external_mass=True,
            enable_geodetic_output=False,
        ),
        name="eom",
    )
    sensor = Sensor(cfg.sensor, name="sensor")

    u >> actuator.command
    d >> wind.noise

    eom.pos_ned_out >> wind.position
    eom.vel_body_out >> wind.velocity
    eom.quat_out >> wind.quat

    actuator.position_out >> aero.actuators
    eom.pos_ned_out >> aero.position
    eom.vel_body_out >> aero.velocity
    eom.quat_out >> aero.quat
    eom.omega_body_out >> aero.omega_body
    wind.uvw >> aero.wind_uvw
    wind.omega >> aero.wind_omega

    actuator.position_out >> engine.actuators
    eom.omega_body_out >> engine.omega_body
    aero.pressure_ratio >> engine.pressure_ratio
    aero.qbar >> engine.qbar

    mass.mass >> eom.mass
    mass.inertia >> eom.inertia
    mass.inertia_inv >> eom.inertia_inv
    gravity_body = eom.DCM.T @ fr.math.vector([0.0, 0.0, mass.weight[0]])
    force_body = aero.force + engine.force + gravity_body
    moment_body = aero.moment + engine.moment
    force_body >> eom.force_body
    moment_body >> eom.moment_body

    load_factor = (engine.force + aero.force) / (-mass.weight[0])
    nz = fr.math.vector([load_factor[2]])

    aero.gamma >> sensor.gamma
    aero.alpha >> sensor.alpha
    aero.beta >> sensor.beta
    aero.V >> sensor.V
    aero.altitude >> sensor.altitude
    eom.euler_angles >> sensor.euler
    eom.omega_body_out >> sensor.omega_body
    nz >> sensor.nz
    aero.qbar >> sensor.qbar
    actuator.position_out >> sensor.act_pos
    u >> sensor.act_cmd
    n >> sensor.noise

    quat_dot_reference = Quaternion.derivative(eom.quat_out, eom.omega_body_out)

    fr.export(
        y=sensor.measurement,
        clean_y=sensor.clean_y,
        pos_ned=eom.pos_ned_out,
        vel_body=eom.vel_body_out,
        vel_ned=eom.velocity_ned,
        quat=eom.quat_out,
        omega_body=eom.omega_body_out,
        euler=eom.euler_angles,
        gamma=aero.gamma,
        altitude=aero.altitude,
        alpha=aero.alpha,
        beta=aero.beta,
        V=aero.V,
        VE=aero.VE,
        mach=aero.mach,
        qbar=aero.qbar,
        nz=nz,
        uvw=wind.uvw,
        wind_omega=wind.omega,
        force_aero=aero.force,
        force_engine=engine.force,
        force_gravity=gravity_body,
        moment_aero=aero.moment,
        moment_engine=engine.moment,
        coefficients=aero.coefficients,
        reynolds=aero.reynolds,
        quat_dot_reference=quat_dot_reference,
        actuator_position=actuator.position_out,
    )


def build_initial_state(
    model: fr.CompiledModel[Any],
    cfg: GTMConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    """Build the default 25-element physical state using complete paths."""
    del cfg
    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": np.zeros(3),
                "Root.eom::vel_body": np.array([50.0, 0.0, 0.0]),
                "Root.eom::quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "Root.eom::omega_body": np.zeros(3),
                "Root.actuator::position": np.zeros(4),
                "Root.wind.dryden::u_gw": np.zeros(1),
                "Root.wind.dryden::v_gw": np.zeros(2),
                "Root.wind.dryden::w_gw": np.zeros(2),
                "Root.wind.dryden::p_gw": np.zeros(1),
                "Root.wind.dryden::q_gw": np.zeros(1),
                "Root.wind.dryden::r_gw": np.zeros(1),
            }
        ),
        dtype=float,
    )
