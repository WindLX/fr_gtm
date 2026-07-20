"""Top-level GTM wiring in NED navigation and FRD body coordinates.

The root system only connects semantic actuator, wind, mass, aerodynamic,
engine, rigid-body and sensor components. GTM-specific table/sign conventions
remain inside their owning blocks; std owns actuator and 6-DOF integration.
"""

from __future__ import annotations

from typing import Any, cast

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.blocks.actuator import SimpleActuator, SimpleActuatorConfig
from fly_ruler_std.blocks.physics import (
    GroundContact,
    RigidBody6DOFConfig,
    RigidBody6DOFQuaternion,
    solve_static_ground_pose,
)
from fly_ruler_std.libs.rotation import Quaternion

from .blocks import Aero, Engine, Mass, Sensor, Wind
from .config import GTMConfig

_DEFAULT_CONFIG = GTMConfig.default()


@fr.system
def GTMSystem(cfg: GTMConfig = _DEFAULT_CONFIG) -> None:
    """Assemble the 25-state GTM plant and fixed 21-channel observation.

    ``u`` follows ``[elevator_deg, aileron_deg, rudder_deg, throttle_percent]``;
    ``d`` drives the four Dryden filters and ``n`` supplies standardized sensor
    samples. All randomness is external, so zero inputs are deterministic.
    """
    u = fr.input(
        "u",
        (4,),
        description=(
            "Commands [elevator_deg, aileron_deg, rudder_deg, throttle_percent]"
        ),
    )
    d = fr.input("d", (4,), description="Dryden white-noise inputs [nu, nv, nw, np]")
    n = fr.input(
        "n", (21,), description="Standardized noise for the 21-channel sensor layout"
    )

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
    eom = RigidBody6DOFQuaternion(
        RigidBody6DOFConfig(
            enable_external_mass=True,
            enable_geodetic_output=False,
        ),
        name="eom",
    )
    sensor = Sensor(cfg.sensor, name="sensor")

    # Command dynamics and environmental disturbances remain independent root
    # inputs, which permits deterministic tests and externally sampled noise.
    u >> actuator.command
    d >> wind.noise

    # Altitude is positive above the NED origin. Dryden intentionally receives
    # body-speed magnitude inside the wind subsystem to avoid aero feedback.
    altitude_from_state = fr.math.vector([-eom.pos_ned_out[2]])
    altitude_from_state >> wind.altitude
    eom.vel_body_out >> wind.velocity
    eom.DCM >> wind.DCM

    # Aero consumes wind-relative translational/angular motion internally and
    # preserves the gtm_ad coefficient-table ordering and Kalviste rate split.
    actuator.position_out >> aero.actuators
    eom.pos_ned_out >> aero.position
    eom.vel_body_out >> aero.velocity
    eom.quat_out >> aero.quat
    eom.omega_body_out >> aero.omega_body
    wind.uvw >> aero.wind_uvw
    wind.omega >> aero.wind_omega

    # The twin-engine block owns thrust, ram drag, offset moments and rotor
    # gyroscopic effects; it shares air-data produced by the aero block.
    actuator.position_out >> engine.actuators
    eom.omega_body_out >> engine.omega_body
    aero.pressure_ratio >> engine.pressure_ratio
    aero.qbar >> engine.qbar

    # Runtime mass/inertia uncertainty feeds every load-to-acceleration path.
    # Gravity is positive Down in NED and rotated to FRD for the rigid body.
    mass.mass >> eom.mass
    mass.inertia >> eom.inertia
    mass.inertia_inv >> eom.inertia_inv
    gravity_body = eom.DCM.T @ fr.math.vector([0.0, 0.0, mass.weight[0]])
    if cfg.ground.enabled:
        ground = cast(fr.SubsystemHandle, GroundContact(cfg.ground, name="ground"))
        eom.pos_ned_out >> ground.position
        eom.vel_body_out >> ground.velocity_body
        eom.DCM >> ground.dcm
        eom.omega_body_out >> ground.omega_body
        ground_force_body = ground.force_body
        ground_moment_body = ground.moment_body
        force_body = aero.force + engine.force + ground_force_body + gravity_body
        moment_body = aero.moment + engine.moment + ground_moment_body
        non_gravitational_force = aero.force + engine.force + ground_force_body
    else:
        force_body = aero.force + engine.force + gravity_body
        moment_body = aero.moment + engine.moment
        non_gravitational_force = aero.force + engine.force
    force_body >> eom.force_body
    moment_body >> eom.moment_body

    # GTM nz follows the reference sign convention: upward specific load is
    # positive even though FRD body +Z points down.
    load_factor = non_gravitational_force / (-mass.weight[0])
    nz = fr.math.vector([load_factor[2]])

    # Sensor assembles its documented 21 channels, then applies bias, scale and
    # externally supplied noise without feeding measurements back into physics.
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
    if cfg.ground.enabled:
        fr.export(
            ground_force_body=ground_force_body,
            ground_moment_body=ground_moment_body,
            ground_penetrations=ground.penetrations,
            ground_normal_forces=ground.normal_forces,
            ground_longitudinal_forces=ground.longitudinal_forces,
            ground_lateral_forces=ground.lateral_forces,
            ground_contact_count=ground.contact_count,
        )


def build_initial_state(
    model: fr.CompiledModel[Any],
    cfg: GTMConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    """Build the default 25-element physical state using complete paths.

    Layout comprises 13 stored rigid-body values, four actuator states and
    eight Dryden shaping-filter states. Quaternion storage is scalar-first.
    """
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


def build_ground_initial_state(
    model: fr.CompiledModel[Any],
    cfg: GTMConfig,
    *,
    forward_speed_mps: float = 0.0,
    altitude_agl_m: float = 0.0,
    sink_rate_mps: float = 0.0,
) -> np.ndarray:
    """Build a runway-relative state for ground roll or touchdown.

    ``forward_speed_mps`` is horizontal speed along NED North and
    ``sink_rate_mps`` is positive NED Down speed. They are rotated into FRD
    before populating the rigid-body state, so a pitched ground attitude does
    not create an unintended climb or sink rate.
    """

    if not np.isfinite(forward_speed_mps) or forward_speed_mps < 0.0:
        raise ValueError("forward_speed_mps must be finite and nonnegative")
    if not np.isfinite(altitude_agl_m) or altitude_agl_m < 0.0:
        raise ValueError("altitude_agl_m must be finite and nonnegative")
    if not np.isfinite(sink_rate_mps) or sink_rate_mps < 0.0:
        raise ValueError("sink_rate_mps must be finite and nonnegative")
    down, roll, pitch = solve_static_ground_pose(cfg.ground, cfg.aircraft.w0)
    down -= altitude_agl_m
    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    quat = np.array(
        [
            np.cos(roll_half) * np.cos(pitch_half),
            np.sin(roll_half) * np.cos(pitch_half),
            np.cos(roll_half) * np.sin(pitch_half),
            -np.sin(roll_half) * np.sin(pitch_half),
        ]
    )
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    dcm = np.array(
        [
            [cp, sr * sp, cr * sp],
            [0.0, cr, -sr],
            [-sp, sr * cp, cr * cp],
        ]
    )
    velocity_body = dcm.T @ np.array([forward_speed_mps, 0.0, sink_rate_mps])
    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": np.array([0.0, 0.0, down]),
                "Root.eom::vel_body": velocity_body,
                "Root.eom::quat": quat,
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
