"""GTM aerodynamic lookup and load expressions.

The functions in this module are deliberately not FlyRuler blocks. They are
called while the GTM ``Aero`` block is being declared, so intermediate
expressions remain inside one compiler-visible block.
"""

from __future__ import annotations

from typing import NamedTuple

import fly_ruler_core as fr
import numpy as np
from fly_ruler_core import Signal

from ..config import GTMConfig, Mode


class AeroResult(NamedTuple):
    """Aerodynamic coefficients, forces, and moments."""

    coefficients: Signal
    force: Signal
    moment: Signal
    reynolds: Signal


def _uncertainty_scales(
    c_static_unc: Signal,
    c_e_unc: Signal,
    c_a_unc: Signal,
    c_r_unc: Signal,
) -> tuple[Signal, Signal, Signal, Signal]:
    one6 = fr.math.vector([1.0] * 6)
    c_e = fr.math.vector(
        [
            1.0 + c_e_unc[0],
            0.0,
            1.0 + c_e_unc[1],
            0.0,
            1.0 + c_e_unc[2],
            0.0,
        ]
    )
    return one6 + c_static_unc, c_e, one6 + c_a_unc, one6 + c_r_unc


def _reynolds(cfg: GTMConfig, altitude: Signal, mach: Signal) -> Signal:
    table_cfg = cfg.aero.reynolds
    reynolds_values = np.asarray(table_cfg.reynolds_values)
    # gtm_ad flattens this MATLAB table in Fortran order before the backend
    # reshapes it on the regular grid. Preserve that exact orientation.
    reynolds_values = reynolds_values.flatten(order="F").reshape(
        len(table_cfg.altitude_ft), len(table_cfg.mach_bp)
    )
    table = fr.resource.interpolator(
        "reynolds",
        grid_points=[
            np.asarray(table_cfg.altitude_ft),
            np.asarray(table_cfg.mach_bp),
        ],
        values=reynolds_values,
        method="linear",
    )
    return table(altitude / 0.3048, mach)


def _basic_coefficients(
    cfg: GTMConfig,
    alpha: Signal,
    beta: Signal,
    reynolds: Signal,
    scale: Signal,
) -> Signal:
    basic = cfg.aero.basic_airframe
    cx_cz = fr.resource.interpolator(
        "basic_cx_cz",
        grid_points=[
            np.asarray(basic.alpha_c2),
            np.asarray(basic.beta_c2),
            np.asarray(basic.reynolds_c2),
        ],
        values=np.asarray(basic.data_cx_cz),
        method="linear",
        n_out=2,
    )(alpha, beta, reynolds)
    lateral = fr.resource.interpolator(
        "basic_cy_cl_cm_cn",
        grid_points=[np.asarray(basic.alpha_c4), np.asarray(basic.beta_c4)],
        values=np.asarray(basic.data_cy_cl_cm_cn),
        method="linear",
        n_out=4,
    )(alpha, beta)
    return (
        fr.math.vector(
            [
                cx_cz[0],
                lateral[0],
                cx_cz[1],
                lateral[1],
                lateral[2],
                lateral[3],
            ]
        )
        * scale
    )


def _elevator_matrix(cfg: GTMConfig) -> Signal:
    control = cfg.aero.control_surfaces
    matrix = np.zeros((6, 6))
    for arm in (
        control.arm_left_out,
        control.arm_left_in,
        control.arm_right_in,
        control.arm_right_out,
    ):
        contribution = np.eye(6)
        contribution[3, 2] = arm
        contribution[5, 0] = -arm
        matrix += contribution
    return fr.math.matrix(matrix.tolist())


def _configuration_increment(mode: Mode) -> Signal:
    flap = np.array([-0.0282, 0.0, -0.2989, 0.0, 0.0728, 0.0]) * 2.0
    gear = np.array([-0.016, 0.0, 0.014, 0.0, -0.008, 0.0]) * 2.0
    if mode is Mode.HALF_FLAP_GEAR_UP:
        result = flap * 0.5
    elif mode is Mode.FULL_FLAP_GEAR_UP:
        result = flap
    elif mode is Mode.FULL_FLAP_GEAR_DOWN:
        result = flap + gear
    else:
        result = np.zeros(6)
    return fr.math.vector(result.tolist())


def _control_coefficients(
    cfg: GTMConfig,
    alpha: Signal,
    beta: Signal,
    actuators: Signal,
    c_e: Signal,
    c_a: Signal,
    c_r: Signal,
) -> Signal:
    control = cfg.aero.control_surfaces
    elevator_table = fr.resource.interpolator(
        "elevator",
        grid_points=[
            np.asarray(control.alpha_ele),
            np.asarray(control.beta_ele),
            np.asarray(control.stab_ele),
            np.asarray(control.elev_bp),
        ],
        values=np.asarray(control.data_ele),
        method="linear",
        n_out=3,
    )
    aileron_table = fr.resource.interpolator(
        "aileron",
        grid_points=[
            np.asarray(control.alpha_ail),
            np.asarray(control.beta_ail),
            np.asarray(control.ail_bp),
        ],
        values=np.asarray(control.data_ail),
        method="linear",
        n_out=6,
    )
    rudder_table = fr.resource.interpolator(
        "rudder",
        grid_points=[
            np.asarray(control.alpha_rud),
            np.asarray(control.beta_rud),
            np.asarray(control.rud_bp),
        ],
        values=np.asarray(control.data_rud),
        method="linear",
        n_out=6,
    )

    elevator, aileron, rudder = actuators[0], actuators[1], actuators[2]
    stabilizer = elevator_table(alpha, beta, 0.0, 0.0)
    stabilizer_c6 = (
        fr.math.vector([stabilizer[0], 0.0, stabilizer[1], 0.0, stabilizer[2], 0.0])
        * c_e
    )
    elevator_delta = 0.25 * (elevator_table(alpha, beta, 0.0, elevator) - stabilizer)
    elevator_c6 = (
        fr.math.vector(
            [
                elevator_delta[0],
                0.0,
                elevator_delta[1],
                0.0,
                elevator_delta[2],
                0.0,
            ]
        )
        * c_e
    )

    left_aileron = aileron_table(alpha, -beta, -aileron) * fr.math.vector(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )
    right_aileron = aileron_table(alpha, beta, aileron)

    rudder_sign = fr.math.sign(rudder)
    rudder_delta = -rudder_sign * rudder_table(
        alpha, -beta * rudder_sign, -fr.math.abs(rudder)
    )

    return (
        _elevator_matrix(cfg) @ elevator_c6
        + stabilizer_c6
        + (left_aileron + right_aileron) * c_a
        + rudder_delta * c_r
        + _configuration_increment(cfg.mode)
    )


def _hykal(
    alpha: Signal,
    beta: Signal,
    tas: Signal,
    pqr: Signal,
    span: float,
    chord: float,
) -> tuple[Signal, Signal]:
    alpha_rad = fr.math.radians(alpha)
    beta_rad = fr.math.radians(beta)
    p, q, r = pqr[0], pqr[1], pqr[2]
    ca = fr.math.cos(alpha_rad)
    sa = fr.math.sin(alpha_rad)
    cb = fr.math.cos(beta_rad)
    sb = fr.math.sin(beta_rad)
    cacb = ca * cb
    sacb = sa * cb

    case1 = fr.math.logic_and(fr.math.abs(cacb) > 1e-10, fr.math.abs(p) > 1e-10)
    safe_cacb = fr.math.if_else(fr.math.abs(cacb) > 1e-10, cacb, 1.0)
    omega1 = p / safe_cacb
    r_osc1 = r - omega1 * sacb
    case1 = fr.math.logic_and(
        case1,
        fr.math.logic_and(fr.math.abs(r_osc1) <= fr.math.abs(r), r_osc1 * r >= 0.0),
    )

    case2 = fr.math.logic_and(fr.math.abs(sacb) > 1e-10, fr.math.abs(r) > 1e-10)
    safe_sacb = fr.math.if_else(fr.math.abs(sacb) > 1e-10, sacb, 1.0)
    omega2 = r / safe_sacb
    p_osc2 = p - omega2 * cacb
    case2 = fr.math.logic_and(
        case2,
        fr.math.logic_and(fr.math.abs(p_osc2) <= fr.math.abs(p), p_osc2 * p >= 0.0),
    )
    case2 = fr.math.logic_and(case2, fr.math.logic_not(case1))

    omega = fr.math.if_else(case1, omega1, fr.math.if_else(case2, omega2, 0.0))
    p_osc = fr.math.if_else(case1, 0.0, fr.math.if_else(case2, p_osc2, p))
    q_osc = fr.math.if_else(fr.math.logic_or(case1, case2), q - omega * sb, q)
    r_osc = fr.math.if_else(case1, r_osc1, fr.math.if_else(case2, 0.0, r))

    span_scale = span / (2.0 * tas)
    chord_scale = chord / (2.0 * tas)
    return (
        omega * span_scale,
        fr.math.vector([p_osc, q_osc, r_osc])
        * fr.math.vector([span_scale, chord_scale, span_scale]),
    )


def _dynamic_coefficients(
    cfg: GTMConfig,
    alpha: Signal,
    beta: Signal,
    tas: Signal,
    pqr: Signal,
) -> Signal:
    dynamic = cfg.aero.dynamic_derivatives
    p_table = fr.resource.interpolator(
        "dynamic_p",
        grid_points=[np.asarray(dynamic.alpha_p), np.asarray(dynamic.phat_bp)],
        values=np.asarray(dynamic.data_p),
        method="linear",
        n_out=3,
    )
    q_table = fr.resource.interpolator(
        "dynamic_q",
        grid_points=[np.asarray(dynamic.alpha_q), np.asarray(dynamic.qhat_bp)],
        values=np.asarray(dynamic.data_q),
        method="linear",
        n_out=3,
    )
    r_table = fr.resource.interpolator(
        "dynamic_r",
        grid_points=[np.asarray(dynamic.alpha_r), np.asarray(dynamic.rhat_bp)],
        values=np.asarray(dynamic.data_r),
        method="linear",
        n_out=3,
    )
    w_table = fr.resource.interpolator(
        "dynamic_w",
        grid_points=[
            np.asarray(dynamic.alpha_w),
            np.asarray(dynamic.what_bp),
            np.asarray(dynamic.beta_w),
        ],
        values=np.asarray(dynamic.data_w),
        method="linear",
        n_out=6,
    )

    safe_tas = fr.math.clip(tas, 1.0, 10000.0)
    omega, oscillatory = _hykal(alpha, beta, safe_tas, pqr, dynamic.b, dynamic.Cbar)
    omega = fr.math.clip(omega, -0.5, 0.5)
    p_delta = p_table(alpha, oscillatory[0])
    q_delta = q_table(alpha, oscillatory[1])
    r_delta = r_table(alpha, oscillatory[2])
    w_delta = w_table(alpha, omega, beta)
    return fr.math.vector(
        [
            q_delta[0] + w_delta[0],
            p_delta[0] + r_delta[0] + w_delta[1],
            q_delta[1] + w_delta[2],
            p_delta[1] + r_delta[1] + w_delta[3],
            q_delta[2] + w_delta[4],
            p_delta[2] + r_delta[2] + w_delta[5],
        ]
    )


def aerodynamic_loads(
    cfg: GTMConfig,
    *,
    altitude: Signal,
    mach: Signal,
    qbar: Signal,
    alpha: Signal,
    beta: Signal,
    tas: Signal,
    pqr: Signal,
    actuators: Signal,
    c_static_unc: Signal,
    c_e_unc: Signal,
    c_a_unc: Signal,
    c_r_unc: Signal,
) -> AeroResult:
    """Build the aerodynamic coefficient and load expressions."""
    c_static, c_e, c_a, c_r = _uncertainty_scales(
        c_static_unc, c_e_unc, c_a_unc, c_r_unc
    )
    reynolds = _reynolds(cfg, altitude, mach)
    coefficients = (
        _basic_coefficients(cfg, alpha, beta, reynolds, c_static)
        + _control_coefficients(cfg, alpha, beta, actuators, c_e, c_a, c_r)
        + _dynamic_coefficients(cfg, alpha, beta, tas, pqr)
    )

    aircraft = cfg.aircraft
    aero_all = coefficients * qbar * aircraft.S
    force = aero_all[0:3]
    reference_lengths = fr.math.vector([aircraft.b, aircraft.Cbar, aircraft.b])
    arm = fr.math.vector(
        (np.asarray(aircraft.ref_cg) - np.asarray(aircraft.cg_pos0)).tolist()
    )
    moment = aero_all[3:6] * reference_lengths + fr.math.cross(arm, force)
    return AeroResult(coefficients, force, moment, reynolds)
