from functools import lru_cache
from pathlib import Path
from typing import Any

import casadi as ca
import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm import GTMConfig, GTMSystem, build_ground_initial_state

BACKEND_REFERENCE = np.load(
    Path(__file__).parent / "data" / "gtm_backend_trajectory.npz",
    allow_pickle=True,
)
TRAJECTORY_BACKENDS = ("numpy", "casadi", "native", "jax")


def _builder(*, enable_ground: bool = False) -> fr.PyIrBuilder:
    builder = GTMSystem(GTMConfig.default(enable_ground=enable_ground), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    return builder


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, ca.DM):
        return np.asarray(value.full(), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def _state(model: Any, x: np.ndarray) -> np.ndarray:
    return _as_numpy(
        model.x.build(
            {
                "Root.eom::pos_ned": x[0:3],
                "Root.eom::vel_body": x[3:6],
                "Root.eom::quat": x[6:10],
                "Root.eom::omega_body": x[10:13],
                "Root.actuator::position": x[13:17],
                "Root.wind.dryden::u_gw": x[17:18],
                "Root.wind.dryden::v_gw": x[18:20],
                "Root.wind.dryden::w_gw": x[20:22],
                "Root.wind.dryden::p_gw": x[22:23],
                "Root.wind.dryden::q_gw": x[23:24],
                "Root.wind.dryden::r_gw": x[24:25],
            }
        )
    )


def _parameters(model: Any, p: np.ndarray) -> np.ndarray:
    # gtm_ad has 31 runtime parameters.  fr_gtm exposes the two Dryden
    # configuration constants as additional parameters with package defaults.
    return _as_numpy(
        model.p.build(
            {
                "Root.wind.params::wind_speed": p[0:1],
                "Root.wind.params::wind_direction": np.deg2rad(p[1:2]),
                "Root.wind.params::turbulence_level": p[2:3],
                "Root.mass::weight_unc": p[3:4],
                "Root.mass::inertia_unc": p[4:10],
                "Root.aero::c_static_unc": p[10:16],
                "Root.aero::c_e_unc": p[16:19],
                "Root.aero::c_a_unc": p[19:25],
                "Root.aero::c_r_unc": p[25:31],
            }
        )
    )


def _inputs(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = _as_numpy(
        model.u.build(
            {
                "Root::u": BACKEND_REFERENCE["u"],
                "Root::d": BACKEND_REFERENCE["d"],
                "Root::n": BACKEND_REFERENCE["n"],
            }
        )
    )
    xd = _as_numpy(model.xd.build())
    p = _parameters(model, BACKEND_REFERENCE["p"])
    return xd, u, p


@lru_cache(maxsize=None)
def _compiled_backend(name: str, *, enable_ground: bool = False) -> Any:
    dt = float(BACKEND_REFERENCE["dt"])
    if name == "jax":
        pytest.importorskip("jax")
        backend: fr.Backend[Any] = fr.backends.jax
        return fr.Compiler(
            fr.CompilerSpec(backend=backend, integrator="rk4", dt=dt)
        ).compile(_builder(enable_ground=enable_ground))
    if name == "native":
        try:
            from fly_ruler_core.codegen import NativeCodegenSpec, codegen
        except ImportError as exc:
            pytest.skip(f"native codegen dependencies are not available: {exc}")
        casadi_model = fr.Compiler(
            fr.CompilerSpec(backend=fr.backends.casadi, integrator="rk4", dt=dt)
        ).compile(_builder(enable_ground=enable_ground))
        return codegen(casadi_model, NativeCodegenSpec(batch_size=1))

    backend = {"numpy": fr.backends.numpy, "casadi": fr.backends.casadi}[name]
    return fr.Compiler(
        fr.CompilerSpec(backend=backend, integrator="rk4", dt=dt)
    ).compile(_builder(enable_ground=enable_ground))


@lru_cache(maxsize=None)
def _casadi_functions(model_id: int, n_x: int, n_xd: int, n_u: int, n_p: int) -> Any:
    del model_id
    model = _compiled_backend("casadi")
    t_sym = ca.SX.sym("t")
    x_sym = ca.SX.sym("x", n_x)
    xd_sym = ca.SX.sym("xd", n_xd)
    u_sym = ca.SX.sym("u", n_u)
    p_sym = ca.SX.sym("p", n_p)
    x_next, xd_next, y_disc, effects = model.discrete_fn(
        t_sym, x_sym, xd_sym, u_sym, p_sym
    )
    xdot, xd_cont, y_cont, effects_cont = model.continuous_fn(
        t_sym, x_sym, xd_sym, u_sym, p_sym
    )
    return (
        ca.Function(
            "fr_gtm_casadi_discrete_reference_test",
            [t_sym, x_sym, xd_sym, u_sym, p_sym],
            [x_next, xd_next, y_disc, effects],
        ),
        ca.Function(
            "fr_gtm_casadi_continuous_reference_test",
            [t_sym, x_sym, xd_sym, u_sym, p_sym],
            [xdot, xd_cont, y_cont, effects_cont],
        ),
    )


@lru_cache(maxsize=None)
def _evaluate_backend(
    name: str,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = _compiled_backend(name)
    dt = float(BACKEND_REFERENCE["dt"])
    step = int(BACKEND_REFERENCE["sample_steps"][sample_index])
    t = step * dt
    x = _state(model, BACKEND_REFERENCE["x_samples"][sample_index])
    xd, u, p = _inputs(model)

    if name == "casadi":
        discrete_fn, continuous_fn = _casadi_functions(
            id(model),
            model.module.n_x,
            model.module.n_xd,
            model.module.n_u,
            model.module.n_p,
        )
        x_next, _, _, _ = discrete_fn(t, x, xd, u, p)
        _, _, y, _ = continuous_fn(t, x, xd, u, p)
        y_outputs = model.y.read(_as_numpy(y), by="path")
        return _as_numpy(x_next), _as_numpy(y_outputs["Root::y"])

    x_next, _, _, _ = model.discrete_fn(t, x, xd, u, p)
    _, _, y, _ = model.continuous_fn(t, x, xd, u, p)
    y_outputs = model.y.read(_as_numpy(y), by="path")
    return _as_numpy(x_next), _as_numpy(y_outputs["Root::y"])


def test_casadi_matches_numpy() -> None:
    numpy_model = fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.numpy, integrator="euler", dt=0.01)
    ).compile(_builder())
    casadi_model = fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.casadi, integrator="euler", dt=0.01)
    ).compile(_builder())

    x = numpy_model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -1000.0],
            "Root.eom::vel_body": [100.0, 0.0, 5.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [-2.0, 1.0, 0.0, 40.0],
        }
    )
    u = numpy_model.u.build(
        {
            "Root::u": [-2.0, 1.0, 0.0, 40.0],
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    p = numpy_model.p.build()
    numpy_xdot, _, numpy_y, _ = numpy_model.continuous_fn(
        0.0, x, numpy_model.xd.build(), u, p
    )

    x_symbol = ca.SX.sym("x", casadi_model.module.n_x)
    u_symbol = ca.SX.sym("u", casadi_model.module.n_u)
    p_symbol = ca.SX.sym("p", casadi_model.module.n_p)
    casadi_xdot, _, casadi_y, _ = casadi_model.continuous_fn(
        0.0,
        x_symbol,
        ca.SX.zeros(casadi_model.module.n_xd),
        u_symbol,
        p_symbol,
    )
    evaluate = ca.Function(
        "gtm_backend_test",
        [x_symbol, u_symbol, p_symbol],
        [casadi_xdot, casadi_y],
    )
    xdot_value, y_value = evaluate(x, u, p)
    np.testing.assert_allclose(
        _as_numpy(xdot_value), _as_numpy(numpy_xdot), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        _as_numpy(y_value), _as_numpy(numpy_y), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("backend", TRAJECTORY_BACKENDS)
def test_ground_contact_one_step_matches_numpy(backend: str) -> None:
    cfg = GTMConfig.default(enable_ground=True)
    reference = _compiled_backend("numpy", enable_ground=True)
    candidate = _compiled_backend(backend, enable_ground=True)
    reference_x = build_ground_initial_state(
        reference,
        cfg,
        forward_speed_mps=20.0,
    )
    candidate_x = build_ground_initial_state(
        candidate,
        cfg,
        forward_speed_mps=20.0,
    )
    reference_u = reference.u.build(
        {"Root::u": np.zeros(4), "Root::d": np.zeros(4), "Root::n": np.zeros(21)}
    )
    candidate_u = candidate.u.build(
        {"Root::u": np.zeros(4), "Root::d": np.zeros(4), "Root::n": np.zeros(21)}
    )
    reference_result = reference.discrete_fn(
        0.0,
        reference_x,
        reference.xd.build(),
        reference_u,
        reference.p.default(),
    )
    if backend == "casadi":
        t_sym = ca.SX.sym("ground_t")
        x_sym = ca.SX.sym("ground_x", candidate.module.n_x)
        xd_sym = ca.SX.sym("ground_xd", candidate.module.n_xd)
        u_sym = ca.SX.sym("ground_u", candidate.module.n_u)
        p_sym = ca.SX.sym("ground_p", candidate.module.n_p)
        step_fn = ca.Function(
            "fr_gtm_ground_backend_test",
            [t_sym, x_sym, xd_sym, u_sym, p_sym],
            list(candidate.discrete_fn(t_sym, x_sym, xd_sym, u_sym, p_sym)),
        )
        candidate_result = step_fn(
            0.0,
            candidate_x,
            candidate.xd.build(),
            candidate_u,
            candidate.p.default(),
        )
    else:
        candidate_result = candidate.discrete_fn(
            0.0,
            candidate_x,
            candidate.xd.build(),
            candidate_u,
            candidate.p.default(),
        )
    rtol = 1.0e-7 if backend == "jax" else 1.0e-9
    atol = 1.0e-8 if backend == "jax" else 1.0e-9
    np.testing.assert_allclose(
        _as_numpy(candidate_result[0]),
        _as_numpy(reference_result[0]),
        rtol=rtol,
        atol=atol,
    )
    reference_outputs = reference.y.read(_as_numpy(reference_result[2]), by="path")
    candidate_outputs = candidate.y.read(_as_numpy(candidate_result[2]), by="path")
    for path in (
        "Root::ground_force_body",
        "Root::ground_moment_body",
        "Root::ground_normal_forces",
        "Root::ground_longitudinal_forces",
        "Root::ground_lateral_forces",
    ):
        np.testing.assert_allclose(
            _as_numpy(candidate_outputs[path]),
            _as_numpy(reference_outputs[path]),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("backend", TRAJECTORY_BACKENDS)
@pytest.mark.parametrize(
    "sample_index",
    range(len(BACKEND_REFERENCE["sample_steps"])),
    ids=[f"t={step / 120.0:.0f}s" for step in BACKEND_REFERENCE["sample_steps"]],
)
def test_backend_one_step_matches_frozen_gtm_ad_trajectory(
    backend: str,
    sample_index: int,
) -> None:
    x_next, y = _evaluate_backend(backend, sample_index)
    ref_x_rtol = 1e-6
    ref_x_atol = 3e-4
    ref_y_rtol = 1e-6
    ref_y_atol = 1e-3
    backend_rtol = 1e-9
    backend_atol = 1e-9
    if backend == "jax":
        # JAX still agrees closely with x64 enabled, but its generated code
        # can take a slightly different floating-point path from NumPy/CasADi.
        backend_rtol = 1e-7
        backend_atol = 1e-8

    if backend != "numpy":
        numpy_x_next, numpy_y = _evaluate_backend("numpy", sample_index)
        np.testing.assert_allclose(
            x_next,
            numpy_x_next,
            rtol=backend_rtol,
            atol=backend_atol,
        )
        np.testing.assert_allclose(
            y,
            numpy_y,
            rtol=backend_rtol,
            atol=backend_atol,
        )

    model = _compiled_backend(backend)
    expected_x_next = _state(model, BACKEND_REFERENCE["x_next_samples"][sample_index])
    np.testing.assert_allclose(
        x_next,
        expected_x_next,
        rtol=ref_x_rtol,
        atol=ref_x_atol,
    )
    np.testing.assert_allclose(
        y,
        BACKEND_REFERENCE["y_samples"][sample_index],
        rtol=ref_y_rtol,
        atol=ref_y_atol,
    )


def test_zero_rudder_retains_local_control_effectiveness() -> None:
    """The piecewise rudder table must have a usable derivative at trim."""

    model = _compiled_backend("casadi")
    x_sym = ca.SX.sym("x", model.n_x)
    u_sym = ca.SX.sym("u", model.n_u)
    _, _, y_sym, _ = model.continuous_fn(
        ca.SX(0.0), x_sym, ca.SX.zeros(model.n_xd), u_sym, model.p.default()
    )
    actuator_row = next(
        row for row in model.x.layout_rows() if row["path"] == "Root.actuator::position"
    )
    nz_row = next(row for row in model.y.layout_rows() if row["path"] == "Root::nz")
    derivative = ca.jacobian(
        y_sym[int(nz_row["start"])],
        x_sym[int(actuator_row["start"]) + 2],
    )
    derivative_fn = ca.Function(
        "zero_rudder_control_effectiveness", [x_sym, u_sym], [derivative]
    )
    x = _state(model, BACKEND_REFERENCE["x_samples"][0])
    x[int(actuator_row["start"]) + 2] = 0.0
    _, u, _ = _inputs(model)
    value = float(derivative_fn(x, u))
    assert np.isfinite(value)
    assert abs(value) > 1.0e-10


def test_zero_speed_airdata_jacobian_is_finite() -> None:
    """Regularized fixed-wing air data must remain differentiable at rest."""

    model = _compiled_backend("casadi")
    x_sym = ca.SX.sym("zero_speed_x", model.n_x)
    u_sym = ca.SX.sym("zero_speed_u", model.n_u)
    xdot, _, y_sym, _ = model.continuous_fn(
        ca.SX(0.0), x_sym, ca.SX.zeros(model.n_xd), u_sym, model.p.default()
    )
    selected_outputs = []
    for path in ("Root::V", "Root::alpha", "Root::beta"):
        row = next(item for item in model.y.layout_rows() if item["path"] == path)
        selected_outputs.append(y_sym[int(row["start"]) : int(row["stop"])])
    derivative = ca.Function(
        "gtm_zero_speed_derivative",
        [x_sym, u_sym],
        [ca.jacobian(ca.vertcat(xdot, *selected_outputs), x_sym), y_sym],
    )
    x = _state(model, BACKEND_REFERENCE["x_samples"][0])
    velocity = next(
        row for row in model.x.layout_rows() if row["path"] == "Root.eom::vel_body"
    )
    x[int(velocity["start"]) : int(velocity["stop"])] = 0.0
    _, u, _ = _inputs(model)
    jacobian, outputs = derivative(x, u)
    assert np.isfinite(np.asarray(jacobian)).all()
    assert np.isfinite(np.asarray(outputs)).all()
