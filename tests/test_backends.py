from typing import Any

import casadi as ca
import fly_ruler_core as fr
import numpy as np
from fly_ruler_core._core import PyCompiler
from fly_ruler_core.codegen import JAXCodeGen

from fr_gtm import GTMConfig, GTMSystem


def _builder() -> fr.PyIrBuilder:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    return builder


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "full"):
        return np.asarray(value.full(), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


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


def test_jax_source_is_valid_python() -> None:
    plan = PyCompiler.compile_from_builder(_builder())
    source = JAXCodeGen().generate_source(plan, dt=0.01, integrator="euler")
    compile(source, "<fr_gtm_jax>", "exec")
