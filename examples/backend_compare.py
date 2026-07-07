"""Compare NumPy and CasADi on one GTM continuous evaluation."""

from typing import Any

import casadi as ca
import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "full"):
        return np.asarray(value.full(), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def _compile(backend: fr.Backend[Any]) -> fr.CompiledModel[Any]:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    return fr.Compiler(
        fr.CompilerSpec(backend=backend, integrator="euler", dt=0.01)
    ).compile(builder)


def _inputs(model: fr.CompiledModel[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -1000.0],
            "Root.eom::vel_body": [100.0, 0.0, 5.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [-2.0, 1.0, 0.0, 40.0],
        }
    )
    u = model.u.build(
        {
            "Root::u": [-2.0, 1.0, 0.0, 40.0],
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    return x, u, model.p.build()


def main() -> None:
    numpy_model = _compile(fr.backends.numpy)
    x, u, p = _inputs(numpy_model)
    nx, _, ny, _ = numpy_model.continuous_fn(0.0, x, numpy_model.xd.build(), u, p)

    casadi_model = _compile(fr.backends.casadi)
    x_sym = ca.SX.sym("x", casadi_model.module.n_x)
    u_sym = ca.SX.sym("u", casadi_model.module.n_u)
    p_sym = ca.SX.sym("p", casadi_model.module.n_p)
    cx, _, cy, _ = casadi_model.continuous_fn(
        0.0,
        x_sym,
        ca.SX.zeros(casadi_model.module.n_xd),
        u_sym,
        p_sym,
    )
    evaluate = ca.Function("gtm_compare", [x_sym, u_sym, p_sym], [cx, cy])
    cx_value, cy_value = evaluate(x, u, p)

    xdot_error = np.max(np.abs(_array(nx) - _array(cx_value)))
    output_error = np.max(np.abs(_array(ny) - _array(cy_value)))
    print(f"max |xdot_numpy-xdot_casadi|: {xdot_error:.3e}")
    print(f"max |y_numpy-y_casadi|:       {output_error:.3e}")


if __name__ == "__main__":
    main()
