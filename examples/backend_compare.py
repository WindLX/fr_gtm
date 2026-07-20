"""Compare one GTM evaluation across NumPy, CasADi, and optional JAX."""

import os
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    backends: list[tuple[str, fr.Backend[Any]]] = [
        ("numpy", fr.backends.numpy),
        ("casadi", fr.backends.casadi),
    ]
    if find_spec("jax") is not None:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_ENABLE_X64"] = "1"
        backends.append(("jax", fr.backends.jax))
    else:
        print("JAX is not installed; skipping it.")

    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name, backend in backends:
        started = perf_counter()
        model = fr.Compiler(
            fr.CompilerSpec(
                backend=backend,
                integrator="euler",
                dt=0.01,
                artifact_dir=ROOT / "build" / "flyruler",
            )
        ).compile(cast(fr.PyIrBuilder, GTMSystem(GTMConfig.default(), name="Root")))
        compile_ms = (perf_counter() - started) * 1000.0
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
        if name == "casadi":
            import casadi as ca

            x_sym = ca.SX.sym("x", model.module.n_x)
            u_sym = ca.SX.sym("u", model.module.n_u)
            p_sym = ca.SX.sym("p", model.module.n_p)
            xdot_expr, _, y_expr, _ = model.continuous_fn(
                0.0, x_sym, ca.SX.zeros(model.module.n_xd), u_sym, p_sym
            )
            evaluate = ca.Function(
                "gtm_compare", [x_sym, u_sym, p_sym], [xdot_expr, y_expr]
            )
            xdot, y = evaluate(x, u, model.p.default())
        else:
            xdot, _, y, _ = model.continuous_fn(
                0.0, x, model.xd.build(), u, model.p.default()
            )
        results[name] = (
            compile_ms,
            np.asarray(xdot, dtype=float).reshape(-1),
            np.asarray(y, dtype=float).reshape(-1),
        )

    _, xdot_ref, y_ref = results["numpy"]
    print(f"{'backend':>8}  {'compile/ms':>10}  {'max|xdot|':>12}  {'max|y|':>12}")
    for name, (compile_ms, xdot, y) in results.items():
        print(
            f"{name:>8}  {compile_ms:10.1f}  "
            f"{np.max(np.abs(xdot - xdot_ref)):12.3e}  "
            f"{np.max(np.abs(y - y_ref)):12.3e}"
        )


if __name__ == "__main__":
    main()
