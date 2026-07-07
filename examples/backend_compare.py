"""Compare fr_gtm CodeGen backends on one continuous dynamics evaluation."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import casadi as ca
import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DT = 0.01
INTEGRATOR = "euler"


@dataclass
class BackendResult:
    backend: str
    compile_ms: float
    xdot: np.ndarray
    y: np.ndarray
    source_path: Path


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "full"):
        return np.asarray(value.full(), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def _compile(backend: fr.Backend[Any]) -> tuple[fr.CompiledModel[Any], float]:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    t0 = perf_counter()
    model = fr.Compiler(
        fr.CompilerSpec(
            backend=backend,
            integrator=INTEGRATOR,
            dt=DT,
            output_dir=str(PROJECT_ROOT),
        )
    ).compile(builder)
    return model, (perf_counter() - t0) * 1000.0


def _backend_obj(name: str) -> fr.Backend[Any]:
    return {
        "numpy": fr.backends.numpy,
        "jax": fr.backends.jax,
        "casadi": fr.backends.casadi,
    }[name]


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


def _generated_source_path(backend: str) -> Path:
    return PROJECT_ROOT / f"generated_{backend}_flyruler.py"


def _compile_and_eval(backend: str) -> BackendResult:
    model, compile_ms = _compile(_backend_obj(backend))
    x, u, p = _inputs(model)

    if backend == "casadi":
        x_sym = ca.SX.sym("x", model.module.n_x)
        u_sym = ca.SX.sym("u", model.module.n_u)
        p_sym = ca.SX.sym("p", model.module.n_p)
        xdot_expr, _, y_expr, _ = model.continuous_fn(
            0.0,
            x_sym,
            ca.SX.zeros(model.module.n_xd),
            u_sym,
            p_sym,
        )
        evaluate = ca.Function(
            "gtm_compare",
            [x_sym, u_sym, p_sym],
            [xdot_expr, y_expr],
        )
        xdot, y = evaluate(x, u, p)
    else:
        xdot, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, p)

    return BackendResult(
        backend=backend,
        compile_ms=compile_ms,
        xdot=_array(xdot),
        y=_array(y),
        source_path=_generated_source_path(backend),
    )


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def main() -> None:
    backends = ["casadi", "numpy"]
    if importlib.util.find_spec("jax") is not None:
        backends.append("jax")
    else:
        print("JAX is not installed; skipping JAX runtime comparison.")

    results: list[BackendResult] = []
    for backend in backends:
        print(f"Compiling {backend} backend...")
        results.append(_compile_and_eval(backend))

    ref = next(result for result in results if result.backend == "numpy")
    print("\nBackend comparison against NumPy")
    print(f"{'backend':>8}  {'compile/ms':>10}  {'max|xdot|':>12}  {'max|y|':>12}")
    for result in results:
        print(
            f"{result.backend:>8}  "
            f"{result.compile_ms:10.1f}  "
            f"{_max_abs_diff(result.xdot, ref.xdot):12.3e}  "
            f"{_max_abs_diff(result.y, ref.y):12.3e}"
        )

    print("\nGenerated files:")
    for result in results:
        print(f"  {result.backend}: {result.source_path}")


if __name__ == "__main__":
    main()
