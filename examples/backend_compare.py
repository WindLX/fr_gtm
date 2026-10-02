"""在 NumPy、CasADi 与可选 JAX 后端之间比较同一次 GTM 求值。"""

from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import TypeVar

import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem

ROOT = Path(__file__).resolve().parent.parent
ArrayT = TypeVar("ArrayT")


def _evaluate(model: fr.CompiledModel[ArrayT]) -> tuple[np.ndarray, np.ndarray]:
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
    xdot, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.default())
    return (
        np.asarray(xdot, dtype=float).reshape(-1),
        np.asarray(y, dtype=float).reshape(-1),
    )


def main() -> None:
    names = ["numpy", "casadi"]
    if find_spec("jax") is not None:
        fr.configure_jax_runtime(
            fr.JaxRuntimePolicy(
                platforms=("cpu",),
                preallocate=False,
                enable_x64=True,
            )
        )
        names.append("jax")
    else:
        print("JAX is not installed; skipping it.")

    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name in names:
        started = perf_counter()
        compiler = fr.Compiler(
            fr.CompilerSpec(
                integrator="euler", dt=0.01, artifact_dir=ROOT / "build" / "flyruler"
            )
        )
        builder = GTMSystem.build("Root", GTMConfig.default())
        if name == "numpy":
            xdot, y = _evaluate(compiler.compile(builder, target=fr.targets.numpy))
        elif name == "casadi":
            xdot, y = _evaluate(compiler.compile(builder, target=fr.targets.casadi))
        else:
            xdot, y = _evaluate(compiler.compile(builder, target=fr.targets.jax))
        compile_ms = (perf_counter() - started) * 1000.0
        results[name] = (
            compile_ms,
            xdot,
            y,
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
