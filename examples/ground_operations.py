"""Simulate a GTM takeoff roll or touchdown with shared std ground contact."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, cast

import casadi as ca
import fly_ruler_core as fr
import numpy as np
import typer
from fly_ruler_core.codegen import NativeCodegenSpec, codegen
from fly_ruler_std.runtime import Recorder

from fr_gtm import GTMConfig, GTMSystem, build_ground_initial_state


def main(
    scenario: Annotated[str, typer.Option(help="takeoff or touchdown")] = "takeoff",
    backend: Annotated[
        str, typer.Option(help="numpy, casadi, jax, or native")
    ] = "numpy",
    duration: Annotated[float, typer.Option(min=0.001)] = 8.0,
    dt: Annotated[float, typer.Option(min=0.0001)] = 0.01,
    output_dir: Annotated[Path, typer.Option()] = Path("results/gtm_ground"),
) -> None:
    """Run one deterministic runway-relative GTM scenario."""

    if scenario not in {"takeoff", "touchdown"}:
        raise typer.BadParameter("scenario must be takeoff or touchdown")
    cfg = GTMConfig.default(enable_ground=True)
    builder = cast(fr.PyIrBuilder, GTMSystem(cfg, name="Root"))
    model: fr.CompiledModel[object]
    casadi_step: Callable[..., tuple[object, object, object, object]] | None = None
    if backend == "numpy":
        model = cast(
            fr.CompiledModel[object],
            fr.Compiler(
                fr.CompilerSpec(backend=fr.backends.numpy, integrator="rk4", dt=dt)
            ).compile(builder),
        )
    elif backend == "casadi":
        model = cast(
            fr.CompiledModel[object],
            fr.Compiler(
                fr.CompilerSpec(backend=fr.backends.casadi, integrator="rk4", dt=dt)
            ).compile(builder),
        )
        t_symbol = ca.SX.sym("t")
        x_symbol = ca.SX.sym("x", model.module.n_x)
        xd_symbol = ca.SX.sym("xd", model.module.n_xd)
        u_symbol = ca.SX.sym("u", model.module.n_u)
        p_symbol = ca.SX.sym("p", model.module.n_p)
        casadi_step = cast(
            Callable[..., tuple[object, object, object, object]],
            ca.Function(
                "fr_gtm_ground_step",
                [t_symbol, x_symbol, xd_symbol, u_symbol, p_symbol],
                list(
                    model.discrete_fn(t_symbol, x_symbol, xd_symbol, u_symbol, p_symbol)
                ),
            ),
        )
    elif backend == "jax":
        fr.configure_jax_runtime(
            fr.JaxRuntimePolicy(
                platforms=("cpu",),
                preallocate=False,
                enable_x64=True,
            )
        )
        model = cast(
            fr.CompiledModel[object],
            fr.Compiler(
                fr.CompilerSpec(backend=fr.backends.jax, integrator="rk4", dt=dt)
            ).compile(builder),
        )
    elif backend == "native":
        symbolic = fr.Compiler(
            fr.CompilerSpec(backend=fr.backends.casadi, integrator="rk4", dt=dt)
        ).compile(builder)
        model = cast(fr.CompiledModel[object], codegen(symbolic, NativeCodegenSpec()))
    else:
        raise typer.BadParameter("backend must be numpy, casadi, jax, or native")

    if scenario == "takeoff":
        x = build_ground_initial_state(model, cfg, forward_speed_mps=0.0)
    else:
        x = build_ground_initial_state(
            model,
            cfg,
            forward_speed_mps=70.0,
            altitude_agl_m=1.0,
            sink_rate_mps=1.5,
        )
    xd = np.asarray(model.xd.build(), dtype=float)
    p = np.asarray(model.p.default(), dtype=float)
    recorder = Recorder(model, output_dir)
    y: np.ndarray | None = None
    steps = max(1, round(duration / dt))
    for step in range(steps):
        velocity = model.x.read(x, by="path")["Root.eom::vel_body"]
        speed = float(np.linalg.norm(np.asarray(velocity, dtype=float)))
        command = (
            np.array([-5.0 if speed >= 65.0 else 0.0, 0.0, 0.0, 100.0])
            if scenario == "takeoff"
            else np.array([0.0, 0.0, 0.0, 25.0])
        )
        u = np.asarray(
            model.u.build(
                {
                    "Root::u": command,
                    "Root::d": np.zeros(4),
                    "Root::n": np.zeros(21),
                }
            ),
            dtype=float,
        )
        if casadi_step is None:
            x_next, xd_next, y_next, effects_next = model.discrete_fn(
                step * dt, x, xd, u, p
            )
        else:
            x_next, xd_next, y_next, effects_next = casadi_step(step * dt, x, xd, u, p)
        x = np.asarray(x_next, dtype=float).reshape(-1)
        xd = np.asarray(xd_next, dtype=float).reshape(-1)
        y = np.asarray(y_next, dtype=float).reshape(-1)
        effects = np.asarray(effects_next, dtype=float).reshape(-1)
        recorder.record(
            step, (step + 1) * dt, x=x, xd=xd, y=y, u=u, p=p, effect=effects
        )
    recorder.save(scenario)
    if y is None:
        raise RuntimeError("simulation produced no output")
    outputs = model.y.read(y, by="path")
    print(f"saved: {output_dir / f'{scenario}.parquet'}")
    altitude = float(np.asarray(outputs["Root::altitude"], dtype=float).reshape(-1)[0])
    wheel_loads = np.asarray(
        outputs["Root::ground_normal_forces"], dtype=float
    ).reshape(-1)
    print(f"final altitude: {altitude:.3f} m")
    print(
        "wheel loads:",
        wheel_loads,
    )


if __name__ == "__main__":
    typer.run(main)
