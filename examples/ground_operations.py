"""用共享的 std 地面接触模型仿真 GTM 起飞滑跑或接地。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypeVar

import fly_ruler_core as fr
import numpy as np
import typer
from fly_ruler_std.runtime.record import Recorder

from fr_gtm import GTMConfig, GTMSystem, build_ground_initial_state

ArrayT = TypeVar("ArrayT")


def _run(
    model: fr.CompiledModel[ArrayT],
    cfg: GTMConfig,
    scenario: str,
    duration: float,
    dt: float,
    output_dir: Path,
) -> None:
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
        discrete = model.discrete_fn
        if discrete is None:
            raise RuntimeError("model has no discrete function")
        x_next, xd_next, y_next, effects_next = discrete(step * dt, x, xd, u, p)
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
    print("wheel loads:", wheel_loads)


def main(
    scenario: Annotated[str, typer.Option(help="takeoff or touchdown")] = "takeoff",
    backend: Annotated[
        str, typer.Option(help="numpy, casadi, jax, or native")
    ] = "numpy",
    duration: Annotated[float, typer.Option(min=0.001)] = 8.0,
    dt: Annotated[float, typer.Option(min=0.0001)] = 0.01,
    output_dir: Annotated[Path, typer.Option()] = Path("results/gtm_ground"),
) -> None:
    """运行一次确定的跑道相对 GTM 场景。

    Args:
        scenario: 场景名，取 `takeoff` 或 `touchdown`。
        backend: 数值后端，取 `numpy`、`casadi`、`jax` 或 `native`。
        duration: 仿真时长，单位 s，必须为正。
        dt: 积分步长，单位 s，必须为正。
        output_dir: 结果输出目录。
    """

    if scenario not in {"takeoff", "touchdown"}:
        raise typer.BadParameter("scenario must be takeoff or touchdown")
    cfg = GTMConfig.default(enable_ground=True)
    builder = GTMSystem.build("Root", cfg)
    if backend == "numpy":
        _run(
            fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=dt)).compile(
                builder, target=fr.targets.numpy
            ),
            cfg,
            scenario,
            duration,
            dt,
            output_dir,
        )
    elif backend == "casadi":
        _run(
            fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=dt)).compile(
                builder, target=fr.targets.casadi
            ),
            cfg,
            scenario,
            duration,
            dt,
            output_dir,
        )
    elif backend == "jax":
        fr.configure_jax_runtime(
            fr.JaxRuntimePolicy(
                platforms=("cpu",),
                preallocate=False,
                enable_x64=True,
            )
        )
        _run(
            fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=dt)).compile(
                builder, target=fr.targets.jax
            ),
            cfg,
            scenario,
            duration,
            dt,
            output_dir,
        )
    elif backend == "native":
        compiler = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=dt))
        symbolic = compiler.compile(builder, target=fr.targets.casadi)
        _run(
            compiler.convert(symbolic, target=fr.targets.NativeTarget()),
            cfg,
            scenario,
            duration,
            dt,
            output_dir,
        )
    else:
        raise typer.BadParameter("backend must be numpy, casadi, jax, or native")


if __name__ == "__main__":
    typer.run(main)
