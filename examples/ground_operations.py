"""用共享的 std 地面接触模型仿真 GTM 起飞滑跑或接地。

主题：开启地面接触后编译 GTM，按场景装配跑道相对初值，用固定作动器指令推进若干步，
并把轨迹与地面法向力写入 Parquet。

运行：

    cd models/fr_gtm && just example-ground
    cd models/fr_gtm && just example-ground --scenario touchdown --duration 3

预期结果：打印 `saved: ...parquet`、末态高度与三个起落架接地点（鼻轮、左主轮、右主轮）
的法向力；takeoff 场景在速度达到 65 m/s 后抬前轮，touchdown 场景以 70 m/s、1.5 m/s
下降率接地。

约定：地面接触是编译期开关，必须用 GTMConfig.default(enable_ground=True)；位置取
世界系 NED，速度在机体系 FRD，高度为 -pos_ned[2]，作动器指令末位为油门百分比。
"""

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
    """按场景装配跑道相对初值，推进指定时长并记录轨迹与地面力。"""
    # #region 02-state-input-1
    # takeoff 从静止接地状态开始滑跑；touchdown 从 1 m 高度、70 m/s 前向速度、
    # 1.5 m/s 下降率开始接地，两种初值都让最低接地点贴近跑道参考面。
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
    # #endregion 02-state-input-1
    # 显式转成 float 数组，避免不同后端返回类型在循环里混用。
    xd = model.to_numpy(model.xd.build())
    p = model.to_numpy(model.p.default())
    recorder = Recorder(model, output_dir)
    y: np.ndarray | None = None
    steps = max(1, round(duration / dt))
    # 每步按机体系速度模值决定指令：takeoff 到 65 m/s 时给 -5 度升降舵抬前轮并全油门，
    # touchdown 保持零升降舵与 25% 油门；这样可以复现一条确定的跑道轨迹。
    for step in range(steps):
        velocity = model.x.read(x, by="path")["Root.eom::vel_body"]
        speed = float(np.linalg.norm(model.to_numpy(velocity)))
        command = (
            np.array([-5.0 if speed >= 65.0 else 0.0, 0.0, 0.0, 100.0])
            if scenario == "takeoff"
            else np.array([0.0, 0.0, 0.0, 25.0])
        )
        u = model.to_numpy(
            model.u.build(
                {
                    "Root::u": command,
                    "Root::d": np.zeros(4),
                    "Root::n": np.zeros(21),
                }
            )
        )
        discrete = model.discrete_fn
        if discrete is None:
            raise RuntimeError("model has no discrete function")
        # 一步推进后用 reshape(-1) 抹平后端差异，再把状态与代数量写入记录器。
        x_next, xd_next, y_next, effects_next = discrete(step * dt, x, xd, u, p)
        x = model.to_numpy(x_next).reshape(-1)
        xd = model.to_numpy(xd_next).reshape(-1)
        y = model.to_numpy(y_next).reshape(-1)
        effects = model.to_numpy(effects_next).reshape(-1)
        recorder.record(
            step, (step + 1) * dt, x=x, xd=xd, y=y, u=u, p=p, effect=effects
        )
    recorder.save(scenario)
    if y is None:
        raise RuntimeError("simulation produced no output")
    # 开启地面接触后输出里才有地面诊断通道，这里取高度与三个接地点的法向力。
    outputs = model.y.read(y, by="path")
    print(f"saved: {output_dir / f'{scenario}.parquet'}")
    altitude = float(model.to_numpy(outputs["Root::altitude"]).reshape(-1)[0])
    wheel_loads = model.to_numpy(outputs["Root::ground_normal_forces"]).reshape(-1)
    print(f"final altitude: {altitude:.3f} m")
    print("wheel loads:", wheel_loads)


def main(
    scenario: Annotated[str, typer.Option(help="takeoff or touchdown")] = "takeoff",
    backend: Annotated[
        str, typer.Option(help="numpy, casadi, jax, or native")
    ] = "casadi",
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
    # 地面接触是编译期开关，必须先在这里开启再构建系统，输出才会带地面诊断通道。
    cfg = GTMConfig.default(enable_ground=True)
    builder = GTMSystem.build("Root", cfg)
    # 按后端分别编译同一条 IR；每个分支保留自己的精确模型类型。
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
    # #region 04-scenarios-1
    # native 不能直接编译，需要先生成 CasADi 符号模型，再转换成本地目标。
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
    # #endregion 04-scenarios-1
    else:
        raise typer.BadParameter("backend must be numpy, casadi, jax, or native")


if __name__ == "__main__":
    typer.run(main)
