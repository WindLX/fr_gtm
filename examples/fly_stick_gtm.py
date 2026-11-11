"""读取 fly_stick、推进 GTM，再通过 Proto 发送真值与选定诊断。

运行 just example-fly-stick；无硬件、无网络用 --dry-run --backend numpy。
接口转换分别由 fr_gtm.interface.stick/proto 负责，示例管理连接、调度和记录。
"""

from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core.runtime.instance import RuntimeInstance
from fly_ruler_std.runtime.loop import LoopTask, SimulationContext, run_async_realtime
from fly_ruler_std.runtime.record import Recorder

from fr_gtm import GTMConfig, GTMPaths, GTMSystem, build_input, init, read_observation

if TYPE_CHECKING:
    from fly_ruler_proto_python import FlyRulerClient
    from fly_stick import DeviceDescription, PyDevicePool

    from fr_gtm.interface.proto import EffectsTelemetry

DEFAULT_PROFILE = (
    Path(__file__).resolve().parent / "fly_stick_profiles" / "t16000m.toml"
)


@dataclass
class Options:
    address: str
    name: str
    hz: float
    record_hz: float
    integrator: str
    backend: str
    duration: float
    profile: Path
    logical_name: str
    origin_lat: float
    origin_lon: float
    heartbeat: float
    control_scale: float
    axis_deadband: float
    initial_altitude: float
    initial_speed: float
    nominal_throttle: float
    manual_command: tuple[float, float, float, float] | None
    invert_roll: bool
    invert_pitch: bool
    invert_rudder: bool
    invert_throttle: bool
    output_dir: Path
    record_name: str
    no_save: bool
    no_stick: bool
    no_proto: bool
    dry_run: bool


async def _run[T](model: RuntimeInstance[T], cfg: GTMConfig, args: Options) -> int:
    """运行已编译模型；硬件故障直接结束循环，并在 finally 中释放资源。"""
    paths = GTMPaths()
    lower = np.array([*cfg.actuator.position_lower, cfg.engine.lag.command_lower])
    upper = np.array([*cfg.actuator.position_upper, cfg.engine.lag.command_upper])
    command = np.clip(
        args.manual_command
        if args.manual_command is not None
        else (0.0, 0.0, 0.0, args.nominal_throttle),
        lower,
        upper,
    )
    x, xd, u, p = init(
        model,
        cfg,
        position=(0.0, 0.0, -args.initial_altitude),
        airdata=(0.0, 0.0, args.initial_speed),
        command=command,
    )
    initial = model.continuous_fn(0.0, x, xd, u, p)
    y, effects = initial.y, initial.effects
    pool: PyDevicePool | None = None
    description: DeviceDescription | None = None
    client: FlyRulerClient | None = None
    telemetry: EffectsTelemetry[T] | None = None
    ctx = SimulationContext(dt=1.0 / args.hz)
    recorder = Recorder(model, args.output_dir)
    wall_start = time.monotonic()
    trigger = False
    previous_trigger = False

    def stop(signum: int, _frame: FrameType | None) -> None:
        ctx.request_stop(f"signal {signum}")

    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        if not args.no_stick:
            from fly_stick import DeviceDescription, PyDevicePool

            from fr_gtm.interface.stick import StickConfig, to_command

            mapping = StickConfig(
                invert_roll=args.invert_roll,
                invert_pitch=args.invert_pitch,
                invert_yaw=args.invert_rudder,
                invert_throttle=args.invert_throttle,
                control_scale=args.control_scale,
                nominal_throttle=args.nominal_throttle,
                axis_deadband=args.axis_deadband,
            )
            description = DeviceDescription.from_toml(str(args.profile))
            pool = PyDevicePool({args.logical_name: description}, debounce_seconds=0.02)
            await pool.reset()
        if not args.no_proto:
            from fly_ruler_proto_python import FlyRulerClient

            from fr_gtm.interface.proto import (
                VISUAL_MODEL_ID,
                EffectsTelemetry,
                to_aircraft_state,
            )

            telemetry = EffectsTelemetry(
                model,
                paths=(
                    paths.effects.aero_force,
                    paths.effects.engine_force,
                    paths.effects.aero_coefficients,
                ),
                nominal_rate_hz=args.hz,
            )
            client = FlyRulerClient(
                args.address,
                args.name,
                initial_state=to_aircraft_state(
                    model,
                    cfg,
                    x=x,
                    y=y,
                    effects=effects,
                    origin=(args.origin_lat, args.origin_lon, 0.0),
                    paths=paths,
                ),
                toml_config=f"[aircraft]\nmodel='{VISUAL_MODEL_ID}'\nrole='stick_open_loop'",
                heartbeat_interval_secs=args.heartbeat,
                telemetry_schemas=(telemetry.schema,),
            )
            print(f"connected: aircraft_uuid={client.aircraft_uuid}")

        def input_task(_ctx: SimulationContext) -> None:
            nonlocal command, u, trigger
            if pool is not None and description is not None:
                snapshot = pool.fetch_nowait()[args.logical_name]
                trigger = bool(
                    snapshot.get_alias_buttons(description).get("BTN_TRIGGER", 0)
                )
                if args.manual_command is None:
                    command = to_command(snapshot, description, cfg, config=mapping)
            u = build_input(model, command, paths=paths)

        def aircraft_task(task_ctx: SimulationContext) -> None:
            nonlocal x, xd, y, effects
            x, xd, y, effects = model.discrete_fn(task_ctx.time, x, xd, u, p)

        def output_task(_ctx: SimulationContext) -> None:
            nonlocal previous_trigger
            if client is not None and telemetry is not None:
                timestamp = time.time()
                client.update_state(
                    to_aircraft_state(
                        model,
                        cfg,
                        x=x,
                        y=y,
                        effects=effects,
                        origin=(args.origin_lat, args.origin_lon, 0.0),
                        paths=paths,
                    ),
                    timestamp=timestamp,
                )
                client.publish_telemetry(
                    telemetry.stream_id, telemetry.values(effects), timestamp=timestamp
                )
                if trigger and not previous_trigger:
                    client.create_event("flyruler.gtm.trigger", timestamp=timestamp)
            previous_trigger = trigger

        def record_task(task_ctx: SimulationContext) -> None:
            if not args.no_save:
                recorder.record(
                    task_ctx.step,
                    task_ctx.time + task_ctx.dt,
                    x=model.to_numpy(x),
                    xd=model.to_numpy(xd),
                    y=model.to_numpy(y),
                    u=model.to_numpy(u),
                    p=model.to_numpy(p),
                    effect=model.to_numpy(effects),
                )

        def status_task(task_ctx: SimulationContext) -> None:
            observation = read_observation(model, y, paths=paths)
            realtime_factor = task_ctx.time / max(time.monotonic() - wall_start, 1e-9)
            print(
                f"t={task_ctx.time + task_ctx.dt:.1f}s rt={realtime_factor:.2f}x "
                f"alt={observation.altitude_m:.1f} V={observation.tas_mps:.1f} "
                f"cmd={np.array2string(command, precision=3)}"
            )

        tasks = [
            LoopTask("input", input_task, args.hz, stage="input"),
            LoopTask(
                "aircraft",
                aircraft_task,
                args.hz,
                stage="dynamics",
                wait_for=("input",),
            ),
            LoopTask(
                "output", output_task, args.hz, stage="io", wait_for=("aircraft",)
            ),
            LoopTask(
                "record",
                record_task,
                args.record_hz,
                stage="record",
                wait_for=("aircraft",),
            ),
            LoopTask(
                "status",
                status_task,
                1.0,
                stage="record",
                order=1,
                wait_for=("output",),
            ),
        ]
        await run_async_realtime(
            ctx, tasks, duration=None if args.duration <= 0 else args.duration
        )
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if pool is not None:
            await pool.stop()
        if client is not None:
            client.close()
        if not args.no_save and recorder.samples:
            recorder.save(args.record_name)
    print("closed.")
    return 0


async def main_async(args: Options) -> int:
    """校验时序设置，选择后端并开始实时循环。"""
    if (
        not np.isfinite([args.hz, args.record_hz, args.duration]).all()
        or args.hz <= 0
        or args.record_hz <= 0
    ):
        raise ValueError("rates must be finite and positive; duration must be finite")
    if args.integrator not in {"euler", "rk4"}:
        raise ValueError("--integrator must be euler or rk4")
    if args.dry_run:
        args.no_stick = args.no_proto = args.no_save = True
        if args.duration <= 0:
            args.duration = 0.05
    cfg = GTMConfig.fullscale()
    compiler_settings = ModelSettings(integrator=args.integrator, dt=1.0 / args.hz)
    builder = GTMSystem.build("Root", cfg)
    if args.backend == "numpy":
        return await _run(
            runtime.bind(
                compiler.compile(
                    compiler.prepare(builder, settings=compiler_settings),
                    target=targets.numpy,
                )
            ),
            cfg,
            args,
        )
    if args.backend == "casadi":
        return await _run(
            runtime.bind(
                compiler.compile(
                    compiler.prepare(builder, settings=compiler_settings),
                    target=targets.casadi,
                )
            ),
            cfg,
            args,
        )
    if args.backend == "native":
        program = compiler.prepare(builder, settings=compiler_settings)
        return await _run(
            runtime.bind(compiler.compile(program, target=targets.NativeTarget())),
            cfg,
            args,
        )
    raise ValueError("--backend must be native, numpy, or casadi")


def main(
    address: str = "127.0.0.1:18002",
    name: str = "Boeing 737-800 GTM Stick",
    # 120 Hz 控制回路配 60 Hz 记录，兼顾手感与实时余量。
    hz: float = 120.0,
    record_hz: float = 60.0,
    integrator: str = "rk4",
    backend: str = "native",
    duration: float = 0.0,
    profile: Path = DEFAULT_PROFILE,
    logical_name: str = "stick",
    # 原点默认取上海虹桥附近经纬度，只影响 Proto 的经纬映射。
    origin_lat: float = 31.1434,
    origin_lon: float = 121.8052,
    heartbeat: float = 1.0,
    # 只用 60% 舵面行程，给大飞机留操纵余量。
    control_scale: float = 0.6,
    # 3% 死区过滤摇杆回中抖动。
    axis_deadband: float = 0.03,
    # 默认在 1000 m 高度、100 m/s 巡航速度起步。
    initial_altitude: float = 1000.0,
    initial_speed: float = 100.0,
    nominal_throttle: float = 40.0,
    # 给出该参数时忽略摇杆，始终发送这一条固定指令。
    manual_command: tuple[float, float, float, float] | None = None,
    invert_roll: Annotated[bool, typer.Option("--invert-roll")] = False,
    invert_pitch: Annotated[bool, typer.Option("--invert-pitch")] = False,
    invert_rudder: Annotated[bool, typer.Option("--invert-rudder")] = False,
    invert_throttle: Annotated[bool, typer.Option("--invert-throttle")] = False,
    output_dir: Path = Path("results/fly_stick"),
    record_name: str = "fly_stick",
    no_save: Annotated[bool, typer.Option("--no-save")] = False,
    no_stick: Annotated[bool, typer.Option("--no-stick")] = False,
    no_proto: Annotated[bool, typer.Option("--no-proto")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> int:
    """运行 GTM 实时摇杆仿真。"""
    return asyncio.run(
        main_async(
            Options(
                address=address,
                name=name,
                hz=hz,
                record_hz=record_hz,
                integrator=integrator,
                backend=backend,
                duration=duration,
                profile=profile,
                logical_name=logical_name,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                heartbeat=heartbeat,
                control_scale=control_scale,
                axis_deadband=axis_deadband,
                initial_altitude=initial_altitude,
                initial_speed=initial_speed,
                nominal_throttle=nominal_throttle,
                manual_command=manual_command,
                invert_roll=invert_roll,
                invert_pitch=invert_pitch,
                invert_rudder=invert_rudder,
                invert_throttle=invert_throttle,
                output_dir=output_dir,
                record_name=record_name,
                no_save=no_save,
                no_stick=no_stick,
                no_proto=no_proto,
                dry_run=dry_run,
            )
        )
    )


if __name__ == "__main__":
    typer.run(main)
