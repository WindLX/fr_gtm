"""用双摇杆、本地记录与 schema 优先的实时遥测运行 GTM。

本示例直接导入 `fly_stick` 与 `fly_ruler_proto_python`，属于可选依赖；
在 `models/fr_gtm` 下先执行 `just setup-stick`，再用
`just example-active-sidestick` 运行，冒烟测试用 `just test-examples-stick`。
"""

from __future__ import annotations

import asyncio
import math
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Annotated, Any

import fly_ruler_core as fr
import fly_ruler_proto_python as proto
import numpy as np
import typer
from fly_ruler_core import PyIrBuilder
from fly_ruler_core.codegen import NativeCodegenSpec, codegen
from fly_ruler_std.runtime import (
    LoopTask,
    Recorder,
    SimulationContext,
    make_discrete_model_task,
    run_async_realtime,
)
from fly_stick import (
    ActiveSidestick,
    ActiveSidestickConfig,
    DeviceDescription,
    PyDevicePool,
)

from fr_gtm import GTMConfig, GTMSystem, airdata_angles_to_proto_radians
from fr_gtm.stick_input import (
    STICK_VISUAL_MODEL_ID,
    StickAxes,
    merge_active_sidestick_axes,
)

DEFAULT_PROFILE = (
    Path(__file__).resolve().parent / "fly_stick_profiles" / "t16000m.toml"
)


@dataclass
class Options:
    hz: float
    record_hz: float
    duration: float
    integrator: str
    profile: Path
    logical_name: str
    active_bind_host: str
    active_teensy_host: str
    active_command_port: int
    active_logic_port: int
    active_state_port: int
    active_stale_after_ms: int
    active_stick_index: int
    active_roll_limit_deg: float
    active_pitch_limit_deg: float
    control_scale: float
    axis_deadband: float
    nominal_throttle: float
    initial_altitude: float
    initial_speed: float
    invert_roll: bool
    invert_pitch: bool
    invert_rudder: bool
    invert_throttle: bool
    proto_address: str
    aircraft_name: str
    aircraft_telemetry_hz: float
    controller_telemetry_hz: float
    sensor_telemetry_hz: float
    runtime_telemetry_hz: float
    output_dir: Path
    record_name: str
    no_proto: bool
    no_save: bool
    no_stick: bool
    dry_run: bool


def compile_native_model(
    cfg: GTMConfig, dt: float, integrator: str
) -> fr.CompiledModel[np.ndarray]:
    builder = GTMSystem(cfg, name="Root")
    if not isinstance(builder, PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    print(f"compiling native GTM (dt={dt:.6g}, integrator={integrator})...")
    compiled = fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.casadi, integrator=integrator, dt=dt)
    ).compile(builder)
    started = time.monotonic()
    native = codegen(compiled, NativeCodegenSpec(batch_size=1))
    cache_state = (
        "cached"
        if native.artifact is not None and native.artifact.cache_hit
        else "built"
    )
    print(f"native module ready: {cache_state} in {time.monotonic() - started:.2f}s")
    return native


def build_initial_state(
    model: fr.CompiledModel[Any],
    *,
    altitude_m: float,
    speed_mps: float,
    command: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": np.array([0.0, 0.0, -altitude_m]),
                "Root.eom::vel_body": np.array([speed_mps, 0.0, 0.0]),
                "Root.eom::quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "Root.eom::omega_body": np.zeros(3),
                "Root.actuator::position": command,
                "Root.wind.dryden::u_gw": np.zeros(1),
                "Root.wind.dryden::v_gw": np.zeros(2),
                "Root.wind.dryden::w_gw": np.zeros(2),
                "Root.wind.dryden::p_gw": np.zeros(1),
                "Root.wind.dryden::q_gw": np.zeros(1),
                "Root.wind.dryden::r_gw": np.zeros(1),
            }
        ),
        dtype=float,
    )


def build_model_input(model: fr.CompiledModel[Any], command: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.u.build(
            {
                "Root::u": command,
                "Root::d": np.zeros(4),
                "Root::n": np.zeros(21),
            }
        ),
        dtype=float,
    )


def telemetry_schemas(
    aircraft_rate_hz: float,
    controller_rate_hz: float,
    sensor_rate_hz: float,
    runtime_rate_hz: float,
) -> list[proto.TelemetryStreamSchema]:
    f64 = proto.TelemetryValueType.F64
    return [
        proto.TelemetryStreamSchema(
            "aircraft",
            [
                proto.TelemetryField(
                    "flight.altitude", "Altitude", "Flight", "m", value_type=f64
                ),
                proto.TelemetryField(
                    "flight.tas", "True airspeed", "Flight", "m/s", value_type=f64
                ),
                proto.TelemetryField(
                    "flight.alpha", "Angle of attack", "Air data", "deg", value_type=f64
                ),
                proto.TelemetryField(
                    "flight.beta", "Sideslip angle", "Air data", "deg", value_type=f64
                ),
            ],
            name="Aircraft",
            nominal_rate_hz=aircraft_rate_hz,
        ),
        proto.TelemetryStreamSchema(
            "controller",
            [
                proto.TelemetryField(
                    "actuator.elevator.command",
                    "Elevator command",
                    "Actuator",
                    "deg",
                    value_type=f64,
                ),
                proto.TelemetryField(
                    "actuator.aileron.command",
                    "Aileron command",
                    "Actuator",
                    "deg",
                    value_type=f64,
                ),
                proto.TelemetryField(
                    "actuator.rudder.command",
                    "Rudder command",
                    "Actuator",
                    "deg",
                    value_type=f64,
                ),
                proto.TelemetryField(
                    "propulsion.throttle.command",
                    "Throttle command",
                    "Propulsion",
                    "%",
                    value_type=f64,
                ),
            ],
            name="Controller",
            nominal_rate_hz=controller_rate_hz,
        ),
        proto.TelemetryStreamSchema(
            "sensor",
            [
                proto.TelemetryField(
                    f"sensor.output.{index}",
                    f"Sensor {index}",
                    "Sensor",
                    value_type=f64,
                )
                for index in range(21)
            ],
            name="Sensor",
            nominal_rate_hz=sensor_rate_hz,
        ),
        proto.TelemetryStreamSchema(
            "runtime",
            [
                proto.TelemetryField(
                    "runtime.realtime_factor",
                    "Realtime factor",
                    "Runtime",
                    value_type=f64,
                ),
                proto.TelemetryField(
                    "runtime.step",
                    "Simulation step",
                    "Runtime",
                    value_type=proto.TelemetryValueType.I64,
                ),
            ],
            name="Runtime",
            nominal_rate_hz=runtime_rate_hz,
        ),
    ]


def proto_state(output: Mapping[str, object]) -> proto.AircraftState:
    pos_ned = np.asarray(output["Root::pos_ned"], dtype=float).reshape(3)
    vel_body = np.asarray(output["Root::vel_body"], dtype=float).reshape(3)
    quat = np.asarray(output["Root::quat"], dtype=float).reshape(4)
    omega = np.asarray(output["Root::omega_body"], dtype=float).reshape(3)
    actuator = np.asarray(output["Root::actuator_position"], dtype=float).reshape(4)
    altitude = -float(pos_ned[2])
    tas = float(np.asarray(output["Root::V"], dtype=float).reshape(-1)[0])
    alpha, beta = airdata_angles_to_proto_radians(
        alpha_deg=float(np.asarray(output["Root::alpha"], dtype=float).reshape(-1)[0]),
        beta_deg=float(np.asarray(output["Root::beta"], dtype=float).reshape(-1)[0]),
    )
    return proto.create_aircraft_state(
        position=(float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])),
        velocity=(float(vel_body[0]), float(vel_body[1]), float(vel_body[2])),
        attitude=proto.Attitude.from_quaternion(
            (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        ),
        angular_velocity=(float(omega[0]), float(omega[1]), float(omega[2])),
        derived=proto.DerivedState(
            lat=0.0,
            lon=0.0,
            altitude=altitude,
            alpha=alpha,
            beta=beta,
            tas=tas,
            eas=float(np.asarray(output["Root::VE"], dtype=float).reshape(-1)[0]),
            dynamic_pressure=float(
                np.asarray(output["Root::qbar"], dtype=float).reshape(-1)[0]
            ),
        ),
        control_surfaces=proto.ControlSurfaceState(
            elevator_rad=math.radians(float(actuator[0])),
            aileron_left_rad=math.radians(float(actuator[1])),
            aileron_right_rad=math.radians(float(-actuator[1])),
            rudder_rad=math.radians(float(actuator[2])),
        ),
        propulsors=[
            proto.PropulsorState(
                "engine.left",
                kind=proto.PropulsorKind.JET,
                throttle_ratio=float(np.clip(actuator[3] / 100.0, 0.0, 1.0)),
                index=1,
            ),
            proto.PropulsorState(
                "engine.right",
                kind=proto.PropulsorKind.JET,
                throttle_ratio=float(np.clip(actuator[3] / 100.0, 0.0, 1.0)),
                index=2,
            ),
        ],
    )


def read_t16000_axes(
    pool: PyDevicePool | None,
    description: DeviceDescription | None,
    logical_name: str,
) -> StickAxes:
    if pool is None or description is None:
        return StickAxes()
    state = pool.fetch_nowait().get(logical_name)
    if state is None:
        return StickAxes()
    axes = state.get_alias_axes(description)
    throttle = axes.get("ABS_THROTTLE")
    return StickAxes(
        yaw=float(np.clip(axes.get("ABS_RZ", 0.0), -1.0, 1.0)),
        throttle=None if throttle is None else float(np.clip(throttle, -1.0, 1.0)),
    )


def deadband(value: float, threshold: float) -> float:
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    return math.copysign((magnitude - threshold) / (1.0 - threshold), value)


def axes_to_command(
    axes: StickAxes,
    cfg: GTMConfig,
    args: Options,
) -> np.ndarray:
    lower = np.asarray(cfg.actuator.position_lower, dtype=float)
    upper = np.asarray(cfg.actuator.position_upper, dtype=float)
    scale = float(np.clip(args.control_scale, 0.0, 1.0))
    roll = deadband(axes.roll if args.invert_roll else -axes.roll, args.axis_deadband)
    pitch = deadband(
        axes.pitch if args.invert_pitch else -axes.pitch, args.axis_deadband
    )
    yaw = deadband(axes.yaw if args.invert_rudder else -axes.yaw, args.axis_deadband)
    if axes.throttle is None:
        throttle = args.nominal_throttle
    else:
        throttle_axis = axes.throttle if args.invert_throttle else -axes.throttle
        throttle = float(np.clip((throttle_axis + 1.0) * 50.0, 0.0, 100.0))
    command = np.array(
        [
            pitch * min(abs(lower[0]), abs(upper[0])) * scale,
            roll * min(abs(lower[1]), abs(upper[1])) * scale,
            yaw * min(abs(lower[2]), abs(upper[2])) * scale,
            throttle,
        ]
    )
    return np.asarray(np.clip(command, lower, upper), dtype=float)


async def open_devices(
    args: Options,
) -> tuple[PyDevicePool | None, DeviceDescription | None, ActiveSidestick | None]:
    if args.no_stick:
        return None, None, None
    description = DeviceDescription.from_toml(str(args.profile))
    pool = PyDevicePool(
        device_descs={args.logical_name: description},
        debounce_seconds=0.02,
    )
    devices = await pool.reset()
    if args.logical_name not in devices:
        await pool.stop()
        raise RuntimeError(f"no T16000 matched profile {args.profile}")
    sidestick = ActiveSidestick(
        ActiveSidestickConfig(
            bind_host=args.active_bind_host,
            teensy_host=args.active_teensy_host,
            command_port=args.active_command_port,
            logic_port=args.active_logic_port,
            state_port=args.active_state_port,
            stale_after_ms=args.active_stale_after_ms,
        )
    )
    await sidestick.start()
    return pool, description, sidestick


async def main_async(args: Options) -> int:
    rates = (
        args.hz,
        args.record_hz,
        args.aircraft_telemetry_hz,
        args.controller_telemetry_hz,
        args.sensor_telemetry_hz,
        args.runtime_telemetry_hz,
    )
    if any(rate <= 0.0 for rate in rates):
        raise ValueError("simulation, record, and telemetry rates must be > 0")
    if not 0.0 <= args.axis_deadband < 1.0:
        raise ValueError("--axis-deadband must be in [0, 1)")
    if args.active_roll_limit_deg <= 0.0 or args.active_pitch_limit_deg <= 0.0:
        raise ValueError("active sidestick angle limits must be > 0")
    if args.dry_run:
        args.no_stick = True
        args.no_proto = True
        args.no_save = True
        if args.duration <= 0.0:
            args.duration = 0.05

    cfg = GTMConfig.default()
    dt = 1.0 / args.hz
    native_model = compile_native_model(cfg, dt, args.integrator)
    model = native_model
    command = np.array([0.0, 0.0, 0.0, args.nominal_throttle], dtype=float)
    x = build_initial_state(
        model,
        altitude_m=args.initial_altitude,
        speed_mps=args.initial_speed,
        command=command,
    )
    xd = np.asarray(model.xd.build(), dtype=float)
    p = np.asarray(model.p.default(), dtype=float)
    u = build_model_input(model, command)
    _, _, y, effects = native_model.continuous_fn(0.0, x, xd, u, p)
    ctx = SimulationContext(dt=dt)
    ctx.state.update({"x": x, "xd": xd, "u": u, "p": p, "y": y, "effects": effects})
    ctx.data["command"] = command

    recorder = Recorder(model, output_dir=args.output_dir)
    roll_limit_rad = math.radians(args.active_roll_limit_deg)
    pitch_limit_rad = math.radians(args.active_pitch_limit_deg)
    pool: PyDevicePool | None = None
    description: DeviceDescription | None = None
    sidestick: ActiveSidestick | None = None
    output = model.y.read(np.asarray(y, dtype=float), by="path")
    client = None
    if not args.no_proto:
        client = proto.FlyRulerClient(
            args.proto_address,
            args.aircraft_name,
            initial_state=proto_state(output),
            toml_config=(
                f"[aircraft]\nmodel='{STICK_VISUAL_MODEL_ID}'\nrole='active_sidestick'"
            ),
            telemetry_schemas=telemetry_schemas(
                args.aircraft_telemetry_hz,
                args.controller_telemetry_hz,
                args.sensor_telemetry_hz,
                args.runtime_telemetry_hz,
            ),
            spawn_timestamp=ctx.time,
        )
        print(f"connected telemetry: aircraft_uuid={client.aircraft_uuid}")
    ctx.data["output"] = output
    telemetry_wall_start = time.monotonic()

    def stop(signum: int, _frame: FrameType | None) -> None:
        ctx.request_stop(f"signal {signum}")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def input_task(task_ctx: SimulationContext) -> None:
        t16000 = read_t16000_axes(pool, description, args.logical_name)
        active_state = None if sidestick is None else sidestick.fetch_nowait()
        axes = merge_active_sidestick_axes(
            t16000,
            active_state,
            stick_index=args.active_stick_index,
            roll_limit_rad=roll_limit_rad,
            pitch_limit_rad=pitch_limit_rad,
        )
        target = axes_to_command(axes, cfg, args)
        task_ctx.data["command"] = target.copy()
        task_ctx.state["u"] = build_model_input(model, target)

    def feedback_task(task_ctx: SimulationContext) -> None:
        output = model.y.read(np.asarray(task_ctx.state["y"], dtype=float), by="path")
        task_ctx.data["output"] = output
        if sidestick is None:
            return
        actuator = np.asarray(output["Root::actuator_position"], dtype=float).reshape(4)
        alpha_deg = float(np.asarray(output["Root::alpha"], dtype=float).reshape(-1)[0])
        sidestick.send_aircraft_state(
            math.radians(alpha_deg),
            math.radians(float(actuator[0])),
            math.radians(float(actuator[1])),
        )

    def record_task(task_ctx: SimulationContext) -> None:
        recorder.record(
            step=task_ctx.step,
            time=task_ctx.time,
            x=np.asarray(task_ctx.state["x"], dtype=float),
            xd=np.asarray(task_ctx.state["xd"], dtype=float),
            y=np.asarray(task_ctx.state["y"], dtype=float),
            u=np.asarray(task_ctx.state["u"], dtype=float),
            effect=np.asarray(task_ctx.state["effects"], dtype=float),
        )

    def status_task(task_ctx: SimulationContext) -> None:
        output = task_ctx.data["output"]
        pos_ned = np.asarray(output["Root::pos_ned"], dtype=float).reshape(3)
        speed = float(np.asarray(output["Root::V"], dtype=float).reshape(-1)[0])
        command_now = np.asarray(task_ctx.data["command"], dtype=float)
        print(
            f"t={task_ctx.time:.1f}s alt={-pos_ned[2]:.1f}m V={speed:.1f}m/s "
            f"cmd={np.array2string(command_now, precision=2)}"
        )

    def aircraft_telemetry_task(task_ctx: SimulationContext) -> None:
        if client is None:
            return
        output_now = task_ctx.data["output"]
        client.update_state(proto_state(output_now), timestamp=task_ctx.time)
        pos_ned = np.asarray(output_now["Root::pos_ned"], dtype=float).reshape(3)
        client.publish_telemetry(
            "aircraft",
            (
                -float(pos_ned[2]),
                float(np.asarray(output_now["Root::V"], dtype=float).reshape(-1)[0]),
                float(
                    np.asarray(output_now["Root::alpha"], dtype=float).reshape(-1)[0]
                ),
                float(np.asarray(output_now["Root::beta"], dtype=float).reshape(-1)[0]),
            ),
            timestamp=task_ctx.time,
        )

    def controller_telemetry_task(task_ctx: SimulationContext) -> None:
        if client is None:
            return
        command_now = np.asarray(task_ctx.data["command"], dtype=float).reshape(4)
        client.publish_telemetry(
            "controller",
            tuple(float(value) for value in command_now),
            timestamp=task_ctx.time,
        )

    def sensor_telemetry_task(task_ctx: SimulationContext) -> None:
        if client is None:
            return
        output_now = task_ctx.data["output"]
        sensor = np.asarray(output_now["Root::y"], dtype=float).reshape(21)
        client.publish_telemetry(
            "sensor",
            tuple(float(value) for value in sensor),
            timestamp=task_ctx.time,
        )

    def runtime_telemetry_task(task_ctx: SimulationContext) -> None:
        if client is None:
            return
        wall_elapsed = max(time.monotonic() - telemetry_wall_start, 1e-9)
        client.publish_telemetry(
            "runtime",
            (task_ctx.time / wall_elapsed, task_ctx.step),
            timestamp=task_ctx.time,
        )

    tasks = [
        LoopTask("input", input_task, rate_hz=args.hz, stage="input"),
        make_discrete_model_task(
            native_model,
            name="aircraft",
            rate_hz=args.hz,
            wait_for=("input",),
        ),
        LoopTask(
            "feedback",
            feedback_task,
            rate_hz=args.hz,
            stage="io",
            wait_for=("aircraft",),
        ),
        LoopTask(
            "record",
            record_task,
            rate_hz=args.record_hz,
            stage="record",
            wait_for=("aircraft",),
        ),
        LoopTask(
            "aircraft_telemetry",
            aircraft_telemetry_task,
            rate_hz=args.aircraft_telemetry_hz,
            stage="io",
            order=1,
            wait_for=("feedback",),
        ),
        LoopTask(
            "controller_telemetry",
            controller_telemetry_task,
            rate_hz=args.controller_telemetry_hz,
            stage="io",
            order=2,
            wait_for=("input",),
        ),
        LoopTask(
            "sensor_telemetry",
            sensor_telemetry_task,
            rate_hz=args.sensor_telemetry_hz,
            stage="io",
            order=3,
            wait_for=("feedback",),
        ),
        LoopTask(
            "runtime_telemetry",
            runtime_telemetry_task,
            rate_hz=args.runtime_telemetry_hz,
            stage="io",
            order=4,
            wait_for=("aircraft",),
        ),
        LoopTask(
            "status",
            status_task,
            rate_hz=1.0,
            stage="record",
            order=1,
            wait_for=("feedback",),
        ),
    ]

    try:
        pool, description, sidestick = await open_devices(args)
        stats = await run_async_realtime(
            ctx,
            tasks,
            duration=None if args.duration <= 0.0 else args.duration,
        )
        if not args.no_save:
            recorder.save(args.record_name)
        print(f"closed: steps={ctx.step}, max_lag={stats['aircraft'].max_lag_s:.4f}s")
        return 0
    finally:
        if sidestick is not None:
            await sidestick.stop()
        if pool is not None:
            await pool.stop()
        if client is not None:
            client.despawn("simulation_stopped", timestamp=ctx.time)
            client.close()


def main(
    hz: float = 120.0,
    record_hz: float = 60.0,
    duration: float = 0.0,
    integrator: str = "rk4",
    profile: Path = DEFAULT_PROFILE,
    logical_name: str = "t16000",
    active_bind_host: str = "0.0.0.0",
    active_teensy_host: str = "30.30.30.6",
    active_command_port: int = 5405,
    active_logic_port: int = 5406,
    active_state_port: int = 5407,
    active_stale_after_ms: int = 100,
    active_stick_index: int = 1,
    active_roll_limit_deg: float = 20.0,
    active_pitch_limit_deg: float = 15.0,
    control_scale: float = 0.6,
    axis_deadband: float = 0.03,
    nominal_throttle: float = 40.0,
    initial_altitude: float = 1000.0,
    initial_speed: float = 100.0,
    invert_roll: Annotated[bool, typer.Option("--invert-roll")] = False,
    invert_pitch: Annotated[bool, typer.Option("--invert-pitch")] = False,
    invert_rudder: Annotated[bool, typer.Option("--invert-rudder")] = False,
    invert_throttle: Annotated[bool, typer.Option("--invert-throttle")] = False,
    proto_address: str = "127.0.0.1:18002",
    aircraft_name: str = "Boeing 737-800 GTM Active Sidestick",
    aircraft_telemetry_hz: float = 60.0,
    controller_telemetry_hz: float = 120.0,
    sensor_telemetry_hz: float = 60.0,
    runtime_telemetry_hz: float = 10.0,
    output_dir: Path = Path("results_active_sidestick"),
    record_name: str = "active_sidestick",
    no_proto: Annotated[bool, typer.Option("--no-proto")] = False,
    no_save: Annotated[bool, typer.Option("--no-save")] = False,
    no_stick: Annotated[bool, typer.Option("--no-stick")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> int:
    """运行 ActiveSidestick/T16000 的 GTM 实时仿真。"""
    options = Options(
        hz=hz,
        record_hz=record_hz,
        duration=duration,
        integrator=integrator,
        profile=profile,
        logical_name=logical_name,
        active_bind_host=active_bind_host,
        active_teensy_host=active_teensy_host,
        active_command_port=active_command_port,
        active_logic_port=active_logic_port,
        active_state_port=active_state_port,
        active_stale_after_ms=active_stale_after_ms,
        active_stick_index=active_stick_index,
        active_roll_limit_deg=active_roll_limit_deg,
        active_pitch_limit_deg=active_pitch_limit_deg,
        control_scale=control_scale,
        axis_deadband=axis_deadband,
        nominal_throttle=nominal_throttle,
        initial_altitude=initial_altitude,
        initial_speed=initial_speed,
        invert_roll=invert_roll,
        invert_pitch=invert_pitch,
        invert_rudder=invert_rudder,
        invert_throttle=invert_throttle,
        proto_address=proto_address,
        aircraft_name=aircraft_name,
        aircraft_telemetry_hz=aircraft_telemetry_hz,
        controller_telemetry_hz=controller_telemetry_hz,
        sensor_telemetry_hz=sensor_telemetry_hz,
        runtime_telemetry_hz=runtime_telemetry_hz,
        output_dir=output_dir,
        record_name=record_name,
        no_proto=no_proto,
        no_save=no_save,
        no_stick=no_stick,
        dry_run=dry_run,
    )
    return asyncio.run(main_async(options))


if __name__ == "__main__":
    typer.run(main)
