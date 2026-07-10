"""Fly the GTM interactively with fly_stick and stream it to fly_ruler_proto.

This example intentionally keeps fly_stick and fly_ruler_proto_python optional.
For a real joystick + server run:

    uv sync --group fly-stick
    uv run --group fly-stick python examples/fly_stick_gtm.py

For a dependency-free smoke test:

    uv run python examples/fly_stick_gtm.py --dry-run --no-stick --no-proto

The default runtime backend is native: the model is first compiled with the
CasADi backend, then lowered to a cached C++/pybind11 extension via
fly_ruler_std.codegen.  The first run may spend some time compiling; later runs
reuse the cache.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fly_ruler_core as fr
import numpy as np
from fly_ruler_core import PyIrBuilder

from fr_gtm import GTMConfig, GTMSystem
from fr_gtm.stick_input import StickAxes, merge_active_sidestick_axes

EARTH_RADIUS_M = 6_378_137.0
DEFAULT_PROFILE = (
    Path(__file__).resolve().parent / "fly_stick_profiles" / "t16000m.toml"
)


@dataclass(frozen=True)
class StickButtons:
    trigger: bool = False


@dataclass
class CommandFilter:
    """Small real-time command filter for noisy joystick axes."""

    value: np.ndarray
    tau: float

    def update(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self.tau <= 0.0:
            self.value = np.asarray(target, dtype=float)
            return self.value
        alpha = 1.0 - math.exp(-max(dt, 0.0) / self.tau)
        self.value = self.value + alpha * (np.asarray(target, dtype=float) - self.value)
        return self.value


def _optional_import(module: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional dependency '{module}' is required. Run: {install_hint}"
        ) from exc


def _clip_unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return float(np.clip(number, -1.0, 1.0))


def _deadband(value: float, threshold: float) -> float:
    threshold = float(np.clip(threshold, 0.0, 0.95))
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    return math.copysign((magnitude - threshold) / (1.0 - threshold), value)


def _throttle_percent(value: float | None, *, invert: bool) -> float | None:
    if value is None:
        return None
    axis = value if invert else -value
    return float(np.clip((axis + 1.0) * 50.0, 0.0, 100.0))


def _symmetric_limit(lower: float, upper: float) -> float:
    return min(abs(float(lower)), abs(float(upper)))


def map_stick_to_command(
    axes: StickAxes,
    cfg: GTMConfig,
    *,
    invert_roll: bool,
    invert_pitch: bool,
    invert_rudder: bool,
    invert_throttle: bool,
    control_scale: float,
    nominal_throttle: float,
    axis_deadband: float,
) -> np.ndarray:
    """Map normalized stick axes to GTM [elevator, aileron, rudder, throttle]."""

    lower = cfg.actuator.position_lower
    upper = cfg.actuator.position_upper
    scale = float(np.clip(control_scale, 0.0, 1.0))

    roll = _deadband(axes.roll if invert_roll else -axes.roll, axis_deadband)
    pitch = _deadband(axes.pitch if invert_pitch else -axes.pitch, axis_deadband)
    yaw = _deadband(axes.yaw if invert_rudder else -axes.yaw, axis_deadband)

    elevator_limit = _symmetric_limit(lower[0], upper[0])
    aileron_limit = _symmetric_limit(lower[1], upper[1])
    rudder_limit = _symmetric_limit(lower[2], upper[2])

    throttle = _throttle_percent(axes.throttle, invert=invert_throttle)
    if throttle is None:
        throttle = nominal_throttle

    command = np.array(
        [
            pitch * elevator_limit * scale,
            roll * aileron_limit * scale,
            yaw * rudder_limit * scale,
            throttle,
        ],
        dtype=float,
    )
    return np.asarray(
        np.clip(
            command, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        ),
        dtype=float,
    )


def build_input(model: fr.CompiledModel[Any], command: np.ndarray) -> np.ndarray:
    """Build the GTM root input vector."""

    return np.asarray(
        model.u.build(
            {
                "Root::u": np.asarray(command, dtype=float),
                "Root::d": np.zeros(4),
                "Root::n": np.zeros(21),
            }
        ),
        dtype=float,
    )


def build_stick_initial_state(
    model: fr.CompiledModel[Any],
    *,
    altitude_m: float,
    speed_mps: float,
    command: np.ndarray,
) -> np.ndarray:
    """Build a conservative initial flight state for interactive GTM control."""

    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": np.array([0.0, 0.0, -altitude_m]),
                "Root.eom::vel_body": np.array([speed_mps, 0.0, 0.0]),
                "Root.eom::quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "Root.eom::omega_body": np.zeros(3),
                "Root.actuator::position": np.asarray(command, dtype=float),
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


def _read_alias_input(
    state: Any | None,
    desc: Any | None,
) -> tuple[StickAxes, StickButtons]:
    if state is None or desc is None:
        return StickAxes(), StickButtons()

    axes_by_alias = state.get_alias_axes(desc)
    buttons_by_alias = state.get_alias_buttons(desc)
    throttle_value = axes_by_alias.get("ABS_THROTTLE")
    return (
        StickAxes(
            roll=_clip_unit(axes_by_alias.get("ABS_X")),
            pitch=_clip_unit(axes_by_alias.get("ABS_Y")),
            yaw=_clip_unit(axes_by_alias.get("ABS_RZ")),
            throttle=None if throttle_value is None else _clip_unit(throttle_value),
        ),
        StickButtons(trigger=bool(buttons_by_alias.get("BTN_TRIGGER", 0))),
    )


def _ned_to_geodetic(
    pos_ned: np.ndarray,
    *,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> tuple[float, float, float]:
    north, east, down = (float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]))
    lat = origin_lat_deg + math.degrees(north / EARTH_RADIUS_M)
    lon = origin_lon_deg + math.degrees(
        east / (EARTH_RADIUS_M * max(math.cos(math.radians(origin_lat_deg)), 1.0e-6))
    )
    return lat, lon, -down


def _flight_path_angles(vel_ned: np.ndarray) -> tuple[float, float]:
    north, east, down = (float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]))
    horizontal = math.hypot(north, east)
    gamma = math.atan2(-down, max(horizontal, 1.0e-9))
    chi = math.atan2(east, north)
    return gamma, chi


def build_proto_state(
    proto: Any,
    output: dict[str, Any],
    *,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> Any:
    """Convert GTM outputs into a fly_ruler_proto aircraft state."""

    pos_ned = np.asarray(output["pos_ned"], dtype=float).reshape(3)
    vel_body = np.asarray(output["vel_body"], dtype=float).reshape(3)
    vel_ned = np.asarray(output["vel_ned"], dtype=float).reshape(3)
    quat = np.asarray(output["quat"], dtype=float).reshape(4)
    omega = np.asarray(output["omega_body"], dtype=float).reshape(3)
    control = np.asarray(output["actuator_position"], dtype=float).reshape(4)
    alpha = float(np.asarray(output["alpha"], dtype=float).reshape(-1)[0])
    beta = float(np.asarray(output["beta"], dtype=float).reshape(-1)[0])
    tas = float(np.asarray(output["V"], dtype=float).reshape(-1)[0])
    eas = float(np.asarray(output["VE"], dtype=float).reshape(-1)[0])
    mach = float(np.asarray(output["mach"], dtype=float).reshape(-1)[0])
    qbar_pa = float(np.asarray(output["qbar"], dtype=float).reshape(-1)[0])

    lat, lon, altitude = _ned_to_geodetic(
        pos_ned, origin_lat_deg=origin_lat_deg, origin_lon_deg=origin_lon_deg
    )
    gamma, chi = _flight_path_angles(vel_ned)
    throttle_ratio = float(np.clip(control[3] / 100.0, 0.0, 1.0))

    controls = proto.ControlSurfaceState(
        aileron_left_rad=math.radians(float(control[1])),
        aileron_right_rad=math.radians(float(-control[1])),
        elevator_rad=math.radians(float(control[0])),
        rudder_rad=math.radians(float(control[2])),
        flaps_left_ratio=0.0,
        flaps_right_ratio=0.0,
        spoilers_ratio=0.0,
    )
    custom_fields = {
        "flyruler.gtm.control.elevator_deg": float(control[0]),
        "flyruler.gtm.control.aileron_deg": float(control[1]),
        "flyruler.gtm.control.rudder_deg": float(control[2]),
        "flyruler.gtm.airdata.mach": mach,
        "flyruler.gtm.airdata.qbar_pa": qbar_pa,
        "flyruler.gtm.navigation.altitude_m": altitude,
        "flyruler.gtm.navigation.down_m": float(pos_ned[2]),
    }
    return proto.create_aircraft_state(
        position=tuple(float(v) for v in pos_ned),
        velocity=tuple(float(v) for v in vel_body),
        attitude=tuple(float(v) for v in quat),
        angular_velocity=tuple(float(v) for v in omega),
        derived=proto.DerivedState(
            lat=lat,
            lon=lon,
            altitude=altitude,
            alpha=alpha,
            beta=beta,
            tas=tas,
            eas=eas,
            gamma=gamma,
            chi=chi,
            mach=mach,
        ),
        control_surfaces=controls,
        engines=[
            proto.EngineState(1, throttle_lever_ratio=throttle_ratio),
            proto.EngineState(2, throttle_lever_ratio=throttle_ratio),
        ],
        custom_fields=custom_fields,
    )


def compile_model(
    cfg: GTMConfig,
    dt: float,
    integrator: str,
    backend: str,
) -> Any:
    builder = GTMSystem(cfg, name="Root")
    if not isinstance(builder, PyIrBuilder):
        raise TypeError("expected top-level GTM builder")

    if backend == "native":
        from fly_ruler_std.codegen import CodegenConfig, codegen

        print(
            "compiling GTM with CasADi for native codegen "
            f"(dt={dt:.6g}, integrator={integrator})..."
        )
        compiled = fr.Compiler(
            fr.CompilerSpec(backend=fr.backends.casadi, integrator=integrator, dt=dt)
        ).compile(builder)
        started = time.monotonic()
        native = codegen(compiled, CodegenConfig(batch_size=1))
        elapsed = time.monotonic() - started
        cache_state = "cached" if native.build_info.get("cached") else "built"
        print(f"native module ready: {cache_state} in {elapsed:.2f}s")
        return native

    backend_obj = {
        "numpy": fr.backends.numpy,
        "casadi": fr.backends.casadi,
    }[backend]
    print(f"compiling GTM with {backend} backend (Python runtime)...")
    return fr.Compiler(
        fr.CompilerSpec(backend=backend_obj, integrator=integrator, dt=dt)
    ).compile(builder)


def parse_args(*, active_sidestick_default: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:18002")
    parser.add_argument("--name", default="FlyRulerGTMStick")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--integrator", choices=["euler", "rk4"], default="rk4")
    parser.add_argument(
        "--backend",
        choices=["native", "numpy", "casadi"],
        default="native",
        help=(
            "Runtime backend. 'native' uses CasADi codegen + cached C++ and is "
            "intended for real-time MSFS streaming."
        ),
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--logical-name", default="stick")
    parser.add_argument(
        "--active-sidestick",
        action=argparse.BooleanOptionalAction,
        default=active_sidestick_default,
        help="Use ActiveSidestick roll/pitch together with T16000 rudder/throttle.",
    )
    parser.add_argument("--active-bind-host", default="0.0.0.0")
    parser.add_argument("--active-teensy-host", default="30.30.30.6")
    parser.add_argument("--active-command-port", type=int, default=5405)
    parser.add_argument("--active-logic-port", type=int, default=5406)
    parser.add_argument("--active-state-port", type=int, default=5407)
    parser.add_argument("--active-stale-after-ms", type=int, default=100)
    parser.add_argument("--active-stick-index", type=int, choices=[1, 2], default=1)
    parser.add_argument("--active-roll-limit-deg", type=float, default=20.0)
    parser.add_argument("--active-pitch-limit-deg", type=float, default=15.0)
    parser.add_argument("--origin-lat", type=float, default=31.1434)
    parser.add_argument("--origin-lon", type=float, default=121.8052)
    parser.add_argument("--heartbeat", type=float, default=1.0)
    parser.add_argument("--control-scale", type=float, default=0.6)
    parser.add_argument("--axis-deadband", type=float, default=0.03)
    parser.add_argument("--command-filter-tau", type=float, default=0.12)
    parser.add_argument("--initial-altitude", type=float, default=1000.0)
    parser.add_argument("--initial-speed", type=float, default=100.0)
    parser.add_argument("--nominal-throttle", type=float, default=40.0)
    parser.add_argument(
        "--manual-command",
        nargs=4,
        type=float,
        metavar=("ELEV", "AIL", "RUD", "THR"),
        help=(
            "Fixed [elevator, aileron, rudder, throttle] command; "
            "useful with --no-stick."
        ),
    )
    parser.add_argument("--invert-roll", action="store_true")
    parser.add_argument("--invert-pitch", action="store_true")
    parser.add_argument("--invert-rudder", action="store_true")
    parser.add_argument("--invert-throttle", action="store_true")
    parser.add_argument("--no-stick", action="store_true")
    parser.add_argument("--no-proto", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    return parser.parse_args(argv)


async def _open_stick(args: argparse.Namespace) -> tuple[Any | None, Any | None]:
    if args.no_stick:
        return None, None
    fly_stick = _optional_import("fly_stick", "uv sync --group fly-stick")
    desc = fly_stick.DeviceDescription.from_toml(str(args.profile))
    pool = fly_stick.PyDevicePool(
        device_descs={args.logical_name: desc},
        debounce_seconds=0.02,
    )
    devices = await pool.reset()
    if args.logical_name not in devices:
        await pool.stop()
        raise RuntimeError(
            f"No input device matched profile '{args.profile}'. "
            "Check the profile or run with --no-stick for dry testing."
        )
    return pool, desc


async def _open_active_sidestick(args: argparse.Namespace) -> Any | None:
    if args.no_stick or not args.active_sidestick:
        return None
    fly_stick = _optional_import("fly_stick", "uv sync --group fly-stick")
    config = fly_stick.ActiveSidestickConfig(
        bind_host=args.active_bind_host,
        teensy_host=args.active_teensy_host,
        command_port=args.active_command_port,
        logic_port=args.active_logic_port,
        state_port=args.active_state_port,
        stale_after_ms=args.active_stale_after_ms,
    )
    sidestick = fly_stick.ActiveSidestick(config)
    await sidestick.start()
    return sidestick


def _latest_stick_state(pool: Any | None, logical_name: str) -> Any | None:
    if pool is None:
        return None
    try:
        return pool.fetch_nowait().get(logical_name)
    except Exception:
        return None


async def main_async(*, active_sidestick_default: bool = False) -> int:
    args = parse_args(active_sidestick_default=active_sidestick_default)
    if args.hz <= 0:
        raise ValueError("--hz must be > 0")
    if args.active_roll_limit_deg <= 0 or args.active_pitch_limit_deg <= 0:
        raise ValueError("active sidestick angle limits must be > 0")
    if args.dry_run:
        args.no_stick = True
        args.no_proto = True
        if args.duration <= 0:
            args.duration = 0.05

    cfg = GTMConfig.default()
    dt = 1.0 / args.hz
    model = compile_model(cfg, dt, args.integrator, args.backend)
    command = (
        np.asarray(args.manual_command, dtype=float)
        if args.manual_command is not None
        else np.array([0.0, 0.0, 0.0, args.nominal_throttle], dtype=float)
    )
    command = np.clip(
        command,
        np.asarray(cfg.actuator.position_lower, dtype=float),
        np.asarray(cfg.actuator.position_upper, dtype=float),
    )
    x = build_stick_initial_state(
        model,
        altitude_m=args.initial_altitude,
        speed_mps=args.initial_speed,
        command=command,
    )
    xd = model.xd.build()
    p = model.p.default()
    u = build_input(model, command)
    _, _, y0, _ = model.continuous_fn(0.0, x, xd, u, p)
    output = model.y.read(np.asarray(y0, dtype=float))

    proto = None
    client = None
    if not args.no_proto:
        proto = _optional_import("fly_ruler_proto_python", "uv sync --group fly-stick")
        initial_state = build_proto_state(
            proto,
            output,
            origin_lat_deg=args.origin_lat,
            origin_lon_deg=args.origin_lon,
        )
        client = proto.FlyRulerClient(
            args.address,
            args.name,
            initial_state=initial_state,
            toml_config="[aircraft]\nmodel='fr_gtm'\nrole='stick_open_loop'",
            heartbeat_interval_secs=args.heartbeat,
        )
        print(f"connected: aircraft_uuid={client.aircraft_uuid}")

    pool = None
    desc = None
    active_sidestick = None
    active_roll_limit_rad = math.radians(args.active_roll_limit_deg)
    active_pitch_limit_rad = math.radians(args.active_pitch_limit_deg)
    command_filter = CommandFilter(value=command.copy(), tau=args.command_filter_tau)
    running = True

    def stop(_signal: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    start = time.monotonic()
    next_tick = start
    step = 0
    try:
        pool, desc = await _open_stick(args)
        active_sidestick = await _open_active_sidestick(args)
        while running:
            elapsed = time.monotonic() - start
            if args.duration > 0 and elapsed >= args.duration:
                break

            state = _latest_stick_state(pool, args.logical_name)
            axes, buttons = _read_alias_input(state, desc)
            if active_sidestick is not None:
                axes = merge_active_sidestick_axes(
                    axes,
                    active_sidestick.fetch_nowait(),
                    stick_index=args.active_stick_index,
                    roll_limit_rad=active_roll_limit_rad,
                    pitch_limit_rad=active_pitch_limit_rad,
                )
            if args.manual_command is not None:
                raw_command = np.asarray(
                    np.clip(
                        np.asarray(args.manual_command, dtype=float),
                        np.asarray(cfg.actuator.position_lower, dtype=float),
                        np.asarray(cfg.actuator.position_upper, dtype=float),
                    ),
                    dtype=float,
                )
            else:
                raw_command = map_stick_to_command(
                    axes,
                    cfg,
                    invert_roll=args.invert_roll,
                    invert_pitch=args.invert_pitch,
                    invert_rudder=args.invert_rudder,
                    invert_throttle=args.invert_throttle,
                    control_scale=args.control_scale,
                    nominal_throttle=args.nominal_throttle,
                    axis_deadband=args.axis_deadband,
                )
            command = command_filter.update(raw_command, dt)
            u = build_input(model, command)
            step_started = time.monotonic()
            x, xd, y, _effects = model.discrete_fn(step * dt, x, xd, u, p)
            step_elapsed = time.monotonic() - step_started
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            output = model.y.read(y)

            if active_sidestick is not None:
                actuator_position = np.asarray(
                    output["actuator_position"], dtype=float
                ).reshape(4)
                alpha_deg = float(
                    np.asarray(output["alpha"], dtype=float).reshape(-1)[0]
                )
                active_sidestick.send_aircraft_state(
                    math.radians(alpha_deg),
                    math.radians(float(actuator_position[0])),
                    math.radians(float(actuator_position[1])),
                )

            if client is not None and proto is not None:
                wall_ts = time.time()
                client.update_state(
                    build_proto_state(
                        proto,
                        output,
                        origin_lat_deg=args.origin_lat,
                        origin_lon_deg=args.origin_lon,
                    ),
                    timestamp=wall_ts,
                )
                if buttons.trigger:
                    client.create_event("flyruler.gtm.trigger", timestamp=wall_ts)

            if step % max(int(args.hz), 1) == 0:
                sim_time = (step + 1) * dt
                realtime_factor = sim_time / max(time.monotonic() - start, 1.0e-9)
                print(
                    "t={:.2f}s sim={:.2f}s rt={:.2f}x step={:.3f}s "
                    "pos_ned={} alt={:.1f} V={:.1f} att={} cmd={}".format(
                        elapsed,
                        sim_time,
                        realtime_factor,
                        step_elapsed,
                        np.array2string(
                            np.asarray(output["pos_ned"]).squeeze(), precision=3
                        ),
                        float(-np.asarray(output["pos_ned"]).reshape(3)[2]),
                        float(np.asarray(output["V"]).reshape(-1)[0]),
                        np.array2string(
                            np.asarray(output["euler"]).squeeze(), precision=3
                        ),
                        np.array2string(command, precision=3),
                    )
                )

            step += 1
            next_tick += dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            else:
                if step % max(int(args.hz), 1) == 0:
                    lag = -sleep_s
                    print(
                        "warning: simulation cannot keep real time; "
                        f"lag={lag:.3f}s. Lower --hz, use --integrator euler, "
                        "or switch to --backend native."
                    )
                next_tick = time.monotonic()
    finally:
        if active_sidestick is not None:
            await active_sidestick.stop()
        if pool is not None:
            await pool.stop()
        if client is not None:
            client.close()

    print("closed.")
    return 0


def main(*, active_sidestick_default: bool = False) -> int:
    return asyncio.run(main_async(active_sidestick_default=active_sidestick_default))


if __name__ == "__main__":
    raise SystemExit(main())
