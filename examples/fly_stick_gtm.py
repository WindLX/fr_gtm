"""用 fly_stick 交互驾驶 GTM，并把状态流推送到 fly_ruler_proto。

本示例有意把 fly_stick 与 fly_ruler_proto_python 保持为可选依赖。

设备与操作方式与 F-16 版本一致：默认 profile 是 T16000M，俯仰、滚转、
偏航三轴加扳机开关，没有油门轴时用 --nominal-throttle。终端每秒打印
t、rt、alt、V 与当前指令；V 是真空速，alt 取 -pos_ned[2]。指令单位是
[elevator_deg, aileron_deg, rudder_deg, throttle_percent]，与 F-16 的
SI 向量不同。用 Ctrl+C 退出，未加 --no-save 时结果写进
results/fly_stick/fly_stick.parquet。

Note:
    默认运行时后端是 native：先用 CasADi 后端编译模型，再通过
    `fly_ruler_core.codegen` 下沉为带缓存的 C++/pybind11 扩展。首次运行
    需要编译，之后的运行复用缓存。

Examples:
    真实摇杆加服务端（`just setup-stick` 用 `--with-editable` 引入
    `fly_stick` 与 `fly_ruler_proto_python` 两个可选依赖）：

    ```bash
    just setup-stick
    just example-fly-stick
    ```

    免依赖冒烟测试：

    ```bash
    uv run python examples/fly_stick_gtm.py --dry-run --no-stick --no-proto
    ```
"""

from __future__ import annotations

import asyncio
import importlib
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Annotated, Any

import fly_ruler_core as fr
import numpy as np
import typer
from fly_ruler_std.runtime.loop import (
    LoopTask,
    SimulationContext,
    make_discrete_model_task,
    run_async_realtime,
)
from fly_ruler_std.runtime.record import Recorder

from fr_gtm import GTMConfig, GTMSystem, airdata_angles_to_proto_radians
from fr_gtm.stick_input import STICK_VISUAL_MODEL_ID, StickAxes

# WGS84 赤道半径，用于把 NED 位移换算成经纬度增量。
EARTH_RADIUS_M = 6_378_137.0
# 默认 profile 描述 T16000M 的轴与按键别名，可用 --profile 换设备。
DEFAULT_PROFILE = (
    Path(__file__).resolve().parent / "fly_stick_profiles" / "t16000m.toml"
)


# 目前只用到扳机，用于向 Proto 发送一次事件。
@dataclass(frozen=True)
class StickButtons:
    trigger: bool = False


# 可选依赖在真正用到时才导入，缺依赖时给出可执行的安装命令。
def _optional_import(module: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional dependency '{module}' is required. Run: {install_hint}"
        ) from exc


# 手柄上报值先夹到 [-1, 1]，非数值输入退回默认值。
def _clip_unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return float(np.clip(number, -1.0, 1.0))


# 死区保证松杆时指令严格为零，输出按剩余行程线性重标定。
def _deadband(value: float, threshold: float) -> float:
    threshold = float(np.clip(threshold, 0.0, 0.95))
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    return math.copysign((magnitude - threshold) / (1.0 - threshold), value)


# 把归一化摇杆轴映射成 [升降舵, 副翼, 方向舵, 油门] 指令。
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
    """把归一化摇杆轴映射为 GTM 的根控制指令。

    各轴先减去死区，再乘以对应舵面的可用行程与 `control_scale`；油门轴由
    `[-1, 1]` 线性映射到 `[0, 100]` 百分比，未接油门时保持
    `nominal_throttle`。

    Returns:
        np.ndarray: 形状 `(4,)` 的指令向量
            `[elevator_deg, aileron_deg, rudder_deg, throttle_percent]`，
            前三个舵面通道已按 `cfg.actuator.position_lower`/`position_upper`
            限幅，油门通道按 `cfg.engine.lag.command_lower`/`command_upper` 限幅。
    """

    lower = np.asarray(cfg.actuator.position_lower, dtype=float)
    upper = np.asarray(cfg.actuator.position_upper, dtype=float)
    scale = float(np.clip(control_scale, 0.0, 1.0))
    lag = cfg.engine.lag

    # 三条舵面轴默认取反，与操纵面正方向对齐，可用 --invert-* 纠正。
    roll = _deadband(axes.roll if invert_roll else -axes.roll, axis_deadband)
    pitch = _deadband(axes.pitch if invert_pitch else -axes.pitch, axis_deadband)
    yaw = _deadband(axes.yaw if invert_rudder else -axes.yaw, axis_deadband)

    elevator_limit = min(abs(float(lower[0])), abs(float(upper[0])))
    aileron_limit = min(abs(float(lower[1])), abs(float(upper[1])))
    rudder_limit = min(abs(float(lower[2])), abs(float(upper[2])))

    # 没有油门轴时保持标称油门百分比；油门是 Engine 的指令，不经作动器。
    throttle = nominal_throttle
    if axes.throttle is not None:
        throttle_axis = axes.throttle if invert_throttle else -axes.throttle
        throttle = (throttle_axis + 1.0) * 50.0

    surfaces = np.clip(
        np.array(
            [
                pitch * elevator_limit * scale,
                roll * aileron_limit * scale,
                yaw * rudder_limit * scale,
            ],
            dtype=float,
        ),
        lower,
        upper,
    )
    return np.concatenate(
        [
            surfaces,
            np.array([np.clip(throttle, lag.command_lower, lag.command_upper)]),
        ]
    )


def clip_command(cfg: GTMConfig, command: np.ndarray) -> np.ndarray:
    """把四通道指令钳位到配置允许的区间。

    前三个舵面通道按 `cfg.actuator.position_lower`/`position_upper` 钳位，油门
    通道按 `cfg.engine.lag.command_lower`/`command_upper` 钳位：油门是 Engine
    的指令，不是作动器行程。

    Args:
        cfg: GTM 配置，提供舵面行程与油门指令区间。
        command: `[elevator_deg, aileron_deg, rudder_deg, throttle_percent]`。

    Returns:
        np.ndarray: 钳位后的四元素浮点指令。
    """

    lag = cfg.engine.lag
    surfaces = np.clip(
        np.asarray(command[:3], dtype=float),
        np.asarray(cfg.actuator.position_lower, dtype=float),
        np.asarray(cfg.actuator.position_upper, dtype=float),
    )
    throttle = np.clip(float(command[3]), lag.command_lower, lag.command_upper)
    return np.concatenate([surfaces, np.array([throttle])])


# GTM 的根输入是 u(4) + d(4) + n(21) 共 29 个元素。
def build_input(model: fr.CompiledModel[Any], command: np.ndarray) -> np.ndarray:
    """构造 GTM 根输入向量。

    Returns:
        np.ndarray: 由 `u(4,)` 控制指令、零 `d(4,)` 干扰与零 `n(21,)` 噪声
            拼接而成的 29 元素平铺输入。
    """

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


# 构造水平匀速无角速率的保守初值，保证交互起步稳定。
def build_stick_initial_state(
    model: fr.CompiledModel[Any],
    cfg: GTMConfig,
    *,
    altitude_m: float,
    speed_mps: float,
    command: np.ndarray,
) -> np.ndarray:
    """构造用于交互式 GTM 控制的保守初始飞行状态。

    Returns:
        np.ndarray: 26 元素状态向量：高度为 `altitude_m`（NED 下为
            `-altitude_m`），机体系速度为 `[speed_mps, 0, 0]`，姿态为单位
            四元数，三个舵面位置取 `command[:3]`，发动机滞后状态取
            `command[3]` 对应的配平值，各 Dryden 状态为零。
    """

    # 高度存成 NED 的 -altitude_m，Dryden 各阶状态全部清零。
    lag_state = np.asarray(cfg.engine.lag.trim_state(float(command[3])), dtype=float)
    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": np.array([0.0, 0.0, -altitude_m]),
                "Root.eom::vel_body": np.array([speed_mps, 0.0, 0.0]),
                "Root.eom::quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "Root.eom::omega_body": np.zeros(3),
                "Root.actuator::position": np.asarray(command[:3], dtype=float),
                "Root.engine::lag_state": lag_state,
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


# 按 profile 声明的别名取轴与按键；设备未连接时返回中位值。
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


# 把局部 NED 位移按球面近似换算成经纬高，原点由 --origin-lat/lon 给出。
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


# 航迹倾角 gamma 与航迹方位角 chi 都由 NED 速度分量定义。
def _flight_path_angles(vel_ned: np.ndarray) -> tuple[float, float]:
    north, east, down = (float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]))
    horizontal = math.hypot(north, east)
    gamma = math.atan2(-down, max(horizontal, 1.0e-9))
    chi = math.atan2(east, north)
    return gamma, chi


# 把 GTM 输出翻译成 Proto 飞机状态，角度按 deg 转 rad。
def build_proto_state(
    proto: Any,
    output: dict[str, Any],
    *,
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> Any:
    """把 GTM 输出转换为 fly_ruler_proto 飞机状态。

    Returns:
        Any: 由 `proto.create_aircraft_state` 构造的飞机状态对象，角度已由
            GTM 的 deg 转换为 proto 期望的 rad。
    """

    pos_ned = np.asarray(output["Root::pos_ned"], dtype=float).reshape(3)
    vel_body = np.asarray(output["Root::vel_body"], dtype=float).reshape(3)
    vel_ned = np.asarray(output["Root::vel_ned"], dtype=float).reshape(3)
    quat = np.asarray(output["Root::quat"], dtype=float).reshape(4)
    omega = np.asarray(output["Root::omega_body"], dtype=float).reshape(3)
    control = np.asarray(output["Root::actuator_position"], dtype=float).reshape(4)
    # 四个通道依次是升降舵、副翼、方向舵与有效油门；油门已经过二阶滞后。
    # GTM 的 alpha 与 beta 单位是度，Proto 需要弧度。
    alpha, beta = airdata_angles_to_proto_radians(
        alpha_deg=float(np.asarray(output["Root::alpha"], dtype=float).reshape(-1)[0]),
        beta_deg=float(np.asarray(output["Root::beta"], dtype=float).reshape(-1)[0]),
    )
    tas = float(np.asarray(output["Root::V"], dtype=float).reshape(-1)[0])
    eas = float(np.asarray(output["Root::VE"], dtype=float).reshape(-1)[0])
    mach = float(np.asarray(output["Root::mach"], dtype=float).reshape(-1)[0])
    qbar_pa = float(np.asarray(output["Root::qbar"], dtype=float).reshape(-1)[0])

    # Proto 需要经纬高，模型内部只维护 NED 位置。
    lat, lon, altitude = _ned_to_geodetic(
        pos_ned, origin_lat_deg=origin_lat_deg, origin_lon_deg=origin_lon_deg
    )
    gamma, chi = _flight_path_angles(vel_ned)
    # 油门在指令里是百分比，Proto 用 0 到 1 的比例。
    throttle_ratio = float(np.clip(control[3] / 100.0, 0.0, 1.0))

    # 舵面在 Proto 里按左右分开：左副翼为正，右副翼取反。
    controls = proto.ControlSurfaceState(
        aileron_left_rad=math.radians(float(control[1])),
        aileron_right_rad=math.radians(float(-control[1])),
        elevator_rad=math.radians(float(control[0])),
        rudder_rad=math.radians(float(control[2])),
        flaps_left_ratio=0.0,
        flaps_right_ratio=0.0,
        spoilers_ratio=0.0,
    )
    return proto.create_aircraft_state(
        position=tuple(float(v) for v in pos_ned),
        velocity=tuple(float(v) for v in vel_body),
        attitude=proto.Attitude.from_quaternion(tuple(float(v) for v in quat)),
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
            dynamic_pressure=qbar_pa,
        ),
        control_surfaces=controls,
        propulsors=[
            proto.PropulsorState(
                "engine.left",
                kind=proto.PropulsorKind.JET,
                throttle_ratio=throttle_ratio,
                index=1,
            ),
            proto.PropulsorState(
                "engine.right",
                kind=proto.PropulsorKind.JET,
                throttle_ratio=throttle_ratio,
                index=2,
            ),
        ],
    )


# 按后端编译模型；native 走 CasADi 符号图再转 C++。
def compile_model(
    cfg: GTMConfig,
    dt: float,
    integrator: str,
    backend: str,
) -> Any:
    builder = GTMSystem.build("Root", cfg)

    # native 分支打印编译耗时与缓存命中情况。
    if backend == "native":
        print(
            "compiling GTM with CasADi for native codegen "
            f"(dt={dt:.6g}, integrator={integrator})..."
        )
        compiled = fr.Compiler(fr.CompilerSpec(integrator=integrator, dt=dt)).compile(
            builder, target=fr.targets.casadi
        )
        started = time.monotonic()
        native = fr.Compiler(fr.CompilerSpec()).convert(
            compiled, target=fr.targets.NativeTarget()
        )
        elapsed = time.monotonic() - started
        cache_state = (
            "cached"
            if native.artifact is not None and native.artifact.cache_hit
            else "built"
        )
        print(f"native module ready: {cache_state} in {elapsed:.2f}s")
        return native

    print(f"compiling GTM with {backend} backend (Python runtime)...")
    compiler = fr.Compiler(fr.CompilerSpec(integrator=integrator, dt=dt))
    if backend == "numpy":
        return compiler.compile(builder, target=fr.targets.numpy)
    if backend == "casadi":
        return compiler.compile(builder, target=fr.targets.casadi)
    raise ValueError(f"unsupported backend: {backend}")


# 命令行参数在这里汇总，避免异步主流程反复读全局状态。
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


# 连接摇杆：设备描述来自 profile，去抖 20 ms。
async def _open_stick(args: Options) -> tuple[Any | None, Any | None]:
    if args.no_stick:
        return None, None
    fly_stick = _optional_import("fly_stick", "just setup-stick")
    desc = fly_stick.DeviceDescription.from_toml(str(args.profile))
    pool = fly_stick.PyDevicePool(
        device_descs={args.logical_name: desc},
        debounce_seconds=0.02,
    )
    await pool.reset()
    return pool, desc


# 校验参数、编译模型、初始化状态与 Proto 客户端，然后启动实时循环。
async def main_async(args: Options) -> int:
    if args.hz <= 0 or args.record_hz <= 0:
        raise ValueError("--hz and --record-hz must be > 0")
    if args.backend not in {"native", "numpy", "casadi"}:
        raise ValueError("--backend must be native, numpy, or casadi")
    if args.integrator not in {"euler", "rk4"}:
        raise ValueError("--integrator must be euler or rk4")
    # 冒烟测试关闭硬件、Proto 与落盘，并给一个很短的时长。
    if args.dry_run:
        args.no_stick = True
        args.no_proto = True
        args.no_save = True
        if args.duration <= 0:
            args.duration = 0.05

    # 默认配置提供作动器行程与初值。
    cfg = GTMConfig.default()
    dt = 1.0 / args.hz
    model = compile_model(cfg, dt, args.integrator, args.backend)
    # 手动指令优先，其次用中立舵面加标称油门。
    command = (
        np.asarray(args.manual_command, dtype=float)
        if args.manual_command is not None
        else np.array([0.0, 0.0, 0.0, args.nominal_throttle], dtype=float)
    )
    command = clip_command(cfg, command)
    # 初值高度与速度由 --initial-altitude 与 --initial-speed 决定。
    x = build_stick_initial_state(
        model,
        cfg,
        altitude_m=args.initial_altitude,
        speed_mps=args.initial_speed,
        command=command,
    )
    xd = model.xd.build()
    p = model.p.default()
    u = build_input(model, command)
    # 先算一次稳态输出，用于 Proto 初值与记录清零。
    _, _, y0, _ = model.continuous_fn(0.0, x, xd, u, p)
    output = model.y.read(np.asarray(y0, dtype=float), by="path")

    proto = None
    client = None
    # 未禁用 Proto 时建立客户端并推送初始状态。
    if not args.no_proto:
        proto = _optional_import("fly_ruler_proto_python", "just setup-stick")
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
            toml_config=(
                f"[aircraft]\nmodel='{STICK_VISUAL_MODEL_ID}'\nrole='stick_open_loop'"
            ),
            heartbeat_interval_secs=args.heartbeat,
        )
        print(f"connected: aircraft_uuid={client.aircraft_uuid}")

    # 上下文保存实时循环共享的状态、输入、输出与记录器。
    recorder = Recorder(model, args.output_dir)
    ctx = SimulationContext(dt=dt)
    ctx.state.update(
        {"x": x, "xd": xd, "u": u, "p": p, "y": y0, "effects": np.zeros(0)}
    )
    ctx.data.update({"command": command, "output": output, "buttons": StickButtons()})
    pool = None
    desc = None

    # Ctrl+C 与 SIGTERM 都转成上下文停止请求，保证 finally 能落盘。
    def stop(signum: int, _frame: FrameType | None) -> None:
        ctx.request_stop(f"signal {signum}")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    # 单调时钟用于计算实时倍率。
    wall_start = time.monotonic()

    # input：取最新摇杆帧；--manual-command 存在时保持固定指令。
    def input_task(task_ctx: SimulationContext) -> None:
        state = None if pool is None else pool.fetch_nowait().get(args.logical_name)
        axes, buttons = _read_alias_input(state, desc)
        if args.manual_command is not None:
            command = clip_command(cfg, np.asarray(args.manual_command, dtype=float))
        else:
            command = map_stick_to_command(
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
        task_ctx.data["buttons"] = buttons
        task_ctx.data["command"] = command
        task_ctx.state["u"] = build_input(model, command)

    # output：读取输出、推送 Proto，并在扳机按下时发一次事件。
    def output_task(task_ctx: SimulationContext) -> None:
        output = model.y.read(np.asarray(task_ctx.state["y"], dtype=float), by="path")
        task_ctx.data["output"] = output
        buttons = task_ctx.data["buttons"]
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

    # record：按 record_hz 抽样写入记录器。
    def record_task(task_ctx: SimulationContext) -> None:
        recorder.record(
            task_ctx.step,
            task_ctx.time,
            x=np.asarray(task_ctx.state["x"], dtype=float),
            xd=np.asarray(task_ctx.state["xd"], dtype=float),
            y=np.asarray(task_ctx.state["y"], dtype=float),
            u=np.asarray(task_ctx.state["u"], dtype=float),
            effect=np.asarray(task_ctx.state["effects"], dtype=float),
        )

    # status：每秒打印一行进度，方便无界面时判断仿真是否健康。
    def status_task(task_ctx: SimulationContext) -> None:
        output = task_ctx.data["output"]
        realtime_factor = task_ctx.time / max(time.monotonic() - wall_start, 1.0e-9)
        pos_ned = np.asarray(output["Root::pos_ned"], dtype=float).reshape(3)
        speed = float(np.asarray(output["Root::V"]).reshape(-1)[0])
        print(
            f"t={task_ctx.time:.1f}s rt={realtime_factor:.2f}x "
            f"alt={-pos_ned[2]:.1f} V={speed:.1f} "
            f"cmd={np.array2string(np.asarray(task_ctx.data['command']), precision=3)}"
        )

    # 任务按依赖串联：input → aircraft → output，record 与 status 挂在后面。
    tasks = [
        LoopTask("input", input_task, args.hz, stage="input"),
        make_discrete_model_task(
            model, name="aircraft", rate_hz=args.hz, wait_for=("input",)
        ),
        LoopTask("output", output_task, args.hz, stage="io", wait_for=("aircraft",)),
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

    try:
        pool, desc = await _open_stick(args)
        await run_async_realtime(
            ctx, tasks, duration=None if args.duration <= 0 else args.duration
        )
    finally:
        if not args.no_save and recorder.samples:
            recorder.save(args.record_name)
        if pool is not None:
            await pool.stop()
        if client is not None:
            client.close()

    print("closed.")
    return 0


# Typer 入口；默认参数面向真实硬件，--dry-run 用于无硬件冒烟。
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
