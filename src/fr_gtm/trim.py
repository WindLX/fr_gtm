"""以生产 GTM 方程和 std TrimSolver 求指定飞行条件的局部定常配平。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from fly_ruler_core.model.description import VariableInfo
from fly_ruler_core.runtime.backend import NumpyArray
from fly_ruler_core.runtime.instance import RuntimeInstance
from fly_ruler_std.runtime.trim import TrimResult, TrimSolver, TrimSpec

from .config import GTMConfig
from .init import init
from .layout import D_DIM, N_DIM, Y_DIM, Y_INDEX, GTMPaths
from .system import GTMSystem

_ROOT = GTMPaths()
_PLANT = GTMPaths("Root.aircraft")
type _Values = dict[str | VariableInfo, NumpyArray]
type _TrimValues = dict[str, NumpyArray]


@fr.block
def _FlightCondition() -> None:
    """从生产模型无误差观测中提取独立标量条件，不增加物理状态。"""
    measurement = fr.input("measurement", (Y_DIM,))
    fr.output("tas", measurement[Y_INDEX["V"]], unit="m/s")
    fr.output("gamma", measurement[Y_INDEX["gamma"]], unit="deg")
    fr.output("psi", measurement[Y_INDEX["psi"]], unit="deg")
    fr.output("phi", measurement[Y_INDEX["phi"]], unit="deg")


@fr.system(config=GTMConfig)
def _TrimSystem(cfg: GTMConfig) -> None:
    """生产飞机外的配平观测包装；不改变生产接口或力学方程。"""
    command = fr.input("u", (4,))
    disturbance = fr.input("d", (D_DIM,))
    noise = fr.input("n", (N_DIM,))
    aircraft = GTMSystem("aircraft", cfg)
    command >> aircraft.u
    disturbance >> aircraft.d
    noise >> aircraft.n
    condition = _FlightCondition("condition")
    aircraft.clean_y >> condition.measurement
    fr.export(
        tas=condition.tas, gamma=condition.gamma, psi=condition.psi, phi=condition.phi
    )


@dataclass(frozen=True)
class GTMTrimResult[T]:
    """配平结果，向量已按生产模型路径映射，可直接传回该模型。

    Attributes:
        x: 最优连续状态，目标模型后端类型。
        xd: 目标模型默认离散状态；GTM 当前为空。
        u: 最优输入，d/n 为零。
        p: 本次求解固定使用的完整运行时参数。
        residuals: 原物理量单位下的最大残差，含各目标及内部状态导数。
        solver: 未改写的 std TrimResult 与统计信息。
    """

    x: T
    xd: T
    u: T
    p: T
    residuals: dict[str, float]
    solver: TrimResult

    @property
    def success(self) -> bool:
        """求解器成功且物理/内部状态/目标残差均达到 1e-6。"""
        return self.solver.success and all(
            value <= 1e-6 for value in self.residuals.values()
        )


def trim[T](
    model: RuntimeInstance[T],
    cfg: GTMConfig,
    *,
    altitude: float,
    tas: float,
    gamma: float | None = 0.0,
    psi: float | None = 0.0,
    phi: float | None = 0.0,
    p: T | None = None,
    guess: tuple[T, T, T] | None = None,
    paths: GTMPaths = _ROOT,
    verbose: bool = False,
) -> GTMTrimResult[T]:
    """以 altitude/TAS/gamma/psi/phi 为目标求完整三轴直线配平。

    Args:
        model: 生产模型，必须与 cfg 对应；结果向量映射回该模型及后端。
        cfg: 生产飞机配置，提供真实作动器/油门边界和发动机阶数。
        altitude: 正向上高度，m，固定 NED 位置为 (0,0,-altitude)。
        tas: 目标真空速，m/s，必须为正。
        gamma: NED 航迹角，deg；None 表示自由。
        psi: 机头偏航角，deg，不是地面航向角 chi；None 表示自由。
        phi: 滚转角，deg；None 表示自由。侧滑角不额外固定。
        p: 固定运行时参数，省略时用 model.p.default()。
        guess: 可选 (x,xd,u) 初猜；位置和零角速率仍按配平条件固定。
        paths: 生产实例的路径目录。
        verbose: 是否打印 NLP 求解过程。

    Returns:
        GTMTrimResult，失败时保留求解器统计与实际残差，不放松指定条件。

    Raises:
        ValueError: 目标包含非有限值或 TAS 不大于零。

    Note:
        内部用 CasADi 编译小型条件包装；编译使用临时工作区，不缓存配平答案。
        p/q/r 固定零，平动/转动加速度与内部状态导数为零，位置导数自由。
        gamma 非零时为给定高度处的局部爬升配平，不表示高度/大气永远不变；
        非零 phi 不自动引入协调转弯。输入扰动 d/n 固定零，风/不确定度由 p 提供。
    """
    flight_targets = {"tas": tas, "gamma": gamma, "psi": psi, "phi": phi}
    finite = [
        altitude,
        *(value for value in flight_targets.values() if value is not None),
    ]
    if not np.isfinite(finite).all() or tas <= 0.0:
        raise ValueError("trim targets must be finite and TAS must be positive")
    parameters = model.p.default() if p is None else p
    parameter_values = model.p.read(parameters, by="path")
    if guess is None:
        speed = float(model.to_numpy(parameter_values[paths.p.wind_speed]).item())
        direction = float(
            np.rad2deg(model.to_numpy(parameter_values[paths.p.wind_direction]).item())
        )
        level = float(model.to_numpy(parameter_values[paths.p.turbulence_level]).item())
        x0, _, u0, _ = init(
            model,
            cfg,
            position=(0.0, 0.0, -altitude),
            airdata=(3.0, 0.0, tas),
            euler=(phi or 0.0, (gamma or 0.0) + 3.0, psi or 0.0),
            command=(-2.0, 0.0, 0.0, 40.0),
            wind=(speed, direction, level),
            paths=paths,
        )
    else:
        x0, _, u0 = guess
    wrapped = runtime.bind(
        compiler.compile(
            compiler.prepare(_TrimSystem.build("Root", cfg), settings=ModelSettings()),
            target=targets.casadi,
        )
    )
    # 两套模型用显式路径映射，绝不把包装模型的向量下标当作生产布局。
    wrapped_parameters: _TrimValues = {}
    for key in wrapped.p.keys:
        production_key = paths.root + key.removeprefix(_PLANT.root)
        wrapped_parameters[key] = model.to_numpy(
            parameter_values[production_key]
        ).reshape(-1)
    initial_guess: _TrimValues = {}
    x_values = model.x.read(x0, by="path")
    for key in wrapped.x.keys:
        production_key = paths.root + key.removeprefix(_PLANT.root)
        initial_guess[key] = model.to_numpy(x_values[production_key]).reshape(-1)
    u_values = model.u.read(u0, by="path")
    initial_guess[_ROOT.u.command] = model.to_numpy(u_values[paths.u.command]).reshape(
        -1
    )
    zeros = {
        _PLANT.x.vel_body: np.zeros(3),
        _PLANT.x.omega_body: np.zeros(3),
        _PLANT.x.actuator_position: np.zeros(3),
        _PLANT.x.engine_lag: np.zeros(len(cfg.engine.lag.trim_state(0.0))),
        _PLANT.x.dryden_u: np.zeros(1),
        _PLANT.x.dryden_v: np.zeros(2),
        _PLANT.x.dryden_w: np.zeros(2),
        _PLANT.x.dryden_p: np.zeros(1),
        _PLANT.x.dryden_q: np.zeros(1),
        _PLANT.x.dryden_r: np.zeros(1),
    }
    lower = np.array([*cfg.actuator.position_lower, cfg.engine.lag.command_lower])
    upper = np.array([*cfg.actuator.position_upper, cfg.engine.lag.command_upper])
    solved = TrimSolver(
        wrapped,
        TrimSpec(
            param_value=wrapped_parameters,
            state_value={
                _PLANT.x.pos_ned: np.array([0.0, 0.0, -altitude]),
                _PLANT.x.omega_body: np.zeros(3),
            },
            state_derivative=zeros,
            state_bounds={_PLANT.x.actuator_position: (lower[:3], upper[:3])},
            input_bounds={_ROOT.u.command: (lower, upper)},
            input_value={
                _ROOT.u.disturbance: np.zeros(D_DIM),
                _ROOT.u.noise: np.zeros(N_DIM),
            },
            output_value={
                f"Root::{name}": value
                for name, value in flight_targets.items()
                if value is not None
            },
            initial_guess=initial_guess,
            # 不惩罚正常的前进/爬升位置导数，自由条件下以控制量正则选择解。
            objective_weights={"u": 1e-6, "x_dot": 0.0},
            solver_opts={"ipopt.tol": 1e-10, "ipopt.max_iter": 1000},
        ),
    ).solve(verbose=verbose)
    destination: _Values = {}
    for key, value in wrapped.x.read(solved.x_opt, by="path").items():
        destination[paths.root + key.removeprefix(_PLANT.root)] = wrapped.to_numpy(
            value
        ).reshape(-1)
    x = model.x.build(destination)
    command = wrapped.u.read(solved.u_opt, by="path")[_ROOT.u.command]
    inputs: _Values = {
        paths.u.command: wrapped.to_numpy(command).reshape(-1),
        paths.u.disturbance: np.zeros(D_DIM),
        paths.u.noise: np.zeros(N_DIM),
    }
    u, xd = model.u.build(inputs), model.xd.build()
    evaluation = model.continuous_fn(0.0, x, xd, u, parameters)
    rates = model.xdot.read(evaluation.xdot, by="path")
    residuals: dict[str, float] = {}
    for key in zeros:
        actual = paths.root + key.removeprefix(_PLANT.root)
        residuals[actual] = float(np.max(np.abs(model.to_numpy(rates[actual]))))
    conditions = wrapped.y.read(solved.y_opt, by="path")
    for name, target in flight_targets.items():
        if target is not None:
            residuals[name] = abs(
                float(wrapped.to_numpy(conditions[f"Root::{name}"]).item()) - target
            )
    position = model.to_numpy(model.x.read(x, by="path")[paths.x.pos_ned]).reshape(-1)
    residuals["altitude"] = abs(float(-position[2]) - altitude)
    return GTMTrimResult(x, xd, u, parameters, residuals, solved)


__all__ = ["trim", "GTMTrimResult"]
