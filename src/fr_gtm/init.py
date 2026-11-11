"""从物理初值构造 GTM 向量、外部噪声与可读观测。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from fly_ruler_core.model.description import VariableInfo
from fly_ruler_core.model.program import TargetProgram
from fly_ruler_core.runtime.backend import BuildValue, NumpyArray
from fly_ruler_core.runtime.instance import RuntimeInstance
from fly_ruler_std.libs.coordinates import aero_to_body_vel
from fly_ruler_std.libs.rotation import Quaternion

from .config import GTMConfig
from .layout import D_DIM, N_DIM, ROOT_PATHS, U_DIM, Y_INDEX, GTMPaths

type Vector = Sequence[float] | NumpyArray
type _Values = dict[str | VariableInfo, NumpyArray]


class Uncertainty(TypedDict, total=False):
    """相对扰动；省略的组使用编译模型的参数默认值。

    Attributes:
        weight: 重量相对增量，例如 0.03 表示增加 3%。
        inertia: 六元 [Ixx,Iyy,Izz,Ixz,Iyz,Ixy] 相对增量。
        c_static: 六元 [CX,CY,CZ,Cl,Cm,Cn] 静态系数相对增量。
        c_e: 三元 [CX,CZ,Cm] 升降舵系数相对增量。
        c_a: 六元副翼系数相对增量。
        c_r: 六元方向舵系数相对增量。
    """

    weight: float
    inertia: Vector
    c_static: Vector
    c_e: Vector
    c_a: Vector
    c_r: Vector


def _vector(values: Vector, size: int, name: str) -> NumpyArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def sample_disturbance(rng: np.random.Generator) -> NumpyArray:
    """生成四路均值 0、标准差 π 的 Dryden 激励，匹配 gtm_ad.create_disturbance。

    Args:
        rng: 调用方持有的随机数发生器，不修改全局随机状态。

    Returns:
        形状 (4,)；不额外按 dt 缩放，外部每次更新后在整个积分步内保持。
    """
    return rng.normal(0.0, np.pi, size=D_DIM)


def sample_sensor_noise(rng: np.random.Generator) -> NumpyArray:
    """生成 17 路标准正态传感器噪声，通道标准差由 Sensor 内部施加。

    Args:
        rng: 调用方持有的随机数发生器。

    Returns:
        形状 (17,)；与 gtm_ad.create_sensor_noise 分布一致，去掉旧命令通道。
    """
    return rng.normal(0.0, 1.0, size=N_DIM)


def build_input[T](
    model: RuntimeInstance[T],
    command: Vector | T,
    *,
    d: Vector | T | None = None,
    n: Vector | T | None = None,
    paths: GTMPaths = ROOT_PATHS,
) -> T:
    """按路径打包命令、Dryden 激励和传感器噪声；省略的噪声置零。

    Args:
        model: 目标后端的编译模型。
        command: [升降舵deg,副翼deg,方向舵deg,油门%]。
        d: 四路风激励，默认零。
        n: 17 路标准化传感器噪声，默认零。
        paths: 模型实例前缀。

    Returns:
        目标后端的完整 u 向量，不生成随机数或裁剪命令。

    Note:
        JAX 输入保留原生数组，支持 jit/vmap，调用方保证数值有限；
        其他后端执行主机数值校验。
    """
    if model.backend.name == "jax":
        # 后端为构造期常量；原生路径的形状由 VariableView 校验，不读取 tracer 的值。
        return model.u.build(
            {
                paths.u.command: command,
                paths.u.disturbance: (0.0,) * D_DIM if d is None else d,
                paths.u.noise: (0.0,) * N_DIM if n is None else n,
            }
        )
    values: _Values = {
        paths.u.command: _vector(model.to_numpy(command), U_DIM, "command"),
        paths.u.disturbance: np.zeros(D_DIM)
        if d is None
        else _vector(model.to_numpy(d), D_DIM, "d"),
        paths.u.noise: np.zeros(N_DIM)
        if n is None
        else _vector(model.to_numpy(n), N_DIM, "n"),
    }
    return model.u.build(values)


@fr.system(config=GTMConfig)
def _InitialKinematics(cfg: GTMConfig) -> None:
    """共享姿态/来流/油门初值图，编译后使用目标后端的数值与微分语义。"""
    euler = fr.input("euler", (3,))
    airdata = fr.input("airdata", (3,))
    rates = fr.input("rates", (3,))
    command = fr.input("command", (4,))
    wind = fr.input("wind", (3,))
    steady = fr.input("steady", (3,))
    fr.export(
        quat=Quaternion.from_euler(fr.math.radians(euler)),
        vel_body=aero_to_body_vel(
            airdata[2], fr.math.radians(airdata[0]), fr.math.radians(airdata[1])
        )
        + steady,
        omega_body=fr.math.radians(rates),
        surfaces=command[:3],
        engine_lag=fr.math.vector(cfg.engine.lag.trim_state(1.0)) * command[3],
        wind_speed=wind[0],
        wind_direction=fr.math.radians(wind[1]),
        turbulence_level=wind[2],
    )


class Initializer[T]:
    """预编译的后端原生初值构造器；可在外部套 JAX jit/vmap/jacfwd。

    Note:
        在 JAX 变换外构造一次，调用时不转 NumPy、不抽取 Python float、不生成随机数。
        输入须有限且 TAS 非负；该原生路径只做静态形状/参数名称检查。
        需要数值校验的普通脚本使用 init。
    """

    def __init__(
        self,
        model: RuntimeInstance[T],
        cfg: GTMConfig,
        *,
        paths: GTMPaths = ROOT_PATHS,
    ) -> None:
        """绑定生产模型与配置并编译共享运动学图。

        Args:
            model: 生产模型，输入输出保持其后端类型。
            cfg: 与生产模型一致的配置。
            paths: 生产实例路径目录。

        Raises:
            ValueError: 模型不是支持的 NumPy、CasADi 或 JAX 后端。
        """
        self._model, self._paths = model, paths
        program = compiler.prepare(
            _InitialKinematics.build("Root", cfg), settings=ModelSettings()
        )
        generated: TargetProgram[object]
        match model.backend.name:
            case "numpy":
                generated = compiler.compile(program, target=targets.numpy)
            case "casadi":
                generated = compiler.compile(program, target=targets.casadi)
            case "jax":
                generated = compiler.compile(program, target=targets.jax)
            case _:
                raise ValueError("Initializer requires a NumPy, CasADi or JAX model")
        self._kernel = runtime.bind(generated, backend=model.backend)
        self._empty_x = self._kernel.x.build()
        self._empty_xd = self._kernel.xd.build()
        self._parameters = self._kernel.p.default()
        self._zero3 = model.backend.runtime_factory().zeros((3,))

    def __call__(
        self,
        *,
        position: BuildValue[T] = (0.0, 0.0, -1000.0),
        airdata: BuildValue[T] = (0.0, 0.0, 100.0),
        euler: BuildValue[T] = (0.0, 0.0, 0.0),
        command: BuildValue[T] = (0.0, 0.0, 0.0, 40.0),
        wind: BuildValue[T] = (0.0, 0.0, 0.0),
        uncertainty: Mapping[str, BuildValue[T]] | None = None,
        rates: BuildValue[T] = (0.0, 0.0, 0.0),
    ) -> tuple[T, T, T, T]:
        """从原生数组或普通数值序列构造 x/xd/u/p。

        Args:
            position: NED 位置，m。
            airdata: alpha/beta（deg）与 TAS（m/s）。
            euler: phi/theta/psi，deg。
            command: 三舵面 deg 与油门百分比。
            wind: 参考风速 m/s、风向 deg、湍流等级。
            uncertainty: weight/inertia/c_static/c_e/c_a/c_r 的相对扰动。
            rates: 机体角速率 p/q/r，deg/s。

        Returns:
            后端原生 (x,xd,u,p)，可直接传给生产模型。

        Raises:
            KeyError: 不确定度参数名或模型路径未知。
            ValueError: 输入形状与模型声明不匹配。
        """
        model, paths, kernel = self._model, self._paths, self._kernel
        inputs: dict[str | VariableInfo, BuildValue[T]] = {
            "Root::euler": euler,
            "Root::airdata": airdata,
            "Root::rates": rates,
            "Root::command": command,
            "Root::wind": wind,
            "Root::steady": self._zero3,
        }
        output = kernel.continuous_fn(
            0.0, self._empty_x, self._empty_xd, kernel.u.build(inputs), self._parameters
        )
        k = kernel.y.read(output.y, by="path")
        parameters: dict[str | VariableInfo, BuildValue[T]] = {
            paths.p.wind_speed: k["Root::wind_speed"],
            paths.p.wind_direction: k["Root::wind_direction"],
            paths.p.turbulence_level: k["Root::turbulence_level"],
        }
        names = {
            "weight": paths.p.weight,
            "inertia": paths.p.inertia,
            "c_static": paths.p.c_static,
            "c_e": paths.p.c_e,
            "c_a": paths.p.c_a,
            "c_r": paths.p.c_r,
        }
        if uncertainty is not None:
            for name, value in uncertainty.items():
                parameters[names[name]] = value
        p = model.p.build(parameters)
        u = model.u.build({paths.u.command: command})
        xd = model.xd.build()
        values: dict[str | VariableInfo, BuildValue[T]] = {
            paths.x.pos_ned: position,
            paths.x.vel_body: k["Root::vel_body"],
            paths.x.quat: k["Root::quat"],
            paths.x.omega_body: k["Root::omega_body"],
            paths.x.actuator_position: k["Root::surfaces"],
            paths.x.engine_lag: k["Root::engine_lag"],
            paths.x.dryden_u: (0.0,),
            paths.x.dryden_v: (0.0, 0.0),
            paths.x.dryden_w: (0.0, 0.0),
            paths.x.dryden_p: (0.0,),
            paths.x.dryden_q: (0.0,),
            paths.x.dryden_r: (0.0,),
        }
        provisional = model.x.build(values)
        result = model.continuous_fn(0.0, provisional, xd, u, p)
        inputs["Root::steady"] = model.effects.read(result.effects, by="path")[
            paths.effects.wind_steady
        ]
        corrected = kernel.continuous_fn(
            0.0, self._empty_x, self._empty_xd, kernel.u.build(inputs), self._parameters
        )
        values[paths.x.vel_body] = kernel.y.read(corrected.y, by="path")[
            "Root::vel_body"
        ]
        return model.x.build(values), xd, u, p


def init[T](
    model: RuntimeInstance[T],
    cfg: GTMConfig,
    *,
    position: Vector = (0.0, 0.0, -1000.0),
    airdata: Vector = (0.0, 0.0, 100.0),
    euler: Vector = (0.0, 0.0, 0.0),
    command: Vector = (0.0, 0.0, 0.0, 40.0),
    wind: Vector = (0.0, 0.0, 0.0),
    uncertainty: Uncertainty | None = None,
    rates: Vector = (0.0, 0.0, 0.0),
    paths: GTMPaths = ROOT_PATHS,
) -> tuple[T, T, T, T]:
    """把位置、气流、姿态、命令和参数直接转换为 x、xd、u、p。

    Args:
        model: 与 cfg 对应的编译模型，返回值保留该后端的数组类型。
        cfg: 生产模型配置，决定发动机滞后状态的阶数和配平映射。
        position: NED (x,y,z)，m，高度为 -z。
        airdata: (alpha_deg,beta_deg,VTAS_mps)，TAS 不小于零。
        euler: (phi,theta,psi)，deg。
        command: 三个实际舵面初值和命令（deg），以及油门初值和命令（%）。
        wind: (参考风速m/s,风来自方向deg,湍流等级)，风向转为参数中的 rad。
        uncertainty: 运行时相对增量，省略的项使用模型参数默认值。
        rates: 初始机体系 (p,q,r)，deg/s。
        paths: 默认 Root，也可指向嵌套实例。

    Returns:
        (x,xd,u,p)，噪声和 Dryden 状态为零。初始化不是飞行配平。

    Raises:
        ValueError: 初值形状不匹配、包含非有限值或 TAS 为负。

    Note:
        v_air=VTAS*[cos(alpha)cos(beta),sin(beta),sin(alpha)cos(beta)]。
        给定姿态/高度/参数后，读取现有 Wind effects 的稳态风，再令
        v_body=v_air+wind_body，不复制风切变公式。零速气流角沿用模型正则化约定。
        其他子系统状态使用 model.x 的默认值；不裁剪调用方提供的命令。
    """
    xyz = _vector(position, 3, "position")
    abv = _vector(airdata, 3, "airdata")
    if abv[2] < 0.0:
        raise ValueError("VTAS must be nonnegative")
    parameters: dict[str, BuildValue[T]] = {}
    if uncertainty is not None:
        if "weight" in uncertainty:
            parameters["weight"] = _vector([uncertainty["weight"]], 1, "weight")
        groups = {
            "inertia": uncertainty.get("inertia"),
            "c_static": uncertainty.get("c_static"),
            "c_e": uncertainty.get("c_e"),
            "c_a": uncertainty.get("c_a"),
            "c_r": uncertainty.get("c_r"),
        }
        for name, size in (
            ("inertia", 6),
            ("c_static", 6),
            ("c_e", 3),
            ("c_a", 6),
            ("c_r", 6),
        ):
            value = groups[name]
            if value is not None:
                parameters[name] = _vector(value, size, name)
    return Initializer(model, cfg, paths=paths)(
        position=xyz,
        airdata=abv,
        euler=_vector(euler, 3, "euler"),
        command=_vector(command, U_DIM, "command"),
        wind=_vector(wind, 3, "wind"),
        rates=_vector(rates, 3, "rates"),
        uncertainty=parameters,
    )


@dataclass(frozen=True)
class Observation:
    """17 个物理观测的主机数值表示，所有角度/角速率使用 deg、deg/s。

    Attributes:
        gamma_deg: 正爬升航迹角。
        alpha_deg: 迎角。
        beta_deg: 侧滑角。
        tas_mps: 真空速，m/s。
        altitude_m: 正向上高度，m。
        euler_deg: (phi,theta,psi)。
        body_rates_deg_s: (p,q,r)。
        nz: 向上为正的法向过载。
        qbar_pa: 动压，Pa。
        surfaces_deg: (elevator,aileron,rudder) 实际位置。
        throttle_percent: 实际有效油门，%。
    """

    gamma_deg: float
    alpha_deg: float
    beta_deg: float
    tas_mps: float
    altitude_m: float
    euler_deg: tuple[float, float, float]
    body_rates_deg_s: tuple[float, float, float]
    nz: float
    qbar_pa: float
    surfaces_deg: tuple[float, float, float]
    throttle_percent: float


def read_observation[T](
    model: RuntimeInstance[T],
    packed_y: T | NumpyArray,
    *,
    measured: bool = False,
    paths: GTMPaths = ROOT_PATHS,
) -> Observation:
    """解析打包输出为带单位的物理观测；默认 clean_y，可选择带误差的 y。

    Args:
        model: 输出所属的编译模型。
        packed_y: continuous_fn/discrete_fn 返回的完整 y 向量。
        measured: True 读取 y，False 读取 clean_y。
        paths: 模型实例前缀。

    Returns:
        主机标量/元组组成的 Observation，不重复解析状态或 effects。
    """
    path = paths.y.measurement if measured else paths.y.clean
    values = model.to_numpy(model.y.read(packed_y, by="path")[path]).reshape(-1)
    i = Y_INDEX
    return Observation(
        gamma_deg=float(values[i["gamma"]]),
        alpha_deg=float(values[i["alpha"]]),
        beta_deg=float(values[i["beta"]]),
        tas_mps=float(values[i["V"]]),
        altitude_m=float(values[i["altitude"]]),
        euler_deg=(
            float(values[i["phi"]]),
            float(values[i["theta"]]),
            float(values[i["psi"]]),
        ),
        body_rates_deg_s=(
            float(values[i["p"]]),
            float(values[i["q"]]),
            float(values[i["r"]]),
        ),
        nz=float(values[i["nz"]]),
        qbar_pa=float(values[i["qbar"]]),
        surfaces_deg=(
            float(values[i["elevator"]]),
            float(values[i["aileron"]]),
            float(values[i["rudder"]]),
        ),
        throttle_percent=float(values[i["throttle"]]),
    )


__all__ = [
    "init",
    "Initializer",
    "build_input",
    "sample_disturbance",
    "sample_sensor_noise",
    "read_observation",
    "Observation",
    "Uncertainty",
]
