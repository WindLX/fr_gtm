"""GTM 真值状态与选定 effects 到 fly_ruler_proto 的显式转换边界。"""

import math
from collections.abc import Sequence

import numpy as np
from fly_ruler_core.runtime.instance import RuntimeInstance
from fly_ruler_proto_python import (
    AircraftState,
    Attitude,
    ControlSurfaceState,
    DerivedState,
    PropulsorKind,
    PropulsorState,
    TelemetryField,
    TelemetryStreamSchema,
    create_aircraft_state,
)

from ..config import GTMConfig, Mode
from ..init import read_observation
from ..layout import ROOT_PATHS, GTMPaths

VISUAL_MODEL_ID = "boeing_737_800"
"""实时示例使用的可视化机型，不参与动力学计算。"""


def to_aircraft_state[T](
    model: RuntimeInstance[T],
    cfg: GTMConfig,
    *,
    x: T,
    y: T,
    effects: T,
    origin: tuple[float, float, float],
    paths: GTMPaths = ROOT_PATHS,
) -> AircraftState:
    """由状态、clean_y 与 effects 构造协议真值；不读取带噪观测。

    Args:
        model: 数据所属的编译模型，后端数值在此转为主机标量。
        cfg: 对应飞机配置，用于襟翼构型。
        x: 完整连续状态。
        y: 完整打包输出，始终从中选择 clean_y。
        effects: 同一时刻的完整诊断向量。
        origin: NED 原点 (纬度deg,经度deg,海拔m)。
        paths: 飞机实例的路径目录。

    Returns:
        NED 位置、BODY-FRD 速度、scalar-first 姿态及实际操纵面的 AircraftState。

    Raises:
        ValueError: 原点不是有限值，或经纬度超出局部坐标换算范围。

    Note:
        地理坐标采用与 std ned_position_to_geodetic 相同的 WGS-84 局部曲率近似，
        适用于原点附近飞行。高度为 origin.altitude-down，垂直速度向上为正。
        deg→rad 只发生在气流角、航迹角与舵面边界；x 中角速度本来就是 rad/s。
    """
    lat0, lon0, h0 = origin
    if (
        not np.isfinite(origin).all()
        or not -90.0 < lat0 < 90.0
        or not -180 <= lon0 <= 180
    ):
        raise ValueError(
            "origin must be finite with latitude in (-90,90), longitude in [-180,180]"
        )
    states = model.x.read(x, by="path")
    diagnostics = model.effects.read(effects, by="path")
    position = model.to_numpy(states[paths.x.pos_ned]).reshape(3)
    velocity = model.to_numpy(states[paths.x.vel_body]).reshape(3)
    quat = model.to_numpy(states[paths.x.quat]).reshape(4)
    omega = model.to_numpy(states[paths.x.omega_body]).reshape(3)
    ned = model.to_numpy(diagnostics[paths.effects.vel_ned]).reshape(3)
    obs = read_observation(model, y, paths=paths)
    # WGS-84 子午圈/卯酉圈曲率半径；与 std coordinates 的局部 NED 定义一致。
    eccentricity_squared = 6.6943799901413165e-3
    temp = 1.0 - eccentricity_squared * math.sin(math.radians(lat0)) ** 2
    rm = 6378137.0 * (1.0 - eccentricity_squared) / temp**1.5
    rn = 6378137.0 / math.sqrt(temp)
    lat = lat0 + math.degrees(float(position[0]) / (rm + h0))
    lon = lon0 + math.degrees(
        float(position[1]) / ((rn + h0) * math.cos(math.radians(lat0)))
    )
    match cfg.aero.mode:
        case Mode.NORMAL:
            flap = 0.0
        case Mode.HALF_FLAP_GEAR_UP:
            flap = 0.5
        case Mode.FULL_FLAP_GEAR_UP | Mode.FULL_FLAP_GEAR_DOWN:
            flap = 1.0
    elevator, aileron, rudder = map(math.radians, obs.surfaces_deg)
    # 协议油门为 [0,1] 比例；可选二阶模型的原始有效油门可从 clean_y 独立记录。
    throttle = float(np.clip(obs.throttle_percent / 100.0, 0.0, 1.0))
    return create_aircraft_state(
        position=(float(position[0]), float(position[1]), float(position[2])),
        velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
        attitude=Attitude.from_quaternion(tuple(float(v) for v in quat)),
        angular_velocity=(float(omega[0]), float(omega[1]), float(omega[2])),
        derived=DerivedState(
            lat=lat,
            lon=lon,
            altitude=h0 - float(position[2]),
            alpha=math.radians(obs.alpha_deg),
            beta=math.radians(obs.beta_deg),
            tas=obs.tas_mps,
            eas=float(
                model.to_numpy(diagnostics[paths.effects.equivalent_airspeed]).item()
            ),
            gamma=math.radians(obs.gamma_deg),
            chi=float(model.to_numpy(diagnostics[paths.effects.chi]).item()),
            mach=float(model.to_numpy(diagnostics[paths.effects.mach]).item()),
            ground_speed=math.hypot(float(ned[0]), float(ned[1])),
            vertical_speed=-float(ned[2]),
            dynamic_pressure=obs.qbar_pa,
            normal_load_factor=obs.nz,
        ),
        control_surfaces=ControlSurfaceState(
            aileron_left_rad=aileron,
            aileron_right_rad=-aileron,
            elevator_rad=elevator,
            rudder_rad=rudder,
            flaps_left_ratio=flap,
            flaps_right_ratio=flap,
            spoilers_ratio=0.0,
        ),
        propulsors=[
            PropulsorState(
                name, kind=PropulsorKind.JET, throttle_ratio=throttle, index=i
            )
            for i, name in enumerate(("engine.left", "engine.right"), start=1)
        ],
    )


class EffectsTelemetry[T]:
    """绑定诊断布局，保证连接前声明与逐帧取值使用完全相同的字段顺序。"""

    schema: TelemetryStreamSchema
    """连接前向 FlyRulerClient 注册的诊断流结构。"""

    def __init__(
        self,
        model: RuntimeInstance[T],
        *,
        paths: Sequence[str],
        stream_id: str = "gtm.effects",
        nominal_rate_hz: float | None = None,
    ) -> None:
        """按显式路径选择诊断；向量/矩阵以 C 顺序逐标量展开。

        Args:
            model: 诊断向量所属模型。
            paths: 有序且不重复的完整 effects 路径；不自动发送所有诊断。
            stream_id: Proto 遥测流名称。
            nominal_rate_hz: 标称发送频率，实际调度由调用方负责。

        Raises:
            ValueError: 路径列表为空、重复，流名为空或标称频率不是有限正数。
            KeyError: 所选 effects 不存在，例如未启用的地面诊断。
        """
        if not paths or len(set(paths)) != len(paths) or not stream_id:
            raise ValueError("select nonempty unique effects paths and a stream_id")
        if nominal_rate_hz is not None and (
            not math.isfinite(nominal_rate_hz) or nominal_rate_hz <= 0
        ):
            raise ValueError("nominal_rate_hz must be finite and positive")
        self._model = model
        for path in paths:
            if path not in model.effects.keys:
                raise KeyError(f"Unknown complete effects path: {path}")
        self._infos = tuple(model.effects.info(path) for path in paths)
        fields: list[TelemetryField] = []
        for info in self._infos:
            for index in np.ndindex(info.dim):
                suffix = "" if info.size == 1 else "[" + ",".join(map(str, index)) + "]"
                fields.append(
                    TelemetryField(
                        info.path + suffix,
                        label=info.name + suffix,
                        group=info.path.split("::")[0],
                        unit=info.unit,
                        description=info.description,
                    )
                )
        self.schema = TelemetryStreamSchema(
            stream_id,
            fields=fields,
            name="GTM diagnostics",
            nominal_rate_hz=nominal_rate_hz,
        )

    @property
    def stream_id(self) -> str:
        """发布时使用的遥测流标识。"""
        return self.schema.stream_id

    def values(self, effects: T) -> tuple[float, ...]:
        """取出与 schema 字段一一对应的主机浮点数。

        Args:
            effects: 构造绑定时那个模型的完整 effects 向量。

        Returns:
            路径选择顺序、各变量 C 顺序展开的标量元组。
        """
        values = self._model.effects.read(effects, by="path")
        return tuple(
            float(value)
            for info in self._infos
            for value in self._model.to_numpy(values[info.path]).reshape(-1)
        )


__all__ = ["to_aircraft_state", "EffectsTelemetry", "VISUAL_MODEL_ID"]
