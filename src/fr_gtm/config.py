"""FlyRuler GTM 模型的配置对象。

源 `gtm_ad` 包构建一个大型运行时配置对象；本模块保留相同的默认参数推导，
但把数据暴露为 FlyRuler `BaseModel`，使每个子系统只接收自己拥有的配置。
"""

from __future__ import annotations

from math import pi, sqrt
from pathlib import Path

import numpy as np
from fly_ruler_std.blocks.physics import (
    GroundContactConfig,
    GroundContactPointConfig,
)
from fly_ruler_std.libs.units import Force, Inertia, Length
from pydantic import BaseModel, ConfigDict, Field

from .components.aero import AeroConfig, Mode
from .components.engine import (
    EngineConfig,
    EngineLoadsConfig,
    FirstOrderThrottleLagConfig,
    JetCatThrottleLagConfig,
)
from .components.sensor import SensorConfig, SensorSingleConfig
from .components.wind import WindConfig

SCALE_FACTOR = 1.0 / 0.055
"""从缩比 GTM 到全尺寸的固定长度倍率，仅用于两套标定配置。"""


class AircraftConfig(BaseModel):
    """标称飞机几何与质量特性。

    默认值即全尺寸模型；`AircraftConfig.fullscale()` 等价于无参构造，
    `AircraftConfig.subscale()` 返回缩比模型。

    Attributes:
        w0: 标称飞机重量，单位 N。
        inertia0: `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]`，单位 kg·m²。
        S: 参考机翼面积，单位 m²。
        Cbar: 平均气动弦长，单位 m。
        b: 翼展，单位 m。
        le_mac: 平均气动弦前缘到机头的距离，单位 m。
        cg_pos0: 重心相对机头位置，单位 m，机体三元素。
        ref_cg: 气动参考重心相对机头位置，单位 m，机体三元素。

    Note:
        除字段名特别说明外均为 SI 单位。
    """

    model_config = ConfigDict(frozen=True)
    """飞机几何与质量配置构造后不可修改。"""

    w0: float = Force.lbf_to_n(185_000.0)
    inertia0: tuple[float, ...] = tuple(
        float(value)
        for value in Inertia.slug_ft2_to_kg_m2(
            (1_770_000.0, 5_680_000.0, 7_270_000.0, 160_000.0, 0.0, 0.0)
        )
    )
    S: float = 5.9018 * SCALE_FACTOR**2 * Length.ft_to_m(1.0) ** 2
    Cbar: float = Length.ft_to_m(0.9153 * SCALE_FACTOR)
    b: float = Length.ft_to_m(6.8488 * SCALE_FACTOR)
    le_mac: float = Length.ft_to_m(4.5462 * SCALE_FACTOR)
    cg_pos0: tuple[float, float, float] = (
        -(
            Length.ft_to_m(4.5462 * SCALE_FACTOR)
            + Length.ft_to_m(0.9153 * SCALE_FACTOR) * 0.2199
        ),
        Length.ft_to_m(-0.1416 / 12.0 * SCALE_FACTOR),
        Length.ft_to_m(-0.9761 * SCALE_FACTOR),
    )
    ref_cg: tuple[float, float, float] = (
        -(
            Length.ft_to_m(4.5462 * SCALE_FACTOR)
            + Length.ft_to_m(0.9153 * SCALE_FACTOR) * 0.25
        ),
        0.0,
        Length.ft_to_m(-0.9401 * SCALE_FACTOR),
    )

    @classmethod
    def fullscale(cls) -> AircraftConfig:
        """返回全尺寸模型配置（与默认值一致）。

        Returns:
            全尺寸飞机的几何与质量配置。
        """
        return cls()

    @classmethod
    def subscale(cls) -> AircraftConfig:
        """返回缩比模型配置。

        Returns:
            缩比飞机的几何与质量配置。
        """
        S = 5.9018 * Length.ft_to_m(1.0) ** 2
        Cbar = Length.ft_to_m(0.9153)
        b = Length.ft_to_m(6.8488)
        le_mac = Length.ft_to_m(4.5462)

        return cls(
            w0=Force.lbf_to_n(57.75),
            inertia0=tuple(
                float(v)
                for v in Inertia.slug_ft2_to_kg_m2(
                    (1.221, 4.655, 5.587, 0.274, 0.0, 0.006)
                )
            ),
            S=S,
            Cbar=Cbar,
            b=b,
            le_mac=le_mac,
            cg_pos0=(
                -(le_mac + Cbar * 0.2199),
                Length.ft_to_m(-0.1416 / 12.0),
                Length.ft_to_m(-0.9761),
            ),
            ref_cg=(
                -(le_mac + Cbar * 0.25),
                0.0,
                Length.ft_to_m(-0.9401),
            ),
        )


class ActuatorConfig(BaseModel):
    """`[升降舵, 副翼左右, 方向舵]` 的作动器动力学限幅。

    Attributes:
        gain: 一阶作动器增益，三元素。
        rate_limit: 速率上限，单位 deg/s，三元素且必须为正。
        position_lower: 位置下限，单位 deg，三元素。
        position_upper: 位置上限，单位 deg，三元素。

    Note:
        三个向量的排列均为 `[elevator, aileron, rudder]`；位置下限必须严格
        小于上限。油门不是作动器通道，它的指令直接进入 `Engine` 的可配置滞后。
    """

    gain: list[float] = Field(
        default_factory=lambda: [
            5.0 * 2.0 * pi / sqrt(SCALE_FACTOR),
            5.0 * 2.0 * pi / sqrt(SCALE_FACTOR),
            5.0 * 2.0 * pi / sqrt(SCALE_FACTOR),
        ]
    )  # First-order actuator gains.
    rate_limit: list[float] = Field(
        default_factory=lambda: [
            300.0 / sqrt(SCALE_FACTOR),
            300.0 / sqrt(SCALE_FACTOR),
            300.0 / sqrt(SCALE_FACTOR),
        ]
    )  # Rate limits [deg/s].
    position_lower: list[float] = Field(
        default_factory=lambda: [-30.0, -20.0, -30.0]
    )  # Lower position limits [deg].
    position_upper: list[float] = Field(
        default_factory=lambda: [20.0, 20.0, 30.0]
    )  # Upper position limits [deg].

    @classmethod
    def fullscale(cls) -> ActuatorConfig:
        """返回全尺寸作动器增益与速率限幅。

        Returns:
            全尺寸三通道舵面配置。
        """
        return cls()

    @classmethod
    def subscale(cls) -> ActuatorConfig:
        """返回缩比作动器增益与速率限幅。

        Returns:
            缩比三通道舵面配置。
        """
        return cls(
            gain=[5.0 * 2.0 * pi, 5.0 * 2.0 * pi, 5.0 * 2.0 * pi],
            rate_limit=[300.0, 300.0, 300.0],
        )


def _build_ground_config(
    is_fullscale: bool,
    aircraft: AircraftConfig,
    *,
    enabled: bool = False,
) -> GroundContactConfig:
    """由公开的 NASA GTM T2 参数推导 FRD/SI 起落架数据。"""

    length_ratio = SCALE_FACTOR if is_fullscale else 1.0
    weight_ratio = aircraft.w0 / Force.lbf_to_n(57.75)
    source_cg_ft = np.array(
        [
            -(4.5462 + 0.9153 * 0.2199),
            -0.1416 / 12.0,
            -0.9761,
        ]
    )
    source_points_ft = np.array(
        [
            [-22.24 / 12.0, 0.0, -1.7 / 12.0],
            [-61.31 / 12.0, -8.0 / 12.0, -2.64 / 12.0],
            [-61.31 / 12.0, 7.8 / 12.0, -2.64 / 12.0],
        ]
    )
    points_body = Length.ft_to_m((source_points_ft - source_cg_ft) * length_ratio)
    force_per_length = Force.lbf_to_n(1.0) / Length.ft_to_m(1.0)
    stiffness_ratio = weight_ratio / length_ratio
    mass_ratio = weight_ratio
    stiffness = np.array([25.0, 41.6, 42.4]) * 12.0 * force_per_length * stiffness_ratio
    damping = np.full(3, 7.0) * force_per_length * np.sqrt(stiffness_ratio * mass_ratio)
    support_matrix = np.vstack([np.ones(3), points_body[:, 0], points_body[:, 1]])
    static_loads = np.linalg.solve(support_matrix, np.array([aircraft.w0, 0.0, 0.0]))
    names = ("nose", "main_left", "main_right")
    contacts = [
        GroundContactPointConfig(
            name=name,
            point_body=point.tolist(),
            normal_stiffness=float(wheel_stiffness),
            normal_damping=float(wheel_damping),
            rolling_resistance_coefficient=0.02,
            lateral_friction_coefficient=0.5,
            max_normal_force=float(10.0 * static_load),
        )
        for name, point, wheel_stiffness, wheel_damping, static_load in zip(
            names, points_body, stiffness, damping, static_loads, strict=True
        )
    ]
    return GroundContactConfig(enabled=enabled, contacts=contacts)


class GTMConfig(BaseModel):
    """整机配置树，由 fullscale 或 subscale 一次组装各组件。

    Attributes:
        is_fullscale: 工厂所选的尺度标识。
        aircraft: 标称飞机几何与质量特性。
        wind: 风切变与 Dryden 标定值。
        ground: 可选的跑道接触配置。
        engine: 独立的双发载荷与油门滞后配置。
        actuator: 三通道舵面作动器配置。
        aero: 独立的气动表、构型和几何配置。
        sensor: 传感器误差配置。

    Note:
        气动构型仅存放在 aero.mode。两个工厂把标称飞机几何同步提供给风、
        发动机、气动与地面组件；构造后各组件拥有自己的配置。
    """

    is_fullscale: bool
    aircraft: AircraftConfig
    wind: WindConfig
    ground: GroundContactConfig
    engine: EngineConfig
    actuator: ActuatorConfig
    aero: AeroConfig
    sensor: SensorConfig

    @classmethod
    def fullscale(
        cls,
        mode: Mode | str = Mode.NORMAL,
        *,
        data_path: str | Path | None = None,
        enable_ground: bool = False,
    ) -> GTMConfig:
        """构造全尺寸整机配置，显式传递各组件所需的几何。

        Args:
            mode: 襟翼与起落架气动构型。
            data_path: 气动 NPZ 文件或其所在目录，省略时使用包内资源。
            enable_ground: 是否挂载跑道接触子系统。

        Returns:
            面积、翼展、重心和各组件参数一致的全尺寸整机配置。
        """
        aircraft = AircraftConfig.fullscale()
        return cls(
            is_fullscale=True,
            aircraft=aircraft,
            wind=WindConfig(wing_span=aircraft.b),
            ground=_build_ground_config(True, aircraft, enabled=enable_ground),
            engine=EngineConfig(
                loads=EngineLoadsConfig.fullscale(S=aircraft.S, cg_pos=aircraft.cg_pos0)
            ),
            actuator=ActuatorConfig.fullscale(),
            aero=AeroConfig.load(
                S=aircraft.S,
                b=aircraft.b,
                Cbar=aircraft.Cbar,
                cg_pos=aircraft.cg_pos0,
                ref_cg=aircraft.ref_cg,
                mode=mode,
                data_path=data_path,
            ),
            sensor=SensorConfig.fullscale(),
        )

    @classmethod
    def subscale(
        cls,
        mode: Mode | str = Mode.NORMAL,
        *,
        data_path: str | Path | None = None,
        enable_ground: bool = False,
    ) -> GTMConfig:
        """构造缩比整机配置，显式传递各组件所需的几何。

        Args:
            mode: 襟翼与起落架气动构型。
            data_path: 气动 NPZ 文件或其所在目录，省略时使用包内资源。
            enable_ground: 是否挂载跑道接触子系统。

        Returns:
            面积、翼展、重心和各组件参数一致的缩比整机配置。
        """
        aircraft = AircraftConfig.subscale()
        return cls(
            is_fullscale=False,
            aircraft=aircraft,
            wind=WindConfig(wing_span=aircraft.b),
            ground=_build_ground_config(False, aircraft, enabled=enable_ground),
            engine=EngineConfig(
                loads=EngineLoadsConfig.subscale(S=aircraft.S, cg_pos=aircraft.cg_pos0)
            ),
            actuator=ActuatorConfig.subscale(),
            aero=AeroConfig.load(
                S=aircraft.S,
                b=aircraft.b,
                Cbar=aircraft.Cbar,
                cg_pos=aircraft.cg_pos0,
                ref_cg=aircraft.ref_cg,
                mode=mode,
                data_path=data_path,
            ),
            sensor=SensorConfig.subscale(),
        )


__all__ = [
    "Mode",
    "AircraftConfig",
    "ActuatorConfig",
    "GTMConfig",
    "AeroConfig",
    "EngineConfig",
    "EngineLoadsConfig",
    "FirstOrderThrottleLagConfig",
    "JetCatThrottleLagConfig",
    "SensorConfig",
    "SensorSingleConfig",
    "WindConfig",
]
