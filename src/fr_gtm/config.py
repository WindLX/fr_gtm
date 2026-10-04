"""FlyRuler GTM 模型的配置对象。

源 `gtm_ad` 包构建一个大型运行时配置对象；本模块保留相同的默认参数推导，
但把数据暴露为 FlyRuler `BaseModel`，使每个子系统只接收自己拥有的配置。
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from importlib import resources
from math import pi, sqrt
from pathlib import Path
from typing import cast

import numpy as np
import rtoml
from fly_ruler_std.blocks.physics import (
    GroundContactConfig,
    GroundContactPointConfig,
)
from fly_ruler_std.libs.units import Force, Inertia, Length, Pressure, Speed
from pydantic import BaseModel, Field, field_validator, model_validator

from .resources import AeroTableResources, load_aero_tables


class Mode(str, Enum):
    """GTM 襟翼与起落架气动构型。

    Attributes:
        NORMAL: 光洁构型，无襟翼、起落架收起。
        HALF_FLAP_GEAR_UP: 半襟翼、起落架收起。
        FULL_FLAP_GEAR_UP: 全襟翼、起落架收起。
        FULL_FLAP_GEAR_DOWN: 全襟翼、起落架放下。
    """

    NORMAL = "normal"
    HALF_FLAP_GEAR_UP = "0.5_flap_gear_up"
    FULL_FLAP_GEAR_UP = "1_flap_gear_up"
    FULL_FLAP_GEAR_DOWN = "1_flap_gear_down"


class UncertaintyConfig(BaseModel):
    """运行时摄动使用的标称全尺寸/缩比质量特性。

    Attributes:
        w0: 标称重量，单位 N，必须为正。
        Ixx0: 绕机体 x 轴的标称转动惯量，单位 kg·m²，必须为正。
        Iyy0: 绕机体 y 轴的标称转动惯量，单位 kg·m²，必须为正。
        Izz0: 绕机体 z 轴的标称转动惯量，单位 kg·m²，必须为正。
        Ixz0: 惯性积 Ixz，单位 kg·m²。
        Iyz0: 惯性积 Iyz，单位 kg·m²。
        Ixy0: 惯性积 Ixy，单位 kg·m²。
        gravity_acc: 重力加速度，单位 m/s²，必须为正。

    Note:
        惯性积保留源模型的符号约定；Mass Block 会把这份六元表示转换为对称的
        FRD 惯量张量。
    """

    w0: float = Field(default=Force.lbf_to_n(185_000.0), gt=0.0)  # Weight [N].
    Ixx0: float = Field(default=Inertia.slug_ft2_to_kg_m2(1_770_000.0), gt=0.0)
    Iyy0: float = Field(default=Inertia.slug_ft2_to_kg_m2(5_680_000.0), gt=0.0)
    Izz0: float = Field(default=Inertia.slug_ft2_to_kg_m2(7_270_000.0), gt=0.0)
    Ixz0: float = Inertia.slug_ft2_to_kg_m2(160_000.0)  # Ixz (kg*m^2).
    Iyz0: float = 0.0  # Iyz (kg*m^2).
    Ixy0: float = 0.0  # Ixy (kg*m^2).
    gravity_acc: float = Field(default=9.80665, gt=0.0)  # m/s^2


class EngineLagConfig(BaseModel):
    """原始 NASA 缩比 GTM 的 JetCat P70 二阶油门滞后。

    状态空间为 `ẋ_e = (A x_e + B δ_c) / c`、`δ = C x_e`，其中 `δ_c` 是钳位后的
    油门指令百分比，`δ` 是有效油门百分比，`c` 是时间尺度系数。矩阵按行优先
    展开存放：`state_matrix` 是 `[a11, a12, a21, a22]`。

    Attributes:
        state_matrix: 状态矩阵 `A` 的行优先展开，四元素。
        input_matrix: 输入矩阵 `B`，两元素。
        output_matrix: 输出矩阵 `C`，两元素。
        time_scale: 时间尺度系数，必须为正；`A` 与 `B` 同时除以它。
        command_lower: 油门指令下限，单位 %。
        command_upper: 油门指令上限，单位 %，必须大于下限。

    Note:
        默认矩阵取自 `gtm_design/gtm_design.slx` 的 `Left_Engine / Jetcat P70`
        `Throttle Lag` 子系统，直流增益为 1，因此静态映射的输入与输出都是
        有效油门百分比。`time_scale` 在 fullscale 与 scale 下都取 1，两侧滞后
        动态相同，不按几何扩比因子缩放。
    """

    state_matrix: list[float] = Field(
        default_factory=lambda: [-1.336, -0.7314, 1.0, 0.0]
    )  # A, row-major.
    input_matrix: list[float] = Field(default_factory=lambda: [1.0, 0.0])  # B.
    output_matrix: list[float] = Field(default_factory=lambda: [-0.1474, 0.7314])  # C.
    time_scale: float = Field(default=1.0, gt=0.0)
    command_lower: float = 0.0  # Throttle command lower limit [%].
    command_upper: float = 100.0  # Throttle command upper limit [%].

    @model_validator(mode="after")
    def validate_lag(self) -> "EngineLagConfig":
        """校验滞后矩阵形状、限幅关系与单位直流增益。

        Returns:
            通过校验的自身实例。

        Raises:
            ValueError: 矩阵长度或有限性不合法、指令下限不低于上限、状态矩阵
                奇异，或直流增益不为 1。
        """

        for name, value, size in (
            ("state_matrix", self.state_matrix, 4),
            ("input_matrix", self.input_matrix, 2),
            ("output_matrix", self.output_matrix, 2),
        ):
            if len(value) != size or not np.all(np.isfinite(value)):
                raise ValueError(f"engine lag {name} must contain {size} finite values")
        if self.command_lower >= self.command_upper:
            raise ValueError("engine lag command lower limit must be below upper limit")
        if abs(self.dc_gain - 1.0) > 1e-9:
            raise ValueError(
                "engine lag must have unit DC gain so the effective throttle "
                "matches the commanded throttle at trim"
            )
        return self

    @property
    def trim_map(self) -> tuple[float, float]:
        """`-A⁻¹ B`：把有效油门映射到配平一致状态的两位向量。

        Returns:
            `(m1, m2)`，满足 `x_{e,★} = (m1, m2) · δ_★`。

        Raises:
            ValueError: 状态矩阵奇异，配平状态不唯一。
        """

        a11, a12, a21, a22 = self.state_matrix
        b1, b2 = self.input_matrix
        det = a11 * a22 - a12 * a21
        if det == 0.0:
            raise ValueError("engine lag state matrix must be nonsingular")
        return ((a12 * b2 - a22 * b1) / det, (a21 * b1 - a11 * b2) / det)

    @property
    def dc_gain(self) -> float:
        """`C (-A⁻¹ B)`，二阶滞后从有效油门到有效油门的直流增益。

        Returns:
            直流增益，必须为 1 才使有效油门在配平时等于指令油门。
        """

        m1, m2 = self.trim_map
        c1, c2 = self.output_matrix
        return c1 * m1 + c2 * m2

    def trim_state(self, throttle: float) -> tuple[float, float]:
        """返回有效油门为 `throttle` 时的配平一致滞后状态。

        把两个内部状态都置零会人为制造一段推力暂态，因此任何非零油门初值都
        必须用本函数给出 `x_{e,★}`。

        Args:
            throttle: 配平有效油门百分比。

        Returns:
            `x_{e,★} = -A⁻¹ B δ_★`，两位内部状态。
        """

        m1, m2 = self.trim_map
        return (m1 * throttle, m2 * throttle)


class EngineConfig(BaseModel):
    """GTM 双发推力与陀螺力矩配置。

    Attributes:
        T_gain: 推力增益，把模型推力换算为 N。
        h_gain: 角动量增益，单位 (kg·m²)/(slug·ft²)。
        RPMref_gain: 参考 RPM 缩放增益。
        ram_drag_coeff: 冲压阻力系数。
        S: 冲压阻力使用的参考机翼面积，单位 m²。
        shaft_diameter: 转轴直径，单位 ft，沿用源模型。
        shaft_length: 转轴长度，单位 ft，沿用源模型。
        density: 转轴材料密度，单位 slug/ft³。
        thrust_coefficients: 由参考 RPM 计算推力的多项式系数，最高次在前。
        throttle_coefficients: 由有效油门计算参考 RPM 的多项式系数，最高次在前。
        lag: 油门指令到有效油门的二阶滞后。
        engl_pos: 左发安装位置，单位 m，机体三元素。
        engr_pos: 右发安装位置，单位 m，机体三元素。
        engl_ang: 左发 `[roll, pitch, yaw]` 安装角，单位 deg。
        engr_ang: 右发 `[roll, pitch, yaw]` 安装角，单位 deg。
        cg_pos: 重心相对机头位置，单位 m，机体三元素。
    """

    T_gain: float = 1.0  # Thrust gain converting model thrust to Newtons.
    h_gain: float = 1.0  # Angular momentum gain (kg*m^2)/(slug*ft^2).
    RPMref_gain: float = 1.0  # Reference RPM scaling gain.
    ram_drag_coeff: float = 0.01  # Ram drag coefficient.
    S: float = 1.0  # Reference wing area used for ram drag (m^2).
    shaft_diameter: float = 0.75 / 12.0  # Shaft diameter (ft, per source model).
    shaft_length: float = 6.25 / 12.0  # Shaft length (ft, per source model).
    density: float = 15.2  # Shaft material density (slug/ft^3).
    thrust_coefficients: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.41266]
    )  # Polynomial coefficients for thrust from reference RPM.
    throttle_coefficients: list[float] = Field(
        default_factory=lambda: [0.0, -0.00835, 1.19113, 31.519]
    )  # Polynomial coefficients for reference RPM from throttle.
    lag: EngineLagConfig = Field(default_factory=EngineLagConfig)
    engl_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    engr_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    engl_ang: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    engr_ang: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    cg_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @field_validator(
        "engl_pos",
        "engr_pos",
        "engl_ang",
        "engr_ang",
        "cg_pos",
    )
    @classmethod
    def validate_vector3(cls, value: list[float]) -> list[float]:
        """校验发动机安装位置、安装角与重心向量都是 3 个有限值。

        Args:
            value: 待校验的机体三元素向量。

        Returns:
            通过校验的原向量。

        Raises:
            ValueError: 长度不为 3，或存在非有限值。
        """

        if len(value) != 3 or not np.all(np.isfinite(value)):
            raise ValueError(
                "engine position/alignment vectors must contain 3 finite values"
            )
        return value

    @field_validator("thrust_coefficients", "throttle_coefficients")
    @classmethod
    def validate_polynomial(cls, value: list[float]) -> list[float]:
        """校验推力与参考转速多项式都是 4 个有限系数（最高次在前）。

        Args:
            value: 待校验的多项式系数列表。

        Returns:
            通过校验的原系数列表。

        Raises:
            ValueError: 长度不为 4，或存在非有限值。
        """

        if len(value) != 4 or not np.all(np.isfinite(value)):
            raise ValueError(
                "engine polynomial coefficients must contain 4 finite values"
            )
        return value


class SensorSingleConfig(BaseModel):
    """单个传感器通道的偏置、比例与白噪声标准差。

    Attributes:
        bias: 加性偏置，单位与对应物理量一致。
        scale: 乘性比例系数。
        noise_std: 白噪声标准差，单位与对应物理量一致，必须非负。

    Note:
        `bias`、`scale` 与 `noise_std` 都必须是有限值。
    """

    bias: float = 0.0
    scale: float = 1.0
    noise_std: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> "SensorSingleConfig":
        """校验偏置、比例与噪声标准差三个字段都是有限值。

        Returns:
            通过校验的自身实例。

        Raises:
            ValueError: 任一字段不是有限值。
        """

        if not np.isfinite([self.bias, self.scale, self.noise_std]).all():
            raise ValueError("sensor bias, scale and noise_std must be finite")
        return self


class SensorConfig(BaseModel):
    """传感器输出向量的 GTM 传感器误差配置。

    Attributes:
        gamma: 航迹角通道误差，单位 deg。
        alpha: 迎角通道误差，单位 deg。
        beta: 侧滑角通道误差，单位 deg。
        V: 真空速通道误差，单位 m/s。
        altitude: 高度通道误差，单位 m。
        phi: 滚转角通道误差，单位 deg。
        theta: 俯仰角通道误差，单位 deg。
        psi: 偏航角通道误差，单位 deg。
        p: 滚转角速率通道误差，单位 deg/s。
        q: 俯仰角速率通道误差，单位 deg/s。
        r: 偏航角速率通道误差，单位 deg/s。
        nz: 法向过载通道误差，无量纲。
        qbar: 动压通道误差，单位 Pa。
        elevator: 升降舵位置通道误差，单位 deg。
        aileron: 副翼位置通道误差，单位 deg。
        rudder: 方向舵位置通道误差，单位 deg。

    Note:
        各通道误差按 `bias + scale * clean + noise * noise_std` 施加，其中
        `noise` 是外部输入的标准化噪声。
    """

    gamma: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    alpha: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    beta: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    V: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    altitude: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=3.0)
    )
    phi: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    theta: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    psi: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    p: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    q: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    r: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    nz: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.015)
    )
    qbar: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(
            noise_std=Pressure.psi_to_pa(0.00028)
        )
    )
    elevator: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.01)
    )
    aileron: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.01)
    )
    rudder: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.01)
    )


class ControlSurfacesConfig(BaseModel):
    """舵面展向力臂。

    Attributes:
        arm_left_out: 左外副翼力臂，单位 m。
        arm_left_in: 左内副翼力臂，单位 m。
        arm_right_in: 右内副翼力臂，单位 m。
        arm_right_out: 右外副翼力臂，单位 m。
    """

    arm_left_out: float = -0.07
    arm_left_in: float = -0.03
    arm_right_in: float = 0.03
    arm_right_out: float = 0.07


class DynamicDerivativesConfig(BaseModel):
    """Kalviste 角速率分解使用的参考长度。

    Attributes:
        b: 翼展，单位 m，用于归一化 p/r。
        Cbar: 平均气动弦长，单位 m，用于归一化 q。
    """

    b: float = 6.0  # Wing span (m).
    Cbar: float = 1.0  # Mean aerodynamic chord (m).


class AeroForceMomentConfig(BaseModel):
    """国际单位制下的参考面积与长度及气动中心偏置。

    Attributes:
        Sref: 参考机翼面积，单位 m²。
        b: 翼展，单位 m。
        Cbar: 平均气动弦长，单位 m。
        arm_x: 压心相对重心的偏置 x，单位 m。
        arm_y: 压心相对重心的偏置 y，单位 m。
        arm_z: 压心相对重心的偏置 z，单位 m。
    """

    Sref: float = 1.0  # Reference wing area (m^2).
    b: float = 1.0  # Wing span (m).
    Cbar: float = 1.0  # Mean aerodynamic chord (m).
    arm_x: float = 0.0  # CP - CG offset x (m).
    arm_y: float = 0.0  # CP - CG offset y (m).
    arm_z: float = 0.0  # CP - CG offset z (m).


class AircraftConfig(BaseModel):
    """标称飞机几何与质量特性。

    Attributes:
        w0: 标称飞机重量，单位 N。
        inertia0: `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]`，单位 kg·m²。
        S: 参考机翼面积，单位 m²。
        Cbar: 平均气动弦长，单位 m。
        b: 翼展，单位 m。
        le_mac: 平均气动弦前缘到机头的距离，单位 m。
        cg_pos0: 重心相对机头位置，单位 m，机体三元素。
        ref_cg: 气动参考重心相对机头位置，单位 m，机体三元素。
        engl_pos: 左发安装位置，单位 m，机体三元素。
        engr_pos: 右发安装位置，单位 m，机体三元素。
        engl_ang: 左发 `[roll, pitch, yaw]` 安装角，单位 deg。
        engr_ang: 右发 `[roll, pitch, yaw]` 安装角，单位 deg。

    Note:
        除字段名特别说明外均为 SI 单位。默认值由 `gtm_ad` 缩比模型数据与
        全尺寸缩放规则生成。重量与参考几何必须为正，`inertia0` 必须给出
        正定的惯量张量，各三元素向量必须是有限值。
    """

    w0: float = Force.lbf_to_n(185_000.0)  # Nominal aircraft weight (N).
    inertia0: list[float] = Field(
        default_factory=lambda: (
            Inertia.slug_ft2_to_kg_m2(
                np.array(
                    [
                        1_770_000.0,
                        5_680_000.0,
                        7_270_000.0,
                        160_000.0,
                        0.0,
                        0.0,
                    ]
                )
            )
        ).tolist()
    )  # [Ixx, Iyy, Izz, Ixz, Iyz, Ixy] in kg*m^2.
    S: float = 5.9018 * (1.0 / 0.055) ** 2 * Length.ft_to_m(1.0) ** 2
    Cbar: float = Length.ft_to_m(0.9153 * (1.0 / 0.055))  # Mean aero chord (m).
    b: float = Length.ft_to_m(6.8488 * (1.0 / 0.055))  # Wing span (m).
    le_mac: float = Length.ft_to_m(4.5462 * (1.0 / 0.055))  # MAC LE from nose (m).
    cg_pos0: list[float] = Field(default_factory=list)  # CG from nose (m).
    ref_cg: list[float] = Field(
        default_factory=list
    )  # Aero reference CG from nose (m).
    engl_pos: list[float] = Field(default_factory=list)  # Left engine position (m).
    engr_pos: list[float] = Field(default_factory=list)  # Right engine position (m).
    engl_ang: list[float] = Field(
        default_factory=lambda: [0.0, 1.98, 2.22]
    )  # Left engine [roll, pitch, yaw] alignment (deg).
    engr_ang: list[float] = Field(
        default_factory=lambda: [0.0, 1.95, -1.18]
    )  # Right engine [roll, pitch, yaw] alignment (deg).

    @model_validator(mode="after")
    def validate_geometry_and_inertia(self) -> "AircraftConfig":
        """校验重量与参考几何为正、惯量张量正定。

        `w0`、`S`、`Cbar`、`b` 必须为正；`inertia0` 必须是 6 个有限值且按源
        模型符号约定展开的惯量张量正定；`cg_pos0`、`ref_cg` 与四个发动机
        位置/安装角向量必须为空或 3 个有限值。

        Returns:
            通过校验的自身实例。

        Raises:
            ValueError: 重量或参考几何非正、惯量向量非法或张量非正定，
                或几何向量长度不是 3、含非有限值。
        """

        if self.w0 <= 0.0 or self.S <= 0.0 or self.Cbar <= 0.0 or self.b <= 0.0:
            raise ValueError("aircraft weight and reference geometry must be positive")
        inertia = np.asarray(self.inertia0, dtype=float)
        if inertia.shape != (6,) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia0 must contain 6 finite values")
        tensor = np.array(
            [
                [inertia[0], -inertia[5], -inertia[3]],
                [-inertia[5], inertia[1], -inertia[4]],
                [-inertia[3], -inertia[4], inertia[2]],
            ]
        )
        if np.min(np.linalg.eigvalsh(tensor)) <= 0.0:
            raise ValueError("inertia0 must define a positive-definite tensor")
        for name, value in (
            ("cg_pos0", self.cg_pos0),
            ("ref_cg", self.ref_cg),
            ("engl_pos", self.engl_pos),
            ("engr_pos", self.engr_pos),
            ("engl_ang", self.engl_ang),
            ("engr_ang", self.engr_ang),
        ):
            if value and (len(value) != 3 or not np.all(np.isfinite(value))):
                raise ValueError(f"{name} must contain 3 finite values")
        return self


class WindConfig(BaseModel):
    """编译期风切变与 Dryden 标定值。

    Attributes:
        runway_altitude: 风切变使用的跑道高度，单位 m。
        wind_speed_low: 低空风强度，单位 m/s，必须为正。
        scale_length_high: 高空尺度长度，单位 m，必须为正。
        noise_sample_time: 噪声采样时间，单位 s，必须为正。

    Note:
        平均风速、风向与湍流强度仍是运行时参数；风向在公开模型中统一使用
        弧度。
    """

    runway_altitude: float = 0.0  # Runway altitude used by wind shear (m).
    wind_speed_low: float = Field(default=15.0, gt=0.0)  # Low-altitude intensity [m/s].
    scale_length_high: float = Field(default=533.4, gt=0.0)  # High scale [m].
    noise_sample_time: float = Field(default=0.1, gt=0.0)  # Noise sample time [s].


class ActuatorConfig(BaseModel):
    """`[升降舵, 副翼左右, 方向舵]` 的作动器动力学限幅。

    Attributes:
        gain: 一阶作动器增益，三元素。
        rate_limit: 速率上限，单位 deg/s，三元素且必须为正。
        position_lower: 位置下限，单位 deg，三元素。
        position_upper: 位置上限，单位 deg，三元素。

    Note:
        三个向量的排列均为 `[elevator, aileron, rudder]`；位置下限必须严格
        小于上限。油门不是作动器通道，它的指令直接进入 `Engine` 的二阶滞后。
    """

    gain: list[float] = Field(default_factory=list)  # First-order actuator gains.
    rate_limit: list[float] = Field(default_factory=list)  # Rate limits [deg/s].
    position_lower: list[float] = Field(
        default_factory=lambda: [-30.0, -20.0, -30.0]
    )  # Lower position limits [deg].
    position_upper: list[float] = Field(
        default_factory=lambda: [20.0, 20.0, 30.0]
    )  # Upper position limits [deg].

    @model_validator(mode="after")
    def validate_channels(self) -> "ActuatorConfig":
        """校验三个作动器通道向量同长、有限，且限幅关系自洽。

        Returns:
            通过校验的自身实例。

        Raises:
            ValueError: 非空向量长度不为 3 或含非有限值、位置下限不低于
                上限，或速率上限非正。
        """

        vectors = [self.gain, self.rate_limit, self.position_lower, self.position_upper]
        for value in vectors:
            if value and (len(value) != 3 or not np.all(np.isfinite(value))):
                raise ValueError("actuator vectors must contain 3 finite values")
        if (
            self.position_lower
            and self.position_upper
            and np.any(
                np.asarray(self.position_lower) >= np.asarray(self.position_upper)
            )
        ):
            raise ValueError("actuator lower limits must be below upper limits")
        if self.rate_limit and np.any(np.asarray(self.rate_limit) <= 0.0):
            raise ValueError("actuator rate limits must be positive")
        return self


class AeroConfig(BaseModel):
    """全部 GTM 气动表与力/力矩几何。

    Attributes:
        data_path: 可选的 `aero_tables.npz` 路径或其所在目录；为 `None` 时
            使用包内资源。
        tables: 经过校验的冻结气动表资源。
        control_surfaces: 舵面展向力臂。
        dynamic_derivatives: Kalviste 动导参考长度。
        force_moment: 参考面积/长度与气动中心偏置。

    Note:
        `tables` 被排除在配置序列化之外，避免 JSON/TOML 配置复制大型数组。
    """

    # Optional aero_tables.npz path or containing directory.
    data_path: str | None = None
    tables: AeroTableResources = Field(exclude=True)
    control_surfaces: ControlSurfacesConfig = Field(
        default_factory=ControlSurfacesConfig
    )
    dynamic_derivatives: DynamicDerivativesConfig = Field(
        default_factory=DynamicDerivativesConfig
    )
    force_moment: AeroForceMomentConfig = Field(default_factory=AeroForceMomentConfig)


class GTMConfig(BaseModel):
    """按缩比/构型选择一致派生的顶层 GTM 配置。

    Attributes:
        is_fullscale: 是否使用全尺寸飞机参数。
        mode: 襟翼与起落架气动构型。
        aircraft: 标称几何与质量特性。
        wind: 风切变与 Dryden 标定值。
        ground: 跑道接触子系统配置。
        uncertainty: 质量与惯量不确定性标称值。
        engine: 双发推力与陀螺力矩配置。
        actuator: 作动器动力学限幅。
        aero: 气动表与力/力矩几何。
        sensor: 传感器误差配置。

    Note:
        切换全尺寸/缩比模型或襟翼/起落架构型时，应调用 `default` 而不是独立
        构造嵌套配置；几何、发动机、作动器、传感器与气动资源都由该选择耦合
        决定。`is_fullscale` 与 `mode` 属于编译期配置，运行时参数 `p` 无法
        覆盖它们。
    """

    is_fullscale: bool = True
    mode: Mode = Mode.NORMAL
    aircraft: AircraftConfig = Field(
        default_factory=lambda: _build_aircraft_config(True)
    )
    wind: WindConfig = Field(default_factory=WindConfig)
    ground: GroundContactConfig = Field(
        default_factory=lambda: _build_ground_config(True, _build_aircraft_config(True))
    )
    uncertainty: UncertaintyConfig = Field(
        default_factory=lambda: _build_uncertainty_config(_build_aircraft_config(True))
    )
    engine: EngineConfig = Field(
        default_factory=lambda: _build_engine_config(True, _build_aircraft_config(True))
    )
    actuator: ActuatorConfig = Field(
        default_factory=lambda: _build_actuator_config(True)
    )
    aero: AeroConfig = Field(
        default_factory=lambda: _build_aero_config(_build_aircraft_config(True))
    )
    sensor: SensorConfig = Field(default_factory=lambda: _build_sensor_config(True))

    @classmethod
    def default(
        cls,
        is_fullscale: bool = True,
        mode: Mode | str = Mode.NORMAL,
        *,
        data_path: str | Path | None = None,
        enable_ground: bool = False,
    ) -> "GTMConfig":
        """创建按选择一致派生的默认 GTM 配置。

        Args:
            is_fullscale: 为 `True` 时使用全尺寸飞机参数，否则使用缩比模型参数。
            mode: 襟翼与起落架气动构型。
            data_path: 可选的 `aero_tables.npz` 路径或其所在目录。
            enable_ground: 是否实例化固定翼跑道接触子系统。

        Returns:
            GTMConfig: 几何、发动机、作动器、传感器与气动表已按 `is_fullscale`
                和 `mode` 一致派生的配置对象。

        Examples:
            ```python
            from fr_gtm import GTMConfig, Mode

            cfg = GTMConfig.default(is_fullscale=True, mode=Mode.NORMAL)
            ```
        """
        mode_value = _parse_mode(mode)
        aircraft = _build_aircraft_config(is_fullscale)
        return cls(
            is_fullscale=is_fullscale,
            mode=mode_value,
            aircraft=aircraft,
            wind=WindConfig(),
            ground=_build_ground_config(
                is_fullscale,
                aircraft,
                enabled=enable_ground,
            ),
            uncertainty=_build_uncertainty_config(aircraft),
            engine=_build_engine_config(is_fullscale, aircraft),
            actuator=_build_actuator_config(is_fullscale),
            aero=_build_aero_config(aircraft, data_path=data_path),
            sensor=_build_sensor_config(is_fullscale),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "GTMConfig":
        """从 TOML 文件加载默认 GTM 配置并叠加稀疏覆盖。

        Args:
            path: TOML 文件路径。

        Returns:
            GTMConfig: 以 `default` 为基线，叠加文件中 `aircraft`、`wind`、
                `ground`、`uncertainty`、`engine`、`actuator`、`aero`、`sensor`
                小节后的配置。

        Raises:
            ValueError: 文件仍包含已移除的 `trim_model_type` 键时抛出。

        Note:
            `aero.data_path` 的相对路径按 TOML 文件所在目录解析；修改
            `aircraft` 小节会重新派生发动机、不确定性与气动表配置。
        """
        config_path = Path(path)
        raw = cast(
            dict[str, object],
            rtoml.loads(config_path.read_text(encoding="utf-8")),
        )
        gtm_raw = _as_mapping(raw.get("gtm", {}))
        if "trim_model_type" in gtm_raw:
            raise ValueError(
                "trim_model_type has been removed from fr_gtm; remove it from TOML."
            )
        mode_value = gtm_raw.get("mode", Mode.NORMAL)
        if not isinstance(mode_value, (Mode, str)):
            raise TypeError("gtm.mode must be a mode string")
        mode = _parse_mode(mode_value)
        is_fullscale = bool(gtm_raw.get("is_fullscale", True))

        aero_raw = dict(_as_mapping(raw.get("aero", {})))
        data_path = aero_raw.pop("data_path", None)
        if data_path is not None:
            if not isinstance(data_path, str):
                raise TypeError("aero.data_path must be a string")
            data_path = Path(data_path)
            if not data_path.is_absolute():
                data_path = config_path.parent / data_path

        cfg = cls.default(
            is_fullscale=is_fullscale,
            mode=mode,
            data_path=data_path,
        )

        if "aircraft" in raw:
            cfg.aircraft = _merge_model(cfg.aircraft, _as_mapping(raw["aircraft"]))
            cfg.uncertainty = _build_uncertainty_config(cfg.aircraft)
            cfg.engine = _build_engine_config(cfg.is_fullscale, cfg.aircraft)
            cfg.aero = _build_aero_config(
                cfg.aircraft,
                data_path=cfg.aero.data_path,
            )

        if "wind" in raw:
            cfg.wind = _merge_model(cfg.wind, _as_mapping(raw["wind"]))
        if "ground" in raw:
            cfg.ground = _merge_model(cfg.ground, _as_mapping(raw["ground"]))
        if "uncertainty" in raw:
            cfg.uncertainty = _merge_model(
                cfg.uncertainty, _as_mapping(raw["uncertainty"])
            )
        if "engine" in raw:
            cfg.engine = _merge_model(cfg.engine, _as_mapping(raw["engine"]))
        if "actuator" in raw:
            cfg.actuator = _merge_model(cfg.actuator, _as_mapping(raw["actuator"]))
        if aero_raw:
            cfg.aero = _merge_model(cfg.aero, aero_raw)
        if "sensor" in raw:
            cfg.sensor = _merge_model(cfg.sensor, _as_mapping(raw["sensor"]))

        return cfg


def _scale_factor(is_fullscale: bool) -> float:
    return 1.0 / 0.055 if is_fullscale else 1.0


def _build_aircraft_config(is_fullscale: bool) -> AircraftConfig:
    scale = _scale_factor(is_fullscale)

    if is_fullscale:
        w0 = Force.lbf_to_n(185_000.0)
        inertia0 = Inertia.slug_ft2_to_kg_m2(
            np.array([1_770_000.0, 5_680_000.0, 7_270_000.0, 160_000.0, 0.0, 0.0])
        )
    else:
        w0 = Force.lbf_to_n(57.75)
        inertia0 = Inertia.slug_ft2_to_kg_m2(
            np.array([1.221, 4.655, 5.587, 0.274, 0.0, 0.006])
        )

    S = 5.9018 * scale**2 * Length.ft_to_m(1.0) ** 2
    Cbar = Length.ft_to_m(0.9153 * scale)
    b = Length.ft_to_m(6.8488 * scale)
    le_mac = Length.ft_to_m(4.5462 * scale)

    cg_pos0 = np.array(
        [
            -(le_mac + Cbar * 0.2199),
            Length.ft_to_m(-0.1416 / 12.0 * scale),
            Length.ft_to_m(-0.9761 * scale),
        ]
    )
    ref_cg = np.array([-(le_mac + Cbar * 0.25), 0.0, Length.ft_to_m(-0.9401 * scale)])
    engl_pos = np.array(
        [
            Length.ft_to_m(-51.903 / 12.0 * scale),
            Length.ft_to_m(-14.20 / 12.0 * scale),
            Length.ft_to_m(-7.71 / 12.0 * scale),
        ]
    )
    engr_pos = np.array(
        [
            Length.ft_to_m(-51.903 / 12.0 * scale),
            Length.ft_to_m(14.20 / 12.0 * scale),
            Length.ft_to_m(-7.71 / 12.0 * scale),
        ]
    )

    return AircraftConfig(
        w0=w0,
        inertia0=inertia0.tolist(),
        S=S,
        Cbar=Cbar,
        b=b,
        le_mac=le_mac,
        cg_pos0=cg_pos0.tolist(),
        ref_cg=ref_cg.tolist(),
        engl_pos=engl_pos.tolist(),
        engr_pos=engr_pos.tolist(),
        engl_ang=[0.0, 1.98, 2.22],
        engr_ang=[0.0, 1.95, -1.18],
    )


def _build_uncertainty_config(aircraft: AircraftConfig) -> UncertaintyConfig:
    inertia = aircraft.inertia0
    return UncertaintyConfig(
        w0=aircraft.w0,
        Ixx0=inertia[0],
        Iyy0=inertia[1],
        Izz0=inertia[2],
        Ixz0=inertia[3],
        Iyz0=inertia[4],
        Ixy0=inertia[5],
    )


def _build_ground_config(
    is_fullscale: bool,
    aircraft: AircraftConfig,
    *,
    enabled: bool = False,
) -> GroundContactConfig:
    """由公开的 NASA GTM T2 参数推导 FRD/SI 起落架数据。"""

    length_ratio = _scale_factor(is_fullscale)
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


def _build_engine_config(
    is_fullscale: bool,
    aircraft: AircraftConfig,
) -> EngineConfig:
    scale = _scale_factor(is_fullscale)
    if is_fullscale:
        T_gain = Force.lbf_to_n(40000.0 / 14.0)
        h_gain = scale**5 * Inertia.slug_ft2_to_kg_m2(1.0)
        RPMref_gain = 20000.0 / 123000.0
        ram_drag_coeff = 0.004
    else:
        T_gain = Force.lbf_to_n(1.0)
        h_gain = Inertia.slug_ft2_to_kg_m2(1.0)
        RPMref_gain = 1.0
        ram_drag_coeff = 0.01

    # 滞后结构恢复为原始二阶后，扩比时间尺度系数取 c = 1：两侧使用同一份
    # 滞后动态，不再按 sqrt(scale) 缩放，因此这里不区分全尺寸与缩比。
    return EngineConfig(
        T_gain=T_gain,
        h_gain=h_gain,
        RPMref_gain=RPMref_gain,
        ram_drag_coeff=ram_drag_coeff,
        S=aircraft.S,
        shaft_diameter=0.75 / 12.0,
        shaft_length=6.25 / 12.0,
        density=15.2,
        thrust_coefficients=[0.000014847593013, 0.0, 0.0, 0.412662356277121],
        throttle_coefficients=[
            0.000033035538475,
            -0.008354385314499,
            1.191126728328792,
            31.518959847101669,
        ],
        lag=EngineLagConfig(),
        engl_pos=aircraft.engl_pos,
        engr_pos=aircraft.engr_pos,
        engl_ang=aircraft.engl_ang,
        engr_ang=aircraft.engr_ang,
        cg_pos=aircraft.cg_pos0,
    )


def _build_actuator_config(is_fullscale: bool) -> ActuatorConfig:
    scale = _scale_factor(is_fullscale)
    if is_fullscale:
        return ActuatorConfig(
            gain=[
                5.0 * 2.0 * pi / sqrt(scale),
                5.0 * 2.0 * pi / sqrt(scale),
                5.0 * 2.0 * pi / sqrt(scale),
            ],
            rate_limit=[
                300.0 / sqrt(scale),
                300.0 / sqrt(scale),
                300.0 / sqrt(scale),
            ],
        )
    return ActuatorConfig(
        gain=[5.0 * 2.0 * pi, 5.0 * 2.0 * pi, 5.0 * 2.0 * pi],
        rate_limit=[300.0, 300.0, 300.0],
    )


def _build_sensor_config(is_fullscale: bool) -> SensorConfig:
    if is_fullscale:
        return SensorConfig(
            gamma=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            alpha=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            beta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            V=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            altitude=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=3.0),
            phi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            theta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            psi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            p=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            q=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            r=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            nz=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.015),
            qbar=SensorSingleConfig(
                bias=0.0,
                scale=1.0,
                noise_std=Pressure.psi_to_pa(0.00028),
            ),
            elevator=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
            aileron=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
            rudder=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
        )

    return SensorConfig(
        gamma=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.033),
        alpha=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.031),
        beta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.033),
        V=SensorSingleConfig(
            bias=Speed.kts_to_mps(0.07),
            scale=1.0,
            noise_std=Speed.kts_to_mps(0.06),
        ),
        altitude=SensorSingleConfig(
            bias=0.0,
            scale=1.0,
            noise_std=Length.ft_to_m(2.36),
        ),
        phi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.02),
        theta=SensorSingleConfig(bias=-0.09, scale=1.0, noise_std=0.02),
        psi=SensorSingleConfig(bias=-0.04, scale=1.0, noise_std=0.02),
        p=SensorSingleConfig(bias=-0.0057, scale=1.0, noise_std=0.5350),
        q=SensorSingleConfig(bias=-0.0115, scale=1.0, noise_std=0.4737),
        r=SensorSingleConfig(bias=-0.0286, scale=1.0, noise_std=0.3397),
        nz=SensorSingleConfig(bias=0.00086, scale=1.0, noise_std=0.0082),
        qbar=SensorSingleConfig(
            bias=0.0,
            scale=1.0,
            noise_std=Pressure.psi_to_pa(0.00028),
        ),
        elevator=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
        aileron=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
        rudder=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
    )


def _build_aero_config(
    aircraft: AircraftConfig,
    *,
    data_path: str | Path | None = None,
) -> AeroConfig:
    tables, resolved_data_path = _load_aero_tables(data_path)
    arm = np.asarray(aircraft.ref_cg) - np.asarray(aircraft.cg_pos0)

    return AeroConfig(
        data_path=str(resolved_data_path) if data_path is not None else None,
        tables=tables,
        control_surfaces=ControlSurfacesConfig(),
        dynamic_derivatives=DynamicDerivativesConfig(
            b=aircraft.b,
            Cbar=aircraft.Cbar,
        ),
        force_moment=AeroForceMomentConfig(
            Sref=aircraft.S,
            b=aircraft.b,
            Cbar=aircraft.Cbar,
            arm_x=float(arm[0]),
            arm_y=float(arm[1]),
            arm_z=float(arm[2]),
        ),
    )


def _load_aero_tables(data_path: str | Path | None) -> tuple[AeroTableResources, Path]:
    if data_path is None:
        resource = resources.files("fr_gtm") / "data" / "aero_tables.npz"
        with resources.as_file(resource) as package_data_path:
            return load_aero_tables(package_data_path), package_data_path

    resolved = Path(data_path)
    if resolved.is_dir():
        resolved /= "aero_tables.npz"
    return load_aero_tables(resolved), resolved


def _parse_mode(mode: Mode | str) -> Mode:
    if isinstance(mode, Mode):
        return mode
    return Mode(mode)


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError("configuration sections must be TOML tables")
    return cast(dict[str, object], value)


def _merge_model[T: BaseModel](model: T, updates: Mapping[str, object]) -> T:
    data = cast(dict[str, object], model.model_dump())
    _deep_update(data, updates)
    if isinstance(model, AeroConfig):
        data["tables"] = model.tables
    return type(model).model_validate(data)


def _deep_update(base: dict[str, object], updates: Mapping[str, object]) -> None:
    for key, value in updates.items():
        existing = base.get(key)
        if isinstance(value, Mapping) and isinstance(existing, dict):
            _deep_update(existing, value)
        else:
            base[key] = value


__all__ = [
    "ActuatorConfig",
    "AeroConfig",
    "AeroForceMomentConfig",
    "AircraftConfig",
    "ControlSurfacesConfig",
    "DynamicDerivativesConfig",
    "EngineConfig",
    "EngineLagConfig",
    "GTMConfig",
    "Mode",
    "SensorConfig",
    "SensorSingleConfig",
    "UncertaintyConfig",
    "WindConfig",
]
