"""GTM 发动机：独立油门滞后、双发载荷与组装系统。"""

from __future__ import annotations

from typing import Literal

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.libs.units import Force, Inertia, Length
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator


class JetCatThrottleLagConfig(BaseModel):
    """原始 NASA 缩比 GTM 的 JetCat P70 二阶油门滞后。

    状态空间为 `ẋ_e = (A x_e + B δ_c) / c`、`δ = C x_e`，其中 `δ_c` 是钳位后的
    油门指令百分比，`δ` 是有效油门百分比，`c` 是时间尺度系数。矩阵按行优先
    展开存放：`state_matrix` 是 `[a11, a12, a21, a22]`。

    Attributes:
        model: 配置模型标识 `second_order`。
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

    model_config = ConfigDict(frozen=True)
    """配置字段构造后不可重新赋值。"""

    model: Literal["second_order"] = "second_order"

    state_matrix: list[float] = Field(
        default_factory=lambda: [-1.336, -0.7314, 1.0, 0.0]
    )  # A, row-major.
    input_matrix: list[float] = Field(default_factory=lambda: [1.0, 0.0])  # B.
    output_matrix: list[float] = Field(default_factory=lambda: [-0.1474, 0.7314])  # C.
    time_scale: float = Field(default=1.0, gt=0.0)
    command_lower: float = 0.0  # Throttle command lower limit [%].
    command_upper: float = 100.0  # Throttle command upper limit [%].

    @model_validator(mode="after")
    def validate_lag(self) -> "JetCatThrottleLagConfig":
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


class FirstOrderThrottleLagConfig(BaseModel):
    """一阶有效油门滞后，默认时间常数为 1.5 s。

    指令先限幅，再通过 `τ δ̇ = δ_c − δ` 驱动有效油门状态。全尺寸和缩比
    使用相同时间常数；这是可选近似模型，不是 JetCat 的辨识结果。

    Attributes:
        model: 配置模型标识 `first_order`。
        time_constant_s: 正且有限的时间常数，单位 s。
        command_lower: 指令下限，单位 %。
        command_upper: 指令上限，单位 %。
    """

    model_config = ConfigDict(frozen=True)
    """配置字段构造后不可重新赋值。"""

    model: Literal["first_order"] = "first_order"

    time_constant_s: float = Field(default=1.5, gt=0.0, allow_inf_nan=False)
    command_lower: float = Field(default=0.0, allow_inf_nan=False)
    command_upper: float = Field(default=100.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_limits(self) -> "FirstOrderThrottleLagConfig":
        """校验指令下限低于上限。

        Returns:
            通过校验的配置。

        Raises:
            ValueError: 下限不低于上限。
        """
        if self.command_lower >= self.command_upper:
            raise ValueError("engine lag command lower limit must be below upper limit")
        return self

    def trim_state(self, throttle: float) -> tuple[float]:
        """返回指定有效油门的配平状态。

        Args:
            throttle: 有效油门百分比，应在配置的指令范围内。

        Returns:
            仅包含有效油门的一维状态元组。
        """
        return (throttle,)


class EngineLoadsConfig(BaseModel):
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
        engl_pos: 左发安装位置，单位 m，机体三元素。
        engr_pos: 右发安装位置，单位 m，机体三元素。
        engl_ang: 左发 `[roll, pitch, yaw]` 安装角，单位 deg。
        engr_ang: 右发 `[roll, pitch, yaw]` 安装角，单位 deg。
        cg_pos: 重心相对机头位置，单位 m，机体三元素。
    """

    model_config = ConfigDict(frozen=True)
    """配置字段构造后不可重新赋值。"""

    T_gain: float = Field(default=Force.lbf_to_n(40000.0 / 14.0))
    h_gain: float = Field(default=2693940.077776904)
    RPMref_gain: float = Field(default=20000.0 / 123000.0)
    ram_drag_coeff: float = Field(default=0.004)
    shaft_diameter: float = 0.75 / 12.0  # Shaft diameter (ft, per source model).
    shaft_length: float = 6.25 / 12.0  # Shaft length (ft, per source model).
    density: float = 15.2  # Shaft material density (slug/ft^3).
    thrust_coefficients: list[float] = Field(
        default_factory=lambda: [0.000014847593013, 0.0, 0.0, 0.412662356277121]
    )  # Polynomial coefficients for thrust from reference RPM.
    throttle_coefficients: list[float] = Field(
        default_factory=lambda: [
            0.000033035538475,
            -0.008354385314499,
            1.191126728328792,
            31.518959847101669,
        ]
    )  # Polynomial coefficients for reference RPM from throttle.
    # 两套标定模型的固定 SI 数据；全尺寸值由原 1/0.055 几何扩比换算，
    # h_gain 同时按原倍率五次方换算。这里不引入可变 scale，也不依赖顶层 config。
    engl_pos: tuple[float, ...] = (
        -23.969749090909094,
        -6.557818181818182,
        -3.560618181818182,
    )
    engr_pos: tuple[float, ...] = (
        -23.969749090909094,
        6.557818181818182,
        -3.560618181818182,
    )
    engl_ang: tuple[float, ...] = (0.0, 1.98, 2.22)
    engr_ang: tuple[float, ...] = (0.0, 1.95, -1.18)

    S: float
    cg_pos: tuple[float, ...]

    @classmethod
    def fullscale(cls, *, S: float, cg_pos: tuple[float, ...]) -> EngineLoadsConfig:
        """返回全尺寸载荷配置，面积与重心由调用方提供。

        Args:
            S: 参考机翼面积，单位 m²。
            cg_pos: 重心位置，单位 m，机体三元素。

        Returns:
            全尺寸发动机载荷配置。
        """
        return cls(
            S=S,
            cg_pos=cg_pos,
        )

    @classmethod
    def subscale(cls, *, S: float, cg_pos: tuple[float, ...]) -> EngineLoadsConfig:
        """返回缩比载荷配置，面积与重心由调用方提供。

        Args:
            S: 参考机翼面积，单位 m²。
            cg_pos: 重心位置，单位 m，机体三元素。

        Returns:
            缩比发动机载荷配置。
        """
        return cls(
            T_gain=Force.lbf_to_n(1.0),
            h_gain=Inertia.slug_ft2_to_kg_m2(1.0),
            RPMref_gain=1.0,
            ram_drag_coeff=0.01,
            engl_pos=(
                Length.ft_to_m(-51.903 / 12.0),
                Length.ft_to_m(-14.20 / 12.0),
                Length.ft_to_m(-7.71 / 12.0),
            ),
            engr_pos=(
                Length.ft_to_m(-51.903 / 12.0),
                Length.ft_to_m(14.20 / 12.0),
                Length.ft_to_m(-7.71 / 12.0),
            ),
            S=S,
            cg_pos=cg_pos,
        )


class EngineConfig(BaseModel):
    """发动机组装配置：滞后模型与唯一的载荷参数快照。

    Attributes:
        loads: 双发载荷配置，不再从顶层飞机配置隐式读取几何。
        lag: 默认 τ=1.5 s 一阶滞后，可显式选择原始 JetCat 二阶模型。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    """组装配置不可重新赋值且拒绝未知字段；载荷设置必须位于 loads。"""

    loads: EngineLoadsConfig
    lag: FirstOrderThrottleLagConfig | JetCatThrottleLagConfig = Field(
        default_factory=FirstOrderThrottleLagConfig, discriminator="model"
    )


@fr.block(config=EngineLoadsConfig)
def EngineLoads(cfg: EngineLoadsConfig) -> None:
    """无状态双发载荷，供连续反馈图将滞后状态与载荷求值分开。

    输入为实际有效油门（百分点）、机体系角速率（rad/s）、当地压力比及
    动压（Pa）。它们决定当前推力、安装力矩和转子陀螺力矩；油门命令不
    直接参与当前载荷。Engine 系统直接复用此 Block，不新增发动机动态、
    不复制多项式或在下游重建推力。状态及命令限幅由调用方的滞后块拥有。

    Args:
        cfg: 唯一的发动机载荷参数，含安装几何及参考面积。
    """
    effective_throttle = fr.input(
        "effective_throttle",
        (1,),
        unit="%",
        description="Effective throttle after lag; not output-clipped",
    )
    omega_body = fr.input(
        "omega_body",
        (3,),
        unit="rad/s",
        description="Body angular velocity [p, q, r] in FRD axes",
    )
    pressure_ratio = fr.input(
        "pressure_ratio",
        (1,),
        description="Local static pressure divided by sea-level reference pressure",
    )
    qbar = fr.input(
        "qbar", (1,), unit="Pa", description="Dynamic pressure used for engine ram drag"
    )

    def _engine_direction(angles_deg: tuple[float, ...]) -> NDArray[np.float64]:
        angles = np.radians(angles_deg)
        return np.array(
            [
                np.cos(angles[1]) * np.cos(angles[2]),
                np.sin(angles[2]),
                -np.sin(angles[1]) * np.cos(angles[2]),
            ]
        )

    left_direction = fr.math.vector(_engine_direction(cfg.engl_ang))
    right_direction = fr.math.vector(_engine_direction(cfg.engr_ang))
    left_arm = fr.math.vector(cfg.engl_pos) - fr.math.vector(cfg.cg_pos)
    right_arm = fr.math.vector(cfg.engr_pos) - fr.math.vector(cfg.cg_pos)

    # Coefficients are highest-order first, matching fr.math.polyval and gtm_ad.
    rpm_reference = fr.math.polyval(cfg.throttle_coefficients, effective_throttle[0])
    ram_drag = cfg.ram_drag_coeff * qbar[0] * cfg.S
    thrust = (
        fr.math.polyval(cfg.thrust_coefficients, rpm_reference)
        * cfg.T_gain
        * pressure_ratio[0]
        - ram_drag
    )
    thrust = fr.math.max(thrust, 0.0)

    shaft_momentum = (
        np.pi
        * cfg.shaft_diameter**4
        / 32.0
        * cfg.shaft_length
        * cfg.density
        * (2.0 * np.pi / 60.0)
    )
    momentum = (
        cfg.RPMref_gain
        * rpm_reference
        * (123_000.0 / 100.0)
        * shaft_momentum
        * cfg.h_gain
    )

    left_thrust = thrust * left_direction
    right_thrust = thrust * right_direction
    force = left_thrust + right_thrust

    angular_momentum = momentum * (left_direction + right_direction)
    # Moments combine body-rate/shaft-momentum coupling with r x F from each
    # engine's position relative to the center of mass.
    moment = (
        fr.math.cross(omega_body, angular_momentum)
        + fr.math.cross(left_arm, left_thrust)
        + fr.math.cross(right_arm, right_thrust)
    )

    fr.output(
        "force",
        force,
        dim=(3,),
        unit="N",
        description="Combined left and right engine force in body FRD axes",
    )
    fr.output(
        "moment",
        moment,
        dim=(3,),
        unit="N*m",
        description=(
            "Engine moment about CG, including installation arms and rotor "
            "gyroscopic coupling"
        ),
    )


@fr.block(config=FirstOrderThrottleLagConfig)
def FirstOrderThrottleLag(cfg: FirstOrderThrottleLagConfig) -> None:
    """指令限幅后驱动一阶有效油门状态，不计算载荷。

    Args:
        cfg: 一阶时间常数与指令范围。
    """
    throttle_command = fr.input(
        "throttle_command",
        (1,),
        unit="%",
        description="Throttle command before configured input clipping",
    )
    lag_state = fr.state(
        "lag_state",
        (1,),
        default=[0.0],
        description="First-order effective throttle state in percent",
    )

    command = fr.math.clip(throttle_command, cfg.command_lower, cfg.command_upper)

    fr.dynamics({lag_state: (command - lag_state) / cfg.time_constant_s})
    fr.output(
        "effective_throttle",
        lag_state,
        dim=(1,),
        unit="%",
        description="Effective throttle after first-order lag; not output-clipped",
    )


@fr.block(config=JetCatThrottleLagConfig)
def JetCatThrottleLag(cfg: JetCatThrottleLagConfig) -> None:
    """保留原 JetCat P70 二阶矩阵、时间尺度与无输出限幅语义。

    Args:
        cfg: 二阶状态空间参数与指令范围。
    """
    throttle_command = fr.input(
        "throttle_command",
        (1,),
        unit="%",
        description="Throttle command before configured input clipping",
    )
    lag_state = fr.state(
        "lag_state",
        (2,),
        default=[0.0, 0.0],
        description=(
            "JetCat fitted internal coordinates; effective throttle is C @ lag_state"
        ),
    )

    command = fr.math.clip(throttle_command[0], cfg.command_lower, cfg.command_upper)
    A = fr.math.matrix(np.asarray(cfg.state_matrix).reshape(2, 2))
    B = fr.math.vector(cfg.input_matrix)
    C = fr.math.matrix([cfg.output_matrix])

    lag_rate = (A @ lag_state + B * command) / cfg.time_scale
    effective_throttle = C @ lag_state

    fr.dynamics({lag_state: lag_rate})
    fr.output(
        "effective_throttle",
        effective_throttle,
        dim=(1,),
        unit="%",
        description="JetCat effective throttle C @ lag_state; not output-clipped",
    )


@fr.system(config=EngineConfig)
def Engine(cfg: EngineConfig) -> None:
    """选择油门滞后并连接无状态载荷，不在组装层重复物理公式。

    Args:
        cfg: 滞后配置与双发载荷配置。

    Note:
        状态路径为 Root.engine.lag::lag_state；初态使用 cfg.lag.trim_state(throttle)。
        默认一阶含 1 个状态，JetCat 二阶含 2 个状态；有效油门不作输出限幅。
    """
    throttle_command = fr.input(
        "throttle_command",
        (1,),
        unit="%",
        description="Throttle command before configured input clipping",
    )
    omega_body = fr.input(
        "omega_body",
        (3,),
        unit="rad/s",
        description="Body angular velocity [p, q, r] in FRD axes",
    )
    pressure_ratio = fr.input(
        "pressure_ratio",
        (1,),
        description="Local static pressure divided by sea-level reference pressure",
    )
    qbar = fr.input(
        "qbar", (1,), unit="Pa", description="Dynamic pressure used for engine ram drag"
    )

    if isinstance(cfg.lag, FirstOrderThrottleLagConfig):
        lag = FirstOrderThrottleLag("lag", cfg.lag)
    else:
        lag = JetCatThrottleLag("lag", cfg.lag)

    loads = EngineLoads("loads", cfg.loads)

    throttle_command >> lag.throttle_command
    lag.effective_throttle >> loads.effective_throttle
    omega_body >> loads.omega_body
    pressure_ratio >> loads.pressure_ratio
    qbar >> loads.qbar

    fr.export(
        effective_throttle=lag.effective_throttle,
        force=loads.force,
        moment=loads.moment,
    )


__all__ = [
    "EngineConfig",
    "EngineLoadsConfig",
    "FirstOrderThrottleLagConfig",
    "JetCatThrottleLagConfig",
    "Engine",
    "EngineLoads",
    "FirstOrderThrottleLag",
    "JetCatThrottleLag",
]
