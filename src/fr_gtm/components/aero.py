"""GTM 大气、空速与气动载荷 Block。"""

from __future__ import annotations

from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Literal, NamedTuple

import fly_ruler_core as fr
import numpy as np
from fly_ruler_core import Signal
from fly_ruler_std.libs.atmosphere import atmos76
from fly_ruler_std.libs.coordinates import body_vel_to_aero_regularized
from pydantic import BaseModel, ConfigDict, Field

from ..resources import AeroTableResources, load_aero_tables


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


class AeroConfig(BaseModel):
    """GTM 气动表、构型与载荷几何的单一配置。

    Attributes:
        tables: 校验过的气动资源，不参与配置序列化。
        data_path: 气动资源文件路径；由 load 解析。
        mode: 襟翼与起落架气动构型。
        interpolation_method: 表内插值，默认 linear，可选 cubic。
        extrapolation_method: cubic 表外延拓，默认 polynomial，可选 linear。
        S: 参考机翼面积，单位 m²。
        b: 翼展，单位 m，同时用于动导归一化和力矩换算。
        Cbar: 平均气动弦长，单位 m，同时用于动导归一化和力矩换算。
        cg_pos: 重心位置，单位 m，机体三元素。
        ref_cg: 气动参考点位置，单位 m，机体三元素。
        control_arm_ratios: 四个升降舵分段的无量纲展向力臂 y/b，从左外侧到右外侧。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    """允许 NumPy 资源，并拒绝已移除的嵌套配置字段。"""

    tables: AeroTableResources = Field(exclude=True)
    data_path: str | None = None
    mode: Mode = Mode.NORMAL
    interpolation_method: Literal["linear", "cubic"] = "linear"
    extrapolation_method: Literal["polynomial", "linear"] = "polynomial"
    S: float
    b: float
    Cbar: float
    cg_pos: tuple[float, float, float]
    ref_cg: tuple[float, float, float]
    control_arm_ratios: tuple[float, float, float, float] = (-0.07, -0.03, 0.03, 0.07)

    @classmethod
    def load(
        cls,
        *,
        S: float,
        b: float,
        Cbar: float,
        cg_pos: tuple[float, float, float],
        ref_cg: tuple[float, float, float],
        mode: Mode | str = Mode.NORMAL,
        data_path: str | Path | None = None,
    ) -> AeroConfig:
        """加载气动表并接收外部显式提供的几何，不依赖整机或飞机配置。

        Args:
            S: 参考机翼面积，单位 m²。
            b: 翼展，单位 m。
            Cbar: 平均气动弦长，单位 m。
            cg_pos: 重心位置，单位 m，三个机体系分量。
            ref_cg: 气动参考点位置，单位 m，三个机体系分量。
            mode: 襟翼与起落架气动构型。
            data_path: NPZ 文件或其所在目录；省略时使用包内资源。

        Returns:
            表格与几何一致的气动配置。
        """
        if data_path is None:
            resource = resources.files("fr_gtm") / "data" / "aero_tables.npz"
            with resources.as_file(resource) as path:
                tables = load_aero_tables(path)
                resolved_path = str(path)
        else:
            path = Path(data_path)
            if path.is_dir():
                path /= "aero_tables.npz"
            tables = load_aero_tables(path)
            resolved_path = str(path)
        return cls(
            tables=tables,
            data_path=resolved_path,
            S=S,
            b=b,
            Cbar=Cbar,
            cg_pos=cg_pos,
            ref_cg=ref_cg,
            mode=Mode(mode),
        )


class AeroResult(NamedTuple):
    """气动系数与 FRD 载荷结果。

    Attributes:
        coefficients: 六元气动系数，顺序为 `[CX, CY, CZ, Cl, Cm, Cn]`。
        force: 机体系气动力，单位 N。
        moment: 机体系气动力矩，单位 N·m。
        reynolds: 雷诺数，无量纲。
    """

    coefficients: Signal
    force: Signal
    moment: Signal
    reynolds: Signal


def _reynolds(cfg: AeroConfig, altitude: Signal, mach: Signal) -> Signal:
    """构造雷诺数表格插值器并计算雷诺数。

    Args:
        cfg: 气动配置，提供雷诺数表格与插值方法。
        altitude: 几何高度，单位 m，向上为正。
        mach: 马赫数，无量纲。

    Returns:
        雷诺数，无量纲。
    """
    table_cfg = cfg.tables.reynolds
    reynolds_values = table_cfg.values
    # 源模型按 MATLAB/Fortran 列优先展开后，再按当前网格重排；不能直接使用
    # 原二维数组，否则高度/马赫数与表值的对应关系会变。高度轴保留 ft，调用时 m→ft。
    reynolds_values = reynolds_values.flatten(order="F").reshape(
        len(table_cfg.altitude_ft), len(table_cfg.mach)
    )
    table = fr.resource.interpolator(
        "reynolds",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[
            table_cfg.altitude_ft,
            table_cfg.mach,
        ],
        values=reynolds_values,
    )
    return table(altitude / 0.3048, mach)


def _basic_coefficients(
    cfg: AeroConfig,
    alpha: Signal,
    beta: Signal,
    reynolds: Signal,
    scale: Signal,
) -> Signal:
    """构造基本气动表格插值器并计算六元静态系数。

    Args:
        cfg: 气动配置，提供基本气动表格。
        alpha: 迎角，单位度。
        beta: 侧滑角，单位度。
        reynolds: 雷诺数，无量纲。
        scale: 缩放修正因子，无量纲。

    Returns:
        六元静态系数，顺序为 `[CX, CY, CZ, Cl, Cm, Cn]`。
    """
    basic = cfg.tables.basic_airframe
    # CX/CZ 随迎角、侧滑角、雷诺数变化；其余四个静态系数只查迎角/侧滑角。
    cx_cz = fr.resource.interpolator(
        "basic_cx_cz",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[
            basic.alpha_c2,
            basic.beta_c2,
            basic.reynolds_c2,
        ],
        values=basic.cx_cz,
        n_out=2,
    )(alpha, beta, reynolds)
    cy_cl_cm_cn = fr.resource.interpolator(
        "basic_cy_cl_cm_cn",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[basic.alpha_c4, basic.beta_c4],
        values=basic.cy_cl_cm_cn,
        n_out=4,
    )(alpha, beta)
    # 两张表的输出并非连续的六元通道，必须交织成 [CX, CY, CZ, Cl, Cm, Cn]；
    # scale 是各通道的 1+相对扰动，不是面积、长度或动压的量纲换算。
    return (
        fr.math.vector(
            [
                cx_cz[0],
                cy_cl_cm_cn[0],
                cx_cz[1],
                cy_cl_cm_cn[1],
                cy_cl_cm_cn[2],
                cy_cl_cm_cn[3],
            ]
        )
        * scale
    )


def _control_coefficients(
    cfg: AeroConfig,
    alpha: Signal,
    beta: Signal,
    surfaces: Signal,
    c_e: Signal,
    c_a: Signal,
    c_r: Signal,
) -> Signal:
    """
    构造控制面气动表格插值器并计算六元控制系数。

    Args:
        cfg: 气动配置，提供控制面气动表格。
        alpha: 迎角，单位度。
        beta: 侧滑角，单位度。
        surfaces: 三元控制面偏转量，顺序为 `[elevator, aileron, rudder]`，单位度。
        c_e: 升降舵 [CX,CZ,Cm] 三通道缩放因子，均为 1+相对扰动。
        c_a: 副翼六通道缩放因子，均为 1+相对扰动。
        c_r: 方向舵六通道缩放因子，均为 1+相对扰动。

    Returns:
        六元控制系数，顺序为 `[CX, CY, CZ, Cl, Cm, Cn]`。
    """
    control = cfg.tables.control_surfaces
    elevator_table = fr.resource.interpolator(
        "elevator",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[
            control.alpha_elevator,
            control.beta_elevator,
            control.stabilizer,
            control.elevator,
        ],
        values=control.elevator_data,
        n_out=3,
    )
    aileron_table = fr.resource.interpolator(
        "aileron",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[
            control.alpha_aileron,
            control.beta_aileron,
            control.aileron,
        ],
        values=control.aileron_data,
        n_out=6,
    )
    rudder_table = fr.resource.interpolator(
        "rudder",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[
            control.alpha_rudder,
            control.beta_rudder,
            control.rudder,
        ],
        values=control.rudder_data,
        n_out=6,
    )

    elevator, aileron, rudder = surfaces[0], surfaces[1], surfaces[2]

    # 零安定面偏角、零升降舵偏角时的三元基准系数 [CX, CZ, Cm]。
    # 后续从偏舵查表结果中扣除它，只把“偏舵增量”分配到四段，避免重复计入基准。
    stabilizer = elevator_table(alpha, beta, 0.0, 0.0)
    # 6×3 嵌入矩阵：把 [CX, CZ, Cm] 放入统一的 [CX, CY, CZ, Cl, Cm, Cn]，
    # 即 [CX, 0, CZ, 0, Cm, 0]。它只调整通道排列，不做坐标旋转或力矩平移。
    # c_e 的三个分量分别缩放 CX、CZ、Cm；基准项在这里计入一次，不乘分段矩阵。
    longitudinal = fr.math.matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    stabilizer_c6 = longitudinal @ (stabilizer * c_e)

    # 查表差值是整副升降舵相对于零偏角的 [ΔCX, ΔCZ, ΔCm]。
    # 沿用源模型的四段等权近似：左外、左内、右内、右外各分配 1/4，
    # 四段共享同一 elevator 偏角；这里没有四个可独立控制的升降舵输入。
    elevator_delta = 0.25 * (elevator_table(alpha, beta, 0.0, elevator) - stabilizer)

    def _elevator_matrix(cfg: AeroConfig) -> Signal:
        """累加四段升降舵的系数平移矩阵，作用于单段的六元系数增量。

        每段的 arm = y/b 为无量纲展向位置：FRD 的 y 向右为正，左侧为负。
        对纯展向偏置 r = [0, y, 0]，r × F = [y Fz, 0, -y Fx]；结合
        F = qbar*S*[CX, CY, CZ] 和滚转/偏航力矩的参考量 qbar*S*b，得到
        ΔCl = arm*ΔCZ、ΔCn = -arm*ΔCX。该矩阵不额外改变俯仰系数 Cm。
        因此单段矩阵 K(arm) 为（行、列均按 CX, CY, CZ, Cl, Cm, Cn）：

            [    1, 0,   0, 0, 0, 0]
            [    0, 1,   0, 0, 0, 0]
            [    0, 0,   1, 0, 0, 0]
            [    0, 0, arm, 1, 0, 0]
            [    0, 0,   0, 0, 1, 0]
            [-arm, 0,   0, 0, 0, 1]

        返回的是 K_total = Σ K(arm_i)，不是上面的单段矩阵；其对角线为 4，
        与外部 elevator_delta 的 1/4 配对，恢复整副升降舵的系数增量。
        默认四个 arm 为 (-0.07, -0.03, 0.03, 0.07)，总和为零；相同偏角、
        等权分配时，由展向偏置额外产生的滚转和偏航项相互抵消。若比值不对称，
        则保留这些耦合项。全尺寸下 y 与 b 同比放大，arm 不再另乘尺度因子。

        Args:
            cfg: 气动配置，提供从左外到右外的四个展向位置比值 y/b。

        Returns:
            四个单段矩阵之和，形状为 6×6。
        """
        matrix = np.zeros((6, 6))
        for arm in cfg.control_arm_ratios:
            contribution = np.eye(6)
            contribution[3, 2] = arm
            contribution[5, 0] = -arm
            matrix += contribution
        return fr.math.matrix(matrix)

    # 从右到左：三通道不确定度缩放 → 嵌入六通道 → 各段展向力矩平移并求和。
    # 这里只计入分段展向偏置；气动参考点到整机重心的平移在载荷合成阶段统一处理。
    elevator_c6 = _elevator_matrix(cfg) @ longitudinal @ (elevator_delta * c_e)

    # 副翼表描述单侧响应：左侧通过 (beta, delta_a)→(-beta, -delta_a) 镜像，
    # 再用 diag(1,-1,1,-1,1,-1) 翻转侧向力、滚转/偏航力矩的符号。
    # 纵向 CX/CZ/Cm 在镜像下为偶分量；左右系数相加后才施加六通道 c_a 扰动。
    left_aileron = aileron_table(alpha, -beta, -aileron) * fr.math.vector(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )
    right_aileron = aileron_table(alpha, beta, aileron)

    # 方向舵两侧沿用原表的镜像约定，并在相同 alpha/beta 分支上扣除零舵基准。
    # delta_r=0 选择正侧分支，使自动微分保留该侧舵效；不使用 sign(delta_r)
    # 乘查表结果，因为 sign(0)=0 会错误消去零舵位置的局部导数。
    rudder_delta = fr.math.where(
        rudder >= 0.0,
        -(rudder_table(alpha, -beta, -rudder) - rudder_table(alpha, -beta, 0.0)),
        rudder_table(alpha, beta, rudder) - rudder_table(alpha, beta, 0.0),
    )

    def _configuration_increment(mode: Mode) -> Signal:
        """构造襟翼与起落架气动构型增量。

        Args:
            mode: 襟翼与起落架气动构型。

        Returns:
            六元气动系数增量，顺序为 `[CX, CY, CZ, Cl, Cm, Cn]`。
        """
        # 源模型的构型增量是无量纲六元系数；×2 是原始数据约定，非几何扩比。
        # 此处的“起落架放下”只改变气动阻力等系数，不等于开启跑道接触求解。
        flap = np.array([-0.0282, 0.0, -0.2989, 0.0, 0.0728, 0.0]) * 2.0
        gear = np.array([-0.016, 0.0, 0.014, 0.0, -0.008, 0.0]) * 2.0
        if mode is Mode.HALF_FLAP_GEAR_UP:
            result = flap * 0.5
        elif mode is Mode.FULL_FLAP_GEAR_UP:
            result = flap
        elif mode is Mode.FULL_FLAP_GEAR_DOWN:
            result = flap + gear
        else:
            result = np.zeros(6)
        return fr.math.vector(result)

    return (
        elevator_c6
        + stabilizer_c6
        + (left_aileron + right_aileron) * c_a
        + rudder_delta * c_r
        + _configuration_increment(cfg.mode)
    )


def _hykal(
    alpha: Signal,
    beta: Signal,
    tas: Signal,
    pqr: Signal,
    span: float,
    chord: float,
) -> tuple[Signal, Signal]:
    """按源模型的 Kalviste 规则，把相对气流角速率分成流向轴稳转与剩余运动。

    机体系来流方向为 e_V = [cos(alpha)cos(beta), sin(beta),
    sin(alpha)cos(beta)]，是单位向量。先在有量纲的 rad/s 下寻找
    [p,q,r] = omega*e_V + [p_osc,q_osc,r_osc]，再分别无量纲化。
    omega 表示沿来流轴的稳转分量；osc 表示供各轴动导表使用的剩余分量，
    不是直接测得的振荡频率，也不要求与 e_V 正交。这不是点积正交投影。

    分解规则先试用 p 确定 omega 并令 p_osc=0；仅当剩余 r 不反号且幅值
    不增大时接受。随后独立试用 r 确定 omega 并令 r_osc=0，要求剩余 p
    同样不反号、不增大；若也有效则覆盖 p 候选。两者均不满足则 omega=0。

    Args:
        alpha: 迎角，deg；进入三角函数前转为 rad。
        beta: 侧滑角，deg；进入三角函数前转为 rad。
        tas: 调用方已限幅的真空速，m/s，供 b/(2V)、Cbar/(2V) 使用。
        pqr: 相对气流的机体系角速率 [p,q,r]，rad/s；已扣除突风角速率。
        span: 参考翼展 b，m。
        chord: 参考平均气动弦长 Cbar，m。

    Returns:
        (omega_hat, oscillatory_hat)：均无量纲。前者为 omega*b/(2V)，
        后者为 [p_osc*b/(2V), q_osc*Cbar/(2V), r_osc*b/(2V)]。
        第一项用于 dynamic_w 表，不是机体系垂向速度 w。
    """
    alpha_rad = fr.math.radians(alpha)
    beta_rad = fr.math.radians(beta)
    p, q, r = pqr[0], pqr[1], pqr[2]
    ca = fr.math.cos(alpha_rad)
    sa = fr.math.sin(alpha_rad)
    cb = fr.math.cos(beta_rad)
    sb = fr.math.sin(beta_rad)
    # 来流单位向量的 x/z 分量；y 分量为 sb，三者均无量纲。
    cacb = ca * cb
    sacb = sa * cb

    # 候选一：omega1=p/e_Vx，消去剩余滚转角速率。
    # NASA 原规则只排除精确零。Signal 的 ==/!= 是结构比较，故用 abs(x)>0
    # 构造数值非零判据；只替换真正为零的分母，避免未选中符号支路出现 0/0。
    case1 = fr.math.logic_and(fr.math.abs(cacb) > 0.0, fr.math.abs(p) > 0.0)
    safe_cacb = fr.math.if_else(fr.math.abs(cacb) > 0.0, cacb, 1.0)
    omega1 = p / safe_cacb
    r_osc1 = r - omega1 * sa * cb
    case1 = fr.math.logic_and(
        case1,
        fr.math.logic_and(fr.math.abs(r_osc1) <= fr.math.abs(r), r_osc1 * r >= 0.0),
    )

    # 候选二：omega2=r/e_Vz，消去剩余偏航角速率。
    # 原模型顺序测试而非 if/else：候选二有效时覆盖候选一。
    case2 = fr.math.logic_and(fr.math.abs(sacb) > 0.0, fr.math.abs(r) > 0.0)
    safe_sacb = fr.math.if_else(fr.math.abs(sacb) > 0.0, sacb, 1.0)
    omega2 = r / safe_sacb
    p_osc2 = p - omega2 * ca * cb
    case2 = fr.math.logic_and(
        case2,
        fr.math.logic_and(fr.math.abs(p_osc2) <= fr.math.abs(p), p_osc2 * p >= 0.0),
    )

    # 选定同一个 omega 后，从三个轴都扣除 omega*e_V。
    # q_osc=q-omega*sin(beta) 只在有效候选下修正；不能只扣 p 或 r。
    omega = fr.math.if_else(case2, omega2, fr.math.if_else(case1, omega1, 0.0))
    p_osc = fr.math.if_else(case2, p_osc2, fr.math.if_else(case1, 0.0, p))
    q_osc = fr.math.if_else(fr.math.logic_or(case1, case2), q - omega * sb, q)
    r_osc = fr.math.if_else(case2, 0.0, fr.math.if_else(case1, r_osc1, r))

    # D=diag(b,Cbar,b)/(2V)，单位 s；D @ [p_osc,q_osc,r_osc] 后无量纲。
    # 下面的向量逐元素乘法与这个对角矩阵乘法等价，不是坐标系旋转。
    span_scale = span / (2.0 * tas)
    chord_scale = chord / (2.0 * tas)
    return (
        omega * span_scale,
        fr.math.vector([p_osc, q_osc, r_osc])
        * fr.math.vector([span_scale, chord_scale, span_scale]),
    )


def _dynamic_coefficients(
    cfg: AeroConfig,
    alpha: Signal,
    beta: Signal,
    tas: Signal,
    pqr: Signal,
) -> Signal:
    """查剩余各轴角速率与流向轴稳转表，合成为六元动导增量。

    p/r 表各输出 [ΔCY,ΔCl,ΔCn]，q 表输出 [ΔCX,ΔCZ,ΔCm]，
    w 表输出完整 [ΔCX,ΔCY,ΔCZ,ΔCl,ΔCm,ΔCn]。这里的 w 表对应
    omega_hat（流向轴稳转），不是垂向平动速度或独立的偏航角速率。
    表值已经是系数增量，不是还需乘 p_hat/q_hat/r_hat 的线性导数。

    Args:
        cfg: 气动表、插值选项与参考长度 b/Cbar。
        alpha: 迎角，deg，直接作为表坐标。
        beta: 侧滑角，deg，用于稳转表。
        tas: 真空速，m/s；仅在动导归一化路径中限幅。
        pqr: 扣除突风后的相对气流机体系角速率，rad/s。

    Returns:
        无量纲六元增量 [ΔCX,ΔCY,ΔCZ,ΔCl,ΔCm,ΔCn]；此处不乘动压、面积或长度。
    """
    dynamic = cfg.tables.dynamic_derivatives
    p_table = fr.resource.interpolator(
        "dynamic_p",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[dynamic.alpha_p, dynamic.phat],
        values=dynamic.p,
        n_out=3,
    )
    q_table = fr.resource.interpolator(
        "dynamic_q",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[dynamic.alpha_q, dynamic.qhat],
        values=dynamic.q,
        n_out=3,
    )
    r_table = fr.resource.interpolator(
        "dynamic_r",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[dynamic.alpha_r, dynamic.rhat],
        values=dynamic.r,
        n_out=3,
    )
    w_table = fr.resource.interpolator(
        "dynamic_w",
        method=cfg.interpolation_method,
        extrapolation=cfg.extrapolation_method,
        grid_points=[
            dynamic.alpha_w,
            dynamic.what,
            dynamic.beta_w,
        ],
        values=dynamic.w,
        n_out=6,
    )

    # 保留源模型的 1～10000 m/s 归一化限幅，避免零速分母或极端速率坐标。
    # 不改变用于真实动压/Mach 的 tas；后面的 ±0.5 仅约束稳转查表坐标。
    safe_tas = fr.math.clip(tas, 1.0, 10000.0)
    omega, oscillatory = _hykal(alpha, beta, safe_tas, pqr, cfg.b, cfg.Cbar)
    omega = fr.math.clip(omega, -0.5, 0.5)
    # 三个剩余轴使用各自的无量纲速率查表，稳转项另用 (alpha,omega_hat,beta)。
    p_delta = p_table(alpha, oscillatory[0])
    q_delta = q_table(alpha, oscillatory[1])
    r_delta = r_table(alpha, oscillatory[2])
    w_delta = w_table(alpha, omega, beta)
    # 设 E_lon 把 [CX,CZ,Cm] 嵌入六通道，E_lat 把 [CY,Cl,Cn] 嵌入六通道，
    # 则 ΔC = E_lon @ q_delta + E_lat @ (p_delta+r_delta) + w_delta。
    # 下面显式按行合成，保留原加法顺序；稳转项加到全部六通道，纵/横向项互不串位。
    return fr.math.vector(
        [
            q_delta[0] + w_delta[0],
            p_delta[0] + r_delta[0] + w_delta[1],
            q_delta[1] + w_delta[2],
            p_delta[1] + r_delta[1] + w_delta[3],
            q_delta[2] + w_delta[4],
            p_delta[2] + r_delta[2] + w_delta[5],
        ]
    )


def aerodynamic_loads(
    cfg: AeroConfig,
    *,
    altitude: Signal,
    mach: Signal,
    qbar: Signal,
    alpha: Signal,
    beta: Signal,
    tas: Signal,
    pqr: Signal,
    surfaces: Signal,
    c_static_unc: Signal,
    c_e_unc: Signal,
    c_a_unc: Signal,
    c_r_unc: Signal,
) -> AeroResult:
    """把静态、操纵与动导系数合成为 FRD 载荷。

    Args:
        cfg: 气动配置，提供表格、构型、参考面积/长度与力矩参考点。
        altitude: 几何高度，单位 m。
        mach: 马赫数，无量纲。
        qbar: 动压，单位 Pa。
        alpha: 迎角，单位 deg。
        beta: 侧滑角，单位 deg。
        tas: 真空速，单位 m/s。
        pqr: 相对气流的机体系角速率，单位 rad/s。
        surfaces: 三个舵面位置 `[elevator, aileron, rudder]`，单位 deg；
            油门不参与气动。
        c_static_unc: 六个静态系数的相对不确定性增量。
        c_e_unc: 三个升降舵系数的相对不确定性增量。
        c_a_unc: 六个副翼系数的相对不确定性增量。
        c_r_unc: 六个方向舵系数的相对不确定性增量。

    Returns:
        AeroResult: 六元系数、机体系力（N）、机体系力矩（N·m）与雷诺数。

    Note:
        力系数占前三个通道，力矩系数使用参考长度 `[b, Cbar, b]`；最后的
        `arm x force` 把载荷从气动参考点平移到配置的重心。
    """
    # 相对不确定度以 1+δ 乘系数：静态/副翼/方向舵各六通道，升降舵为 CX/CZ/Cm 三通道。
    # 动导项保持独立，不再乘上述静态/操纵缩放，避免把不同物理来源重复修正。
    one6 = fr.math.ones(6)
    c_static = one6 + c_static_unc
    c_e = fr.math.ones(3) + c_e_unc
    c_a = one6 + c_a_unc
    c_r = one6 + c_r_unc
    reynolds = _reynolds(cfg, altitude, mach)
    coefficients = (
        _basic_coefficients(cfg, alpha, beta, reynolds, c_static)
        + _control_coefficients(cfg, alpha, beta, surfaces, c_e, c_a, c_r)
        + _dynamic_coefficients(cfg, alpha, beta, tas, pqr)
    )

    # 力系数乘 qbar*S 得 N；后三个力矩系数先得到 N，再分别乘 [b,Cbar,b] 得 N·m。
    # ref_cg/cg_pos 在同一机体系原点下，以 r=ref_cg-cg_pos 做 M_CG=M_ref+r×F。
    # 这次三维重心平移不同于升降舵段内 y/b 的系数平移，只在总载荷阶段执行一次。
    aero_all = coefficients * qbar * cfg.S
    force = aero_all[0:3]
    reference_lengths = fr.math.vector([cfg.b, cfg.Cbar, cfg.b])
    arm = fr.math.vector(cfg.ref_cg) - fr.math.vector(cfg.cg_pos)
    moment = aero_all[3:6] * reference_lengths + fr.math.cross(arm, force)
    return AeroResult(coefficients, force, moment, reynolds)


@fr.block(config=AeroConfig)
def Aero(cfg: AeroConfig) -> None:
    """计算相对气流空速与 GTM 气动力/力矩。

    Args:
        cfg: 气动表、构型、参考几何与力矩参考点配置。

    Note:
        高度使用正向上的 m；速度、角速率和风均在机体 FRD 中。气动查表
        坐标保留 deg，因此 alpha/beta 在本块边界处换算；动导分解前先扣除
        突风角速率。位置、姿态变换与 NED 航迹角由系统/刚体负责。
    """
    altitude = fr.input(
        "altitude",
        (1,),
        unit="m",
        description="Altitude above NED origin (positive up)",
    )
    surfaces = fr.input(
        "surfaces", (3,), unit="deg", description="Control surface positions"
    )
    velocity_body = fr.input(
        "velocity",
        (3,),
        unit="m/s",
        description="Body-FRD velocity [u, v, w] before wind correction",
    )
    omega_body = fr.input(
        "omega_body",
        (3,),
        unit="rad/s",
        description="Body angular velocity [p, q, r] in FRD axes",
    )
    wind_uvw = fr.input(
        "wind_uvw",
        (3,),
        unit="m/s",
        description="Wind velocity [u, v, w] resolved in body FRD axes",
    )
    wind_omega = fr.input(
        "wind_omega",
        (3,),
        unit="rad/s",
        description="Dryden angular gust [p, q, r] in body FRD axes",
    )

    # 这四组是无量纲运行时相对扰动，合计 21 维；通道顺序对应源模型
    # wind/mass 后的 static(6)、elevator(3)、aileron(6)、rudder(6)。
    c_static_unc = fr.param(
        "c_static_unc",
        (6,),
        default=[0.0] * 6,
        description="Relative static coefficient increments [CX, CY, CZ, Cl, Cm, Cn]",
    )
    c_e_unc = fr.param(
        "c_e_unc",
        (3,),
        default=[0.0] * 3,
        description="Relative elevator coefficient increments [CX, CZ, Cm]",
    )
    c_a_unc = fr.param(
        "c_a_unc",
        (6,),
        default=[0.0] * 6,
        description="Relative aileron coefficient increments [CX, CY, CZ, Cl, Cm, Cn]",
    )
    c_r_unc = fr.param(
        "c_r_unc",
        (6,),
        default=[0.0] * 6,
        description="Relative rudder coefficient increments [CX, CY, CZ, Cl, Cm, Cn]",
    )

    # 平动速度与风速均在 FRD 中，直接相减得到相对空气运动；角速率同样扣除角阵风。
    # 姿态/导航系变换由系统和刚体负责，本块不再接收位置或四元数。
    air_velocity = velocity_body - wind_uvw
    air_omega = omega_body - wind_omega
    # 正则化气流角在零速附近保持有限；只在查表边界将 rad 转 deg。
    tas, alpha_rad, beta_rad = body_vel_to_aero_regularized(
        air_velocity, speed_epsilon=1.0e-3
    )
    alpha = fr.math.degrees(alpha_rad)
    beta = fr.math.degrees(beta_rad)
    # 大气使用正向上的米制高度；qbar=ρV²/2，Mach=V/a，VEAS=VTAS*sqrt(ρ/ρ0)。
    atmosphere = atmos76(altitude)
    qbar = 0.5 * atmosphere.density * tas**2
    mach = tas / atmosphere.speed_of_sound
    equivalent_airspeed = tas * atmosphere.sqrt_density_ratio

    aero = aerodynamic_loads(
        cfg,
        altitude=altitude,
        mach=mach,
        qbar=qbar,
        alpha=alpha,
        beta=beta,
        tas=tas,
        pqr=air_omega,
        surfaces=surfaces,
        c_static_unc=c_static_unc,
        c_e_unc=c_e_unc,
        c_a_unc=c_a_unc,
        c_r_unc=c_r_unc,
    )

    fr.output(
        "force",
        aero.force,
        dim=(3,),
        unit="N",
        description="Aerodynamic force [X, Y, Z] in body FRD axes",
    )
    fr.output(
        "moment",
        aero.moment,
        dim=(3,),
        unit="N*m",
        description="Aerodynamic moment [L, M, N] about CG in body FRD axes",
    )

    fr.output(
        "VTAS",
        tas,
        dim=(1,),
        unit="m/s",
        description="Regularized wind-relative true airspeed",
    )
    fr.output(
        "alpha",
        alpha,
        dim=(1,),
        unit="deg",
        description="Wind-relative angle of attack used for GTM table lookup",
    )
    fr.output(
        "beta",
        beta,
        dim=(1,),
        unit="deg",
        description="Wind-relative sideslip angle used for GTM table lookup",
    )

    fr.output(
        "qbar",
        qbar,
        dim=(1,),
        unit="Pa",
        description="Dynamic pressure from atmospheric density and true airspeed",
    )
    fr.output(
        "mach",
        mach,
        dim=(1,),
        description="True airspeed divided by local speed of sound",
    )
    fr.output(
        "VEAS",
        equivalent_airspeed,
        dim=(1,),
        unit="m/s",
        description="Equivalent airspeed using sea-level reference density",
    )
    fr.output(
        "pressure_ratio",
        atmosphere.pressure_ratio,
        dim=(1,),
        description="Local static pressure divided by sea-level reference pressure",
    )

    # 系数和雷诺数属于诊断数据，不作为组件控制/接线输出；从 model.effects 按块路径读取。
    fr.effects(
        coefficients=aero.coefficients,
        reynolds=aero.reynolds,
    )


__all__ = ["AeroConfig", "AeroResult", "aerodynamic_loads", "Aero"]
