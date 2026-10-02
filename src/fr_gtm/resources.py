"""冻结 GTM 气动表的有类型读取与校验。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

_DATACLASS_CONFIG = ConfigDict(arbitrary_types_allowed=True)

FloatArray = NDArray[np.float64]


@dataclass(config=_DATACLASS_CONFIG, frozen=True)
class ReynoldsTables:
    """雷诺数换算网格和结果表。

    Attributes:
        altitude_ft: 英尺高度坐标轴。
        mach: 马赫数坐标轴。
        values: 高度/马赫数网格上的雷诺数，轴顺序为高度、马赫数。
    """

    altitude_ft: FloatArray
    mach: FloatArray
    values: FloatArray


@dataclass(config=_DATACLASS_CONFIG, frozen=True)
class BasicAirframeTables:
    """基线机体 C2/C4 系数表及其插值轴。

    Attributes:
        alpha_c2: C2 迎角坐标，单位 deg。
        beta_c2: C2 侧滑角坐标，单位 deg。
        reynolds_c2: C2 雷诺数坐标。
        cx_cz: `[CX, CZ]` 系数表，轴顺序为迎角、侧滑角、雷诺数、系数。
        alpha_c4: C4 迎角坐标，单位 deg。
        beta_c4: C4 侧滑角坐标，单位 deg。
        cy_cl_cm_cn: `[CY, Cl, Cm, Cn]` 系数表，轴顺序为迎角、侧滑角、系数。
    """

    alpha_c2: FloatArray
    beta_c2: FloatArray
    reynolds_c2: FloatArray
    cx_cz: FloatArray
    alpha_c4: FloatArray
    beta_c4: FloatArray
    cy_cl_cm_cn: FloatArray


@dataclass(config=_DATACLASS_CONFIG, frozen=True)
class ControlSurfaceTables:
    """升降舵、副翼和方向舵增量表及其插值轴。

    Attributes:
        alpha_elevator: 升降舵迎角坐标，单位 deg。
        beta_elevator: 升降舵侧滑角坐标，单位 deg。
        stabilizer: 平尾偏角坐标，单位 deg。
        elevator: 升降舵偏角坐标，单位 deg。
        elevator_data: 升降舵三系数增量表。
        alpha_aileron: 副翼迎角坐标，单位 deg。
        beta_aileron: 副翼侧滑角坐标，单位 deg。
        aileron: 副翼偏角坐标，单位 deg。
        aileron_data: 副翼六系数增量表。
        alpha_rudder: 方向舵迎角坐标，单位 deg。
        beta_rudder: 方向舵侧滑角坐标，单位 deg。
        rudder: 方向舵偏角坐标，单位 deg。
        rudder_data: 方向舵六系数增量表。
    """

    alpha_elevator: FloatArray
    beta_elevator: FloatArray
    stabilizer: FloatArray
    elevator: FloatArray
    elevator_data: FloatArray
    alpha_aileron: FloatArray
    beta_aileron: FloatArray
    aileron: FloatArray
    aileron_data: FloatArray
    alpha_rudder: FloatArray
    beta_rudder: FloatArray
    rudder: FloatArray
    rudder_data: FloatArray


@dataclass(config=_DATACLASS_CONFIG, frozen=True)
class DynamicDerivativeTables:
    """Kalviste p/q/r 和 w 轴动态导数表及其插值轴。

    Attributes:
        alpha_p: p 表迎角坐标，单位 deg。
        phat: 归一化滚转角速率坐标。
        p: p 轴三系数导数表。
        alpha_q: q 表迎角坐标，单位 deg。
        qhat: 归一化俯仰角速率坐标。
        q: q 轴三系数导数表。
        alpha_r: r 表迎角坐标，单位 deg。
        rhat: 归一化偏航角速率坐标。
        r: r 轴三系数导数表。
        alpha_w: w 表迎角坐标，单位 deg。
        what: 归一化 w 坐标。
        beta_w: w 表侧滑角坐标，单位 deg。
        w: 六系数耦合导数表。
    """

    alpha_p: FloatArray
    phat: FloatArray
    p: FloatArray
    alpha_q: FloatArray
    qhat: FloatArray
    q: FloatArray
    alpha_r: FloatArray
    rhat: FloatArray
    r: FloatArray
    alpha_w: FloatArray
    what: FloatArray
    beta_w: FloatArray
    w: FloatArray


@dataclass(config=_DATACLASS_CONFIG, frozen=True)
class AeroTableResources:
    """模型私有冻结气动资源。

    Attributes:
        reynolds: 高度/马赫数到雷诺数的二维表。
        basic_airframe: 基线机体 C2/C4 系数及各自坐标轴。
        control_surfaces: 三类舵面增量系数及各自坐标轴。
        dynamic_derivatives: 角速率导数系数及各自坐标轴。
    """

    reynolds: ReynoldsTables
    basic_airframe: BasicAirframeTables
    control_surfaces: ControlSurfaceTables
    dynamic_derivatives: DynamicDerivativeTables


def load_aero_tables(path: str | Path) -> AeroTableResources:
    """加载 GTM NPZ 气动资源并校验坐标轴、表形状和有限值。

    Args:
        path: `aero_tables.npz` 文件路径。

    Returns:
        校验过且按物理分组组织的气动表资源。

    Raises:
        FileNotFoundError: 文件不存在时由 NumPy 抛出。
        ValueError: 坐标轴非法、表格形状不符或数据包含非有限值时抛出。
        KeyError: NPZ 缺少必需数组时抛出。
    """
    data_path = Path(path)
    with np.load(data_path, allow_pickle=False) as archive:
        reynolds_altitude = _axis(archive, "reynolds__altitude_ft")
        reynolds_mach = _axis(archive, "reynolds__mach")
        c2_alpha = _axis(archive, "C2_bas_Re_dep__alpha")
        c2_beta = _axis(archive, "C2_bas_Re_dep__beta")
        c2_reynolds = _axis(archive, "C2_bas_Re_dep__reynolds")
        c4_alpha = _axis(archive, "C4_bas_Re_indep__alpha")
        c4_beta = _axis(archive, "C4_bas_Re_indep__beta")
        elevator_alpha = _axis(archive, "dC3_ele__alpha")
        elevator_beta = _axis(archive, "dC3_ele__beta")
        stabilizer = _axis(archive, "dC3_ele__stab")
        elevator = _axis(archive, "dC3_ele__elev")
        aileron_alpha = _axis(archive, "dC6_ail__alpha")
        aileron_beta = _axis(archive, "dC6_ail__beta")
        aileron = _axis(archive, "dC6_ail__ail")
        rudder_alpha = _axis(archive, "dC6_rud__alpha")
        rudder_beta = _axis(archive, "dC6_rud__beta")
        rudder = _axis(archive, "dC6_rud__rud")
        p_alpha = _axis(archive, "dC3_p__alpha")
        phat = _axis(archive, "dC3_p__phat")
        q_alpha = _axis(archive, "dC3_q__alpha")
        qhat = _axis(archive, "dC3_q__qhat")
        r_alpha = _axis(archive, "dC3_r__alpha")
        rhat = _axis(archive, "dC3_r__rhat")
        w_alpha = _axis(archive, "dC6_w__alpha")
        what = _axis(archive, "dC6_w__what")
        w_beta = _axis(archive, "dC6_w__beta")

        return AeroTableResources(
            reynolds=ReynoldsTables(
                altitude_ft=reynolds_altitude,
                mach=reynolds_mach,
                values=_values(
                    archive,
                    "reynolds__reynolds",
                    (len(reynolds_altitude), len(reynolds_mach)),
                ),
            ),
            basic_airframe=BasicAirframeTables(
                alpha_c2=c2_alpha,
                beta_c2=c2_beta,
                reynolds_c2=c2_reynolds,
                cx_cz=_values(
                    archive,
                    "C2_bas_Re_dep__data",
                    (len(c2_alpha), len(c2_beta), len(c2_reynolds), 2),
                ),
                alpha_c4=c4_alpha,
                beta_c4=c4_beta,
                cy_cl_cm_cn=_values(
                    archive,
                    "C4_bas_Re_indep__data",
                    (len(c4_alpha), len(c4_beta), 4),
                ),
            ),
            control_surfaces=ControlSurfaceTables(
                alpha_elevator=elevator_alpha,
                beta_elevator=elevator_beta,
                stabilizer=stabilizer,
                elevator=elevator,
                elevator_data=_values(
                    archive,
                    "dC3_ele__data",
                    (
                        len(elevator_alpha),
                        len(elevator_beta),
                        len(stabilizer),
                        len(elevator),
                        3,
                    ),
                ),
                alpha_aileron=aileron_alpha,
                beta_aileron=aileron_beta,
                aileron=aileron,
                aileron_data=_values(
                    archive,
                    "dC6_ail__data",
                    (len(aileron_alpha), len(aileron_beta), len(aileron), 6),
                ),
                alpha_rudder=rudder_alpha,
                beta_rudder=rudder_beta,
                rudder=rudder,
                rudder_data=_values(
                    archive,
                    "dC6_rud__data",
                    (len(rudder_alpha), len(rudder_beta), len(rudder), 6),
                ),
            ),
            dynamic_derivatives=DynamicDerivativeTables(
                alpha_p=p_alpha,
                phat=phat,
                p=_values(archive, "dC3_p__data", (len(p_alpha), len(phat), 3)),
                alpha_q=q_alpha,
                qhat=qhat,
                q=_values(archive, "dC3_q__data", (len(q_alpha), len(qhat), 3)),
                alpha_r=r_alpha,
                rhat=rhat,
                r=_values(archive, "dC3_r__data", (len(r_alpha), len(rhat), 3)),
                alpha_w=w_alpha,
                what=what,
                beta_w=w_beta,
                w=_values(
                    archive,
                    "dC6_w__data",
                    (len(w_alpha), len(what), len(w_beta), 6),
                ),
            ),
        )


def _axis(archive: np.lib.npyio.NpzFile, name: str) -> FloatArray:
    value = np.asarray(archive[name], dtype=np.float64)
    if value.ndim != 1 or value.size < 2:
        raise ValueError(
            f"aerodynamic table axis {name} must be a 1D array with 2+ values"
        )
    if not np.isfinite(value).all():
        raise ValueError(
            f"aerodynamic table axis {name} must contain only finite values"
        )
    if not np.all(np.diff(value) > 0.0):
        raise ValueError(f"aerodynamic table axis {name} must be strictly increasing")
    return value


def _values(
    archive: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    value = np.asarray(archive[name], dtype=np.float64)
    if value.shape != shape:
        raise ValueError(
            f"aerodynamic table {name} has shape {value.shape}; expected shape {shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"aerodynamic table {name} must contain only finite values")
    return value


__all__ = ["AeroTableResources", "load_aero_tables"]
