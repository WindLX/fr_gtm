"""GTM 质量与转动惯量不确定性 Block。"""

import fly_ruler_core as fr

from ..config import GTMConfig


@fr.block(config=GTMConfig)
def Mass(cfg: GTMConfig) -> None:
    """对重量与六个惯量分量施加相对不确定性。

    Args:
        cfg: GTM 配置，提供标称重量、`[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]` 标称惯量与
            重力加速度。

    Note:
        `inertia_unc` 的顺序为 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]`，与 gtm_ad
        一致；惯性积按源模型的负号约定进入 FRD 惯量张量。零扰动复现标称
        参考数据，逆惯量用闭式解计算以避免后端特有的矩阵求逆操作。
    """
    weight_unc = fr.param(
        "weight_unc", (1,), default=[0.0], description="Relative weight uncertainty"
    )
    inertia_unc = fr.param(
        "inertia_unc",
        (6,),
        default=[0.0] * 6,
        description="Relative inertia uncertainty [Ixx, Iyy, Izz, Ixz, Iyz, Ixy]",
    )

    nominal = cfg.aircraft.inertia0
    weight = cfg.aircraft.w0 * (1.0 + weight_unc[0])
    mass = weight / cfg.uncertainty.gravity_acc
    ixx = nominal[0] * (1.0 + inertia_unc[0])
    iyy = nominal[1] * (1.0 + inertia_unc[1])
    izz = nominal[2] * (1.0 + inertia_unc[2])
    ixz = nominal[3] * (1.0 + inertia_unc[3])
    iyz = nominal[4] * (1.0 + inertia_unc[4])
    ixy = nominal[5] * (1.0 + inertia_unc[5])
    inertia = fr.math.matrix([[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]])
    # Closed-form inverse avoids a backend-specific matrix inverse operation.
    determinant = (
        ixx * iyy * izz
        - ixx * iyz**2
        - ixy**2 * izz
        - 2.0 * ixy * ixz * iyz
        - ixz**2 * iyy
    )
    determinant = fr.math.max(determinant, 1e-8)
    inertia_inv = (
        fr.math.matrix(
            [
                [iyy * izz - iyz**2, ixy * izz + ixz * iyz, ixy * iyz + ixz * iyy],
                [ixy * izz + ixz * iyz, ixx * izz - ixz**2, ixx * iyz + ixy * ixz],
                [ixy * iyz + ixz * iyy, ixx * iyz + ixy * ixz, ixx * iyy - ixy**2],
            ]
        )
        / determinant
    )

    fr.output("weight", fr.math.vector([weight]), dim=(1,), unit="N")
    fr.output("mass", fr.math.vector([mass]), dim=(1,), unit="kg")
    fr.output("inertia", inertia, dim=(3, 3), unit="kg*m^2")
    fr.output("inertia_inv", inertia_inv, dim=(3, 3), unit="1/(kg*m^2)")
