"""GTM 质量与转动惯量不确定性 Block。"""

from fly_ruler_core import api as fr

from ..config import AircraftConfig


@fr.block(config=AircraftConfig)
def Mass(cfg: AircraftConfig) -> None:
    """对重量与六个惯量分量施加相对不确定性。

    Args:
        cfg: 飞机配置，提供标称重量与 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]` 标称惯量。

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

    nominal = cfg.inertia0
    weight = cfg.w0 * (1.0 + weight_unc[0])
    mass = weight / 9.80665
    Ixx = nominal[0] * (1.0 + inertia_unc[0])
    Iyy = nominal[1] * (1.0 + inertia_unc[1])
    Izz = nominal[2] * (1.0 + inertia_unc[2])
    Ixz = nominal[3] * (1.0 + inertia_unc[3])
    Iyz = nominal[4] * (1.0 + inertia_unc[4])
    Ixy = nominal[5] * (1.0 + inertia_unc[5])

    # [Ixx, -Ixy, -Ixz]
    # [-Ixy, Iyy, -Iyz]
    # [-Ixz, -Iyz, Izz]
    inertia = fr.math.matrix([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])
    # Closed-form inverse avoids a backend-specific matrix inverse operation.

    # det(inertia) = Ixx*(Iyy*Izz - Iyz^2)
    #                - (-Ixy)*(-Ixy*Izz - (-Ixz)*(-Iyz))
    #                + (-Ixz)*((-Ixy)*(-Iyz) - Iyy*(-Ixz))
    #              = Ixx*Iyy*Izz - Ixx*Iyz^2 - Ixy^2*Izz - 2*Ixy*Ixz*Iyz - Ixz^2*Iyy
    determinant = (
        Ixx * Iyy * Izz
        - Ixx * Iyz**2
        - Ixy**2 * Izz
        - 2.0 * Ixy * Ixz * Iyz
        - Ixz**2 * Iyy
    )
    determinant = fr.math.max(determinant, 1e-8)
    inertia_inv = (
        fr.math.matrix(
            [
                [Iyy * Izz - Iyz**2, Ixy * Izz + Ixz * Iyz, Ixy * Iyz + Ixz * Iyy],
                [Ixy * Izz + Ixz * Iyz, Ixx * Izz - Ixz**2, Ixx * Iyz + Ixy * Ixz],
                [Ixy * Iyz + Ixz * Iyy, Ixx * Iyz + Ixy * Ixz, Ixx * Iyy - Ixy**2],
            ]
        )
        / determinant
    )

    fr.output(
        "weight",
        fr.math.vector([weight]),
        dim=(1,),
        unit="N",
        description="Aircraft weight after relative weight uncertainty",
    )
    fr.output(
        "mass",
        fr.math.vector([mass]),
        dim=(1,),
        unit="kg",
        description=(
            "Aircraft mass derived from uncertain weight using standard "
            "gravity 9.80665 m/s^2"
        ),
    )
    fr.output(
        "inertia",
        inertia,
        dim=(3, 3),
        unit="kg*m^2",
        description=(
            "Uncertain body-FRD inertia tensor with negated products of "
            "inertia off-diagonal"
        ),
    )
    fr.output(
        "inertia_inv",
        inertia_inv,
        dim=(3, 3),
        unit="1/(kg*m^2)",
        description=(
            "Body-FRD inverse inertia from cofactors with determinant floored at 1e-8"
        ),
    )


__all__ = ["Mass"]
