"""GTM mass and inertia uncertainty block."""

import fly_ruler_core as fr
from fly_ruler_core import Signal

from ..config import GTMConfig

_GRAVITY = 9.80665


def _mass_properties(
    cfg: GTMConfig, weight_unc: Signal, inertia_unc: Signal
) -> tuple[Signal, Signal, Signal, Signal]:
    nominal = cfg.aircraft.inertia0
    weight = cfg.aircraft.w0 * (1.0 + weight_unc[0])
    mass = weight / _GRAVITY
    ixx = nominal[0] * (1.0 + inertia_unc[0])
    iyy = nominal[1] * (1.0 + inertia_unc[1])
    izz = nominal[2] * (1.0 + inertia_unc[2])
    ixz = nominal[3] * (1.0 + inertia_unc[3])
    iyz = nominal[4] * (1.0 + inertia_unc[4])
    ixy = nominal[5] * (1.0 + inertia_unc[5])
    inertia = fr.math.matrix([[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]])
    determinant = (
        ixx * iyy * izz
        - ixx * iyz**2
        - ixy**2 * izz
        - 2.0 * ixy * ixz * iyz
        - ixz**2 * iyy
    )
    determinant = fr.math.max(determinant, 1e-8)
    inverse = (
        fr.math.matrix(
            [
                [iyy * izz - iyz**2, ixy * izz + ixz * iyz, ixy * iyz + ixz * iyy],
                [ixy * izz + ixz * iyz, ixx * izz - ixz**2, ixx * iyz + ixy * ixz],
                [ixy * iyz + ixz * iyy, ixx * iyz + ixy * ixz, ixx * iyy - ixy**2],
            ]
        )
        / determinant
    )
    return weight, mass, inertia, inverse


@fr.block
def Mass(cfg: GTMConfig) -> None:
    """Apply GTM mass and inertia uncertainty parameters."""
    weight_unc = fr.param("weight_unc", (1,), default=[0.0])
    inertia_unc = fr.param("inertia_unc", (6,), default=[0.0] * 6)

    weight, mass, inertia, inertia_inv = _mass_properties(cfg, weight_unc, inertia_unc)

    fr.output("weight", fr.math.vector([weight]), dim=(1,), unit="N")
    fr.output("mass", fr.math.vector([mass]), dim=(1,), unit="kg")
    fr.output("inertia", inertia, dim=(3, 3), unit="kg*m^2")
    fr.output("inertia_inv", inertia_inv, dim=(3, 3), unit="1/(kg*m^2)")
