"""Unit-safe projection helpers for FlyRuler Proto producers."""

from __future__ import annotations

import math


def airdata_angles_to_proto_radians(
    *, alpha_deg: float, beta_deg: float
) -> tuple[float, float]:
    """Convert GTM aerodynamic angles in degrees to canonical Proto radians."""

    if not math.isfinite(alpha_deg) or not -180.0 <= alpha_deg <= 180.0:
        raise ValueError(
            "GTM angle of attack must be finite and within [-180, 180] deg"
        )
    if not math.isfinite(beta_deg) or not -90.0 <= beta_deg <= 90.0:
        raise ValueError("GTM sideslip angle must be finite and within [-90, 90] deg")
    return math.radians(alpha_deg), math.radians(beta_deg)
