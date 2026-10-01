"""FlyRuler Proto 生产者使用的单位安全投影助手。"""

from __future__ import annotations

import math


def airdata_angles_to_proto_radians(
    *, alpha_deg: float, beta_deg: float
) -> tuple[float, float]:
    """把 GTM 气动角由度换算为 Proto 规范的弧度。

    Args:
        alpha_deg: 迎角，单位 deg，须为有限值且落在 [-180, 180]。
        beta_deg: 侧滑角，单位 deg，须为有限值且落在 [-90, 90]。

    Returns:
        tuple[float, float]: 依次为迎角与侧滑角的弧度值 `(alpha_rad, beta_rad)`。

    Raises:
        ValueError: `alpha_deg` 非有限或超出 [-180, 180] deg，或 `beta_deg`
            非有限或超出 [-90, 90] deg。

    Examples:
        ```python
        from fr_gtm import airdata_angles_to_proto_radians

        alpha_rad, beta_rad = airdata_angles_to_proto_radians(
            alpha_deg=5.0, beta_deg=-2.0
        )
        ```
    """

    if not math.isfinite(alpha_deg) or not -180.0 <= alpha_deg <= 180.0:
        raise ValueError(
            "GTM angle of attack must be finite and within [-180, 180] deg"
        )
    if not math.isfinite(beta_deg) or not -90.0 <= beta_deg <= 90.0:
        raise ValueError("GTM sideslip angle must be finite and within [-90, 90] deg")
    return math.radians(alpha_deg), math.radians(beta_deg)
