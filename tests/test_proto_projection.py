import math

import pytest

from fr_gtm import airdata_angles_to_proto_radians


def test_gtm_airdata_degrees_are_projected_to_proto_radians() -> None:
    alpha, beta = airdata_angles_to_proto_radians(alpha_deg=5.31, beta_deg=-0.122)

    assert alpha == pytest.approx(math.radians(5.31))
    assert beta == pytest.approx(math.radians(-0.122))


@pytest.mark.parametrize(
    ("alpha_deg", "beta_deg"),
    [(math.nan, 0.0), (181.0, 0.0), (0.0, math.inf), (0.0, -91.0)],
)
def test_invalid_gtm_airdata_angles_are_rejected(
    alpha_deg: float, beta_deg: float
) -> None:
    with pytest.raises(ValueError):
        airdata_angles_to_proto_radians(alpha_deg=alpha_deg, beta_deg=beta_deg)
