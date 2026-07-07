from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm import GTMConfig, GTMSystem, Mode

REFERENCE = np.load(Path(__file__).parent / "data" / "gtm_reference.npz")
CASE_NAMES = REFERENCE["name"].tolist()


@lru_cache(maxsize=None)
def _model(fullscale: bool, mode: str) -> fr.CompiledModel[Any]:
    cfg = GTMConfig.default(fullscale, Mode(mode))
    builder = GTMSystem(cfg, name="Root")
    assert isinstance(builder, fr.PyIrBuilder)
    return fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.numpy, integrator="euler", dt=0.01)
    ).compile(builder)


def _state(model: fr.CompiledModel[Any], x: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": x[0:3],
                "Root.eom::vel_body": x[3:6],
                "Root.eom::quat": x[6:10],
                "Root.eom::omega_body": x[10:13],
                "Root.actuator::position": x[13:17],
                "Root.wind.dryden::u_gw": x[17:18],
                "Root.wind.dryden::v_gw": x[18:20],
                "Root.wind.dryden::w_gw": x[20:22],
                "Root.wind.dryden::p_gw": x[22:23],
                "Root.wind.dryden::q_gw": x[23:24],
                "Root.wind.dryden::r_gw": x[24:25],
            }
        ),
        dtype=float,
    )


def _parameters(model: fr.CompiledModel[Any], p: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.p.build(
            {
                "Root.wind.params::wind_speed": p[0:1],
                "Root.wind.params::wind_direction": p[1:2],
                "Root.wind.params::turbulence_level": p[2:3],
                "Root.mass::weight_unc": p[3:4],
                "Root.mass::inertia_unc": p[4:10],
                "Root.aero::c_static_unc": p[10:16],
                "Root.aero::c_e_unc": p[16:19],
                "Root.aero::c_a_unc": p[19:25],
                "Root.aero::c_r_unc": p[25:31],
            }
        ),
        dtype=float,
    )


def _reference_xdot(
    derivatives: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [
            derivatives["Root.eom::pos_ned"],
            derivatives["Root.eom::vel_body"],
            outputs["quat_dot_reference"],
            derivatives["Root.eom::omega_body"],
            derivatives["Root.actuator::position"],
            derivatives["Root.wind.dryden::u_gw"],
            derivatives["Root.wind.dryden::v_gw"],
            derivatives["Root.wind.dryden::w_gw"],
            derivatives["Root.wind.dryden::p_gw"],
            derivatives["Root.wind.dryden::q_gw"],
            derivatives["Root.wind.dryden::r_gw"],
        ]
    )


@pytest.mark.parametrize("index", range(len(CASE_NAMES)), ids=CASE_NAMES)
def test_matches_frozen_gtm_ad_reference(index: int) -> None:
    model = _model(
        bool(REFERENCE["fullscale"][index]),
        str(REFERENCE["mode"][index]),
    )
    x = _state(model, REFERENCE["x"][index])
    u = model.u.build(
        {
            "Root::u": REFERENCE["u"][index],
            "Root::d": REFERENCE["d"][index],
            "Root::n": REFERENCE["n"][index],
        }
    )
    xdot, _, y, _ = model.continuous_fn(
        0.0,
        x,
        model.xd.build(),
        u,
        _parameters(model, REFERENCE["p"][index]),
    )
    derivatives = model.xdot.read(xdot, by="path")
    outputs = model.y.read(y)

    np.testing.assert_allclose(
        _reference_xdot(derivatives, outputs),
        REFERENCE["xdot"][index],
        rtol=1e-8,
        atol=1e-9,
    )
    for actual, expected in (
        (outputs["y"], REFERENCE["y"][index]),
        (outputs["clean_y"], REFERENCE["clean_y"][index]),
        (outputs["force_engine"], REFERENCE["force_engine"][index]),
        (outputs["force_aero"], REFERENCE["force_aero"][index]),
        (outputs["force_gravity"], REFERENCE["force_gravity"][index]),
        (outputs["moment_engine"], REFERENCE["moment_engine"][index]),
        (outputs["moment_aero"], REFERENCE["moment_aero"][index]),
        (outputs["uvw"], REFERENCE["wind_uvw"][index]),
        (outputs["wind_omega"], REFERENCE["wind_omega"][index]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
