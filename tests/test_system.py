from collections.abc import Mapping
from typing import Any, cast

import fly_ruler_core as fr
import numpy as np
import pytest
from fly_ruler_core import VariableInfo

from fr_gtm import (
    GTMConfig,
    GTMSystem,
    build_ground_initial_state,
    build_initial_state,
)


def _compile(cfg: GTMConfig | None = None) -> fr.CompiledModel[Any]:
    builder = GTMSystem(cfg or GTMConfig.default(), name="Root")
    assert isinstance(builder, fr.PyIrBuilder)
    return fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.numpy, integrator="euler", dt=0.01)
    ).compile(builder)


def test_topology_contains_semantic_physical_blocks() -> None:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    assert isinstance(builder, fr.PyIrBuilder)
    blocks = set(builder._artifact_schema(include_exprs=False)["blocks"])
    assert {
        "Root.actuator",
        "Root.aero",
        "Root.engine",
        "Root.eom",
        "Root.mass",
        "Root.sensor",
    }.issubset(blocks)


def test_layout_and_full_paths() -> None:
    model = _compile()
    assert model.module.n_x == 25
    assert model.module.n_xdot == 24
    assert model.module.n_p == 33
    assert "Root.eom::quat" in model.x.keys
    assert "Root.wind.dryden::w_gw" in model.x.keys
    assert "Root.aero::c_static_unc" in model.p.keys
    assert "Root.wind.params::wind_speed" in model.p.keys


def test_initial_state_and_outputs_are_finite() -> None:
    model = _compile()
    x = build_initial_state(model)
    u = model.u.build(
        {
            "Root::u": np.zeros(4),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    xdot, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.build())
    outputs = model.y.read(y)
    assert x.shape == (25,)
    assert np.isfinite(xdot).all()
    assert np.isfinite(outputs["y"]).all()
    assert outputs["clean_y"].shape == (21,)


def test_short_integration_preserves_quaternion_norm() -> None:
    model = _compile()
    x0 = build_initial_state(model)
    u = model.u.build(
        {
            "Root::u": np.array([0.0, 0.0, 0.0, 30.0]),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    x = x0
    xd = model.xd.build()
    p = model.p.build()
    for step in range(5):
        x, xd, _y, _effects = model.discrete_fn(step * 0.01, x, xd, u, p)
    final_state = model.x.read(np.asarray(x))
    np.testing.assert_allclose(
        np.linalg.norm(final_state["quat"]),
        1.0,
        atol=1e-10,
    )


def test_wind_direction_runtime_parameter_is_radians() -> None:
    model = _compile()
    x = build_initial_state(model)
    u = model.u.build(
        {"Root::u": np.zeros(4), "Root::d": np.zeros(4), "Root::n": np.zeros(21)}
    )
    parameters = model.p.read(model.p.default(), by="path")
    parameters["Root.wind.params::wind_speed"] = np.array([10.0])
    parameters["Root.wind.params::wind_direction"] = np.array([np.pi / 2.0])
    p = model.p.build(cast(Mapping[str | VariableInfo, Any], parameters))
    _, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, p)
    steady = model.y.read(y, by="path")["Root::uvw"]
    assert abs(steady[0]) < 1.0e-12
    assert steady[1] < 0.0


@pytest.mark.parametrize("is_fullscale", [False, True])
def test_optional_ground_contact_layout_and_static_support(
    is_fullscale: bool,
) -> None:
    cfg = GTMConfig.default(is_fullscale, enable_ground=True)
    model = _compile(cfg)
    assert model.module.n_x == 25
    assert model.module.n_u == 29
    assert model.module.n_p == 34
    assert "Root.ground.params::ground_down" in model.p.keys
    assert "Root::ground_normal_forces" in model.y.keys

    x = build_ground_initial_state(model, cfg)
    u = model.u.build(
        {
            "Root::u": np.zeros(4),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    _, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.build())
    outputs = model.y.read(y, by="path")
    np.testing.assert_allclose(
        np.sum(outputs["Root::ground_normal_forces"]),
        cfg.aircraft.w0,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        outputs["Root::ground_moment_body"][:2], 0.0, atol=1.0e-4
    )
    np.testing.assert_allclose(outputs["Root::ground_contact_count"], [3.0])

    airborne_x = build_ground_initial_state(
        model,
        cfg,
        forward_speed_mps=70.0,
        altitude_agl_m=1.0,
        sink_rate_mps=1.5,
    )
    _, _, airborne_y, _ = model.continuous_fn(
        0.0, airborne_x, model.xd.build(), u, model.p.build()
    )
    airborne = model.y.read(airborne_y, by="path")
    np.testing.assert_allclose(airborne["Root::ground_normal_forces"], 0.0)
    np.testing.assert_allclose(airborne["Root::ground_contact_count"], [0.0])
    np.testing.assert_allclose(
        airborne["Root::vel_ned"], [70.0, 0.0, 1.5], atol=1.0e-12
    )


def test_ground_touchdown_short_trajectory_is_finite() -> None:
    cfg = GTMConfig.default(enable_ground=True)
    model = _compile(cfg)
    x = build_ground_initial_state(
        model,
        cfg,
        forward_speed_mps=70.0,
        altitude_agl_m=0.05,
        sink_rate_mps=1.5,
    )
    xd = model.xd.build()
    u = model.u.build(
        {
            "Root::u": np.zeros(4),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    p = model.p.build()
    saw_contact = False
    for step in range(50):
        x, xd, y, _effects = model.discrete_fn(step * 0.01, x, xd, u, p)
        outputs = model.y.read(y, by="path")
        saw_contact = saw_contact or outputs["Root::ground_contact_count"][0] > 0.0
    assert saw_contact
    assert np.all(np.isfinite(np.asarray(x, dtype=float)))
    state = model.x.read(np.asarray(x, dtype=float), by="path")
    np.testing.assert_allclose(
        np.linalg.norm(state["Root.eom::quat"]), 1.0, atol=1.0e-10
    )
