from typing import Any

import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem, build_initial_state


def _compile(cfg: GTMConfig | None = None) -> fr.CompiledModel[Any]:
    builder = GTMSystem(cfg or GTMConfig.default(), name="Root")
    assert isinstance(builder, fr.PyIrBuilder)
    return fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.numpy, integrator="euler", dt=0.01)
    ).compile(builder)


def test_topology_contains_semantic_physical_blocks() -> None:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    assert isinstance(builder, fr.PyIrBuilder)
    blocks = set(builder.to_schema_dict()["blocks"])
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
    xs, ys, _ = model.discrete_fn(
        0.0,
        x0,
        model.xd.build(),
        u,
        model.p.build(),
        n_steps=5,
    )
    final_state = model.x.read(np.asarray(xs[-1]))
    assert len(ys) == 5
    np.testing.assert_allclose(
        np.linalg.norm(final_state["quat"]),
        1.0,
        atol=1e-10,
    )
