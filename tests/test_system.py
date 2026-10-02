from collections.abc import Mapping
from typing import Any, cast

import fly_ruler_core as fr
import numpy as np
import pytest
from fly_ruler_core import VariableInfo
from fly_ruler_core._core import PyCompiler

from fr_gtm import (
    GTMConfig,
    GTMSystem,
    build_ground_initial_state,
    build_initial_state,
)
from fr_gtm.blocks import Engine, Mass


def _compile(cfg: GTMConfig | None = None) -> fr.CompiledModel[Any]:
    builder = GTMSystem.build("Root", cfg or GTMConfig.default())
    assert isinstance(builder, fr.PyIrBuilder)
    return fr.Compiler(fr.CompilerSpec(integrator="euler", dt=0.01)).compile(
        builder, target=fr.targets.numpy
    )


def test_topology_contains_semantic_physical_blocks() -> None:
    builder = GTMSystem.build("Root", GTMConfig.default())
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


def test_system_level_control_effectiveness_query_resolves_through_plant() -> None:
    cfg = GTMConfig.default()

    @fr.system
    def ControlledPlant() -> None:
        command = fr.input("u", (4,))
        disturbance = fr.input("d", (4,))
        noise = fr.input("n", (21,))
        plant = GTMSystem("plant", cfg)
        command >> plant.u
        disturbance >> plant.d
        noise >> plant.n
        effectiveness = fr.jacobian(
            output=plant.moment_aero,
            wrt=plant.actuator_position,
            name="control_effectiveness",
        )
        fr.export(control_effectiveness=effectiveness)

    builder = ControlledPlant.build("ControlledPlant")
    plan = PyCompiler.compile_from_builder(builder)
    assert len(plan.jacobian_plans) == 1
    query = plan.jacobian_plans[0]
    assert query.output_path == "Root.plant::moment_aero"
    assert query.wrt_path == "Root.plant::actuator_position"
    assert query.output_shape == [3]
    assert query.wrt_shape == [4]


def test_layout_and_full_paths() -> None:
    model = _compile()
    assert model.n_x == 25
    assert model.n_xdot == 24
    assert model.n_p == 33
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
    assert outputs["feedback_y"].shape == (17,)
    assert outputs["clean_feedback_y"].shape == (17,)
    np.testing.assert_array_equal(outputs["feedback_y"], outputs["y"][:17])
    np.testing.assert_array_equal(outputs["clean_feedback_y"], outputs["clean_y"][:17])


def test_command_telemetry_is_excluded_from_feedback() -> None:
    """控制器反馈不得对根指令存在直接前馈。"""

    model = _compile()
    x = build_initial_state(model)
    outputs = []
    for command in (np.zeros(4), np.array([2.0, -3.0, 1.0, 75.0])):
        values: Mapping[str | VariableInfo, Any] = {
            "Root::u": command,
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
        u = model.u.build(values)
        _, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.build())
        outputs.append(model.y.read(y))

    np.testing.assert_array_equal(outputs[0]["feedback_y"], outputs[1]["feedback_y"])
    assert not np.array_equal(outputs[0]["y"][17:], outputs[1]["y"][17:])


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
    assert model.n_x == 25
    assert model.n_u == 29
    assert model.n_p == 34
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


def _mass_block_outputs(cfg: GTMConfig) -> Mapping[str, Any]:
    """编译只含 `Mass` 块的系统，便于直接读取重量与质量输出。"""

    @fr.system
    def MassOnly() -> None:
        mass = Mass("mass", cfg)
        fr.export(mass_value=mass.mass, weight_value=mass.weight)

    builder = MassOnly.build("MassOnly")
    model = fr.Compiler(fr.CompilerSpec(integrator="euler", dt=0.01)).compile(
        builder, target=fr.targets.numpy
    )
    _, _, y, _ = model.continuous_fn(
        0.0, model.x.build(), model.xd.build(), model.u.build(), model.p.build()
    )
    return model.y.read(y, by="path")


def test_mass_block_uses_configured_gravity() -> None:
    """`Mass` 用 `UncertaintyConfig.gravity_acc` 换算质量，而非硬编码常量。"""

    cfg = GTMConfig.default()
    baseline = _mass_block_outputs(cfg)
    np.testing.assert_allclose(
        baseline["Root::mass_value"], [cfg.aircraft.w0 / cfg.uncertainty.gravity_acc]
    )

    changed = cfg.model_copy(deep=True)
    changed.uncertainty.gravity_acc = 2.0 * cfg.uncertainty.gravity_acc
    updated = _mass_block_outputs(changed)
    np.testing.assert_allclose(
        updated["Root::mass_value"], [cfg.aircraft.w0 / changed.uncertainty.gravity_acc]
    )
    assert not np.allclose(baseline["Root::mass_value"], updated["Root::mass_value"])


def _engine_block_moment(cfg: GTMConfig) -> np.ndarray:
    """编译只含 `Engine` 块的系统，返回固定工况下的发动机力矩。"""

    @fr.system
    def EngineOnly() -> None:
        engine = Engine("engine", cfg)
        actuators = fr.input("actuators", (4,))
        omega_body = fr.input("omega_body", (3,))
        pressure_ratio = fr.input("pressure_ratio", (1,))
        qbar = fr.input("qbar", (1,))
        actuators >> engine.actuators
        omega_body >> engine.omega_body
        pressure_ratio >> engine.pressure_ratio
        qbar >> engine.qbar
        fr.export(engine_force=engine.force, engine_moment=engine.moment)

    builder = EngineOnly.build("EngineOnly")
    model = fr.Compiler(fr.CompilerSpec(integrator="euler", dt=0.01)).compile(
        builder, target=fr.targets.numpy
    )
    u = model.u.build(
        {
            "Root::actuators": np.array([0.0, 0.0, 0.0, 60.0]),
            "Root::omega_body": np.zeros(3),
            "Root::pressure_ratio": np.ones(1),
            "Root::qbar": np.array([1000.0]),
        }
    )
    _, _, y, _ = model.continuous_fn(
        0.0, model.x.build(), model.xd.build(), u, model.p.build()
    )
    return np.asarray(model.y.read(y, by="path")["Root::engine_moment"])


def test_engine_block_uses_engine_config_cg_pos() -> None:
    """`Engine` 的力臂取自 `EngineConfig.cg_pos`，使该字段真正参与仿真。"""

    cfg = GTMConfig.default()
    assert cfg.engine.cg_pos == cfg.aircraft.cg_pos0

    shifted = cfg.model_copy(deep=True)
    shifted.engine.cg_pos = [
        cfg.aircraft.cg_pos0[0] + 0.5,
        cfg.aircraft.cg_pos0[1],
        cfg.aircraft.cg_pos0[2],
    ]
    baseline = _engine_block_moment(cfg)
    moved = _engine_block_moment(shifted)
    assert not np.allclose(baseline, moved)
