from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from fly_ruler_core._core import PyCompiler
from fly_ruler_core.api.types import PyIrBuilder
from fly_ruler_core.model.description import VariableInfo
from fly_ruler_core.runtime.instance import RuntimeInstance
from ground_helpers import ground_state

from fr_gtm import (
    GTMConfig,
    GTMSystem,
    JetCatThrottleLagConfig,
    init,
)
from fr_gtm.components.engine import Engine
from fr_gtm.components.mass import Mass


def _compile(cfg: GTMConfig) -> RuntimeInstance[Any]:
    builder = GTMSystem.build("Root", cfg)
    assert isinstance(builder, PyIrBuilder)
    return runtime.bind(
        compiler.compile(
            compiler.prepare(
                builder, settings=ModelSettings(integrator="euler", dt=0.01)
            ),
            target=targets.numpy,
        )
    )


@pytest.fixture(scope="module")
def default_model() -> RuntimeInstance[Any]:
    """共享编译结果；每个测试仍独立构建状态、输入和运行参数。"""
    return _compile(GTMConfig.fullscale())


@pytest.fixture(scope="module")
def ground_model() -> RuntimeInstance[Any]:
    return _compile(GTMConfig.fullscale(enable_ground=True))


def test_topology_contains_semantic_physical_blocks() -> None:
    builder = GTMSystem.build("Root", GTMConfig.fullscale())
    assert isinstance(builder, PyIrBuilder)
    blocks = set(builder._artifact_schema(include_exprs=False)["blocks"])
    assert {
        "Root.actuator",
        "Root.aero",
        "Root.eom",
        "Root.mass",
        "Root.sensor",
    }.issubset(blocks)
    schema = builder._artifact_schema(include_exprs=False)
    engine = next(child for child in schema["subsystems"] if child["name"] == "engine")
    assert {"Root.engine.lag", "Root.engine.loads"}.issubset(engine["blocks"])


def test_clean_observation_has_no_instantaneous_command_dependency() -> None:
    cfg = GTMConfig.fullscale()

    @fr.system
    def ControlledPlant() -> None:
        command = fr.input("u", (4,))
        disturbance = fr.input("d", (4,))
        noise = fr.input("n", (17,))
        plant = GTMSystem("plant", cfg)
        command >> plant.u
        disturbance >> plant.d
        noise >> plant.n
        effectiveness = fr.jacobian(
            output=plant.clean_y,
            wrt=command,
            name="control_effectiveness",
        )
        fr.export(control_effectiveness=effectiveness)

    builder = ControlledPlant.build("ControlledPlant")
    # 不再追加命令后，物理观测只由当前状态决定；编译器应明确拒绝无依赖的导数查询。
    with pytest.raises(ValueError, match="jacobian_no_dependency"):
        PyCompiler.compile_from_builder(builder)


def test_layout_and_full_paths(default_model: RuntimeInstance[Any]) -> None:
    model = default_model
    assert model.n_x == 25
    assert model.n_xdot == 24
    assert set(model.y.keys) == {"Root::y", "Root::clean_y"}
    assert model.n_y == 34
    assert model.n_u == 25
    assert "Root.aero::coefficients" in model.effects.keys
    assert set(model.effects.keys) == {
        "Root::weight",
        "Root::mass",
        "Root::inertia",
        "Root::inertia_inv",
        "Root::VE",
        "Root::mach",
        "Root::pressure_ratio",
        "Root::DCM",
        "Root::vel_ned",
        "Root::chi",
        "Root::acceleration_body",
        "Root::angular_acceleration_body",
        "Root::wind_omega",
        "Root::force_aero",
        "Root::force_engine",
        "Root::moment_aero",
        "Root::moment_engine",
        "Root.aero::coefficients",
        "Root.aero::reynolds",
        "Root.wind::steady",
        "Root.wind::gust",
    }
    assert model.n_p == 33
    assert "Root.eom::quat" in model.x.keys
    assert "Root.wind.dryden::w_gw" in model.x.keys
    assert "Root.aero::c_static_unc" in model.p.keys
    assert "Root.wind.params::wind_speed" in model.p.keys


def test_initial_state_and_outputs_are_finite(
    default_model: RuntimeInstance[Any],
) -> None:
    model = default_model
    x = init(
        model,
        GTMConfig.fullscale(),
        position=(0.0, 0.0, 0.0),
        airdata=(0.0, 0.0, 50.0),
        command=(0.0, 0.0, 0.0, 0.0),
    )[0]
    u = model.u.build(
        {
            "Root::u": np.zeros(4),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(17),
        }
    )
    xdot, _, y, effects = model.continuous_fn(
        0.0, x, model.xd.build(), u, model.p.build()
    )
    outputs = model.y.read(y)
    assert x.shape == (25,)
    assert np.isfinite(xdot).all()
    assert np.isfinite(outputs["y"]).all()
    assert outputs["clean_y"].shape == (17,)
    assert set(model.y.keys) == {"Root::y", "Root::clean_y"}


def test_command_telemetry_is_excluded_from_feedback(
    default_model: RuntimeInstance[Any],
) -> None:
    """控制器反馈不得对根指令存在直接前馈。"""

    model = default_model
    x = init(
        model,
        GTMConfig.fullscale(),
        position=(0.0, 0.0, 0.0),
        airdata=(0.0, 0.0, 50.0),
        command=(0.0, 0.0, 0.0, 0.0),
    )[0]
    outputs = []
    for command in (np.zeros(4), np.array([2.0, -3.0, 1.0, 75.0])):
        values: Mapping[str | VariableInfo, Any] = {
            "Root::u": command,
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(17),
        }
        u = model.u.build(values)
        _, _, y, effects = model.continuous_fn(
            0.0, x, model.xd.build(), u, model.p.build()
        )
        outputs.append(model.y.read(y))

    np.testing.assert_array_equal(outputs[0]["y"], outputs[1]["y"])
    assert outputs[0]["y"].shape == (17,)


def test_short_integration_preserves_quaternion_norm(
    default_model: RuntimeInstance[Any],
) -> None:
    model = default_model
    x0 = init(
        model,
        GTMConfig.fullscale(),
        position=(0.0, 0.0, 0.0),
        airdata=(0.0, 0.0, 50.0),
        command=(0.0, 0.0, 0.0, 0.0),
    )[0]
    u = model.u.build(
        {
            "Root::u": np.array([0.0, 0.0, 0.0, 30.0]),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(17),
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


def test_wind_direction_runtime_parameter_is_radians(
    default_model: RuntimeInstance[Any],
) -> None:
    model = default_model
    x = init(
        model,
        GTMConfig.fullscale(),
        position=(0.0, 0.0, 0.0),
        airdata=(0.0, 0.0, 50.0),
        command=(0.0, 0.0, 0.0, 0.0),
    )[0]
    u = model.u.build(
        {"Root::u": np.zeros(4), "Root::d": np.zeros(4), "Root::n": np.zeros(17)}
    )
    parameters = model.p.read(model.p.default(), by="path")
    parameters["Root.wind.params::wind_speed"] = np.array([10.0])
    parameters["Root.wind.params::wind_direction"] = np.array([np.pi / 2.0])
    p = model.p.build(cast(Mapping[str | VariableInfo, Any], parameters))
    _, _, y, effects = model.continuous_fn(0.0, x, model.xd.build(), u, p)
    diagnostics = model.effects.read(effects, by="path")
    steady = diagnostics["Root.wind::steady"] + diagnostics["Root.wind::gust"]
    assert abs(steady[0]) < 1.0e-12
    assert steady[1] < 0.0


@pytest.mark.parametrize("is_fullscale", [False, True])
def test_optional_ground_contact_layout_and_static_support(
    is_fullscale: bool,
    ground_model: RuntimeInstance[Any],
) -> None:
    cfg = (GTMConfig.fullscale if is_fullscale else GTMConfig.subscale)(
        enable_ground=True
    )
    model = ground_model if is_fullscale else _compile(cfg)
    assert model.n_x == 25
    assert model.n_u == 25
    assert model.n_p == 34
    assert set(model.y.keys) == {"Root::y", "Root::clean_y"}
    assert model.n_y == 34
    assert "Root.ground.params::ground_down" in model.p.keys
    assert "Root::ground_normal_forces" in model.effects.keys

    x = ground_state(model, cfg)
    u = model.u.build(
        {
            "Root::u": np.zeros(4),
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(17),
        }
    )
    _, _, y, effects = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.build())
    outputs = {**model.y.read(y, by="path"), **model.effects.read(effects, by="path")}
    np.testing.assert_allclose(
        np.sum(outputs["Root::ground_normal_forces"]),
        cfg.aircraft.w0,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        outputs["Root::ground_moment_body"][:2], 0.0, atol=1.0e-4
    )
    np.testing.assert_allclose(outputs["Root::ground_contact_count"], [3.0])

    airborne_x = ground_state(
        model,
        cfg,
        forward_speed_mps=70.0,
        altitude_agl_m=1.0,
        sink_rate_mps=1.5,
    )
    _, _, airborne_y, airborne_effects = model.continuous_fn(
        0.0, airborne_x, model.xd.build(), u, model.p.build()
    )
    airborne = model.effects.read(airborne_effects, by="path")
    np.testing.assert_allclose(airborne["Root::ground_normal_forces"], 0.0)
    np.testing.assert_allclose(airborne["Root::ground_contact_count"], [0.0])
    np.testing.assert_allclose(
        model.xdot.read(
            model.continuous_fn(
                0.0, airborne_x, model.xd.build(), u, model.p.build()
            ).xdot,
            by="path",
        )["Root.eom::pos_ned"],
        [70.0, 0.0, 1.5],
        atol=1.0e-12,
    )


def test_ground_touchdown_short_trajectory_is_finite(
    ground_model: RuntimeInstance[Any],
) -> None:
    cfg = GTMConfig.fullscale(enable_ground=True)
    model = ground_model
    x = ground_state(
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
            "Root::n": np.zeros(17),
        }
    )
    p = model.p.build()
    saw_contact = False
    for step in range(50):
        x, xd, y, effects = model.discrete_fn(step * 0.01, x, xd, u, p)
        outputs = {
            **model.y.read(y, by="path"),
            **model.effects.read(effects, by="path"),
        }
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
        mass = Mass("mass", cfg.aircraft)
        fr.export(mass_value=mass.mass, weight_value=mass.weight)

    builder = MassOnly.build("MassOnly")
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                builder, settings=ModelSettings(integrator="euler", dt=0.01)
            ),
            target=targets.numpy,
        )
    )
    _, _, y, effects = model.continuous_fn(
        0.0, model.x.build(), model.xd.build(), model.u.build(), model.p.build()
    )
    return model.y.read(y, by="path")


def test_mass_block_uses_aircraft_weight_and_standard_gravity() -> None:
    """质量块消费飞机标称重量，以固定标准重力换算质量。"""
    cfg = GTMConfig.fullscale()
    baseline = _mass_block_outputs(cfg)
    np.testing.assert_allclose(
        baseline["Root::mass_value"], [cfg.aircraft.w0 / 9.80665]
    )
    changed = cfg.model_copy(deep=True)
    changed.aircraft = cfg.aircraft.model_copy(update={"w0": 2.0 * cfg.aircraft.w0})
    updated = _mass_block_outputs(changed)
    np.testing.assert_allclose(
        updated["Root::mass_value"], 2.0 * baseline["Root::mass_value"]
    )


def _engine_block_moment(cfg: GTMConfig) -> np.ndarray:
    """编译只含 `Engine` 块的系统，返回固定工况下的发动机力矩。"""

    @fr.system
    def EngineOnly() -> None:
        engine = Engine("engine", cfg.engine)
        throttle_command = fr.input("throttle_command", (1,))
        omega_body = fr.input("omega_body", (3,))
        pressure_ratio = fr.input("pressure_ratio", (1,))
        qbar = fr.input("qbar", (1,))
        throttle_command >> engine.throttle_command
        omega_body >> engine.omega_body
        pressure_ratio >> engine.pressure_ratio
        qbar >> engine.qbar
        fr.export(
            engine_force=engine.force,
            engine_moment=engine.moment,
            effective_throttle=engine.effective_throttle,
        )

    builder = EngineOnly.build("EngineOnly")
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                builder, settings=ModelSettings(integrator="euler", dt=0.01)
            ),
            target=targets.numpy,
        )
    )
    u = model.u.build(
        {
            "Root::throttle_command": np.array([60.0]),
            "Root::omega_body": np.zeros(3),
            "Root::pressure_ratio": np.ones(1),
            "Root::qbar": np.array([1000.0]),
        }
    )
    # 60% 指令对应的滞后配平状态，使有效油门正好是 60% 而不是从零开始爬升。
    x = model.x.build(
        {"Root.engine.lag::lag_state": np.asarray(cfg.engine.lag.trim_state(60.0))}
    )
    _, _, y, effects = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.build())
    outputs = {**model.y.read(y, by="path"), **model.effects.read(effects, by="path")}
    np.testing.assert_allclose(outputs["Root::effective_throttle"], [60.0], atol=1.0e-9)
    return np.asarray(outputs["Root::engine_moment"])


def test_engine_block_uses_engine_config_cg_pos() -> None:
    """`Engine` 的力臂取自 `EngineLoadsConfig.cg_pos`，使该字段真正参与仿真。"""

    cfg = GTMConfig.fullscale()
    np.testing.assert_array_equal(cfg.engine.loads.cg_pos, cfg.aircraft.cg_pos0)

    shifted = cfg.model_copy(deep=True)
    shifted.engine = shifted.engine.model_copy(
        update={
            "loads": shifted.engine.loads.model_copy(
                update={
                    "cg_pos": (
                        cfg.aircraft.cg_pos0[0] + 0.5,
                        cfg.aircraft.cg_pos0[1],
                        cfg.aircraft.cg_pos0[2],
                    )
                }
            )
        }
    )
    baseline = _engine_block_moment(cfg)
    moved = _engine_block_moment(shifted)
    assert not np.allclose(baseline, moved)


def test_original_engine_layout_and_initialization() -> None:
    cfg = GTMConfig.fullscale()
    cfg.engine = cfg.engine.model_copy(update={"lag": JetCatThrottleLagConfig()})
    model = _compile(cfg)
    assert (model.n_x, model.n_xdot) == (26, 25)
    x = init(
        model,
        cfg,
        position=(0.0, 0.0, 0.0),
        airdata=(0.0, 0.0, 50.0),
        command=(0.0, 0.0, 0.0, 40.0),
    )[0]
    np.testing.assert_allclose(
        model.x.read(x, by="path")["Root.engine.lag::lag_state"],
        [0.0, 40.0 / 0.7314],
    )


def test_flight_path_uses_ned_velocity_and_degrees(
    default_model: RuntimeInstance[Any],
) -> None:
    """机头上仰且机体速度沿x轴时，航迹角应随NED爬升速度变化，而不是恒为零。"""
    model = default_model
    pitch = 0.2
    x = model.x.build(
        {
            "Root.eom::vel_body": [50.0, 0.0, 0.0],
            "Root.eom::quat": [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0],
        }
    )
    result = model.continuous_fn(
        0.0, x, model.xd.build(), model.u.build(), model.p.build()
    )
    clean = model.y.read(result.y, by="path")["Root::clean_y"]
    np.testing.assert_allclose(clean[0], np.rad2deg(pitch), atol=1e-7)
    np.testing.assert_allclose(clean[6], np.rad2deg(pitch), atol=1e-12)
