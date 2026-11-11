"""发动机配置归属、独立载荷公式、子系统结构与路径回归。"""

from typing import assert_type

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from fly_ruler_core.runtime.instance import RuntimeInstance

from fr_gtm import GTMConfig
from fr_gtm.components.engine import (
    Engine,
    EngineConfig,
    EngineLoads,
    EngineLoadsConfig,
    FirstOrderThrottleLag,
    FirstOrderThrottleLagConfig,
    JetCatThrottleLag,
    JetCatThrottleLagConfig,
)


def test_definition_configuration_types() -> None:
    assert_type(Engine.Config, type[EngineConfig])
    assert_type(EngineLoads.Config, type[EngineLoadsConfig])
    assert_type(FirstOrderThrottleLag.Config, type[FirstOrderThrottleLagConfig])
    assert_type(JetCatThrottleLag.Config, type[JetCatThrottleLagConfig])
    assert Engine.Config is EngineConfig
    assert EngineLoads.Config is EngineLoadsConfig
    assert FirstOrderThrottleLag.Config is FirstOrderThrottleLagConfig
    assert JetCatThrottleLag.Config is JetCatThrottleLagConfig


@pytest.mark.parametrize("fullscale", [True, False])
def test_factory_preserves_physical_coefficients(fullscale: bool) -> None:
    cfg = (GTMConfig.fullscale if fullscale else GTMConfig.subscale)()
    loads = cfg.engine.loads
    assert isinstance(cfg.engine.lag, FirstOrderThrottleLagConfig)
    assert cfg.engine.lag.time_constant_s == 1.5
    np.testing.assert_array_equal(
        loads.thrust_coefficients, [0.000014847593013, 0.0, 0.0, 0.412662356277121]
    )
    np.testing.assert_array_equal(
        loads.throttle_coefficients,
        [0.000033035538475, -0.008354385314499, 1.191126728328792, 31.518959847101669],
    )
    assert loads.T_gain == pytest.approx(
        4.4482216152605 * (40000.0 / 14.0 if fullscale else 1.0)
    )
    assert loads.h_gain == pytest.approx(
        (1.0 / 0.055 if fullscale else 1.0) ** 5 * 1.3558179483314
    )
    assert loads.RPMref_gain == (20000.0 / 123000.0 if fullscale else 1.0)
    assert loads.ram_drag_coeff == (0.004 if fullscale else 0.01)
    length_scale = 1.0 / 0.055 if fullscale else 1.0
    np.testing.assert_allclose(
        loads.engl_pos,
        np.array([-51.903, -14.20, -7.71]) / 12.0 * length_scale * 0.3048,
    )
    np.testing.assert_array_equal(loads.cg_pos, cfg.aircraft.cg_pos0)
    assert loads.S == cfg.aircraft.S


def test_nested_configuration_serialization() -> None:
    cfg = GTMConfig.fullscale()
    cfg.engine = EngineConfig(
        loads=EngineLoadsConfig.fullscale(S=42.0, cg_pos=(1.0, 2.0, 3.0)),
        lag=JetCatThrottleLagConfig(time_scale=2.0),
    )
    loaded = EngineConfig.model_validate(cfg.engine.model_dump())
    assert loaded == cfg.engine
    assert loaded.loads.S == 42.0
    assert loaded.loads.cg_pos == (1.0, 2.0, 3.0)
    assert isinstance(loaded.lag, JetCatThrottleLagConfig)
    assert loaded.lag.time_scale == 2.0


@pytest.mark.parametrize("backend", ["numpy", "casadi"])
def test_standalone_loads_follow_original_formula(backend: str) -> None:
    # 用非默认几何/面积验证计算只消费载荷配置，而不是偷偷回读飞机配置。
    data = GTMConfig.subscale().engine.loads.model_dump()
    data.update(
        S=2.5,
        engl_pos=(-1.0, -2.0, 0.3),
        engr_pos=(-1.2, 2.2, -0.4),
        cg_pos=(-0.3, 0.1, -0.2),
        engl_ang=(0.0, 5.0, 2.0),
        engr_ang=(0.0, -3.0, -4.0),
    )
    cfg = EngineLoadsConfig.model_validate(data)

    @fr.system
    def LoadsOnly() -> None:
        loads = EngineLoads("loads", cfg)
        throttle = fr.input("throttle", 1)
        omega = fr.input("omega", 3)
        pressure = fr.input("pressure", 1)
        qbar = fr.input("qbar", 1)
        throttle >> loads.effective_throttle
        omega >> loads.omega_body
        pressure >> loads.pressure_ratio
        qbar >> loads.qbar
        fr.export(force=loads.force, moment=loads.moment)

    compiler_settings = ModelSettings()
    builder = LoadsOnly.build("Root")

    def verify[ArrayT](model: RuntimeInstance[ArrayT]) -> None:
        assert model.n_x == 0
        omega = np.array([0.2, -0.1, 0.3])
        for throttle, pressure, qbar in [
            (40.0, 0.78, 6500.0),
            (115.0, 1.0, 0.0),
            (-1.474, 0.1, 100000.0),
        ]:
            u = model.u.build(
                {
                    "Root::throttle": [throttle],
                    "Root::omega": omega,
                    "Root::pressure": [pressure],
                    "Root::qbar": [qbar],
                }
            )
            result = model.continuous_fn(
                0.0, model.x.build(), model.xd.build(), u, model.p.build()
            )
            outputs = model.y.read(result.y, by="path")
            rpm = np.polyval(cfg.throttle_coefficients, throttle)
            thrust = max(
                np.polyval(cfg.thrust_coefficients, rpm) * cfg.T_gain * pressure
                - cfg.ram_drag_coeff * qbar * cfg.S,
                0.0,
            )
            directions = []
            for angles in (cfg.engl_ang, cfg.engr_ang):
                _, pitch, yaw = np.radians(angles)
                directions.append(
                    np.array(
                        [
                            np.cos(pitch) * np.cos(yaw),
                            np.sin(yaw),
                            -np.sin(pitch) * np.cos(yaw),
                        ]
                    )
                )
            left, right = directions
            shaft_momentum = (
                np.pi
                * cfg.shaft_diameter**4
                / 32.0
                * cfg.shaft_length
                * cfg.density
                * (2.0 * np.pi / 60.0)
            )
            momentum = (
                cfg.RPMref_gain * rpm * (123000.0 / 100.0) * shaft_momentum * cfg.h_gain
            )
            expected_force = thrust * left + thrust * right
            expected_moment = (
                np.cross(omega, momentum * (left + right))
                + np.cross(np.asarray(cfg.engl_pos) - cfg.cg_pos, thrust * left)
                + np.cross(np.asarray(cfg.engr_pos) - cfg.cg_pos, thrust * right)
            )
            np.testing.assert_allclose(
                model.to_numpy(outputs["Root::force"]).reshape(-1),
                expected_force,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                model.to_numpy(outputs["Root::moment"]).reshape(-1),
                expected_moment,
                atol=1e-10,
            )

    if backend == "numpy":
        verify(
            runtime.bind(
                compiler.compile(
                    compiler.prepare(builder, settings=compiler_settings),
                    target=targets.numpy,
                )
            )
        )
    else:
        verify(
            runtime.bind(
                compiler.compile(
                    compiler.prepare(builder, settings=compiler_settings),
                    target=targets.casadi,
                )
            )
        )


def test_old_flat_load_overrides_are_not_silently_ignored() -> None:
    cfg = GTMConfig.fullscale().engine
    data = cfg.model_dump()
    data["ram_drag_coeff"] = 0.5
    with pytest.raises(ValueError, match="ram_drag_coeff"):
        EngineConfig.model_validate(data)


@pytest.mark.parametrize("fullscale", [True, False])
def test_load_factory_uses_explicit_geometry(fullscale: bool) -> None:
    loads = (
        EngineLoadsConfig.fullscale(S=42.0, cg_pos=(1.0, 2.0, 3.0))
        if fullscale
        else EngineLoadsConfig.subscale(S=42.0, cg_pos=(1.0, 2.0, 3.0))
    )
    assert loads.S == 42.0
    assert loads.cg_pos == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("backend", ["numpy", "casadi", "native", "jax"])
def test_jetcat_matrix_operations_match_state_space(backend: str) -> None:
    cfg = JetCatThrottleLagConfig(time_scale=2.5)

    @fr.system
    def LagOnly() -> None:
        lag = JetCatThrottleLag("lag", cfg)
        command = fr.input("command", (1,))
        command >> lag.throttle_command
        fr.export(throttle=lag.effective_throttle)

    compiler_settings = ModelSettings()
    builder = LagOnly.build("Root")

    def verify[T](model: RuntimeInstance[T]) -> None:
        A = np.asarray(cfg.state_matrix).reshape(2, 2)
        B = np.asarray(cfg.input_matrix)
        C = np.asarray(cfg.output_matrix)
        for state in ([10.0, -3.0], [0.0, 0.0], [-5.0, 150.0]):
            x = model.x.build({"Root.lag::lag_state": state})
            for command in (-50.0, 35.0, 150.0):
                u = model.u.build({"Root::command": [command]})
                rates, _, outputs, _ = model.continuous_fn(
                    0.0, x, model.xd.build(), u, model.p.build()
                )
                np.testing.assert_allclose(
                    model.to_numpy(rates).reshape(-1),
                    (A @ state + B * np.clip(command, 0.0, 100.0)) / cfg.time_scale,
                    rtol=1e-7,
                    atol=1e-8,
                )
                throttle = model.y.read(outputs, by="path")["Root::throttle"]
                np.testing.assert_allclose(
                    model.to_numpy(throttle).reshape(-1), [C @ state], rtol=1e-7
                )

    match backend:
        case "numpy":
            verify(
                runtime.bind(
                    compiler.compile(
                        compiler.prepare(builder, settings=compiler_settings),
                        target=targets.numpy,
                    )
                )
            )
        case "casadi":
            verify(
                runtime.bind(
                    compiler.compile(
                        compiler.prepare(builder, settings=compiler_settings),
                        target=targets.casadi,
                    )
                )
            )
        case "native":
            verify(
                runtime.bind(
                    compiler.compile(
                        compiler.prepare(builder, settings=compiler_settings),
                        target=targets.NativeTarget(),
                    )
                )
            )
        case "jax":
            pytest.importorskip("jax")
            verify(
                runtime.bind(
                    compiler.compile(
                        compiler.prepare(builder, settings=compiler_settings),
                        target=targets.jax,
                    )
                )
            )
        case _:
            raise AssertionError(backend)
