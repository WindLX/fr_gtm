"""使用真实 Proto 对象验证真值投影、坐标和 effects 遥测。"""

import math

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from fly_ruler_core.runtime.instance import RuntimeInstance

from fr_gtm import GTMConfig, GTMPaths, GTMSystem, Mode, build_input, init
from fr_gtm.interface.proto import VISUAL_MODEL_ID, EffectsTelemetry, to_aircraft_state


@pytest.mark.parametrize("backend", ["numpy", "casadi"])
@pytest.mark.parametrize("mode", list(Mode))
def test_truth_projection(backend: str, mode: Mode) -> None:
    cfg = GTMConfig.fullscale(mode=mode)
    cfg.sensor.alpha.bias = 15
    cfg.sensor.throttle.scale = 1.5
    builder = GTMSystem.build("Root", cfg)
    compiler_settings = ModelSettings()

    def verify[T](model: RuntimeInstance[T]) -> None:
        paths = GTMPaths()
        x, xd, _, p = init(
            model,
            cfg,
            position=(100, 200, -1000),
            airdata=(5.31, -0.122, 100),
            euler=(4, 8, 20),
            rates=(1, 2, 3),
            command=(-2, 3, 1, 40),
            wind=(8, 35, 2),
        )
        # 命令与实际状态故意不同，接口应读取实际舵面和发动机状态。
        u = build_input(model, (10, 10, 10, 90), n=np.ones(17) * 10)
        result = model.continuous_fn(0, x, xd, u, p)
        state = to_aircraft_state(
            model, cfg, x=x, y=result.y, effects=result.effects, origin=(31, 121, 50)
        )
        assert state.position.z == -1000
        assert state.derived is not None
        assert state.derived.altitude == 1050
        assert state.derived.alpha == pytest.approx(math.radians(5.31), abs=1e-8)
        assert state.derived.beta == pytest.approx(math.radians(-0.122), abs=1e-8)
        assert state.derived.tas == pytest.approx(100)
        assert state.derived.lat > 31 and state.derived.lon > 121
        diagnostics = model.effects.read(result.effects, by="path")
        ned = model.to_numpy(diagnostics[paths.effects.vel_ned]).reshape(3)
        assert state.derived.chi == pytest.approx(math.atan2(ned[1], ned[0]))
        assert state.derived.vertical_speed == pytest.approx(-ned[2])
        assert state.derived.ground_speed == pytest.approx(math.hypot(ned[0], ned[1]))
        assert state.linear_acceleration_body is None
        np.testing.assert_allclose(
            [
                state.angular_velocity.x,
                state.angular_velocity.y,
                state.angular_velocity.z,
            ],
            np.deg2rad([1, 2, 3]),
        )
        expected_quat = model.to_numpy(
            model.x.read(x, by="path")[paths.x.quat]
        ).reshape(4)
        np.testing.assert_allclose(state.attitude.quaternion, expected_quat)
        assert state.control_surfaces is not None
        assert state.control_surfaces.elevator_rad == pytest.approx(math.radians(-2))
        assert state.control_surfaces.aileron_left_rad == pytest.approx(math.radians(3))
        assert state.control_surfaces.aileron_right_rad == pytest.approx(
            math.radians(-3)
        )
        expected_flap = (
            0 if mode == Mode.NORMAL else 0.5 if mode == Mode.HALF_FLAP_GEAR_UP else 1
        )
        assert state.control_surfaces.flaps_left_ratio == expected_flap
        assert state.propulsors is not None
        assert [engine.throttle_ratio for engine in state.propulsors] == [0.4, 0.4]
        # NED 速度来自含风的地速，而不是误用 TAS。
        body = model.to_numpy(model.x.read(x, by="path")[paths.x.vel_body]).reshape(3)
        np.testing.assert_allclose(
            [state.velocity.x, state.velocity.y, state.velocity.z], body
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


def test_effects_schema_and_values() -> None:
    cfg = GTMConfig.subscale()
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(GTMSystem.build("Root", cfg), settings=ModelSettings()),
            target=targets.numpy,
        )
    )
    paths = GTMPaths()
    selection = (paths.effects.mass, paths.effects.aero_force, paths.effects.inertia)
    binding = EffectsTelemetry(model, paths=selection, nominal_rate_hz=20)
    x, xd, u, p = init(model, cfg)
    result = model.continuous_fn(0, x, xd, u, p)
    values = binding.values(result.effects)
    assert binding.stream_id == "gtm.effects"
    assert binding.schema.nominal_rate_hz == 20
    assert len(values) == len(binding.schema.fields) == 13
    assert [field.field_id for field in binding.schema.fields] == [
        selection[0],
        *(selection[1] + f"[{i}]" for i in range(3)),
        *(selection[2] + f"[{i},{j}]" for i in range(3) for j in range(3)),
    ]
    read = model.effects.read(result.effects, by="path")
    np.testing.assert_array_equal(
        values,
        np.concatenate([model.to_numpy(read[key]).reshape(-1) for key in selection]),
    )
    for field, key in zip(binding.schema.fields[:2], selection[:2], strict=True):
        assert field.unit == model.effects.info(key).unit
        assert field.description == model.effects.info(key).description
    with pytest.raises(KeyError, match="ground"):
        EffectsTelemetry(model, paths=(paths.effects.ground_force_body,))
    with pytest.raises(ValueError):
        EffectsTelemetry(model, paths=())
    with pytest.raises(ValueError):
        EffectsTelemetry(model, paths=(selection[0], selection[0]))
    with pytest.raises(ValueError):
        EffectsTelemetry(model, paths=selection, nominal_rate_hz=-1)
    with pytest.raises(ValueError):
        to_aircraft_state(
            model, cfg, x=x, y=result.y, effects=result.effects, origin=(90, 0, 0)
        )
    assert VISUAL_MODEL_ID == "boeing_737_800"


def test_geodetic_projection_matches_std() -> None:
    """地理投影与共享坐标库保持同一个 WGS-84 局部近似。"""
    from fly_ruler_std.libs.coordinates import ned_position_to_geodetic

    @fr.system
    def Geodetic() -> None:
        position = fr.input("position", (3,))
        origin = fr.input("origin", (3,))
        fr.export(lla=ned_position_to_geodetic(position, origin))

    compiler_settings = ModelSettings()
    reference = runtime.bind(
        compiler.compile(
            compiler.prepare(Geodetic.build("Root"), settings=compiler_settings),
            target=targets.numpy,
        )
    )
    cfg = GTMConfig.fullscale()
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(GTMSystem.build("Root", cfg), settings=compiler_settings),
            target=targets.numpy,
        )
    )
    origin = (31.0, 121.0, 50.0)
    position = (100.0, 200.0, -1000.0)
    x, xd, u, p = init(model, cfg, position=position)
    result = model.continuous_fn(0, x, xd, u, p)
    state = to_aircraft_state(
        model, cfg, x=x, y=result.y, effects=result.effects, origin=origin
    )
    expected = reference.continuous_fn(
        0,
        reference.x.build(),
        reference.xd.build(),
        reference.u.build(
            {"Root::position": np.array(position), "Root::origin": np.array(origin)}
        ),
        reference.p.default(),
    )
    assert state.derived is not None
    np.testing.assert_allclose(
        [state.derived.lat, state.derived.lon, state.derived.altitude],
        expected.y,
        atol=1e-12,
    )
