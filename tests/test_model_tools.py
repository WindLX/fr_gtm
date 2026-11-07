"""路径目录、物理初值、参数、Gaussian 激励和观测解析合同。"""

from typing import assert_type
from urllib.request import urlopen

import fly_ruler_core as fr
import numpy as np
import pytest
from fly_ruler_std.runtime import graph as fr_graph

from fr_gtm import (
    D_DIM,
    N_DIM,
    U_DIM,
    Y_DIM,
    GTMConfig,
    GTMPaths,
    GTMSystem,
    JetCatThrottleLagConfig,
    build_input,
    init,
    read_observation,
    sample_disturbance,
    sample_sensor_noise,
)


@pytest.fixture(scope="module")
def nominal() -> tuple[GTMConfig, fr.CompiledModel[fr.NumpyArray]]:
    cfg = GTMConfig.fullscale()
    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        GTMSystem.build("Root", cfg), target=fr.targets.numpy
    )
    return cfg, model


def test_layout_paths_and_channel_dimensions(
    nominal: tuple[GTMConfig, fr.CompiledModel[fr.NumpyArray]],
) -> None:
    _, model = nominal
    paths = GTMPaths()
    assert_type(paths.x.pos_ned, str)
    assert (U_DIM, D_DIM, N_DIM, Y_DIM) == (4, 4, 17, 17)
    assert paths.x.pos_ned in model.x.keys
    assert paths.x.engine_lag in model.x.keys
    assert paths.p.c_e in model.p.keys
    assert paths.effects.aero_coefficients in model.effects.keys
    assert paths.y.clean in model.y.keys
    nested = GTMPaths("Root.aircraft", io_root="Root")
    assert nested.x.pos_ned == "Root.aircraft.eom::pos_ned"
    assert nested.p.weight == "Root.aircraft.mass::weight_unc"
    assert nested.u.command == "Root::u"
    assert nested.y.clean == "Root::clean_y"


@pytest.mark.parametrize("fullscale", [True, False])
@pytest.mark.parametrize("second_order", [True, False])
@pytest.mark.parametrize("backend", ["numpy", "casadi"])
def test_initial_airdata_wind_parameters_and_engine(
    fullscale: bool, second_order: bool, backend: str
) -> None:
    cfg = GTMConfig.fullscale() if fullscale else GTMConfig.subscale()
    if second_order:
        cfg.engine = cfg.engine.model_copy(update={"lag": JetCatThrottleLagConfig()})
    compiler = fr.Compiler(fr.CompilerSpec(cache_policy="off"))
    builder = GTMSystem.build("Root", cfg)
    paths = GTMPaths()

    def verify[T](model: fr.CompiledModel[T]) -> None:
        x, xd, u, p = init(
            model,
            cfg,
            position=(12.0, 23.0, -800.0),
            airdata=(7.0, -2.0, 90.0),
            euler=(5.0, 9.0, 35.0),
            command=(-2.0, 1.0, -0.2, 45.0),
            wind=(12.0, 35.0, 3.0),
            uncertainty={"weight": 0.03, "inertia": (0.01,) * 6, "c_e": (0.02,) * 3},
        )
        result = model.continuous_fn(0.0, x, xd, u, p)
        obs = read_observation(model, result.y)
        np.testing.assert_allclose(
            [obs.alpha_deg, obs.beta_deg, obs.tas_mps], [7.0, -2.0, 90.0], atol=1e-7
        )
        np.testing.assert_allclose(obs.euler_deg, [5.0, 9.0, 35.0], atol=1e-10)
        assert obs.altitude_m == 800.0
        assert obs.throttle_percent == pytest.approx(45.0)
        state = model.x.read(x, by="path")
        np.testing.assert_allclose(
            model.to_numpy(state[paths.x.pos_ned]).reshape(-1), [12.0, 23.0, -800.0]
        )
        np.testing.assert_allclose(
            model.to_numpy(state[paths.x.engine_lag]).reshape(-1),
            cfg.engine.lag.trim_state(45.0),
        )
        np.testing.assert_allclose(model.to_numpy(state[paths.x.omega_body]), 0.0)
        assert model.to_numpy(xd).size == 0
        inputs = model.u.read(u, by="path")
        np.testing.assert_allclose(
            model.to_numpy(inputs[paths.u.command]).reshape(-1), [-2.0, 1.0, -0.2, 45.0]
        )
        np.testing.assert_array_equal(
            model.to_numpy(inputs[paths.u.noise]).reshape(-1), np.zeros(N_DIM)
        )
        params = model.p.read(p, by="path")
        np.testing.assert_allclose(
            model.to_numpy(params[paths.p.wind_direction]), np.deg2rad(35.0)
        )
        np.testing.assert_allclose(model.to_numpy(params[paths.p.weight]), 0.03)
        np.testing.assert_allclose(model.to_numpy(params[paths.p.c_e]), 0.02)
        assert model.n_x == (26 if second_order else 25)

    if backend == "numpy":
        verify(compiler.compile(builder, target=fr.targets.numpy))
    else:
        verify(compiler.compile(builder, target=fr.targets.casadi))


def test_nested_instance_init_and_observation() -> None:
    cfg = GTMConfig.fullscale()

    @fr.system
    def Parent() -> None:
        plant = GTMSystem("aircraft", cfg)
        command = fr.input("u", 4)
        d = fr.input("d", 4)
        n = fr.input("n", 17)
        command >> plant.u
        d >> plant.d
        n >> plant.n
        fr.export(y=plant.y, clean_y=plant.clean_y)

    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        Parent.build("Root"), target=fr.targets.numpy
    )
    paths = GTMPaths("Root.aircraft", io_root="Root")
    x, xd, u, p = init(
        model, cfg, paths=paths, airdata=(4.0, 1.0, 70.0), wind=(8.0, 60.0, 2.0)
    )
    obs = read_observation(model, model.continuous_fn(0.0, x, xd, u, p).y, paths=paths)
    np.testing.assert_allclose(
        [obs.alpha_deg, obs.beta_deg, obs.tas_mps], [4.0, 1.0, 70.0], atol=1e-7
    )


def test_measured_observation_and_validation(
    nominal: tuple[GTMConfig, fr.CompiledModel[fr.NumpyArray]],
) -> None:
    cfg, model = nominal
    x, xd, _, p = init(model, cfg)
    u = build_input(model, (0.0, 0.0, 0.0, 40.0), n=np.ones(N_DIM))
    result = model.continuous_fn(0.0, x, xd, u, p)
    clean = read_observation(model, result.y)
    measured = read_observation(model, result.y, measured=True)
    expected = (
        cfg.sensor.V.bias + cfg.sensor.V.scale * clean.tas_mps + cfg.sensor.V.noise_std
    )
    assert measured.tas_mps == pytest.approx(expected)
    with pytest.raises(ValueError, match="n must contain 17"):
        build_input(model, (0.0, 0.0, 0.0, 40.0), n=np.zeros(16))
    with pytest.raises(ValueError, match="c_e must contain 3"):
        init(model, cfg, uncertainty={"c_e": (0.0,) * 6})
    with pytest.raises(ValueError, match="VTAS"):
        init(model, cfg, airdata=(0.0, 0.0, -1.0))


def test_gaussian_contract_and_reproducibility() -> None:
    rng = np.random.default_rng(42)
    expected = np.random.default_rng(42)
    np.testing.assert_array_equal(
        sample_disturbance(rng), expected.normal(0.0, np.pi, size=4)
    )
    np.testing.assert_array_equal(
        sample_sensor_noise(rng), expected.normal(0.0, 1.0, size=17)
    )
    samples = np.stack([sample_disturbance(rng) for _ in range(10000)])
    assert abs(float(samples.mean())) < 0.06
    assert abs(float(samples.std()) - np.pi) < 0.06
    sensor = np.stack([sample_sensor_noise(rng) for _ in range(10000)])
    assert abs(float(sensor.mean())) < 0.01
    assert abs(float(sensor.std()) - 1.0) < 0.01


def test_inspect_graph_webui(
    nominal: tuple[GTMConfig, fr.CompiledModel[fr.NumpyArray]],
) -> None:
    """与 inspect 示例相同的入口能提供真实页面与模型拓扑。"""
    _, model = nominal
    with fr_graph.start(model, open_browser=False) as server:
        with urlopen(server.url, timeout=5) as response:
            assert response.status == 200
            assert b"<html" in response.read()
        with urlopen(server.url + "/api/system", timeout=5) as response:
            assert response.status == 200
            assert b"aero" in response.read()
