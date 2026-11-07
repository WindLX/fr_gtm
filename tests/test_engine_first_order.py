"""一阶发动机的配置选择、解析响应与公开步进合同。"""

import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm import FirstOrderThrottleLagConfig, GTMConfig, JetCatThrottleLagConfig
from fr_gtm.components.engine import Engine


@pytest.mark.parametrize("fullscale", [True, False])
def test_default_and_config_switch(fullscale: bool) -> None:
    cfg = (GTMConfig.fullscale if fullscale else GTMConfig.subscale)()
    assert isinstance(cfg.engine.lag, FirstOrderThrottleLagConfig)
    assert cfg.engine.lag.time_constant_s == 1.5
    assert cfg.engine.lag.trim_state(40.0) == (40.0,)
    second = JetCatThrottleLagConfig(time_scale=2.0)
    assert second.time_scale == 2.0
    first = FirstOrderThrottleLagConfig(time_constant_s=3.0)
    assert first.time_constant_s == 3.0
    assert isinstance(
        type(cfg.engine).model_validate(cfg.engine.model_dump()).lag,
        FirstOrderThrottleLagConfig,
    )
    cfg.engine = cfg.engine.model_copy(update={"lag": second})
    assert isinstance(
        type(cfg.engine).model_validate(cfg.engine.model_dump()).lag,
        JetCatThrottleLagConfig,
    )


@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_time_constant(tau: float) -> None:
    with pytest.raises(ValueError):
        FirstOrderThrottleLagConfig(time_constant_s=tau)


def test_invalid_limits() -> None:
    with pytest.raises(ValueError, match="lower"):
        FirstOrderThrottleLagConfig(command_lower=100.0, command_upper=50.0)


@pytest.mark.parametrize("backend", ["numpy", "casadi", "native", "jax", "torch"])
@pytest.mark.parametrize("tau", [1.5, 3.0])
def test_first_order_analytic_response(backend: str, tau: float) -> None:
    cfg = GTMConfig.fullscale()
    cfg.engine = cfg.engine.model_copy(
        update={"lag": FirstOrderThrottleLagConfig(time_constant_s=tau)}
    )

    @fr.system
    def EngineOnly() -> None:
        engine = Engine("engine", cfg.engine)
        command = fr.input("command", (1,))
        command >> engine.throttle_command
        fr.math.vector([0.0, 0.0, 0.0]) >> engine.omega_body
        fr.math.vector([1.0]) >> engine.pressure_ratio
        fr.math.vector([0.0]) >> engine.qbar
        fr.export(throttle=engine.effective_throttle, force=engine.force)

    compiler = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=0.01))
    builder = EngineOnly.build("Root")

    def verify[T](model: fr.CompiledModel[T]) -> None:
        assert model.n_x == 1
        xd, p = model.xd.build(), model.p.build()
        for command in (-50.0, 80.0, 150.0, 20.0):
            x = model.x.build(
                {"Root.engine.lag::lag_state": cfg.engine.lag.trim_state(40.0)}
            )
            u = model.u.build({"Root::command": [command]})
            rates, _, outputs, _ = model.continuous_fn(0.0, x, xd, u, p)
            clipped = np.clip(command, 0.0, 100.0)
            np.testing.assert_allclose(
                model.to_numpy(rates).reshape(-1), [(clipped - 40.0) / tau], rtol=1e-7
            )
            np.testing.assert_allclose(
                model.to_numpy(model.y.read(outputs, by="path")["Root::throttle"]), 40.0
            )
            np.testing.assert_array_equal(
                model.to_numpy(model.flow_fn(0.0, x, xd, u, p, 0.0)).reshape(-1), [40.0]
            )
            for step in range(150):
                x, xd, _, _ = model.discrete_fn(step * 0.01, x, xd, u, p)
            expected = clipped + (40.0 - clipped) * np.exp(-1.5 / tau)
            np.testing.assert_allclose(
                model.to_numpy(x).reshape(-1), [expected], rtol=2e-7, atol=1e-8
            )
        x = model.x.build({"Root.engine.lag::lag_state": [40.0]})
        u = model.u.build({"Root::command": [40.0]})
        np.testing.assert_array_equal(
            model.to_numpy(model.flow_fn(0.0, x, xd, u, p, 1.5)).reshape(-1), [40.0]
        )

    match backend:
        case "numpy":
            verify(compiler.compile(builder, target=fr.targets.numpy))
        case "casadi":
            verify(compiler.compile(builder))
        case "native":
            verify(compiler.compile(builder, target=fr.targets.NativeTarget()))
        case "jax":
            pytest.importorskip("jax")
            verify(compiler.compile(builder, target=fr.targets.jax))
        case "torch":
            pytest.importorskip("torch")
            verify(
                compiler.compile(
                    builder, target=fr.targets.TorchTarget(dtype="float64")
                )
            )
        case _:
            raise AssertionError(backend)
