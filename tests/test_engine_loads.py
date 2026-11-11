"""公开无状态发动机载荷的CasADi一致性及连续命令反馈组合回归。"""

from __future__ import annotations

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from numpy.typing import NDArray

from fr_gtm import FirstOrderThrottleLagConfig, GTMConfig, JetCatThrottleLagConfig
from fr_gtm.components.engine import Engine, EngineLoads


@pytest.mark.parametrize("fullscale", [False, True], ids=["scale", "fullscale"])
@pytest.mark.parametrize(
    "lag",
    [FirstOrderThrottleLagConfig(), JetCatThrottleLagConfig()],
    ids=["first_order", "second_order"],
)
def test_engine_loads_matches_engine_at_effective_throttle(
    fullscale: bool, lag: FirstOrderThrottleLagConfig | JetCatThrottleLagConfig
) -> None:
    """指定有效油门与环境，比较公开输出；负值与超过100%均保留。"""
    cfg = (GTMConfig.fullscale if fullscale else GTMConfig.subscale)()
    cfg.engine = cfg.engine.model_copy(update={"lag": lag})

    @fr.system
    def EngineAndLoads() -> None:
        engine = Engine("engine", cfg.engine)
        loads = EngineLoads("loads", cfg.engine.loads)
        command = fr.input("command", (1,))
        effective_throttle = fr.input("effective_throttle", (1,))
        omega = fr.input("omega_body", (3,))
        pressure = fr.input("pressure_ratio", (1,))
        qbar = fr.input("qbar", (1,))
        command >> engine.throttle_command
        effective_throttle >> loads.effective_throttle
        omega >> engine.omega_body
        omega >> loads.omega_body
        pressure >> engine.pressure_ratio
        pressure >> loads.pressure_ratio
        qbar >> engine.qbar
        qbar >> loads.qbar
        fr.export(
            engine_effective_throttle=engine.effective_throttle,
            engine_force=engine.force,
            engine_moment=engine.moment,
            loads_force=loads.force,
            loads_moment=loads.moment,
        )

    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                EngineAndLoads.build("EngineAndLoads"),
                settings=ModelSettings(integrator="rk4", dt=0.005),
            ),
            target=targets.casadi,
        )
    )
    # EngineLoads没有增加连续或数字存储；原发动机保留配置对应的滞后阶数。
    assert model.n_x == (1 if isinstance(lag, FirstOrderThrottleLagConfig) else 2)
    assert model.n_xd == 0
    xd, params = model.xd.build(), model.p.build()
    # 非零p/q/r覆盖转子陀螺项，非零动压与低压力覆盖冲压阻力和推力缩放。
    environments = (
        ([0.0, 0.0, 0.0], 1.0, 0.0),
        ([0.1, -0.2, 0.3], 0.78, 6500.0),
        ([-0.3, 0.4, -0.2], 0.45, 12000.0),
    )
    for throttle in (-1.474, 0.0, 40.0, 100.0, 115.0):
        # 二阶额外取非配平内部状态但保持C*x=throttle；载荷不能依赖隐藏坐标。
        state = np.asarray(lag.trim_state(throttle), dtype=np.float64)
        if isinstance(lag, JetCatThrottleLagConfig):
            state += 10.0 * np.array([lag.output_matrix[1], -lag.output_matrix[0]])
        x = model.x.build({"Root.engine.lag::lag_state": state})
        for omega, pressure, qbar in environments:
            baseline_force: NDArray[np.float64] | None = None
            baseline_moment: NDArray[np.float64] | None = None
            for command in (-50.0, 25.0, 150.0):
                u = model.u.build(
                    {
                        "Root::command": [command],
                        "Root::effective_throttle": [throttle],
                        "Root::omega_body": omega,
                        "Root::pressure_ratio": [pressure],
                        "Root::qbar": [qbar],
                    }
                )
                result = model.continuous_fn(0.0, x, xd, u, params)
                outputs = model.y.read(result.y, by="path")
                np.testing.assert_allclose(
                    model.to_numpy(outputs["Root::engine_effective_throttle"]).reshape(
                        -1
                    ),
                    [throttle],
                    rtol=0.0,
                    atol=3e-14,
                )
                force = model.to_numpy(outputs["Root::engine_force"])
                moment = model.to_numpy(outputs["Root::engine_moment"])
                np.testing.assert_allclose(
                    model.to_numpy(outputs["Root::loads_force"]),
                    force,
                    rtol=1e-12,
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    model.to_numpy(outputs["Root::loads_moment"]),
                    moment,
                    rtol=1e-12,
                    atol=1e-9,
                )
                if baseline_force is not None and baseline_moment is not None:
                    np.testing.assert_array_equal(force, baseline_force)
                    np.testing.assert_array_equal(moment, baseline_moment)
                baseline_force, baseline_moment = force.copy(), moment.copy()


@pytest.mark.parametrize("fullscale", [False, True], ids=["scale", "fullscale"])
@pytest.mark.parametrize(
    "lag",
    [FirstOrderThrottleLagConfig(), JetCatThrottleLagConfig()],
    ids=["first_order", "second_order"],
)
def test_engine_loads_compiles_in_command_feedback_without_feedthrough(
    fullscale: bool, lag: FirstOrderThrottleLagConfig | JetCatThrottleLagConfig
) -> None:
    """真实状态输出接载荷反馈；命令改变滞后导数而不直接改变当前载荷。"""
    cfg = (GTMConfig.fullscale if fullscale else GTMConfig.subscale)()
    cfg.engine = cfg.engine.model_copy(update={"lag": lag})

    @fr.system
    def EngineLoadFeedback() -> None:
        engine = Engine("engine", cfg.engine)
        loads = EngineLoads("loads", cfg.engine.loads)
        request = fr.input("request", (1,))
        omega = fr.input("omega_body", (3,))
        pressure = fr.input("pressure_ratio", (1,))
        qbar = fr.input("qbar", (1,))
        engine.effective_throttle >> loads.effective_throttle
        omega >> engine.omega_body
        omega >> loads.omega_body
        pressure >> engine.pressure_ratio
        pressure >> loads.pressure_ratio
        qbar >> engine.qbar
        qbar >> loads.qbar
        # 测试用小力反馈，增益单位%/N；反馈以连续滞后状态隔开，无代数环。
        command = fr.math.vector([request[0] - 1e-5 * loads.force[0]])
        command >> engine.throttle_command
        fr.export(
            command=command,
            effective_throttle=engine.effective_throttle,
            force=loads.force,
            moment=loads.moment,
            engine_force=engine.force,
            engine_moment=engine.moment,
        )

    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                EngineLoadFeedback.build("EngineLoadFeedback"),
                settings=ModelSettings(integrator="rk4", dt=0.005),
            ),
            target=targets.casadi,
        )
    )
    assert model.n_x == (1 if isinstance(lag, FirstOrderThrottleLagConfig) else 2)
    assert model.n_xd == 0
    x = model.x.build({"Root.engine.lag::lag_state": lag.trim_state(40.0)})
    xd, params = model.xd.build(), model.p.build()
    rates: list[NDArray[np.float64]] = []
    commands: list[float] = []
    forces: list[NDArray[np.float64]] = []
    moments: list[NDArray[np.float64]] = []
    for request in (20.0, 80.0):
        u = model.u.build(
            {
                "Root::request": [request],
                "Root::omega_body": [0.1, -0.2, 0.3],
                "Root::pressure_ratio": [0.78],
                "Root::qbar": [6500.0],
            }
        )
        result = model.continuous_fn(0.0, x, xd, u, params)
        outputs = model.y.read(result.y, by="path")
        command = float(model.to_numpy(outputs["Root::command"]).reshape(-1)[0])
        assert lag.command_lower < command < lag.command_upper
        commands.append(command)
        rates.append(model.to_numpy(result.xdot).reshape(-1))
        forces.append(model.to_numpy(outputs["Root::force"]))
        moments.append(model.to_numpy(outputs["Root::moment"]))
        np.testing.assert_array_equal(
            forces[-1], model.to_numpy(outputs["Root::engine_force"])
        )
        np.testing.assert_array_equal(
            moments[-1], model.to_numpy(outputs["Root::engine_moment"])
        )
        np.testing.assert_allclose(
            model.to_numpy(outputs["Root::effective_throttle"]).reshape(-1), [40.0]
        )
        # 实际调用公开步进边界，确保反馈不仅能构图，也能用于RK内部阶段求值。
        stepped = model.discrete_fn(0.0, x, xd, u, params)
        assert np.all(np.isfinite(model.to_numpy(stepped.x_next)))
        assert np.all(np.isfinite(model.to_numpy(stepped.y)))

    np.testing.assert_array_equal(forces[0], forces[1])
    np.testing.assert_array_equal(moments[0], moments[1])
    assert commands[1] - commands[0] == pytest.approx(60.0)
    # 固定当前状态时Δxdot只由指令入口产生：一阶为Δδc/τ，二阶为BΔδc/c。
    delta = commands[1] - commands[0]
    expected_rate_delta = (
        np.array([delta / lag.time_constant_s])
        if isinstance(lag, FirstOrderThrottleLagConfig)
        else np.asarray(lag.input_matrix) * delta / lag.time_scale
    )
    np.testing.assert_allclose(
        rates[1] - rates[0], expected_rate_delta, rtol=1e-12, atol=1e-12
    )
