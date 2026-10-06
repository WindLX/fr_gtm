"""发动机二阶油门滞后的解析响应与不变量回归。

主题：`Engine` 块把油门指令送进原始 GTM 的二阶状态空间
（`A=[[-1.336,-0.7314],[1,0]]`、`B=[[1],[0]]`、`C=[-0.1474,0.7314]`），本节用只含
`Engine` 块的最小系统核对它的解析阶跃响应、配平初始化、指令钳位与时间尺度缩放。

运行：

    cd models/fr_gtm && just test-one tests/test_engine_lag.py

预期结果：全部通过；数值积分与解析解在采样点上相对误差小于 1e-6，且首次达到
63.2% 与 95% 的时间分别接近 2.346 s 与 4.022 s，10–90% 上升时间接近 2.774 s，
正向阶跃的最初 0.2 s 内还会出现约 -1.26% 的反向响应。分段常值命令下，逐步状态与
`Φ x + Γ u` 的精确递推也是一致的。
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Any

import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm import EngineLagConfig, GTMConfig
from fr_gtm.blocks import Engine

_DT = 0.005
"""积分步长，取小是为让四阶 Runge-Kutta 的误差远低于解析比对的容差。"""

_ANALYTIC_RTOL = 1.0e-6
"""数值解与解析解在采样点上的相对容差。"""


def _lag() -> EngineLagConfig:
    """全尺寸与缩比共用同一份滞后动态，因此默认配置即可代表两者。"""
    return EngineLagConfig()


@lru_cache(maxsize=None)
def _engine_only(time_scale: float) -> tuple[GTMConfig, Any]:
    """编译只含 `Engine` 块的最小系统，便于单独观察有效油门。"""

    cfg = GTMConfig.default()
    cfg.engine.lag = EngineLagConfig(time_scale=time_scale)

    @fr.system
    def EngineOnly() -> None:
        engine = Engine("engine", cfg)
        throttle_command = fr.input("throttle_command", (1,))
        omega_body = fr.input("omega_body", (3,))
        pressure_ratio = fr.input("pressure_ratio", (1,))
        qbar = fr.input("qbar", (1,))
        throttle_command >> engine.throttle_command
        omega_body >> engine.omega_body
        pressure_ratio >> engine.pressure_ratio
        qbar >> engine.qbar
        fr.export(effective_throttle=engine.effective_throttle, force=engine.force)

    builder = EngineOnly.build("EngineOnly")
    model = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=_DT)).compile(
        builder, target=fr.targets.numpy
    )
    return cfg, model


def _simulate(
    time_scale: float,
    *,
    initial_throttle: float,
    command: float,
    times: tuple[float, ...],
) -> dict[float, float]:
    """从 `initial_throttle` 的配平状态出发，记录给定时刻的有效油门。"""

    cfg, model = _engine_only(time_scale)
    x = model.x.build(
        {
            "Root.engine::lag_state": np.asarray(
                cfg.engine.lag.trim_state(initial_throttle), dtype=float
            )
        }
    )
    u = model.u.build(
        {
            "Root::throttle_command": np.array([command]),
            "Root::omega_body": np.zeros(3),
            "Root::pressure_ratio": np.ones(1),
            "Root::qbar": np.array([0.0]),
        }
    )
    p = model.p.build()
    xd = model.xd.build()
    samples: dict[float, float] = {}
    steps = int(round(max(times) / _DT))
    for step in range(steps + 1):
        now = step * _DT
        for target in times:
            if abs(now - target) <= 1.0e-12:
                _, _, y, _ = model.continuous_fn(now, x, xd, u, p)
                outputs = model.y.read(y, by="path")
                samples[target] = float(outputs["Root::effective_throttle"][0])
        x, xd, _, _ = model.discrete_fn(now, x, xd, u, p)
    return samples


def _analytic_step(
    lag: EngineLagConfig,
    *,
    amplitude: float,
    times: tuple[float, ...],
) -> dict[float, float]:
    """解析阶跃响应：复极点留数展开 `y(t)=1+2Re(r e^{λt})`。"""

    a11, a12, a21, a22 = lag.state_matrix
    c1, c2 = lag.output_matrix
    # 状态空间 A 的特征多项式为 s^2 - trace s + det，判别式为负即复共轭极点。
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    discriminant = trace * trace - 4.0 * det
    assert discriminant < 0.0
    pole = complex(trace / 2.0, math.sqrt(-discriminant) / 2.0)
    residue = complex(c1 * pole + c2, 0.0) / (pole * (pole - pole.conjugate()))
    return {
        t: amplitude * (1.0 + 2.0 * (residue * cmath.exp(pole * t)).real) for t in times
    }


def test_analytic_step_matches_numerical_integration() -> None:
    """阶跃响应与解析解一致，说明矩阵、配平与推进都按同一组动态工作。"""

    lag = _lag()
    times = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    simulated = _simulate(1.0, initial_throttle=0.0, command=100.0, times=times)
    analytic = _analytic_step(lag, amplitude=100.0, times=times)
    for t in times:
        assert simulated[t] == pytest.approx(
            analytic[t], rel=_ANALYTIC_RTOL, abs=1.0e-9
        )


@pytest.mark.parametrize("fullscale", [False, True])
def test_second_order_lag_matches_original_jetcat_matrices(fullscale: bool) -> None:
    """独立固定原始 SLX 左右 Jetcat P70 Mask 的系数，不从当前配置推导参考。"""
    lag = EngineLagConfig()
    np.testing.assert_array_equal(lag.state_matrix, [-1.336, -0.7314, 1.0, 0.0])
    np.testing.assert_array_equal(lag.input_matrix, [1.0, 0.0])
    np.testing.assert_array_equal(lag.output_matrix, [-0.1474, 0.7314])
    assert lag.time_scale == 1.0
    np.testing.assert_allclose(lag.trim_state(40.0), [0.0, 40.0 / 0.7314])


def test_command_changes_rates_without_direct_feedthrough_into_loads() -> None:
    """固定内部状态时指令只改变导数，推力读取未裁切的有效油门。"""
    _, model = _engine_only(1.0)
    state = np.array([10.0, 0.0])
    x = model.x.build({"Root.engine::lag_state": state})
    previous_force: np.ndarray | None = None
    for command in (-50.0, 25.0, 150.0):
        u = model.u.build(
            {
                "Root::throttle_command": [command],
                "Root::omega_body": np.zeros(3),
                "Root::pressure_ratio": [1.0],
                "Root::qbar": [0.0],
            }
        )
        rates, _, y, _ = model.continuous_fn(
            0.0, x, model.xd.build(), u, model.p.build()
        )
        outputs = model.y.read(y, by="path")
        np.testing.assert_allclose(outputs["Root::effective_throttle"], [-1.474])
        np.testing.assert_allclose(
            model.xdot.read(rates, by="path")["Root.engine::lag_state"],
            [-13.36 + np.clip(command, 0.0, 100.0), 10.0],
        )
        force = np.asarray(outputs["Root::force"])
        assert force[0] >= 0.0
        if previous_force is not None:
            np.testing.assert_array_equal(force, previous_force)
        previous_force = force.copy()


def test_second_order_crossing_times_match_documented_response() -> None:
    """首次达到 63.2% 与 95% 的时间是文档给出的二阶响应尺度。"""

    lag = _lag()
    steps = int(round(9.0 / _DT))
    times = tuple(step * _DT for step in range(steps + 1))
    simulated = _simulate(1.0, initial_throttle=0.0, command=100.0, times=times)

    def crossing(fraction: float) -> float:
        level = fraction * 100.0
        for index in range(1, len(times)):
            previous = simulated[times[index - 1]]
            current = simulated[times[index]]
            if previous < level <= current:
                span = current - previous
                offset = 0.0 if span == 0.0 else (level - previous) / span
                return times[index - 1] + offset * (times[index] - times[index - 1])
        raise AssertionError(f"response never reaches {level}%")

    assert crossing(0.632) == pytest.approx(2.346, rel=2.0e-3)
    assert crossing(0.95) == pytest.approx(4.022, rel=2.0e-3)
    # 解析解与数值积分给出同一条曲线，交叉时间只差积分误差。
    analytic = _analytic_step(lag, amplitude=100.0, times=times)
    analytic_crossing = 0.0
    for index in range(1, len(times)):
        if analytic[times[index - 1]] < 95.0 <= analytic[times[index]]:
            analytic_crossing = times[index]
            break
    assert analytic_crossing == pytest.approx(crossing(0.95), abs=0.05)


def test_ten_to_ninety_rise_time_matches_documented_response() -> None:
    """10–90% 上升时间同样落在文档给出的二阶响应尺度上。"""

    steps = int(round(9.0 / _DT))
    times = tuple(step * _DT for step in range(steps + 1))
    simulated = _simulate(1.0, initial_throttle=0.0, command=100.0, times=times)

    def crossing(fraction: float) -> float:
        level = fraction * 100.0
        for index in range(1, len(times)):
            previous = simulated[times[index - 1]]
            current = simulated[times[index]]
            if previous < level <= current:
                span = current - previous
                offset = 0.0 if span == 0.0 else (level - previous) / span
                return times[index - 1] + offset * (times[index] - times[index - 1])
        raise AssertionError(f"response never reaches {level}%")

    assert crossing(0.90) - crossing(0.10) == pytest.approx(2.774, rel=2.0e-3)


def _zero_order_hold(lag: EngineLagConfig, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """分段常值命令的精确离散推进矩阵 `Φ=e^{AΔt/c}`、`Γ=-A⁻¹(I-Φ)B`。"""

    matrix = np.asarray(lag.state_matrix, dtype=float).reshape(2, 2)
    input_vector = np.asarray(lag.input_matrix, dtype=float)
    values, vectors = np.linalg.eig(matrix * dt / lag.time_scale)
    transition = ((vectors * np.exp(values)) @ np.linalg.inv(vectors)).real
    gain = np.linalg.solve(matrix, (transition - np.eye(2)) @ input_vector)
    return transition, gain


def test_piecewise_constant_command_matches_exact_step() -> None:
    """分段常值命令下，数值推进逐步等于 `Φ x + Γ u` 的精确递推。"""

    cfg, model = _engine_only(1.0)
    assert isinstance(cfg.engine.lag, EngineLagConfig)
    transition, gain = _zero_order_hold(cfg.engine.lag, _DT)
    commands = (0.0, 0.0, 35.0, 35.0, 35.0, 80.0, 80.0, 20.0, 20.0, 20.0, 100.0)
    x = model.x.build(
        {
            "Root.engine::lag_state": np.asarray(
                cfg.engine.lag.trim_state(0.0), dtype=float
            )
        }
    )
    xd = model.xd.build()
    p = model.p.build()
    expected = np.asarray(
        model.x.read(x, by="path")["Root.engine::lag_state"], dtype=float
    )
    for step, command in enumerate(commands):
        u = model.u.build(
            {
                "Root::throttle_command": np.array([command]),
                "Root::omega_body": np.zeros(3),
                "Root::pressure_ratio": np.ones(1),
                "Root::qbar": np.array([0.0]),
            }
        )
        x, xd, _, _ = model.discrete_fn(step * _DT, x, xd, u, p)
        expected = transition @ expected + gain * command
        actual = np.asarray(
            model.x.read(x, by="path")["Root.engine::lag_state"], dtype=float
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-12)


def test_positive_step_starts_with_a_small_negative_response() -> None:
    """右半平面零点让正向阶跃先给出很小的一段负有效油门，之后才单调上升。"""

    steps = int(round(1.0 / _DT))
    times = tuple(step * _DT for step in range(steps + 1))
    simulated = _simulate(1.0, initial_throttle=0.0, command=100.0, times=times)
    minimum = min(simulated.values())
    assert minimum < 0.0
    assert minimum == pytest.approx(-1.2561, rel=5.0e-3)
    assert simulated[1.0] > 0.0


def test_time_scale_coefficient_stretches_time_only() -> None:
    """时间尺度系数 c 只拉伸时间轴：c 倍系数下的响应等于原响应在 t/c 处取值。"""

    lag = _lag()
    coefficient = math.sqrt(1.0 / 0.055)
    times = (1.0, 2.5, 5.0, 10.0, 17.15)
    stretched = _simulate(coefficient, initial_throttle=0.0, command=100.0, times=times)
    reference_times = tuple(t / coefficient for t in times)
    reference = _analytic_step(lag, amplitude=100.0, times=reference_times)
    for t, scaled in zip(times, reference_times, strict=True):
        assert stretched[t] == pytest.approx(
            reference[scaled], rel=_ANALYTIC_RTOL, abs=1.0e-9
        )


@pytest.mark.parametrize("equilibrium", [0.0, 25.0, 40.0, 100.0])
def test_trim_state_holds_the_equilibrium(equilibrium: float) -> None:
    """配平初始化下 `xdot=0`：不从配平状态出发就会人为制造推力暂态。"""

    lag = _lag()
    state = np.asarray(lag.trim_state(equilibrium), dtype=float)
    matrix = np.asarray(lag.state_matrix, dtype=float).reshape(2, 2)
    input_vector = np.asarray(lag.input_matrix, dtype=float)
    output_vector = np.asarray(lag.output_matrix, dtype=float)
    np.testing.assert_allclose(output_vector @ state, equilibrium, atol=1.0e-12)
    np.testing.assert_allclose(
        matrix @ state + input_vector * equilibrium, 0.0, atol=1.0e-12
    )

    times = (1.0, 5.0, 20.0)
    simulated = _simulate(
        1.0, initial_throttle=equilibrium, command=equilibrium, times=times
    )
    for t in times:
        assert simulated[t] == pytest.approx(equilibrium, abs=1.0e-9)


def test_command_is_clipped_into_the_configured_range() -> None:
    """指令先按 `[command_lower, command_upper]` 钳位，再进入二阶滞后。"""

    times = (1.0, 4.0, 30.0)
    below = _simulate(1.0, initial_throttle=0.0, command=-50.0, times=times)
    at_zero = _simulate(1.0, initial_throttle=0.0, command=0.0, times=times)
    above = _simulate(1.0, initial_throttle=100.0, command=150.0, times=times)
    at_hundred = _simulate(1.0, initial_throttle=100.0, command=100.0, times=times)
    for t in times:
        assert below[t] == pytest.approx(at_zero[t], abs=1.0e-9)
        assert above[t] == pytest.approx(at_hundred[t], abs=1.0e-9)
    assert max(below.values()) == pytest.approx(0.0, abs=1.0e-9)
    assert min(above.values()) == pytest.approx(100.0, abs=1.0e-9)


def test_lag_config_rejects_shapes_and_non_unity_dc_gain() -> None:
    """配置校验拦住维度错误、上下限倒置与直流增益不为 1 的矩阵组。"""

    with pytest.raises(ValueError, match="state_matrix"):
        EngineLagConfig(state_matrix=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="input_matrix"):
        EngineLagConfig(input_matrix=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="output_matrix"):
        EngineLagConfig(output_matrix=[1.0])
    with pytest.raises(ValueError, match="lower"):
        EngineLagConfig(command_lower=50.0, command_upper=40.0)
    with pytest.raises(ValueError, match="DC gain"):
        EngineLagConfig(output_matrix=[-0.1474, 0.9])
    with pytest.raises(ValueError):
        EngineLagConfig(time_scale=0.0)
