from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core.api.types import PyIrBuilder
from fly_ruler_core.runtime.instance import RuntimeInstance

from fr_gtm import GTMConfig, GTMSystem, JetCatThrottleLagConfig, Mode

REFERENCE = np.load(Path(__file__).parent / "data" / "gtm_reference.npz")
CASE_NAMES = REFERENCE["name"].tolist()
# 冻结参考把油门建为一阶作动器状态，本模型把油门换成二阶滞后；两侧共用同一份
# 滞后动态，因此这里显式选用原二阶配置把参考的油门状态换算成滞后状态。
LAG = JetCatThrottleLagConfig()


@lru_cache(maxsize=None)
def _model(fullscale: bool, mode: str) -> RuntimeInstance[Any]:
    cfg = (GTMConfig.fullscale if fullscale else GTMConfig.subscale)(mode=Mode(mode))
    cfg.engine = cfg.engine.model_copy(update={"lag": LAG})
    # gtm_ad/core/noise/sensor.py 把 throttle_act 的 noise_std 固定为 0。
    # 冻结参考采用该标定；当前 GTM 默认油门噪声在独立测试中验证，不改冻结数据。
    cfg.sensor.throttle.noise_std = 0.0
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


def _state(model: RuntimeInstance[Any], x: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.x.build(
            {
                "Root.eom::pos_ned": x[0:3],
                "Root.eom::vel_body": x[3:6],
                "Root.eom::quat": x[6:10],
                "Root.eom::omega_body": x[10:13],
                "Root.actuator::position": x[13:16],
                # 参考的第 17 个状态是有效油门；本模型的内部状态由它做配平反解，
                # 这样有效油门初值与参考一致，力与力矩在 t=0 完全对得上。
                "Root.engine.lag::lag_state": np.asarray(LAG.trim_state(x[16])),
                "Root.wind.dryden::u_gw": x[17:18],
                "Root.wind.dryden::v_gw": x[18:20],
                "Root.wind.dryden::w_gw": x[20:22],
                "Root.wind.dryden::p_gw": x[22:23],
                "Root.wind.dryden::q_gw": x[23:24],
                "Root.wind.dryden::r_gw": x[24:25],
            }
        ),
        dtype=float,
    )


def _parameters(model: RuntimeInstance[Any], p: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.p.build(
            {
                "Root.wind.params::wind_speed": p[0:1],
                # Frozen gtm_ad parameters store engineering wind direction in
                # degrees; FlyRuler's corrected public parameter is radians.
                "Root.wind.params::wind_direction": np.deg2rad(p[1:2]),
                "Root.wind.params::turbulence_level": p[2:3],
                "Root.mass::weight_unc": p[3:4],
                "Root.mass::inertia_unc": p[4:10],
                "Root.aero::c_static_unc": p[10:16],
                "Root.aero::c_e_unc": p[16:19],
                "Root.aero::c_a_unc": p[19:25],
                "Root.aero::c_r_unc": p[25:31],
            }
        ),
        dtype=float,
    )


def _reference_xdot_head(
    derivatives: dict[str, np.ndarray],
    quaternion_rate: np.ndarray,
) -> np.ndarray:
    """拼出参考布局中油门通道之前的 16 个导数分量。"""

    return np.concatenate(
        [
            derivatives["Root.eom::pos_ned"],
            derivatives["Root.eom::vel_body"],
            quaternion_rate,
            derivatives["Root.eom::omega_body"],
            derivatives["Root.actuator::position"],
        ]
    )


def _reference_xdot_tail(derivatives: dict[str, np.ndarray]) -> np.ndarray:
    """拼出参考布局中油门通道之后的 8 个 Dryden 导数分量。"""

    return np.concatenate(
        [
            derivatives["Root.wind.dryden::u_gw"],
            derivatives["Root.wind.dryden::v_gw"],
            derivatives["Root.wind.dryden::w_gw"],
            derivatives["Root.wind.dryden::p_gw"],
            derivatives["Root.wind.dryden::q_gw"],
            derivatives["Root.wind.dryden::r_gw"],
        ]
    )


@pytest.mark.parametrize("index", range(len(CASE_NAMES)), ids=CASE_NAMES)
def test_matches_frozen_gtm_ad_reference(index: int) -> None:
    model = _model(
        bool(REFERENCE["fullscale"][index]),
        str(REFERENCE["mode"][index]),
    )
    x = _state(model, REFERENCE["x"][index])
    u = model.u.build(
        {
            "Root::u": REFERENCE["u"][index],
            "Root::d": REFERENCE["d"][index],
            "Root::n": REFERENCE["n"][index][:17],
        }
    )
    xdot, _, y, effects = model.continuous_fn(
        0.0,
        x,
        model.xd.build(),
        u,
        _parameters(model, REFERENCE["p"][index]),
    )
    derivatives = model.xdot.read(xdot, by="path")
    outputs = {**model.y.read(y), **model.effects.read(effects)}

    # 冻结参考的第 16 槽是油门作动器（一阶）的导数，本模型是二阶滞后的有效油门
    # 导数，两者定义不同，因此这一槽不参与比对，改用配平反解的解析式独立核对：
    # 由 A x_e = -B x[16] 得 xdot_e = B (u[3] - x[16]) / c。
    # 冻结文件保存四维四元数导数；运行时 SO3 导数为三维切向量。
    # 在测试边界用实际四元数和切向量重构，不向模型添加仅供参考比对的 effects。
    state = model.x.read(x, by="path")
    quat = state["Root.eom::quat"]
    omega = derivatives["Root.eom::quat"]
    quaternion_rate = 0.5 * np.concatenate(
        [[-np.dot(quat[1:], omega)], quat[0] * omega + np.cross(quat[1:], omega)]
    )
    head = _reference_xdot_head(derivatives, quaternion_rate)
    tail = _reference_xdot_tail(derivatives)
    xdot_reference = REFERENCE["xdot"][index]
    np.testing.assert_allclose(head, xdot_reference[0:16], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(tail, xdot_reference[17:25], rtol=1e-8, atol=1e-9)
    throttle_delta = float(REFERENCE["u"][index][3] - REFERENCE["x"][index][16])
    np.testing.assert_allclose(
        derivatives["Root.engine.lag::lag_state"],
        [
            LAG.input_matrix[0] * throttle_delta / LAG.time_scale,
            LAG.input_matrix[1] * throttle_delta / LAG.time_scale,
        ],
        rtol=1e-8,
        atol=1e-9,
    )
    # 可由状态导数/载荷重构的重力、以及子模块已有的风分量不再重复导出。
    cfg = (
        GTMConfig.fullscale
        if bool(REFERENCE["fullscale"][index])
        else GTMConfig.subscale
    )()
    mass = cfg.aircraft.w0 * (1.0 + REFERENCE["p"][index][3]) / 9.80665
    force_gravity = (
        mass
        * (
            derivatives["Root.eom::vel_body"]
            + np.cross(state["Root.eom::omega_body"], state["Root.eom::vel_body"])
        )
        - outputs["force_aero"]
        - outputs["force_engine"]
    )
    wind_effects = model.effects.read(effects, by="path")
    wind = wind_effects["Root.wind::steady"] + wind_effects["Root.wind::gust"]
    for actual, expected in (
        (outputs["y"], REFERENCE["y"][index][:17]),
        (outputs["clean_y"], REFERENCE["clean_y"][index][:17]),
        (outputs["force_engine"], REFERENCE["force_engine"][index]),
        (outputs["force_aero"], REFERENCE["force_aero"][index]),
        (force_gravity, REFERENCE["force_gravity"][index]),
        (outputs["moment_engine"], REFERENCE["moment_engine"][index]),
        (outputs["moment_aero"], REFERENCE["moment_aero"][index]),
        (wind, REFERENCE["wind_uvw"][index]),
        (outputs["wind_omega"], REFERENCE["wind_omega"][index]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


def test_current_throttle_noise_calibration() -> None:
    """当前默认油门噪声与无噪声冻结标定分别验证。"""
    from fr_gtm import build_input, init

    cfg = GTMConfig.fullscale()
    assert cfg.sensor.throttle.noise_std == 1.0
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(GTMSystem.build("Root", cfg), settings=ModelSettings()),
            target=targets.numpy,
        )
    )
    x, xd, _, p = init(model, cfg, command=(0, 0, 0, 52))
    noise = np.zeros(17)
    noise[16] = 0.6
    u = build_input(model, (0, 0, 0, 52), n=noise)
    result = model.continuous_fn(0, x, xd, u, p)
    values = model.y.read(result.y, by="path")
    assert values["Root::clean_y"][16] == pytest.approx(52)
    assert values["Root::y"][16] == pytest.approx(52.6)
