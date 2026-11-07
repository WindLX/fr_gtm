"""真实三轴力/力矩平衡、可选条件、阶数及生产布局映射回归。"""

from dataclasses import replace

import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm import (
    GTMConfig,
    GTMPaths,
    GTMSystem,
    JetCatThrottleLagConfig,
    init,
    read_observation,
    trim,
)


@pytest.mark.parametrize(
    "fullscale,second_order,gamma,psi,phi",
    [
        (True, False, 0.0, 0.0, 0.0),
        (False, False, 0.0, 0.0, 0.0),
        (True, True, 0.0, 10.0, 0.0),
        (True, False, 3.0, 20.0, 2.0),
        (True, False, None, 0.0, 0.0),
        (True, False, 0.0, None, 0.0),
        (True, False, 0.0, 0.0, None),
    ],
)
def test_complete_flight_trim(
    fullscale: bool,
    second_order: bool,
    gamma: float | None,
    psi: float | None,
    phi: float | None,
) -> None:
    cfg = GTMConfig.fullscale() if fullscale else GTMConfig.subscale()
    if second_order:
        cfg.engine = cfg.engine.model_copy(update={"lag": JetCatThrottleLagConfig()})
    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        GTMSystem.build("Root", cfg), target=fr.targets.numpy
    )
    tas = 100.0 if fullscale else 40.0
    result = trim(model, cfg, altitude=1000.0, tas=tas, gamma=gamma, psi=psi, phi=phi)
    assert result.success, result.solver.stats
    assert max(result.residuals.values()) < 1e-6
    evaluated = model.continuous_fn(0.0, result.x, result.xd, result.u, result.p)
    paths = GTMPaths()
    rates = model.xdot.read(evaluated.xdot, by="path")
    np.testing.assert_allclose(rates[paths.x.vel_body], 0.0, atol=1e-6)
    np.testing.assert_allclose(rates[paths.x.omega_body], 0.0, atol=1e-6)
    # 没有把正常前进错误约束为静止。
    assert np.linalg.norm(rates[paths.x.pos_ned]) > tas * 0.5
    obs = read_observation(model, evaluated.y)
    assert obs.tas_mps == pytest.approx(tas, abs=1e-6)
    assert obs.altitude_m == 1000.0
    if gamma is not None:
        assert obs.gamma_deg == pytest.approx(gamma, abs=1e-6)
    if psi is not None:
        assert obs.euler_deg[2] == pytest.approx(psi, abs=1e-6)
    if phi is not None:
        assert obs.euler_deg[0] == pytest.approx(phi, abs=1e-6)
    np.testing.assert_allclose(
        model.x.read(result.x, by="path")[paths.x.engine_lag],
        cfg.engine.lag.trim_state(obs.throttle_percent),
        atol=1e-6,
    )
    assert not replace(result, solver=result.solver._replace(success=False)).success


def test_trim_uses_provided_parameters_guess_and_casadi_layout() -> None:
    cfg = GTMConfig.fullscale()
    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        GTMSystem.build("Root", cfg), target=fr.targets.casadi
    )
    x, xd, u, p = init(
        model,
        cfg,
        airdata=(5.0, 0.0, 100.0),
        wind=(8.0, 35.0, 0.0),
        uncertainty={"weight": 0.05},
    )
    result = trim(model, cfg, altitude=1000.0, tas=100.0, p=p, guess=(x, xd, u))
    assert result.success, result.solver.stats
    np.testing.assert_array_equal(model.to_numpy(result.p), model.to_numpy(p))
    assert model.to_numpy(result.x).size == model.n_x
    with pytest.raises(ValueError, match="TAS"):
        trim(model, cfg, altitude=1000.0, tas=0.0)
