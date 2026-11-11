"""真实GTM：大角速率时线性延拓不延续三次外推的爆炸增长。"""

from typing import Any

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core.runtime.instance import RuntimeInstance
from pydantic import ValidationError

from fr_gtm import GTMConfig, GTMSystem


def _model(backend: str, cfg: GTMConfig) -> RuntimeInstance[Any]:
    compiler_settings = ModelSettings(integrator="euler", dt=0.05)
    builder = GTMSystem.build("Root", cfg)
    if backend == "numpy":
        return runtime.bind(
            compiler.compile(
                compiler.prepare(builder, settings=compiler_settings),
                target=targets.numpy,
            )
        )
    if backend == "jax":
        return runtime.bind(
            compiler.compile(
                compiler.prepare(builder, settings=compiler_settings),
                target=targets.jax,
            )
        )
    return runtime.bind(
        compiler.compile(
            compiler.prepare(builder, settings=compiler_settings), target=targets.casadi
        )
    )


@pytest.mark.parametrize("backend", ["numpy", "casadi", "jax"])
def test_high_rate_linear_extension(backend: str) -> None:
    cfg = GTMConfig.fullscale()
    cfg.aero.interpolation_method = "cubic"
    cfg.aero.extrapolation_method = "linear"
    m = _model(backend, cfg)
    x = m.x.build(
        {
            "Root.eom::pos_ned": [7.89, 5.99, -2488.6],
            "Root.eom::vel_body": [46.13226, -7.75136, 44.85422],
            "Root.eom::quat": [0.8523097, -0.2087347, 0.4794317, 0.01197339],
            "Root.eom::omega_body": [65.20225, 0.3323816, -11.64703],
            "Root.actuator::position": [-0.4064, 12.217, 9.1685],
            "Root.engine.lag::lag_state": cfg.engine.lag.trim_state(38.319),
        }
    )
    u = m.u.build({"Root::u": [-0.4014, 12.2777, 9.173, 38.358]})
    result = m.continuous_fn(0.0, x, m.xd.build(), u, m.p.build())
    angular = np.asarray(m.xdot.read(result.xdot, by="path")["Root.eom::omega_body"])
    assert np.isfinite(angular).all() and np.max(abs(angular)) < 1e5
    next_result = m.discrete_fn(0.0, x, m.xd.build(), u, m.p.build())
    assert np.isfinite(np.asarray(next_result.x_next)).all()
    assert np.isfinite(np.asarray(next_result.y)).all()


def test_extrapolation_is_explicit_and_validated() -> None:
    cfg = GTMConfig.fullscale()
    assert cfg.aero.interpolation_method == "linear"
    assert cfg.aero.extrapolation_method == "polynomial"
    with pytest.raises(ValidationError):
        type(cfg.aero).model_validate(
            {
                **cfg.aero.model_dump(),
                "tables": cfg.aero.tables,
                "extrapolation_method": "bad",
            }
        )
