"""显式 cubic 的真实 GTM 气动表和后端一致性。"""

import casadi as ca
import fly_ruler_core as fr
import numpy as np
import pytest
from pydantic import ValidationError

from fr_gtm import GTMConfig, GTMSystem


def test_default_remains_linear_and_invalid_method_rejected() -> None:
    cfg = GTMConfig.fullscale()
    assert cfg.aero.interpolation_method == "linear"
    with pytest.raises(ValidationError):
        type(cfg.aero).model_validate(
            {
                **cfg.aero.model_dump(),
                "tables": cfg.aero.tables,
                "interpolation_method": "nearest",
            }
        )


def test_real_elevator_cubic_public_nested_ad_matches_independent_spline() -> None:
    from scipy.interpolate import CubicSpline

    control = GTMConfig.fullscale().aero.tables.control_surfaces
    grids = [
        control.alpha_elevator,
        control.beta_elevator,
        control.stabilizer,
        control.elevator,
    ]
    values = control.elevator_data

    @fr.block
    def Table() -> None:
        q = fr.input("q", 4)
        table = fr.resource.interpolator(
            "elevator", grids, values, method="cubic", n_out=3
        )
        fr.output("loads", table(*[q[i] for i in range(4)]))

    @fr.system
    def System() -> None:
        q = fr.input("q", 4)
        table = Table("table")
        q >> table.q
        fr.export(loads=table.loads, gain=fr.jacobian(output=table.loads, wrt=table.q))

    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        System.build("Root")
    )
    q = ca.SX.sym("q", 4)
    e = ca.SX.zeros(0, 1)
    gain = model.sx.y.read(model.sx.continuous_fn(0.0, e, e, q, e).y, by="path")[
        "Root::gain"
    ]
    function = ca.Function("mixed", [q], [ca.jacobian(ca.vec(gain), q)])
    point = np.array([6.3, 1.7, 0.0, 2.31])
    actual = np.asarray(function(point))
    for i in range(4):
        for j in range(4):
            ref = values
            for axis, (grid, value) in enumerate(zip(grids, point, strict=True)):
                ref = CubicSpline(
                    grid, ref, axis=0, bc_type="not-a-knot", extrapolate=True
                )(value, nu=int(axis == i) + int(axis == j))
            np.testing.assert_allclose(
                actual[i * 3 : i * 3 + 3, j], ref, rtol=1e-8, atol=1e-10
            )


@pytest.fixture(scope="module")
def cubic_case() -> tuple[GTMConfig, fr.PyIrBuilder, fr.Compiler, fr.NumpyArray]:
    """三种候选后端共用一次 NumPy 构图、编译和参考轨迹。"""
    cfg = GTMConfig.fullscale()
    cfg.aero.interpolation_method = "cubic"
    builder = GTMSystem.build("Root", cfg)
    assert isinstance(builder, fr.PyIrBuilder)
    compiler = fr.Compiler(
        fr.CompilerSpec(cache_policy="off", integrator="rk4", dt=0.01)
    )
    expected = simulate(compiler.compile(builder, target=fr.targets.numpy), cfg)
    return cfg, builder, compiler, expected


@pytest.mark.parametrize("backend", ["casadi", "jax", "torch"])
def test_cubic_gtm_short_trajectory_matches_numpy(
    backend: str,
    cubic_case: tuple[GTMConfig, fr.PyIrBuilder, fr.Compiler, fr.NumpyArray],
) -> None:
    if backend in ("jax", "torch"):
        pytest.importorskip(backend)
    cfg, builder, compiler, expected = cubic_case
    if backend == "casadi":
        actual = simulate(compiler.compile(builder, target=fr.targets.casadi), cfg)
    elif backend == "jax":
        actual = simulate(compiler.compile(builder, target=fr.targets.jax), cfg)
    else:
        actual = simulate(
            compiler.compile(builder, target=fr.targets.TorchTarget(dtype="float64")),
            cfg,
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def simulate[ArrayT](model: fr.CompiledModel[ArrayT], cfg: GTMConfig) -> fr.NumpyArray:
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -1000.0],
            "Root.eom::vel_body": [100.0, 0.0, 0.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.engine.lag::lag_state": cfg.engine.lag.trim_state(40.0),
        }
    )
    xd = model.xd.build()
    u = model.u.build({"Root::u": [0.0, 0.0, 0.0, 40.0]})
    p = model.p.default()
    history = []
    for step in range(10):
        x, xd, _, _ = model.discrete_fn(step * 0.01, x, xd, u, p)
        history.append(model.to_numpy(x).ravel())
    return np.stack(history)
