"""NASA HyKal 原始顺序分支的独立标量对照与符号后端回归。"""

import math
from collections.abc import Callable

import casadi as ca
import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm.components.aero import _hykal

SPAN, CHORD, SPEED = 12.0, 2.0, 80.0


def _nasa_hykal(query: np.ndarray, trig: np.ndarray) -> np.ndarray:
    """逐句转录 stateflow.xml 的判据/覆盖顺序；只把 ft/knots 换成现有 SI。"""
    _, _, p, q, r = map(float, query)
    # 后端的角度换算可差一个 ulp；近90度小分母会放大它。只共享三角值，
    # 分支、覆盖、剩余量和尺度仍由本标量实现独立计算。
    ca_, sa, cb, sb = map(float, trig)
    whichcase = 0
    if ca_ * cb != 0 and p != 0:
        omega = p / (ca_ * cb)
        residual = r - omega * sa * cb
        if not (abs(residual) > abs(r) or residual * r < 0):
            whichcase = 1
    if sa * cb != 0 and r != 0:
        omega = r / (sa * cb)
        residual = p - omega * ca_ * cb
        if not (abs(residual) > abs(p) or residual * p < 0):
            whichcase = 2
    if whichcase == 1:
        omega = p / (ca_ * cb)
        osc = [0.0, q - omega * sb, r - omega * sa * cb]
    elif whichcase == 2:
        omega = r / (sa * cb)
        osc = [p - omega * ca_ * cb, q - omega * sb, 0.0]
    else:
        omega = 0.0
        osc = [p, q, r]
    return np.asarray(
        np.array([omega, *osc]) * np.array([SPAN, SPAN, CHORD, SPAN]) / (2 * SPEED),
        dtype=float,
    )


def _builder() -> fr.PyIrBuilder:
    @fr.system
    def Rates() -> None:
        @fr.block
        def Split() -> None:
            query = fr.input("query", (5,))
            omega, oscillatory = _hykal(
                query[0], query[1], fr.math.vector([SPEED])[0], query[2:], SPAN, CHORD
            )
            fr.output(
                "split",
                fr.math.concatenate(
                    [
                        fr.math.vector([omega]),
                        oscillatory,
                        fr.math.vector(
                            [
                                fr.math.cos(fr.math.radians(query[0])),
                                fr.math.sin(fr.math.radians(query[0])),
                                fr.math.cos(fr.math.radians(query[1])),
                                fr.math.sin(fr.math.radians(query[1])),
                            ]
                        ),
                    ]
                ),
            )

        query = fr.input("query", (5,))
        split = Split("split")
        query >> split.query
        fr.export(split=split.split)

    return Rates.build("Root")


def _model_evaluator[T](
    model: fr.CompiledModel[T],
) -> Callable[[np.ndarray], np.ndarray]:
    def run(values: np.ndarray) -> np.ndarray:
        result = model.continuous_fn(
            0.0,
            model.x.build(),
            model.xd.build(),
            model.u.build({"Root::query": values}),
            model.p.default(),
        )
        return model.to_numpy(result.y).reshape(-1)

    return run


@pytest.fixture(params=["numpy", "casadi_sx", "casadi_mx", "native", "jax"])
def evaluate(request: pytest.FixtureRequest) -> Callable[[np.ndarray], np.ndarray]:
    compiler = fr.Compiler(fr.CompilerSpec(cache_policy="off"))
    backend = request.param
    if backend == "jax":
        pytest.importorskip("jax")
        return _model_evaluator(compiler.compile(_builder(), target=fr.targets.jax))
    if backend == "numpy":
        return _model_evaluator(compiler.compile(_builder(), target=fr.targets.numpy))
    symbolic = compiler.compile(_builder(), target=fr.targets.casadi)
    if backend == "native":
        return _model_evaluator(
            compiler.convert(symbolic, target=fr.targets.NativeTarget())
        )
    if backend == "casadi_sx":
        sx_query = ca.SX.sym("query", 5)
        sx_result = symbolic.sx.continuous_fn(
            0.0, ca.SX.zeros(0), ca.SX.zeros(0), sx_query, ca.SX.zeros(0)
        )
        fn = ca.Function("hykal", [sx_query], [sx_result.y])
    else:
        mx_query = ca.MX.sym("query", 5)
        mx_result = symbolic.mx.continuous_fn(
            0.0, ca.MX.zeros(0), ca.MX.zeros(0), mx_query, ca.MX.zeros(0)
        )
        fn = ca.Function("hykal", [mx_query], [mx_result.y])
    return lambda values: np.asarray(fn(values)).reshape(-1)


def test_hykal_matches_nasa_at_rate_threshold_and_seam(
    evaluate: Callable[[np.ndarray], np.ndarray],
) -> None:
    cases = [
        [20, 10, 1, 0.2, 1],  # P wins
        [80, 10, 0.5, -0.2, 1],  # R wins
        [20, 10, 1, 0.2, -1],  # neither wins
        [0, 0, 0, 0, 0],
        [0, 10, 1, 0.3, 0],  # exact zero R denominator, unselected 0/0
        [0, 10, 0, 0.3, 1],  # zero P rate and zero R denominator
        [90, 0, 1, 0.3, 0.5],  # cos(pi/2) is nonzero, not epsilon-zero
        [45, 10, 1e-200, 0.2, 2e-200],  # both valid (underflow): R overwrites P
    ]
    for rate in (np.nextafter(1e-10, 0), 1e-10, np.nextafter(1e-10, np.inf), 1e-14):
        for sign in (-1, 1):
            cases.extend(
                [
                    [0, 10, sign * rate, 0.2, 0],
                    [45, 10, sign * 2 * rate, 0.2, sign * rate],
                    [45, 10, 0, 0.2, sign * rate],
                ]
            )
    # Slanted seam p=e_Vx*k, r=e_Vz*k, both sides and both rate signs.
    for sign in (-1, 1):
        for factor in (1 - 1e-8, 1.0, 1 + 1e-8):
            a, b = math.radians(20), math.radians(10)
            cases.append(
                [
                    20,
                    10,
                    sign * math.cos(a) * math.cos(b),
                    0.2,
                    sign * math.sin(a) * math.cos(b) * factor,
                ]
            )
    for case in cases:
        query = np.asarray(case, dtype=float)
        result = evaluate(query)
        actual, expected = result[:4], _nasa_hykal(query, result[4:])
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual, expected, rtol=3e-14, atol=0, err_msg=str(case)
        )
        ca_, sa, cb, sb = result[4:]
        flow = np.array([ca_ * cb, sb, sa * cb])
        unscaled = actual * (2 * SPEED) / [SPAN, SPAN, CHORD, SPAN]
        np.testing.assert_allclose(
            unscaled[0] * flow + unscaled[1:], query[2:], rtol=3e-14, atol=1e-215
        )


def test_hykal_keeps_nonzero_direction_denominators(
    evaluate: Callable[[np.ndarray], np.ndarray],
) -> None:
    for magnitude in (0.5e-10, 1e-10, 2e-10):
        for sign in (-1, 1):
            for alpha in (
                math.degrees(math.asin(sign * magnitude)),
                math.degrees(math.acos(sign * magnitude)),
            ):
                a = math.radians(alpha)
                cx, cz = math.cos(a), math.sin(a)
                p, r = (2 * cx, cz) if abs(cz) < abs(cx) else (cx, 2 * cz)
                query = np.array([alpha, 0, p, 0.2, r])
                result = evaluate(query)
                np.testing.assert_allclose(
                    result[:4], _nasa_hykal(query, result[4:]), rtol=3e-14, atol=0
                )


def test_symbolic_zero_denominator_has_finite_values_and_jacobian() -> None:
    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        _builder(), target=fr.targets.casadi
    )
    query = ca.SX.sym("query", 5)
    result = model.sx.continuous_fn(
        0.0, ca.SX.zeros(0), ca.SX.zeros(0), query, ca.SX.zeros(0)
    )
    fn = ca.Function(
        "zero_denominator", [query], [result.y, ca.jacobian(result.y, query)]
    )
    for values in ([0, 0, 0, 0, 0], [0, 10, 1, 0.3, 0], [0, 10, 0, 0.3, 1]):
        y, jacobian = fn(values)
        assert np.isfinite(np.asarray(y)).all()
        assert np.isfinite(np.asarray(jacobian)).all()
