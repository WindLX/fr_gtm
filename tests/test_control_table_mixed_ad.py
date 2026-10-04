"""舵面表的效能必须保留对气动状态的混合导数。"""

from __future__ import annotations

import casadi as ca
import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig


def _definition(grids: list[fr.NumpyArray], values: fr.NumpyArray) -> fr.PyIrBuilder:
    @fr.block
    def Loads() -> None:
        aero = fr.input("aero", len(grids) - 1)
        delta = fr.input("delta", 1)
        table = fr.resource.interpolator("control", grids, values)
        fr.output("loads", table(*[aero[i] for i in range(len(grids) - 1)], delta[0]))

    @fr.system
    def System() -> None:
        feedback = fr.input("feedback", 3)
        loads = Loads("loads")
        # Fixed pitch, changing flight path angle: alpha = theta - gamma.
        coords: list[fr.Signal | float] = [
            fr.math.degrees(0.11 - feedback[0]),
            fr.math.degrees(feedback[1]),
        ]
        if len(grids) == 4:
            coords.append(0.0)  # The elevator table's fixed stabilizer axis.
        fr.math.vector(coords) >> loads.aero
        feedback[2:3] >> loads.delta
        gain = fr.jacobian(output=loads.loads, wrt=loads.delta)
        fr.export(effectiveness=gain)

    return System.build("Root")


def test_real_control_tables_mixed_ad_matches_cell_local_difference() -> None:
    control = GTMConfig.default().aero.tables.control_surfaces
    tables = [
        (
            [
                control.alpha_elevator,
                control.beta_elevator,
                control.stabilizer,
                control.elevator,
            ],
            control.elevator_data,
        ),
        (
            [control.alpha_aileron, control.beta_aileron, control.aileron],
            control.aileron_data,
        ),
        (
            [control.alpha_rudder, control.beta_rudder, control.rudder],
            control.rudder_data,
        ),
    ]
    for grids, values in tables:
        model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
            _definition(grids, values)
        )
        feedback = ca.MX.sym("feedback", 3)
        result = model.mx.continuous_fn(
            0.0, model.mx.x.build(), model.mx.xd.build(), feedback, model.mx.p.build()
        )
        gain = model.mx.y.read(result.y, by="path")["Root::effectiveness"]
        evaluate = ca.Function("gain", [feedback], [ca.vec(gain)])
        differentiate = ca.Function(
            "mixed", [feedback], [ca.jacobian(ca.vec(gain), feedback)]
        )
        for gamma in (0.04, 0.047, 0.031):
            point = np.array([gamma, 0.03, 2.31])
            ad = np.asarray(differentiate(point))[:, :2]
            finite = np.empty_like(ad)
            for axis in range(2):
                step = np.eye(3)[axis] * 1e-6
                finite[:, axis] = np.asarray(
                    (evaluate(point + step) - evaluate(point - step)) / 2e-6
                ).reshape(-1)
            assert np.max(np.abs(finite[:, 0])) > 1e-7
            np.testing.assert_allclose(ad, finite, rtol=1e-6, atol=1e-9)
