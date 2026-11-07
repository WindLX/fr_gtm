"""HyKal 的分支选择、流向轴重构与无量纲尺度回归。"""

import fly_ruler_core as fr
import numpy as np

from fr_gtm.components.aero import _hykal


def test_hykal_decomposes_and_reconstructs_relative_body_rates() -> None:
    span, chord, speed = 12.0, 2.0, 80.0

    @fr.system
    def Rates() -> None:
        @fr.block
        def Split() -> None:
            query = fr.input("query", (5,))
            omega, oscillatory = _hykal(
                query[0], query[1], fr.math.vector([speed])[0], query[2:], span, chord
            )
            fr.output("omega", omega)
            fr.output("oscillatory", oscillatory)

        query = fr.input("query", (5,))
        split = Split("split")
        query >> split.query
        fr.export(omega=split.omega, oscillatory=split.oscillatory)

    model = fr.Compiler(fr.CompilerSpec(cache_policy="off")).compile(
        Rates.build("Root"), target=fr.targets.numpy
    )
    # (alpha_deg,beta_deg,p,q,r,所选分支)：p优先、r候选、反号回退、全零、流向x退化。
    cases = [
        (20, 10, 1, 0.2, 1, 1),
        (80, 10, 0.5, -0.2, 1, 2),
        (20, 10, 1, 0.2, -1, 0),
        (0, 0, 0, 0, 0, 0),
        (90, 0, 1, 0.3, 0.5, 2),
    ]
    for alpha, beta, p, q, r, branch in cases:
        u = model.u.build({"Root::query": [alpha, beta, p, q, r]})
        result = model.continuous_fn(
            0.0, model.x.build(), model.xd.build(), u, model.p.build()
        )
        outputs = model.y.read(result.y, by="path")
        omega = float(outputs["Root::omega"][0]) * 2 * speed / span
        residual = outputs["Root::oscillatory"] * np.array(
            [2 * speed / span, 2 * speed / chord, 2 * speed / span]
        )
        a, b = np.deg2rad([alpha, beta])
        flow = np.array([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)])
        np.testing.assert_allclose(omega * flow + residual, [p, q, r], atol=1e-12)
        if branch == 1:
            np.testing.assert_allclose(residual[0], 0, atol=1e-12)
            assert abs(residual[2]) <= abs(r) and residual[2] * r >= 0
        elif branch == 2:
            np.testing.assert_allclose(residual[2], 0, atol=1e-12)
            assert abs(residual[0]) <= abs(p) and residual[0] * p >= 0
        else:
            assert omega == 0
            np.testing.assert_allclose(residual, [p, q, r], atol=1e-12)
