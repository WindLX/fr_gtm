"""独立气动配置、载荷几何与声明元数据回归。"""

from typing import assert_type

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_core import api as fr
from fly_ruler_core.runtime.instance import RuntimeInstance

from fr_gtm import GTMConfig, Mode
from fr_gtm.components.aero import Aero, AeroConfig


def test_aero_owns_one_flat_configuration() -> None:
    assert_type(Aero.Config, type[AeroConfig])
    assert Aero.Config is AeroConfig
    cfg = GTMConfig.subscale().aero
    assert "tables" not in cfg.model_dump()
    data = {**cfg.model_dump(), "tables": cfg.tables}
    restored = AeroConfig.model_validate(data)
    assert restored.tables.basic_airframe.cx_cz.shape == (32, 27, 10, 2)
    assert restored.b == cfg.b
    with pytest.raises(ValueError, match="force_moment"):
        AeroConfig.model_validate({**data, "force_moment": {"Sref": 99.0}})


@pytest.mark.parametrize("backend", ["numpy", "casadi"])
def test_standalone_aero_uses_its_own_geometry(backend: str) -> None:
    # Deliberately unrelated to either aircraft scale: no parent config is built
    # or passed to Aero. Flat fields control both rate scaling and load geometry.
    cfg = AeroConfig.load(
        S=42.0,
        b=18.0,
        Cbar=2.5,
        cg_pos=(-2.0, 0.1, -0.3),
        ref_cg=(-2.2, 0.0, -0.1),
        mode=Mode.HALF_FLAP_GEAR_UP,
    )

    @fr.system
    def AeroOnly() -> None:
        aero = Aero("aero", cfg)
        fr.math.vector([1000.0]) >> aero.altitude
        fr.math.vector([100.0, 2.0, 5.0]) >> aero.velocity
        fr.math.vector([0.1, -0.05, 0.03]) >> aero.omega_body
        fr.math.vector([0.0, 0.0, 0.0]) >> aero.wind_uvw
        fr.math.vector([0.0, 0.0, 0.0]) >> aero.wind_omega
        fr.math.vector([-2.0, 1.5, -1.0]) >> aero.surfaces
        assert aero.force.description
        assert aero.moment.description
        fr.export(
            force=aero.force,
            moment=aero.moment,
            qbar=aero.qbar,
        )

    compiler_settings = ModelSettings()
    builder = AeroOnly.build("Root")

    def verify[T](model: RuntimeInstance[T]) -> None:
        _, _, y, effects = model.continuous_fn(
            0.0, model.x.build(), model.xd.build(), model.u.build(), model.p.build()
        )
        outputs = model.y.read(y, by="path")
        coefficients = model.to_numpy(
            model.effects.read(effects, by="path")["Root.aero::coefficients"]
        ).reshape(-1)
        qbar = model.to_numpy(outputs["Root::qbar"]).item()
        force = coefficients[:3] * qbar * cfg.S
        moment = coefficients[3:] * qbar * cfg.S * [cfg.b, cfg.Cbar, cfg.b]
        moment += np.cross(np.asarray(cfg.ref_cg) - cfg.cg_pos, force)
        np.testing.assert_allclose(
            model.to_numpy(outputs["Root::force"]).reshape(-1), force, atol=1e-10
        )
        np.testing.assert_allclose(
            model.to_numpy(outputs["Root::moment"]).reshape(-1), moment, atol=1e-10
        )

    if backend == "numpy":
        verify(
            runtime.bind(
                compiler.compile(
                    compiler.prepare(builder, settings=compiler_settings),
                    target=targets.numpy,
                )
            )
        )
    else:
        verify(
            runtime.bind(
                compiler.compile(
                    compiler.prepare(builder, settings=compiler_settings),
                    target=targets.casadi,
                )
            )
        )


def test_asymmetric_control_arm_ratios_use_span_normalization() -> None:
    """非对称力臂检查四段求和；通过控制系数入口覆盖局部矩阵，不依赖嵌套私有函数。"""
    from fr_gtm.components.aero import _control_coefficients

    cfg = GTMConfig.subscale().aero
    cfg.control_arm_ratios = (0.01, 0.02, 0.03, 0.04)
    table = cfg.tables.control_surfaces
    i, j = len(table.alpha_elevator) // 2, len(table.beta_elevator) // 2
    k = int(np.flatnonzero(table.stabilizer == 0.0)[0])
    zero = int(np.flatnonzero(table.elevator == 0.0)[0])
    delta = 0.25 * (
        table.elevator_data[i, j, k, -1] - table.elevator_data[i, j, k, zero]
    )

    @fr.system
    def ElevatorMoment() -> None:
        @fr.block
        def Coefficients() -> None:
            c = _control_coefficients(
                cfg,
                fr.math.vector([float(table.alpha_elevator[i])])[0],
                fr.math.vector([float(table.beta_elevator[j])])[0],
                fr.math.vector([float(table.elevator[-1]), 0.0, 0.0]),
                fr.math.ones(3),
                fr.math.zeros(6),
                fr.math.zeros(6),
            )
            fr.output("coefficients", c)

        block = Coefficients("coefficients")
        fr.export(coefficients=block.coefficients)

    model = runtime.bind(
        compiler.compile(
            compiler.prepare(ElevatorMoment.build("Root"), settings=ModelSettings()),
            target=targets.numpy,
        )
    )
    result = model.continuous_fn(
        0.0, model.x.build(), model.xd.build(), model.u.build(), model.p.build()
    )
    c = model.y.read(result.y, by="path")["Root::coefficients"]
    arm_sum = sum(cfg.control_arm_ratios)
    np.testing.assert_allclose(
        c[[3, 5]], [arm_sum * delta[1], -arm_sum * delta[0]], atol=1e-12
    )
    np.testing.assert_array_equal(
        GTMConfig.fullscale().aero.control_arm_ratios,
        GTMConfig.subscale().aero.control_arm_ratios,
    )
