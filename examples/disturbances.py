"""Evaluate steady wind, Dryden turbulence, uncertainty, and sensor noise."""

import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem


def main() -> None:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    model = fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.numpy, integrator="euler", dt=0.01)
    ).compile(builder)
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -500.0],
            "Root.eom::vel_body": [90.0, 0.0, 5.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [-2.0, 0.0, 0.0, 42.0],
            "Root.wind.dryden::u_gw": [0.2],
            "Root.wind.dryden::v_gw": [-0.1, 0.05],
            "Root.wind.dryden::w_gw": [0.1, -0.05],
        }
    )
    u = model.u.build(
        {
            "Root::u": [-2.0, 0.0, 0.0, 42.0],
            "Root::d": [0.4, -0.2, 0.3, -0.1],
            "Root::n": np.linspace(-1.0, 1.0, 21),
        }
    )
    p = model.p.build(
        {
            "Root.wind.params::wind_speed": [12.0],
            "Root.wind.params::wind_direction": [35.0],
            "Root.wind.params::turbulence_level": [3.0],
            "Root.mass::weight_unc": [0.03],
            "Root.aero::c_static_unc": np.linspace(-0.03, 0.03, 6),
        }
    )
    _, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, p)
    output = model.y.read(y)
    print(f"body wind:  {output['uvw']}")
    print(f"alpha/beta: {output['alpha'][0]:.3f}, {output['beta'][0]:.3f} deg")
    print(f"clean y[:5]: {output['clean_y'][:5]}")
    print(f"noisy y[:5]: {output['y'][:5]}")


if __name__ == "__main__":
    main()
