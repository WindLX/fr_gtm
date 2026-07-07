"""Run a short nominal simulation and save it with the standard Recorder."""

from pathlib import Path

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.runtime import Recorder

from fr_gtm import GTMConfig, GTMSystem


def main() -> None:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    model = fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.numpy, integrator="rk4", dt=0.01)
    ).compile(builder)
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -1000.0],
            "Root.eom::vel_body": [100.0, 0.0, 0.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [0.0, 0.0, 0.0, 40.0],
        }
    )
    u = model.u.build(
        {
            "Root::u": [0.0, 0.0, 0.0, 40.0],
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    p = model.p.build()
    output_dir = Path("results") / "gtm_nominal"
    recorder = Recorder(model, output_dir)
    for step in range(100):
        xs, ys, _ = model.discrete_fn(
            step * 0.01,
            x,
            model.xd.build(),
            u,
            p,
            n_steps=1,
        )
        x = np.asarray(xs[0])
        recorder.record(step=step, time=(step + 1) * 0.01, x=x, y=ys[0], u=u, p=p)
    recorder.save("nominal")
    final = model.y.read(np.asarray(ys[0]))
    print(f"saved: {output_dir / 'nominal.parquet'}")
    print(f"final altitude: {final['altitude'][0]:.3f} m")
    print(f"final airspeed: {final['V'][0]:.3f} m/s")


if __name__ == "__main__":
    main()
