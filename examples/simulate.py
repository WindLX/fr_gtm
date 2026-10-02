"""运行一段标称短仿真并用标准 Recorder 保存结果。"""

from pathlib import Path

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.runtime.record import Recorder

from fr_gtm import GTMConfig, GTMSystem


def main() -> None:
    builder = GTMSystem.build("Root", GTMConfig.default())
    model = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=0.01)).compile(
        builder, target=fr.targets.numpy
    )
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
    xd = model.xd.build()
    output_dir = Path("results") / "gtm_nominal"
    recorder = Recorder(model, output_dir)
    for step in range(100):
        x, xd, y, effects = model.discrete_fn(
            step * 0.01,
            x,
            xd,
            u,
            p,
        )
        recorder.record(
            step,
            (step + 1) * 0.01,
            x=x,
            xd=xd,
            y=y,
            u=u,
            p=p,
            effect=effects,
        )
    recorder.save("nominal")
    final = model.y.read(y, by="path")
    print(f"saved: {output_dir / 'nominal.parquet'}")
    print(f"final altitude: {final['Root::altitude'][0]:.3f} m")
    print(f"final airspeed: {final['Root::V'][0]:.3f} m/s")


if __name__ == "__main__":
    main()
