"""在干净环境验收 GTM 发行包、公开 SDK 来源与气动资源。"""

import sys
from importlib.metadata import distribution, version
from pathlib import Path

import fly_ruler_core as fr
import fly_ruler_std
import numpy as np

import fr_gtm


def main() -> None:
    """拒绝源码遮蔽，读取两尺度配置并运行十步标称仿真。"""
    prefix = Path(sys.prefix).resolve()
    for module in (fr, fly_ruler_std, fr_gtm):
        filename = module.__file__
        assert filename is not None
        path = Path(filename).resolve()
        assert path.is_relative_to(prefix), f"导入不来自验收环境：{path}"
    assert distribution("fly-ruler").read_text("direct_url.json") is None
    assert distribution("fr-gtm").read_text("direct_url.json") is not None
    for factory in (fr_gtm.GTMConfig.fullscale, fr_gtm.GTMConfig.subscale):
        for mode in fr_gtm.Mode:
            factory(mode=mode)
    cfg = fr_gtm.GTMConfig.fullscale()
    model = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=0.01)).compile(
        fr_gtm.GTMSystem.build("Root", cfg), target=fr.targets.numpy
    )
    x, xd, u, p = fr_gtm.init(model, cfg, position=(0, 0, -1000), airdata=(0, 0, 100))
    for step in range(10):
        result = model.discrete_fn(step * 0.01, x, xd, u, p)
        x, xd = result.x_next, result.xd_next
        assert np.isfinite(x).all()
        assert np.isfinite(result.y).all()
    observation = fr_gtm.read_observation(model, result.y)
    print(
        f"fr-gtm={version('fr-gtm')}, fly-ruler={version('fly-ruler')}, "
        f"10步仿真高度={observation.altitude_m:.3f} m"
    )


if __name__ == "__main__":
    main()
