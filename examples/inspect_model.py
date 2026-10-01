"""导出 GTM 拓扑、生成源码与向量布局。"""

from pathlib import Path
from typing import cast

import fly_ruler_core as fr

from fr_gtm import GTMConfig, GTMSystem


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    builder = cast(fr.PyIrBuilder, GTMSystem(GTMConfig.default(), name="Root"))
    model = fr.Compiler(
        fr.CompilerSpec(
            backend=fr.backends.numpy,
            integrator="rk4",
            dt=0.01,
            artifact_dir=root / "build" / "flyruler",
        )
    ).compile(builder)
    assert model.artifact is not None

    print(f"artifact: {model.artifact.path}")
    print(f"topology: {model.artifact.file(fr.ArtifactFile.SYSTEM_IR)}")
    print(f"source:   {model.artifact.file(fr.ArtifactFile.BACKEND_SOURCE)}")
    print(f"layout:   {model.artifact.file(fr.ArtifactFile.LAYOUT)}")
    print(f"cache:    {'hit' if model.artifact.cache_hit else 'miss'}")
    print(
        f"sizes: x={model.n_x}, xdot={model.n_xdot}, "
        f"xd={model.n_xd}, u={model.n_u}, p={model.n_p}, y={model.n_y}"
    )
    for space, paths in (("x", model.x.keys), ("u", model.u.keys), ("y", model.y.keys)):
        print(f"\n{space} paths:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
