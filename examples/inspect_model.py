"""导出 GTM 拓扑、生成源码与向量布局。

主题：编译一次 NumPy 模型，打印编译产物目录、系统 IR、后端源码与布局文件的路径，
再给出各向量空间的大小与完整路径清单，便于对照文档检查模型接口。

运行：

    cd models/fr_gtm && just example-inspect

预期结果：打印 artifact、topology、source、layout、cache 与六个尺寸，随后按 x、u、y
分别列出完整路径。cache 首次多为 miss，产物已存在时为 hit。

说明：本示例只读编译产物、不推进仿真；默认全尺寸 NORMAL 构型下 n_x=26、n_xdot=25、
n_xd=0、n_u=29、n_p=33、n_y=144，其中 xdot 比 x 少一维是因为四元数按 SO3 切空间求导。
"""

from pathlib import Path

import fly_ruler_core as fr

from fr_gtm import GTMConfig, GTMSystem


def main() -> None:
    """编译 NumPy 目标并打印产物路径、向量尺寸与路径清单。"""
    # builder 只描述系统连接关系，编译阶段才把它降为后端代码与布局。
    root = Path(__file__).resolve().parent.parent
    builder = GTMSystem.build("Root", GTMConfig.default())
    # artifact_dir 指定产物落盘位置；compile 返回的模型携带产物元数据。
    model = fr.Compiler(
        fr.CompilerSpec(
            integrator="rk4",
            dt=0.01,
            artifact_dir=root / "build" / "flyruler",
        )
    ).compile(builder, target=fr.targets.numpy)
    assert model.artifact is not None

    # artifact.path 是产物根目录，file() 按类别取具体文件；cache_hit 说明是否复用缓存。
    print(f"artifact: {model.artifact.path}")
    print(f"topology: {model.artifact.file(fr.ArtifactFile.SYSTEM_IR)}")
    print(f"source:   {model.artifact.file(fr.ArtifactFile.BACKEND_SOURCE)}")
    print(f"layout:   {model.artifact.file(fr.ArtifactFile.LAYOUT)}")
    print(f"cache:    {'hit' if model.artifact.cache_hit else 'miss'}")
    print(
        f"sizes: x={model.n_x}, xdot={model.n_xdot}, "
        f"xd={model.n_xd}, u={model.n_u}, p={model.n_p}, y={model.n_y}"
    )
    # keys 给出每个向量空间的顶层完整路径；状态、输入与输出的地址都以此为准。
    for space, paths in (("x", model.x.keys), ("u", model.u.keys), ("y", model.y.keys)):
        print(f"\n{space} paths:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
