"""导出 GTM 可恢复产物与向量布局。

主题：编译一次 CasADi 目标，保存完整产物并通过 reader 打印模型描述，
再给出各向量空间的大小与完整路径清单，便于对照文档检查模型接口。

运行：

    cd models/fr_gtm && just example-inspect

预期结果：打印布局与产物信息，并启动 Graph WebUI；Ctrl+C 退出。
使用 --no-graph 只打印 artifact、description 与六个尺寸，按 x、u、y
分别列出完整路径。

说明：本示例生成并检查产物，不推进仿真；默认全尺寸 NORMAL 构型下 n_x=25、n_xdot=24、
n_xd=0、n_u=25、n_p=33、n_y=34，其中 xdot 比 x 少一维是因为四元数按 SO3 切空间求导。
"""

import argparse
from pathlib import Path

from fly_ruler_core import (
    ModelSettings,
    OperationContext,
    artifacts,
    compiler,
    runtime,
    targets,
)
from fly_ruler_std.runtime import graph as fr_graph

from fr_gtm import GTMConfig, GTMSystem


def main(*, graph: bool = True, port: int = 18001) -> None:
    """打印编译布局；默认阻塞打开 Graph WebUI，Ctrl+C 关闭服务。"""
    # builder 只描述系统连接关系，编译阶段才把它降为后端代码与布局。
    root = Path(__file__).resolve().parent.parent
    builder = GTMSystem.build("Root", GTMConfig.fullscale())
    # Store 是显式可复用缓存；保存目录是可独立恢复的交付产物。
    context = OperationContext(store=root / "build" / "store")
    program = compiler.prepare(
        builder, settings=ModelSettings(integrator="rk4", dt=0.01), context=context
    )
    target = compiler.compile(program, target=targets.casadi, context=context)
    saved = artifacts.save(target, output_dir=root / "build" / "model", overwrite=True)
    model = runtime.bind(target)
    print(f"artifact: {saved}")
    print(f"description: {artifacts.open(saved).describe()}")
    print(
        f"sizes: x={model.n_x}, xdot={model.n_xdot}, "
        f"xd={model.n_xd}, u={model.n_u}, p={model.n_p}, y={model.n_y}"
    )
    # keys 给出每个向量空间的顶层完整路径；状态、输入与输出的地址都以此为准。
    for space, paths in (("x", model.x.keys), ("u", model.u.keys), ("y", model.y.keys)):
        print(f"\n{space} paths:")
        for path in paths:
            print(f"  {path}")

    if graph:
        fr_graph.serve(model, port=port, open_browser=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-graph", action="store_true", help="只打印布局，不启动 WebUI"
    )
    parser.add_argument("--port", type=int, default=18001)
    args = parser.parse_args()
    main(graph=not args.no_graph, port=args.port)
