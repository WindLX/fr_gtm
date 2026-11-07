"""度量生成的 GTM 源码规模与 JAX 编译各阶段耗时。

主题：用 JAX 后端编译一次 GTM，分别测量连续函数与离散函数的跟踪（make_jaxpr）、
lower 与 XLA 编译耗时，并统计生成源码的字节数、行数与最大行宽。

运行：

    cd models/fr_gtm && just benchmark-jax-compile

预期结果：打印一份 JSON 指标，并写入 `models/fr_gtm/build/jax_compile_benchmark.json`；
数字随本机 JAX/XLA 版本与硬件变化，只用于开发者横向对比。

Note:
    这是显式的开发者基准，属于常规示例与 CI 测试之外的独立入口。结果依赖本机的
    JAX/XLA 版本与硬件，因此 JSON 只写入 `build/` 下，永不作为冻结的
    参考数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import fly_ruler_core as fr

from fr_gtm import GTMConfig, GTMSystem, init


def main() -> None:
    """配置 JAX 运行时，编译模型并度量构建、跟踪、lower 与编译耗时。"""
    root = Path(__file__).resolve().parent.parent
    # 运行时策略要在导入 jax 之前设置，才能固定 CPU 平台并启用 64 位浮点。
    fr.configure_jax_runtime(
        fr.JaxRuntimePolicy(
            platforms=("cpu",),
            preallocate=False,
            enable_x64=True,
        )
    )
    import jax
    import jax.numpy as jnp

    # cache_policy="rebuild" 强制重新生成，model_build_s 才表示真实的编译耗时。
    cfg = GTMConfig.fullscale()
    builder = GTMSystem.build("Root", cfg)
    started = perf_counter()
    model = fr.Compiler(
        fr.CompilerSpec(
            integrator="rk4",
            dt=0.01,
            artifact_dir=root / "build" / "flyruler",
            cache_policy="rebuild",
        )
    ).compile(builder, target=fr.targets.jax)
    model_build_s = perf_counter() - started

    # 四个运行向量统一转成 jnp 数组，作为跟踪与 lower 的位置参数。
    x0, xd0, u0, p0 = init(
        model,
        cfg,
        position=(0.0, 0.0, 0.0),
        airdata=(0.0, 0.0, 50.0),
        command=(0.0, 0.0, 0.0, 0.0),
    )
    x, xd, u, p = (jnp.asarray(value) for value in (x0, xd0, u0, p0))
    arguments = (0.0, x, xd, u, p)

    # 连续函数分三阶段度量：make_jaxpr 只跟踪，lower 生成 StableHLO，compile 触发 XLA。
    started = perf_counter()
    continuous_jaxpr = jax.make_jaxpr(model.continuous_fn)(*arguments)
    continuous_trace_s = perf_counter() - started
    started = perf_counter()
    continuous_lowered = jax.jit(model.continuous_fn).lower(*arguments)
    continuous_lower_s = perf_counter() - started
    started = perf_counter()
    continuous_lowered.compile()
    continuous_compile_s = perf_counter() - started

    # 离散函数用同一套参数重复上述三阶段度量，便于与连续路径对比。
    started = perf_counter()
    discrete_jaxpr = jax.make_jaxpr(model.discrete_fn)(*arguments)
    discrete_trace_s = perf_counter() - started
    started = perf_counter()
    discrete_lowered = jax.jit(model.discrete_fn).lower(*arguments)
    discrete_lower_s = perf_counter() - started
    started = perf_counter()
    discrete_lowered.compile()
    discrete_compile_s = perf_counter() - started

    # 生成源码文本用于统计规模；方程数为各函数的 jaxpr 等式条数。
    source = model.inspect.sources["jax"]
    result = {
        "model_build_s": model_build_s,
        "source_bytes": len(source.encode()),
        "source_lines": len(source.splitlines()),
        "source_max_line": max(map(len, source.splitlines())),
        "continuous_equations": len(continuous_jaxpr.jaxpr.eqns),
        "continuous_trace_s": continuous_trace_s,
        "continuous_lower_s": continuous_lower_s,
        "continuous_compile_s": continuous_compile_s,
        "continuous_stablehlo_bytes": len(
            str(continuous_lowered.compiler_ir()).encode()
        ),
        "discrete_equations": len(discrete_jaxpr.jaxpr.eqns),
        "discrete_trace_s": discrete_trace_s,
        "discrete_lower_s": discrete_lower_s,
        "discrete_compile_s": discrete_compile_s,
        "discrete_stablehlo_bytes": len(str(discrete_lowered.compiler_ir()).encode()),
    }
    # 结果写入 build/ 下的 JSON，父目录不存在时先创建。
    output = root / "build" / "jax_compile_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
