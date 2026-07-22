"""Measure generated GTM source size and the stages of JAX compilation.

This is an explicit developer benchmark rather than a normal example or CI
test. Results depend on the local JAX/XLA version and machine, so the JSON is
written below ``build/`` and is never treated as frozen reference data.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import cast

import fly_ruler_core as fr

from fr_gtm import GTMConfig, GTMSystem, build_initial_state


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fr.configure_jax_runtime(
        fr.JaxRuntimePolicy(
            platforms=("cpu",),
            preallocate=False,
            enable_x64=True,
        )
    )
    import jax
    import jax.numpy as jnp

    builder = cast(fr.PyIrBuilder, GTMSystem(GTMConfig.default(), name="Root"))
    started = perf_counter()
    model = fr.Compiler(
        fr.CompilerSpec(
            backend=fr.backends.jax,
            integrator="rk4",
            dt=0.01,
            artifact_dir=root / "build" / "flyruler",
            cache_policy="rebuild",
        )
    ).compile(builder)
    model_build_s = perf_counter() - started

    x = jnp.asarray(build_initial_state(model))
    xd = jnp.asarray(model.xd.build())
    u = jnp.asarray(model.u.build())
    p = jnp.asarray(model.p.build())
    arguments = (0.0, x, xd, u, p)

    started = perf_counter()
    continuous_jaxpr = jax.make_jaxpr(model.continuous_fn)(*arguments)
    continuous_trace_s = perf_counter() - started
    started = perf_counter()
    continuous_lowered = jax.jit(model.continuous_fn).lower(*arguments)
    continuous_lower_s = perf_counter() - started
    started = perf_counter()
    continuous_lowered.compile()
    continuous_compile_s = perf_counter() - started

    started = perf_counter()
    discrete_jaxpr = jax.make_jaxpr(model.discrete_fn)(*arguments)
    discrete_trace_s = perf_counter() - started
    started = perf_counter()
    discrete_lowered = jax.jit(model.discrete_fn).lower(*arguments)
    discrete_lower_s = perf_counter() - started
    started = perf_counter()
    discrete_lowered.compile()
    discrete_compile_s = perf_counter() - started

    source = model.module.source_code
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
    output = root / "build" / "jax_compile_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
