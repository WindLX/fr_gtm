"""后端原生初始化在 JAX jit/vmap/正反向微分下保持物理含义。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from fly_ruler_core import ModelSettings, compiler, runtime, targets

from fr_gtm import (
    GTMConfig,
    GTMPaths,
    GTMSystem,
    Initializer,
    JetCatThrottleLagConfig,
    build_input,
    init,
)

if TYPE_CHECKING:
    import jax


@pytest.mark.parametrize("second_order", [False, True])
def test_native_initializer_jit_vmap_and_differentiation(second_order: bool) -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    cfg = GTMConfig.fullscale()
    if second_order:
        cfg.engine = cfg.engine.model_copy(update={"lag": JetCatThrottleLagConfig()})
    compiler_settings = ModelSettings()
    builder = GTMSystem.build("Root", cfg)
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(builder, settings=compiler_settings), target=targets.jax
        )
    )
    initializer = Initializer(model, cfg)
    data = jnp.array([5.0, 1.0, 100.0])
    wind = jnp.array([8.0, 35.0, 2.0])
    euler = jnp.array([4.0, 8.0, 20.0])

    def construct(v: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return initializer(
            airdata=v,
            wind=wind,
            euler=euler,
            command=jnp.array([-2.0, 1.0, 0.0, 45.0]),
            uncertainty={"weight": v[0] * 0.001},
        )

    result = jax.jit(construct)(data)
    assert all(isinstance(value, jax.Array) for value in result)
    reference = runtime.bind(
        compiler.compile(
            compiler.prepare(builder, settings=compiler_settings), target=targets.numpy
        )
    )
    expected = init(
        reference,
        cfg,
        airdata=(5, 1, 100),
        wind=(8, 35, 2),
        euler=(4, 8, 20),
        command=(-2, 1, 0, 45),
        uncertainty={"weight": 0.005},
    )
    for actual, wanted in zip(result, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-11, atol=1e-10)
    batch = jax.jit(jax.vmap(construct))(jnp.stack([data, data.at[2].set(110.0)]))
    assert batch[0].shape == (2, model.n_x)
    for first, single in zip(batch, result, strict=True):
        np.testing.assert_allclose(first[0], single, atol=1e-10)

    def state(v: jax.Array) -> jax.Array:
        return construct(v)[0]

    forward = jax.jacfwd(state)(data)
    reverse = jax.jacrev(state)(data)
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()
    np.testing.assert_allclose(forward, reverse, atol=1e-10)
    h = 1e-4
    fd = (state(data.at[2].add(h)) - state(data.at[2].add(-h))) / (2 * h)
    np.testing.assert_allclose(forward[:, 2], fd, atol=1e-8)

    # 姿态也必须保留导数，而不是只支持来流数组。
    def attitude(angles: jax.Array) -> jax.Array:
        x, _, _, _ = initializer(euler=angles)
        return model.x.read(x, by="path")[GTMPaths().x.quat]

    assert np.isfinite(jax.jacfwd(attitude)(euler)).all()
    assert np.linalg.norm(jax.jacfwd(attitude)(euler)) > 0
    # PRNG key 由调用方 split，噪声不经过 NumPy Generator 或宿主转换。
    d_key, n_key = jax.random.split(jax.random.key(123))
    u = jax.jit(
        lambda d, n: build_input(model, jnp.array([0.0, 0.0, 0.0, 40.0]), d=d, n=n)
    )(jax.random.normal(d_key, (4,)) * jnp.pi, jax.random.normal(n_key, (17,)))
    assert isinstance(u, jax.Array) and u.shape == (model.n_u,)
