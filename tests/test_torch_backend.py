"""真实飞机资源、SO(3) 和作动器在 PyTorch 上的端点与梯度回归。"""

from __future__ import annotations

from typing import Literal

import fly_ruler_core as fr
import numpy as np
import pytest

from fr_gtm import GTMConfig, GTMSystem, build_initial_state

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gtm_torch_endpoint_batch_and_gradient(
    device: str, dtype: Literal["float32", "float64"]
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cfg = GTMConfig.default()
    compiler = fr.Compiler(fr.CompilerSpec(integrator="rk4", cache_policy="off"))
    reference = compiler.compile(GTMSystem.build("Root", cfg))
    model = compiler.compile(
        GTMSystem.build("Root", cfg),
        target=fr.targets.TorchTarget(device=device, dtype=dtype),
    )
    initial = build_initial_state(reference, cfg, throttle=40.0).reshape(-1)
    x = model.backend.array(initial, (model.n_x,)).requires_grad_()
    u = model.u.build({"Root::u": [1.0, -0.5, 0.3, 40.0]}).requires_grad_()
    p = model.p.build()
    xd = model.xd.build()
    endpoint = model.flow_fn(0.0, x, xd, u, p, 0.0037)
    expected = reference.flow_fn(
        0.0,
        initial,
        reference.xd.build(),
        model.to_numpy(u),
        reference.p.build(),
        0.0037,
    )
    tolerance = 3e-5 if dtype == "float32" else 1e-9
    np.testing.assert_allclose(
        model.to_numpy(endpoint),
        reference.to_numpy(expected).reshape(-1),
        rtol=tolerance,
        atol=tolerance,
    )
    x_gradient, u_gradient = torch.autograd.grad(endpoint.sum(), (x, u))
    assert bool(torch.isfinite(x_gradient).all())
    assert bool(torch.isfinite(u_gradient).all())
    batch = model.batched(batch_size=4)
    xb = x.detach().expand(4, -1)
    durations = model.backend.array([[0.0], [0.0037], [0.005], [0.01]], (4, 1))
    outputs = batch.flow_fn(0.0, xb, batch.xd.build(), u.detach(), p, durations)
    for row, duration in enumerate((0.0, 0.0037, 0.005, 0.01)):
        scalar = model.flow_fn(0.0, x.detach(), xd, u.detach(), p, duration)
        torch.testing.assert_close(outputs[row], scalar, rtol=tolerance, atol=tolerance)
    quat = batch.x.read(outputs, by="path")["Root.eom::quat"]
    torch.testing.assert_close(
        torch.linalg.vector_norm(quat, dim=1),
        torch.ones(4, device=device, dtype=quat.dtype),
        rtol=tolerance,
        atol=tolerance,
    )
