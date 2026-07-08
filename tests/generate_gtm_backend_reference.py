"""Generate frozen backend trajectory reference data from ``gtm_ad``.

The regular test suite reads the generated ``gtm_backend_trajectory.npz`` and
does not import the external ``gtm_ad`` checkout.  Re-run this script manually
when the GTM reference model or the backend-parity scenario changes:

    uv run python tests/generate_gtm_backend_reference.py

Set ``GTM_AD_PATH=/path/to/gtm_ad`` when the reference checkout is not at the
default sibling project path used by the FlyRuler workspace.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import casadi as ca
import numpy as np

DT = 1.0 / 120.0
SAMPLE_STEPS = np.array([0, 120, 240, 360], dtype=np.int64)
COMMAND = np.array([10.0, 0.0, 0.0, 40.0], dtype=float)


def _default_gtm_ad_path() -> Path:
    return Path(__file__).resolve().parents[4] / "GTM" / "gtm_ad"


def _load_gtm_ad() -> tuple[object, object, object, object, object]:
    root = Path(os.environ.get("GTM_AD_PATH", _default_gtm_ad_path()))
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(
            f"gtm_ad source tree not found at {src}; set GTM_AD_PATH explicitly"
        )
    sys.path.insert(0, str(src))

    from gtm_ad.casadi import CasadiBackend
    from gtm_ad.core import GTMConfig, GTMConstraints, GTMModel
    from gtm_ad.core.constants import DDIM, NDIM, PDIM, UDIM, XDIM

    return (
        CasadiBackend,
        GTMConfig,
        GTMConstraints,
        GTMModel,
        (
            XDIM,
            UDIM,
            PDIM,
            DDIM,
            NDIM,
        ),
    )


def _as_vector(value: object) -> np.ndarray:
    if hasattr(value, "full"):
        return np.asarray(value.full(), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def main() -> None:
    CasadiBackend, GTMConfig, GTMConstraints, GTMModel, dims = _load_gtm_ad()
    xdim, udim, pdim, ddim, ndim = dims

    model = GTMModel.build(
        CasadiBackend,
        GTMConfig(),
        GTMConstraints.build(),
        solver_type="rk4",
    )

    x_sym = ca.SX.sym("x", xdim)
    u_sym = ca.SX.sym("u", udim)
    p_sym = ca.SX.sym("p", pdim)
    d_sym = ca.SX.sym("d", ddim)
    n_sym = ca.SX.sym("n", ndim)
    step_fn = ca.Function(
        "gtm_backend_reference_step",
        [x_sym, u_sym, p_sym, d_sym, n_sym],
        [
            model.x_dist_fn(x_sym, u_sym, p_sym, d_sym, DT),
            model.y_fn(x_sym, u_sym, p_sym, d_sym, n_sym),
        ],
    )

    x = np.zeros(xdim, dtype=float)
    x[2] = -1000.0
    x[3] = 100.0
    x[6] = 1.0
    x[13:17] = COMMAND

    u = COMMAND.copy()
    p = np.zeros(pdim, dtype=float)
    d = np.zeros(ddim, dtype=float)
    n = np.zeros(ndim, dtype=float)

    x_samples: list[np.ndarray] = []
    x_next_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []

    for step in range(int(SAMPLE_STEPS[-1]) + 1):
        if step in SAMPLE_STEPS:
            x_next, y = step_fn(x, u, p, d, n)
            x_samples.append(x.copy())
            x_next_samples.append(_as_vector(x_next))
            y_samples.append(_as_vector(y))
        if step == int(SAMPLE_STEPS[-1]):
            break
        x_next, _ = step_fn(x, u, p, d, n)
        x = _as_vector(x_next)

    out = Path(__file__).resolve().parent / "data" / "gtm_backend_trajectory.npz"
    np.savez_compressed(
        out,
        source=np.array(["gtm_ad"], dtype=object),
        dt=np.array(DT, dtype=float),
        sample_steps=SAMPLE_STEPS,
        command=COMMAND,
        x0=x_samples[0],
        u=u,
        p=p,
        d=d,
        n=n,
        x_samples=np.vstack(x_samples),
        x_next_samples=np.vstack(x_next_samples),
        y_samples=np.vstack(y_samples),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
