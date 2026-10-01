"""从外部 `gtm_ad` 生成冻结的数值参考用例。

Examples:
    ```bash
    uv run --with /path/to/gtm_ad python scripts/generate_reference.py \\
        --gtm-ad /path/to/gtm_ad
    ```
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Case:
    name: str
    fullscale: bool
    mode: str
    x: np.ndarray
    u: np.ndarray
    p: np.ndarray
    d: np.ndarray
    n: np.ndarray


def _state(
    *,
    altitude: float,
    speed: float,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rates_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    actuators: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 30.0),
    dryden: np.ndarray | None = None,
) -> np.ndarray:
    from gtm_ad.core.utils.quaternion import euler_to_quat_np

    alpha, beta = np.radians([alpha_deg, beta_deg])
    velocity = np.array(
        [
            speed * np.cos(alpha) * np.cos(beta),
            speed * np.sin(beta),
            speed * np.sin(alpha) * np.cos(beta),
        ]
    )
    quat = euler_to_quat_np(*np.radians(euler_deg))
    return np.concatenate(
        [
            np.array([0.0, 0.0, -altitude]),
            velocity,
            quat,
            np.radians(rates_deg),
            np.asarray(actuators, dtype=float),
            np.zeros(8) if dryden is None else dryden,
        ]
    )


def _cases() -> list[Case]:
    modes = [
        "normal",
        "0.5_flap_gear_up",
        "1_flap_gear_up",
        "1_flap_gear_down",
    ]
    cases: list[Case] = []
    for fullscale in (True, False):
        for index, mode in enumerate(modes):
            actuators = (-2.0 + index, 1.0, -0.5, 38.0)
            x = _state(
                altitude=1200.0 if fullscale else 250.0,
                speed=105.0 if fullscale else 28.0,
                alpha_deg=4.0 + index,
                beta_deg=(-1.0) ** index,
                euler_deg=(2.0, 3.0, 5.0),
                rates_deg=(1.0, -1.5, 0.75),
                actuators=actuators,
            )
            cases.append(
                Case(
                    name=f"{'full' if fullscale else 'scale'}_{mode}",
                    fullscale=fullscale,
                    mode=mode,
                    x=x,
                    u=np.asarray(actuators),
                    p=np.zeros(31),
                    d=np.zeros(4),
                    n=np.zeros(21),
                )
            )

    dryden = np.array([0.2, -0.1, 0.05, 0.15, -0.08, 0.03, -0.02, 0.01])
    for name, altitude, turbulence in (
        ("wind_low", 300.0, 3.0),
        ("wind_high", 3500.0, 5.0),
    ):
        p = np.zeros(31)
        p[:3] = [12.0, 35.0, turbulence]
        x = _state(
            altitude=altitude,
            speed=95.0,
            alpha_deg=6.0,
            beta_deg=2.0,
            euler_deg=(4.0, -2.0, 20.0),
            rates_deg=(2.0, 1.0, -1.0),
            actuators=(-3.0, 2.0, 1.0, 45.0),
            dryden=dryden,
        )
        cases.append(
            Case(
                name=name,
                fullscale=True,
                mode="normal",
                x=x,
                u=np.array([-2.5, 1.5, 0.5, 46.0]),
                p=p,
                d=np.array([0.4, -0.2, 0.3, -0.1]),
                n=np.zeros(21),
            )
        )

    p = np.zeros(31)
    p[3:] = np.linspace(-0.08, 0.08, 28)
    n = np.linspace(-1.0, 1.0, 21)
    x = _state(
        altitude=1800.0,
        speed=115.0,
        alpha_deg=8.0,
        beta_deg=-3.0,
        euler_deg=(-5.0, 6.0, 15.0),
        rates_deg=(3.0, -2.0, 1.0),
        actuators=(-4.0, -1.0, 2.0, 52.0),
    )
    cases.append(
        Case(
            name="uncertainty_sensor",
            fullscale=True,
            mode="normal",
            x=x,
            u=np.array([-3.5, -0.5, 1.5, 53.0]),
            p=p,
            d=np.zeros(4),
            n=n,
        )
    )
    return cases


def _clean_output(context: object) -> np.ndarray:
    return np.concatenate(
        [
            np.atleast_1d(np.asarray(context.kinematics.gamma)),
            np.atleast_1d(np.asarray(context.aerodynamics.alpha)),
            np.atleast_1d(np.asarray(context.aerodynamics.beta)),
            np.atleast_1d(np.asarray(context.aerodynamics.V)),
            np.atleast_1d(np.asarray(context.kinematics.altitude)),
            np.degrees(np.asarray(context.kinematics.euler)),
            np.degrees(np.asarray(context.raw.x.rigid_body.angular_velocity)),
            np.atleast_1d(np.asarray(context.fm.load_factor[2])),
            np.atleast_1d(np.asarray(context.aerodynamics.qbar)),
            np.asarray(context.raw.x.actuators),
            np.asarray(context.raw.u.commands),
        ]
    )


def generate(gtm_ad: Path, output: Path) -> None:
    sys.path.insert(0, str(gtm_ad / "src"))
    from gtm_ad.core.config import GTMConfig, Mode
    from gtm_ad.core.constraints import GTMConstraints
    from gtm_ad.core.gtm import GTMModel
    from gtm_ad.jax.backend import JaxBackend

    results: dict[str, list[np.ndarray] | list[str] | list[bool]] = {
        "name": [],
        "fullscale": [],
        "mode": [],
        "x": [],
        "u": [],
        "p": [],
        "d": [],
        "n": [],
        "xdot": [],
        "y": [],
        "clean_y": [],
        "force_engine": [],
        "force_aero": [],
        "force_gravity": [],
        "moment_engine": [],
        "moment_aero": [],
        "wind_uvw": [],
        "wind_omega": [],
    }
    for case in _cases():
        config = GTMConfig(
            is_fullscale=case.fullscale,
            mode=Mode(case.mode),
        )
        model = GTMModel.build(JaxBackend, config, GTMConstraints.build())
        x, u, p, d, n = map(
            jnp.asarray, (case.x, case.u, case.p, case.d, case.n)
        )
        context, _ = model.fm_fn(x, u, p, d)
        values = {
            "xdot": np.asarray(model.x_dot_fn(x, u, p, d)),
            "y": np.asarray(model.y_fn(x, u, p, d, n)),
            "clean_y": _clean_output(context),
            "force_engine": np.asarray(context.fm.force.engine),
            "force_aero": np.asarray(context.fm.force.aero),
            "force_gravity": np.asarray(context.fm.force.gravity),
            "moment_engine": np.asarray(context.fm.moment.engine),
            "moment_aero": np.asarray(context.fm.moment.aero),
            "wind_uvw": np.asarray(context.wind.uvw),
            "wind_omega": np.asarray(context.wind.omega),
        }
        results["name"].append(case.name)
        results["fullscale"].append(case.fullscale)
        results["mode"].append(case.mode)
        for key in ("x", "u", "p", "d", "n"):
            results[key].append(getattr(case, key))
        for key, value in values.items():
            results[key].append(value)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{
            key: np.asarray(value)
            for key, value in results.items()
        },
    )
    print(f"Wrote {len(results['name'])} reference cases to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtm-ad", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gtm_reference.npz"),
    )
    args = parser.parse_args()
    generate(args.gtm_ad.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
