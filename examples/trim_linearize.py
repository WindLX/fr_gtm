"""用完整变量路径对 GTM 做部分配平与线性化。

下方的配平求解固定一个开环刚体工作点，再让作动器与 Dryden 内部状态收敛，
使其状态导数为零。这样既保持示例确定且快速，又仍会走完整的
`TrimSolver`/`LinearizationSolver` 路径，与更大飞行条件的配平研究所用 API
一致。
"""

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.runtime.linearize import LinearizationSolver, LinearizationSpec
from fly_ruler_std.runtime.trim import TrimSolver, TrimSpec

from fr_gtm import GTMConfig, GTMSystem


def _array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def main() -> None:
    builder = GTMSystem(GTMConfig.default(), name="Root")
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    model = fr.Compiler(
        fr.CompilerSpec(backend=fr.backends.casadi, integrator="rk4", dt=0.01)
    ).compile(builder)

    trim = TrimSolver(
        model,
        TrimSpec(
            param_value={
                path: np.asarray(value, dtype=float).reshape(-1)
                for path, value in model.p.read(model.p.default(), by="path").items()
            },
            state_value={
                "Root.eom::pos_ned": _array([0.0, 0.0, -1000.0]),
                "Root.eom::vel_body": _array([100.0, 0.0, 0.0]),
                "Root.eom::quat": _array([1.0, 0.0, 0.0, 0.0]),
                "Root.eom::omega_body": _array([0.0, 0.0, 0.0]),
            },
            input_value={
                "Root::u": _array([0.0, 0.0, 0.0, 40.0]),
                "Root::d": _array([0.0, 0.0, 0.0, 0.0]),
                "Root::n": np.zeros(21),
            },
            state_derivative={
                "Root.actuator::position": _array([0.0, 0.0, 0.0, 0.0]),
                "Root.wind.dryden::u_gw": _array([0.0]),
                "Root.wind.dryden::v_gw": _array([0.0, 0.0]),
                "Root.wind.dryden::w_gw": _array([0.0, 0.0]),
                "Root.wind.dryden::p_gw": _array([0.0]),
                "Root.wind.dryden::q_gw": _array([0.0]),
                "Root.wind.dryden::r_gw": _array([0.0]),
            },
            initial_guess={
                "Root.eom::pos_ned": _array([0.0, 0.0, -1000.0]),
                "Root.eom::vel_body": _array([100.0, 0.0, 0.0]),
                "Root.eom::quat": _array([1.0, 0.0, 0.0, 0.0]),
                "Root.eom::omega_body": _array([0.0, 0.0, 0.0]),
                "Root.actuator::position": _array([-2.0, 0.0, 0.0, 40.0]),
                "Root.wind.dryden::u_gw": _array([0.0]),
                "Root.wind.dryden::v_gw": _array([0.0, 0.0]),
                "Root.wind.dryden::w_gw": _array([0.0, 0.0]),
                "Root.wind.dryden::p_gw": _array([0.0]),
                "Root.wind.dryden::q_gw": _array([0.0]),
                "Root.wind.dryden::r_gw": _array([0.0]),
                "Root::u": _array([0.0, 0.0, 0.0, 40.0]),
                "Root::d": _array([0.0, 0.0, 0.0, 0.0]),
                "Root::n": np.zeros(21),
            },
        ),
    ).solve(verbose=False)
    if not trim.success:
        raise RuntimeError(f"trim failed: {trim.stats}")
    constrained = [
        "Root.actuator::position",
        "Root.wind.dryden::u_gw",
        "Root.wind.dryden::v_gw",
        "Root.wind.dryden::w_gw",
        "Root.wind.dryden::p_gw",
        "Root.wind.dryden::q_gw",
        "Root.wind.dryden::r_gw",
    ]
    xdot_by_path = model.xdot.read(trim.x_dot_opt, by="path")
    constrained_error = max(
        float(np.max(np.abs(np.asarray(xdot_by_path[key], dtype=float))))
        for key in constrained
    )
    print(f"partial trim constrained max |xdot|: {constrained_error:.3e}")

    x_by_path = model.x.read(trim.x_opt, by="path")
    u_by_path = model.u.read(trim.u_opt, by="path")
    linear = LinearizationSolver(
        model,
        LinearizationSpec(
            x_op={
                key: np.asarray(value).reshape(-1).tolist()
                for key, value in x_by_path.items()
            },
            u_op={
                key: np.asarray(value).reshape(-1).tolist()
                for key, value in u_by_path.items()
            },
            x_keys=[
                "Root.eom::vel_body",
                "Root.eom::omega_body",
                "Root.actuator::position",
            ],
            u_keys=["Root::u"],
            y_keys=["Root::clean_y"],
        ),
    ).solve()
    print(f"A shape: {linear.A.shape}, B shape: {linear.B.shape}")
    print(f"C shape: {linear.C.shape}, D shape: {linear.D.shape}")


if __name__ == "__main__":
    main()
