"""用五个飞行目标求完整三轴 GTM 配平，并在该工作点线性化。

运行：cd models/fr_gtm && just example-trim
预期：平动/角加速度残差小于 1e-6；默认一阶模型 A10×10、B10×4、C17×10、D17×4。
高度 m、TAS m/s、gamma/psi/phi deg；生产模型仍只导出两个 17 维物理观测。
"""

from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_std.runtime.linearize import LinearizationSolver, LinearizationSpec

from fr_gtm import GTMConfig, GTMPaths, GTMSystem, read_observation, trim


def main() -> None:
    """真正求解刚体与内部动态平衡，不再固定刚体工作点只配平滞后。"""
    cfg = GTMConfig.fullscale()
    paths = GTMPaths()
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                GTMSystem.build("Root", cfg),
                settings=ModelSettings(integrator="rk4", dt=0.01),
            ),
            target=targets.casadi,
        )
    )
    # #region 05-analysis-1
    point = trim(model, cfg, altitude=1000.0, tas=100.0, gamma=0.0, psi=0.0, phi=0.0)
    if not point.success:
        raise RuntimeError(
            f"trim failed: {point.solver.stats}; residuals={point.residuals}"
        )
    # #endregion 05-analysis-1
    # #region 05-analysis-2
    print(f"body acceleration: {point.residuals[paths.x.vel_body]:.3e} m/s²")
    print(f"angular acceleration: {point.residuals[paths.x.omega_body]:.3e} rad/s²")
    output = model.continuous_fn(0.0, point.x, point.xd, point.u, point.p)
    observation = read_observation(model, output.y)
    print(
        f"alpha={observation.alpha_deg:.3f} deg, "
        f"throttle={observation.throttle_percent:.3f}%"
    )
    # #endregion 05-analysis-2
    # #region 05-analysis-3
    linear = LinearizationSolver(
        model,
        LinearizationSpec(
            x_op={
                key: model.to_numpy(value).reshape(-1).tolist()
                for key, value in model.x.read(point.x, by="path").items()
            },
            u_op={
                key: model.to_numpy(value).reshape(-1).tolist()
                for key, value in model.u.read(point.u, by="path").items()
            },
            p_op={
                key: model.to_numpy(value).reshape(-1).tolist()
                for key, value in model.p.read(point.p, by="path").items()
            },
            x_keys=[
                paths.x.vel_body,
                paths.x.omega_body,
                paths.x.actuator_position,
                paths.x.engine_lag,
            ],
            u_keys=[paths.u.command],
            y_keys=[paths.y.clean],
        ),
    ).solve()
    print(f"A shape: {linear.A.shape}, B shape: {linear.B.shape}")
    print(f"C shape: {linear.C.shape}, D shape: {linear.D.shape}")
    # #endregion 05-analysis-3


if __name__ == "__main__":
    main()
