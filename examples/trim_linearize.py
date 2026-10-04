"""用完整变量路径对 GTM 做部分配平与线性化。

主题：先固定一个开环刚体工作点（1000 m 高度、100 m/s 前向、零姿态、40% 油门），用
TrimSolver 只让三个舵面、发动机二阶滞后与八个 Dryden 内部状态的导数为零，再以配平点
为工作点线性化。

运行：

    cd models/fr_gtm && just example-trim

预期结果：打印受约束通道的最大 |xdot|（应接近 0）与 A、B、C、D 四个矩阵的形状；配平
失败时脚本抛出带 `trim.stats` 的 RuntimeError。

说明：下方的配平求解固定一个开环刚体工作点，再让舵面、发动机滞后与 Dryden 内部状态
收敛，使其状态导数为零。这样既保持示例确定且快速，又仍会走完整的
`TrimSolver`/`LinearizationSolver` 路径，与更大飞行条件的配平研究所用 API
一致。

依赖：配平与线性化走 CasADi 后端。
约定：位置取世界系 NED，速度与角速度在机体系 FRD，四元数为 scalar-first。
"""

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.runtime.linearize import LinearizationSolver, LinearizationSpec
from fly_ruler_std.runtime.trim import TrimSolver, TrimSpec

from fr_gtm import GTMConfig, GTMSystem


def _array(values: list[float] | np.ndarray) -> np.ndarray:
    """把列表或数组统一转成浮点数组，保证各字段类型一致。"""
    return np.asarray(values, dtype=float)


def main() -> None:
    """求舵面、发动机滞后与 Dryden 状态导数为零的部分配平，并在配平点线性化。"""
    cfg = GTMConfig.default()
    # 顶层构建器需要是 PyIrBuilder 才能直接交给编译器，类型不符时立即报错。
    builder = GTMSystem.build("Root", cfg)
    if not isinstance(builder, fr.PyIrBuilder):
        raise TypeError("expected top-level GTM builder")
    # 配平与线性化需要求解器，因此显式选择 CasADi 后端。
    model = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=0.01)).compile(
        builder, target=fr.targets.casadi
    )

    # #region 05-analysis-1
    # param_value 取参数默认值，state_value 与 input_value 把刚体工作点和输入钉住。
    # state_derivative 只约束三个舵面、发动机滞后两个与 Dryden 八个状态的导数为零，
    # 因此是部分配平；initial_guess 给完整初值，其中升降舵先取 -2 度、油门取 40%，
    # 滞后状态取 40% 对应的配平值，避免配平点落在一段人为推力暂态上。
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
                "Root.actuator::position": _array([0.0, 0.0, 0.0]),
                "Root.engine::lag_state": _array([0.0, 0.0]),
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
                "Root.actuator::position": _array([-2.0, 0.0, 0.0]),
                "Root.engine::lag_state": _array(list(cfg.engine.lag.trim_state(40.0))),
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
    # #endregion 05-analysis-1
    # 配平失败时 stats 给出残差与求解器状态，便于定位约束冲突。
    if not trim.success:
        raise RuntimeError(f"trim failed: {trim.stats}")
    # #region 05-analysis-2
    # 受约束通道用完整路径列出，逐个回读导数并取最大绝对值作为配平残差。
    constrained = [
        "Root.actuator::position",
        "Root.engine::lag_state",
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
    # #endregion 05-analysis-2

    # #region 05-analysis-3
    # 工作点取自配平结果；x_keys 与 u_keys 选关注子集，y_keys 用 clean_y 保持 21 通道，
    # 避免把作动器遥测通道一起纳入输出矩阵。
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
                "Root.engine::lag_state",
            ],
            u_keys=["Root::u"],
            y_keys=["Root::clean_y"],
        ),
    ).solve()
    print(f"A shape: {linear.A.shape}, B shape: {linear.B.shape}")
    print(f"C shape: {linear.C.shape}, D shape: {linear.D.shape}")


# #endregion 05-analysis-3


if __name__ == "__main__":
    main()
