"""在 NumPy、CasADi 与可选 JAX 后端之间比较同一次 GTM 求值。

主题：对同一份 GTM 系统与同一套状态、输入，分别用三种后端编译并调用一次连续函数，
以 NumPy 结果为参考，打印编译耗时与 xdot、y 的最大绝对偏差。

运行：

    cd models/fr_gtm && just example-backends

预期结果：打印 `backend / compile-ms / max|xdot| / max|y|` 表格；NumPy 一行偏差为 0，
CasADi 与 JAX 一行应落在数值容差内；未安装 JAX 时打印跳过信息并只比较前两个后端。

依赖：CasADi 必须可用；JAX 为可选项。
约定：位置取世界系 NED，速度在机体系 FRD，高度为 -pos_ned[2]；姿态四元数为
scalar-first；三个舵面位置为 [升降舵, 副翼, 方向舵]（度），油门指令为百分比、
先进 `Engine` 的二阶滞后再成为有效油门。
"""

from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import TypeVar

import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem

ROOT = Path(__file__).resolve().parent.parent
ArrayT = TypeVar("ArrayT")
# 40% 油门对应的发动机滞后配平状态，供各后端的固定工作点共用。
LAG_STATE_40 = GTMConfig.default().engine.lag.trim_state(40.0)


def _evaluate(model: fr.CompiledModel[ArrayT]) -> tuple[np.ndarray, np.ndarray]:
    """装配固定状态与输入，返回一次连续求值的 xdot 与 y 扁平向量。"""
    # 状态取 1000 m 高度（NED 下向为 -1000）、机体系速度 100 m/s 前向加 5 m/s 下向分量、
    # 零姿态；舵面取升降舵 -2 度、副翼 1 度、方向舵 0 度，油门指令 40%，
    # 发动机二阶滞后状态取 40% 对应的配平值。
    # 未列出的状态分量（角速度、Dryden 阵风等）由 build 默认为零。
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -1000.0],
            "Root.eom::vel_body": [100.0, 0.0, 5.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [-2.0, 1.0, 0.0],
            "Root.engine::lag_state": LAG_STATE_40,
        }
    )
    u = model.u.build(
        {
            "Root::u": [-2.0, 1.0, 0.0, 40.0],
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    xdot, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, model.p.default())
    return (
        np.asarray(xdot, dtype=float).reshape(-1),
        np.asarray(y, dtype=float).reshape(-1),
    )


def main() -> None:
    """依次编译可用后端、计时求值，并与 NumPy 结果的偏差一起打印。"""
    names = ["numpy", "casadi"]
    # #region 05-analysis-1
    # 先按实际安装情况确定后端清单；JAX 需先配置运行时策略才能启用 CPU 与 64 位。
    if find_spec("jax") is not None:
        fr.configure_jax_runtime(
            fr.JaxRuntimePolicy(
                platforms=("cpu",),
                preallocate=False,
                enable_x64=True,
            )
        )
        names.append("jax")
    else:
        print("JAX is not installed; skipping it.")
    # #endregion 05-analysis-1

    # #region 05-analysis-2
    # 每个后端独立编译一次并计时；最后统一以 NumPy 的 xdot、y 作为偏差参考。
    results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name in names:
        started = perf_counter()
        compiler = fr.Compiler(
            fr.CompilerSpec(
                integrator="euler", dt=0.01, artifact_dir=ROOT / "build" / "flyruler"
            )
        )
        builder = GTMSystem.build("Root", GTMConfig.default())
        # 走到最后的分支时 name 只能是 jax，因为清单由上面的可用性检查生成。
        if name == "numpy":
            xdot, y = _evaluate(compiler.compile(builder, target=fr.targets.numpy))
        elif name == "casadi":
            xdot, y = _evaluate(compiler.compile(builder, target=fr.targets.casadi))
        else:
            xdot, y = _evaluate(compiler.compile(builder, target=fr.targets.jax))
        compile_ms = (perf_counter() - started) * 1000.0
        results[name] = (
            compile_ms,
            xdot,
            y,
        )

    # 以 NumPy 后端的结果作为参考，偏差用逐分量最大绝对值表示。
    _, xdot_ref, y_ref = results["numpy"]
    print(f"{'backend':>8}  {'compile/ms':>10}  {'max|xdot|':>12}  {'max|y|':>12}")
    for name, (compile_ms, xdot, y) in results.items():
        print(
            f"{name:>8}  {compile_ms:10.1f}  "
            f"{np.max(np.abs(xdot - xdot_ref)):12.3e}  "
            f"{np.max(np.abs(y - y_ref)):12.3e}"
        )


# #endregion 05-analysis-2


if __name__ == "__main__":
    main()
