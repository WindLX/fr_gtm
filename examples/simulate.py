"""运行一段标称短仿真并用标准 Recorder 保存结果。

主题：从 1000 m 高度、100 m/s 前向速度的巡航状态出发，用 40% 油门推进 100 步
（dt=0.01 s，共 1 s），把状态、输入与输出写成 Parquet。

运行：

    cd models/fr_gtm && just example-simulate

预期结果：打印 `saved: results/gtm_nominal/nominal.parquet`，随后给出末态高度与空速；
产物落在 `models/fr_gtm/results/gtm_nominal/` 下。

约定：位置取世界系 NED，高度为 -pos_ned[2]；速度在机体系 FRD，姿态四元数为
scalar-first；三个舵面位置为 [升降舵, 副翼, 方向舵]（度），油门指令为百分比、
先进 `Engine` 的二阶滞后再成为有效油门。
"""

from pathlib import Path

import fly_ruler_core as fr
import numpy as np
from fly_ruler_std.runtime.record import Recorder

from fr_gtm import GTMConfig, GTMSystem


# #region gtm-1
def main() -> None:
    """用默认配置推进一步长为 100 步的标称仿真并落盘。"""
    # #region 01-build-1
    # 默认配置为全尺寸、NORMAL 构型且关闭地面接触；编译到 CasADi 后端。
    cfg = GTMConfig.default()
    builder = GTMSystem.build("Root", cfg)
    model = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=0.01)).compile(
        builder, target=fr.targets.casadi
    )
    # #endregion 01-build-1
    # #region 02-state-input-1
    # 状态用完整路径装配：NED 下向 -1000 即 1000 m 高度，速度在机体系 FRD。
    # 作动器位置与输入指令一致，三元素依次为升降舵、副翼与方向舵；发动机的
    # 二阶滞后状态取 40% 油门对应的配平值，避免起点出现人为推力暂态。
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -1000.0],
            "Root.eom::vel_body": [100.0, 0.0, 0.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [0.0, 0.0, 0.0],
            "Root.engine::lag_state": cfg.engine.lag.trim_state(40.0),
        }
    )
    u = model.u.build(
        {
            "Root::u": [0.0, 0.0, 0.0, 40.0],
            "Root::d": np.zeros(4),
            "Root::n": np.zeros(21),
        }
    )
    # #endregion 02-state-input-1
    # p.build() 不覆盖任何参数，因此风、不确定度与作动器偏置都保持标称值。
    p = model.p.build()
    # #region 03-simulate-1
    # 模型没有离散状态，xd 仍需显式构造并随每一步回传。
    xd = model.xd.build()
    # #region 03-simulate-2
    # Recorder 绑定模型以确定列布局；discrete_fn 推进一步并返回新状态与输出。
    output_dir = Path("results") / "gtm_nominal"
    recorder = Recorder(model, output_dir)
    for step in range(100):
        x, xd, y, effects = model.discrete_fn(
            step * 0.01,
            x,
            xd,
            u,
            p,
        )
        recorder.record(
            step,
            (step + 1) * 0.01,
            x=x,
            xd=xd,
            y=y,
            u=u,
            p=p,
            effect=effects,
        )
    recorder.save("nominal")
    # 按完整路径回读末态输出，高度与空速取自 21 通道 y 的对应分量。
    final = model.y.read(y, by="path")
    print(f"saved: {output_dir / 'nominal.parquet'}")
    altitude = model.to_numpy(final["Root::altitude"]).reshape(-1)[0]
    print(f"final altitude: {altitude:.3f} m")
    # #endregion 03-simulate-2
    # #endregion 03-simulate-1
    print(f"final airspeed: {model.to_numpy(final['Root::V']).reshape(-1)[0]:.3f} m/s")


# #endregion gtm-1


if __name__ == "__main__":
    main()
