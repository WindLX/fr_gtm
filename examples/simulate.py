"""运行一段标称短仿真并用标准 Recorder 保存结果。

主题：从 1000 m 高度、100 m/s 前向速度的巡航状态出发，用 40% 油门推进 100 步
（dt=0.01 s，共 1 s），把状态、输入与输出写成 Parquet。

运行：

    cd models/fr_gtm && just example-simulate

预期结果：打印 `saved: results/gtm_nominal/nominal.parquet`，随后给出末态高度与空速；
产物落在 `models/fr_gtm/results/gtm_nominal/` 下。

约定：位置取世界系 NED，高度为 -pos_ned[2]；速度在机体系 FRD，姿态四元数为
scalar-first；三个舵面位置为 [升降舵, 副翼, 方向舵]（度），油门指令为百分比、
先进 `Engine` 的可配置滞后再成为有效油门。
"""

from pathlib import Path

from fly_ruler_core import ModelSettings, compiler, runtime, targets
from fly_ruler_std.runtime.record import Recorder

from fr_gtm import GTMConfig, GTMSystem, init, read_observation


# #region gtm-1
def main() -> None:
    """用默认配置推进一步长为 100 步的标称仿真并落盘。"""
    # #region 01-build-1
    # 默认配置为全尺寸、NORMAL 构型且关闭地面接触；编译到 CasADi 后端。
    cfg = GTMConfig.fullscale()
    builder = GTMSystem.build("Root", cfg)
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                builder, settings=ModelSettings(integrator="rk4", dt=0.01)
            ),
            target=targets.casadi,
        )
    )
    # #endregion 01-build-1
    # #region 02-state-input-1
    # 状态用完整路径装配：NED 下向 -1000 即 1000 m 高度，速度在机体系 FRD。
    # 作动器位置与输入指令一致，三元素依次为升降舵、副翼与方向舵；发动机的
    # 滞后状态取 40% 油门对应的配平值，避免起点出现人为推力暂态。
    x, xd, u, p = init(
        model,
        cfg,
        position=(0.0, 0.0, -1000.0),
        airdata=(0.0, 0.0, 100.0),
        euler=(0.0, 0.0, 0.0),
        command=(0.0, 0.0, 0.0, 40.0),
    )
    # #endregion 02-state-input-1
    # 未提供 wind/uncertainty，参数保持标称值。
    # #region 03-simulate-1
    # init 已构造空离散状态 xd，每一步仍需传入并回传。
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
    # 按完整路径回读末态输出，高度与空速取自 17 通道 y 的对应分量。
    final = read_observation(model, y)
    print(f"saved: {output_dir / 'nominal.parquet'}")
    print(f"final altitude: {final.altitude_m:.3f} m")
    # #endregion 03-simulate-2
    # #endregion 03-simulate-1
    print(f"final airspeed: {final.tas_mps:.3f} m/s")


# #endregion gtm-1


if __name__ == "__main__":
    main()
