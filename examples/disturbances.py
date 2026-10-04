"""演示稳态风、Dryden 湍流、参数不确定性与传感器噪声。

主题：在同一次连续求值中叠加四类扰动来源，比较无噪声的 clean_y 与叠加噪声的 y，
并观察风引起的机体系气流与迎角、侧滑角。

运行：

    cd models/fr_gtm && just example-disturbances

预期结果：打印机体系气流 `Root::uvw`、迎角与侧滑角，以及 clean_y、y 的前五个分量；
两者在叠加传感器噪声后出现差异，说明噪声只进入带噪输出通道。

约定：风参数为 wind_speed(m/s)、wind_direction(弧度，自 NED 北向顺时针)、
turbulence_level；不确定度参数是相对量，取 0 表示复现标称模型。
"""

import fly_ruler_core as fr
import numpy as np

from fr_gtm import GTMConfig, GTMSystem


def main() -> None:
    """装配风、不确定度与噪声输入，求一次连续输出并打印关键分量。"""
    # 默认配置为全尺寸、NORMAL 构型且不实例化地面接触；示例只关注气动与风响应。
    cfg = GTMConfig.default()
    builder = GTMSystem.build("Root", cfg)
    model = fr.Compiler(fr.CompilerSpec(integrator="euler", dt=0.01)).compile(
        builder, target=fr.targets.numpy
    )
    # #region 04-scenarios-1
    # 状态里额外给出 Dryden 阵风初值：u_gw 为 1 维，v_gw 与 w_gw 各 2 维，
    # 即使湍流未开启这些状态仍留在状态向量中，因此需要显式构造。
    x = model.x.build(
        {
            "Root.eom::pos_ned": [0.0, 0.0, -500.0],
            "Root.eom::vel_body": [90.0, 0.0, 5.0],
            "Root.eom::quat": [1.0, 0.0, 0.0, 0.0],
            "Root.actuator::position": [-2.0, 0.0, 0.0],
            "Root.engine::lag_state": cfg.engine.lag.trim_state(42.0),
            "Root.wind.dryden::u_gw": [0.2],
            "Root.wind.dryden::v_gw": [-0.1, 0.05],
            "Root.wind.dryden::w_gw": [0.1, -0.05],
        }
    )
    # #endregion 04-scenarios-1
    # 输入分三层：四路作动器指令、四路 Dryden 白噪声驱动、21 路标准化传感器噪声。
    u = model.u.build(
        {
            "Root::u": [-2.0, 0.0, 0.0, 42.0],
            "Root::d": [0.4, -0.2, 0.3, -0.1],
            "Root::n": np.linspace(-1.0, 1.0, 21),
        }
    )
    # #region 02-state-input-1
    # 参数覆盖分两组：风的速度、方向与湍流强度，以及重量与六个静态系数的相对偏差。
    p = model.p.build(
        {
            "Root.wind.params::wind_speed": [12.0],
            "Root.wind.params::wind_direction": [np.deg2rad(35.0)],
            "Root.wind.params::turbulence_level": [3.0],
            "Root.mass::weight_unc": [0.03],
            "Root.aero::c_static_unc": np.linspace(-0.03, 0.03, 6),
        }
    )
    # #endregion 02-state-input-1
    # #region 03-simulate-1
    # 只求一次连续输出：clean_y 为未经传感器噪声的真值，y 叠加了噪声与偏置。
    _, _, y, _ = model.continuous_fn(0.0, x, model.xd.build(), u, p)
    output = model.y.read(y, by="path")
    print(f"body wind:  {output['Root::uvw']}")
    print(
        f"alpha/beta: {output['Root::alpha'][0]:.3f}, {output['Root::beta'][0]:.3f} deg"
    )
    print(f"clean y[:5]: {output['Root::clean_y'][:5]}")
    print(f"noisy y[:5]: {output['Root::y'][:5]}")


# #endregion 03-simulate-1


if __name__ == "__main__":
    main()
