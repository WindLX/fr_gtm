"""演示稳态风、Dryden 湍流、参数不确定性与传感器噪声。

主题：在同一次连续求值中叠加四类扰动来源，比较无噪声的 clean_y 与叠加噪声的 y，
并观察风引起的机体系气流与迎角、侧滑角。

运行：

    cd models/fr_gtm && just example-disturbances

预期结果：打印稳态风与阵风之和、迎角/侧滑角，以及 clean_y、y 的高度与真空速；
两者在叠加传感器噪声后出现差异，说明噪声只进入带噪输出通道。

约定：init 的 wind 为速度(m/s)、风向(度，自 NED 北向顺时针)、
turbulence_level；不确定度参数是相对量，取 0 表示复现标称模型。
"""

import numpy as np
from fly_ruler_core import ModelSettings, compiler, runtime, targets

from fr_gtm import (
    GTMConfig,
    GTMPaths,
    GTMSystem,
    build_input,
    init,
    read_observation,
    sample_disturbance,
    sample_sensor_noise,
)


def main() -> None:
    """装配风、不确定度与噪声输入，求一次连续输出并打印关键分量。"""
    # 默认配置为全尺寸、NORMAL 构型且不实例化地面接触；示例只关注气动与风响应。
    cfg = GTMConfig.fullscale()
    builder = GTMSystem.build("Root", cfg)
    model = runtime.bind(
        compiler.compile(
            compiler.prepare(
                builder, settings=ModelSettings(integrator="euler", dt=0.01)
            ),
            target=targets.casadi,
        )
    )
    # #region 04-scenarios-1
    # 来流、姿态和参数一次构造；Dryden 初态为零，激励从外部高斯噪声进入。
    # #region 02-state-input-1
    x, xd, _, p = init(
        model,
        cfg,
        position=(0.0, 0.0, -500.0),
        airdata=(3.0, 0.0, 100.0),
        command=(-2.0, 0.0, 0.0, 42.0),
        wind=(12.0, 35.0, 3.0),
        uncertainty={"weight": 0.03, "c_static": np.linspace(-0.03, 0.03, 6)},
    )
    rng = np.random.default_rng(42)
    u = build_input(
        model,
        (-2.0, 0.0, 0.0, 42.0),
        d=sample_disturbance(rng),
        n=sample_sensor_noise(rng),
    )
    # #endregion 02-state-input-1
    # #endregion 04-scenarios-1
    # #region 03-simulate-1
    # 只求一次连续输出：clean_y 为未经传感器噪声的真值，y 叠加了噪声与偏置。
    result = model.continuous_fn(0.0, x, xd, u, p)
    diagnostics = model.effects.read(result.effects, by="path")
    paths = GTMPaths()
    wind = diagnostics[paths.effects.wind_steady] + diagnostics[paths.effects.wind_gust]
    clean = read_observation(model, result.y)
    noisy = read_observation(model, result.y, measured=True)
    print(f"body wind: {model.to_numpy(wind).reshape(-1)}")
    print(f"alpha/beta: {clean.alpha_deg:.3f}, {clean.beta_deg:.3f} deg")
    print(f"clean altitude/TAS: {clean.altitude_m:.3f} m, {clean.tas_mps:.3f} m/s")
    print(f"noisy altitude/TAS: {noisy.altitude_m:.3f} m, {noisy.tas_mps:.3f} m/s")


# #endregion 03-simulate-1


if __name__ == "__main__":
    main()
