# 地面运行、风与扰动

GTM 的扰动分三类：环境侧有稳态风与 Dryden 紊流，模型侧有质量与气动系数的不确定度，观测侧有 17 通道传感器噪声。再加上可选的起落架地面接触，这一章给出这些场景的配置方式与观察量。

## 起落架地面接触

默认构型没有起落架。打开它要在构造配置时就传参，然后重新编译：

```python
cfg = GTMConfig.fullscale(enable_ground=True)
builder = GTMSystem.build("Root", cfg)
model = fr.Compiler(fr.CompilerSpec(integrator="rk4", dt=dt)).compile(
    builder
)
```

开启后 effects 中多出七个地面诊断路径，滑跑时可以用 `Root::ground_normal_forces` 看轮载分布，用 `Root::ground_contact_count` 看接触点数量。起飞滑跑的初值取静止、接地的状态，接地场景取 1 m 高度、70 m/s 水平速度与 1.5 m/s 下降率：

<<< @/../models/fr_gtm/examples/ground_operations.py#02-state-input-1

`just example-ground --scenario takeoff` 与 `just example-ground --scenario touchdown` 分别跑这两个场景，脚本的 `--backend` 支持 `numpy`、`casadi`、`jax` 与 `native` 四种取值，其中 native 走「先编译 CasADi 再 `convert`」的路径：

<<< @/../models/fr_gtm/examples/ground_operations.py#04-scenarios-1

## 稳态风与 Dryden 紊流

风由三个运行时参数控制：`wind_speed`（m/s）、`wind_direction`（弧度，从 NED 北向顺时针）与 `turbulence_level`。稳态风按跑道高度做风切变修正；Dryden 紊流由 `Root::d` 的四路白噪声驱动，时间常数用机体系速度模值，避免与气动模型形成代数环。

`turbulence_level` 超过 `1e-3` 才开启激励与输出，等于或低于该阈值时阵风与风角速度都被门控为零，但八个 Dryden 状态仍留在状态向量里。湍流强度与低空风速、高空尺度长度一起决定阵风谱：

<<< @/../models/fr_gtm/examples/disturbances.py#04-scenarios-1

机体系风与气流角从观测读回：

<<< @/../models/fr_gtm/examples/disturbances.py#03-simulate-1

## 不确定度与传感器噪声

质量侧的 `Root.mass::weight_unc` 缩放重量，`Root.mass::inertia_unc` 按 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]` 缩放六个惯量分量；气动侧的四个 `c_*_unc` 分别覆盖静态项、升降舵、副翼与方向舵系数。所有不确定度都是相对量，零值复现标称参考数据，逆惯量用闭式解计算以免引入后端特有的矩阵求逆。

传感器噪声从 `Root::n` 的 17 路进入，对应 17 个物理反馈量，标准差配置在 `SensorConfig`。`Root::clean_y` 是同一布局的真值，全部 17 项均不含指令遥测，便于把噪声影响与模型响应分开观察。

## 组合场景

`models/fr_gtm/examples/disturbances.py` 把稳态风、Dryden 紊流、质量与气动不确定度、传感器噪声放在同一次求值里，用 `continuous_fn` 取一次输出打印机体系风、攻角、侧滑角与噪声前后的高度、真空速：

| 场景 | 关键配置 | 观察量 |
| --- | --- | --- |
| 稳态风 | `wind_speed`、`wind_direction` | effects 中稳态风与阵风之和、观测 `clean_y[1:3]` |
| Dryden 紊流 | `turbulence_level`、`Root.wind.dryden::*` 状态与 `Root::d` | `Root::wind_omega`、`Root.wind::steady/gust` 的时间序列 |
| 重量与惯量不确定度 | `Root.mass::weight_unc`、`Root.mass::inertia_unc` | 质量参数、角速度响应 |
| 气动系数不确定度 | `Root.aero::c_static_unc` 与前三个通道 | `Root.aero::coefficients`、`Root::moment_aero` |
| 传感器噪声 | `Root::n` 与 `SensorConfig` | `Root::y` 与 `Root::clean_y` 之差 |

## 这一章用到的 API

- [`fr_gtm.GTMConfig`](/api/gtm/config)：`ground`、`wind`、`sensor` 配置
- [`fr_gtm.init`](/api/gtm/init)：物理初值与输入工具
- [`fly_ruler_std.blocks.wind.DrydenTurbulence`](/api/std/blocks/wind/index)：紊流滤波器的配置面
- [`fly_ruler_std.blocks.physics.GroundContact`](/api/std/blocks/physics/index)：地面接触的配置面

## 边界与常见坑

| 边界 | 说明 |
| --- | --- |
| 地面接触是编译期开关 | 配置里没开地面就没有 `Root::ground_*` 输出，开了之后状态维数及两个根输出不变，只增加 effects 诊断 |
| 风向单位是弧度 | 从 NED 北向顺时针度量；参考轨迹数据存的是度 |
| 紊流门控 | `turbulence_level <= 1e-3` 时阵风被门控为零，看起来「风没起作用」属于预期行为 |
| 不确定度是相对量 | 0.05 表示 5%，不是绝对质量或惯量 |
| 噪声按标准差缩放 | `Root::n` 是标准化样本，物理噪声由配置的标准差决定 |
| native 后端首次运行慢 | 首次会编译并缓存 C++ 扩展，之后复用缓存 |

## 相关页面

- [组装与配置](/guide/components/gtm/01-build)：`enable_ground` 与气动表来源
- [初值、输入与参数](/guide/components/gtm/02-state-input)：参数路径与顺序
- [配平、线性化与后端对比](/guide/components/gtm/05-analysis)：带扰动的工作点怎么线性化
- [gtm 实现](/dev/components/gtm/02-aero-data)：风子系统与气动表的实现契约
