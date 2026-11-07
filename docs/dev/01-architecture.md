# 装配与语义边界

`models/fr_gtm/src/fr_gtm/system.py` 是唯一的装配点，`config.py` 聚合各组件配置并协调派生。这一章讲清这两层的实现与不能打破的约定。

## 配置派生

`GTMConfig.fullscale(mode=...)` 与 `GTMConfig.subscale(mode=...)` 分别构造全尺寸和缩比整机配置，并统一组装 `AircraftConfig`、`WindConfig`、`EngineConfig`、`ActuatorConfig`、`AeroConfig`、`SensorConfig` 与 `GroundContactConfig`。构型只属于 `AeroConfig.mode`，缩比到全尺寸的固定长度倍率 `SCALE_FACTOR=1/0.055` 位于 `config.py`。

`AeroConfig` 与 `Aero` 定义在 `components/aero.py`。配置是扁平的：资源、构型、插值选项、参考面积、翼展、弦长、重心、气动参考点和四个舵面力臂比值；气动计算不导入 `GTMConfig` 或 `AircraftConfig`。`AeroConfig.load` 接收显式几何并加载 NPZ，表字段以 `Field(exclude=True)` 排除序列化。`b/Cbar` 是动导归一化与力矩换算的共同事实源。

发动机载荷的 `fullscale` / `subscale` 工厂仅接收显式的 `S` 和 `cg_pos`，安装几何由 `EngineLoadsConfig` 拥有。整机两个工厂负责把标称飞机几何传给各组件，构造后直接修改某一组件不会隐式改变其他组件。

## 七个块与接线

`GTMSystem` 创建 `actuator`、`wind`、`mass`、`aero`、`engine`、`eom`、`sensor` 七个块，`cfg.ground.enabled` 为真时再挂 `ground`：

| 块 | 来源 | 关键接线 |
| --- | --- | --- |
| `actuator` | std `SimpleActuator`（`adim=3`，`expose_params=False`） | `Root::u[:3] >> actuator.command` |
| `wind` | 模型侧 `Wind` 子系统 | 高度接 `-eom.pos_ned_out[2]`，机体系速度与 `DCM` 从刚体来，白噪声从 `Root::d` 来 |
| `mass` | 模型侧 `Mass` | 输出 `mass`、`inertia`、`inertia_inv` 直接喂给刚体 |
| `aero` | 模型侧 `Aero` | 消费三个舵面位置、位置、机体系速度、四元数、角速度与风 |
| `engine` | 模型侧 `Engine` System | 消费 `Root::u[3]` 的油门指令、角速度、压力比与动压，内部可配置滞后给出有效油门 |
| `eom` | std `RigidBody6DOFQuaternion`（`enable_external_mass=True`） | 接收合力与合力矩，输出运动学量 |
| `sensor` | 模型侧 `Sensor` | 汇总 17 个物理反馈量，噪声取 `Root::n` 的前 17 路 |
| `ground` | std `GroundContact` | 消费位置、机体系速度、`DCM` 与角速度，输出力与力矩 |

气动力、发动机载荷、旋转到 FRD 的重力与可选地面载荷合成后送入刚体。法向过载是 `-F_non_gravity,z/W`，向上为正，排除重力并包含可选地面接触力；归一化使用质量块的同一份扰动后重量。

std 两种刚体统一从 NED 速度计算航迹角/航向角。GTM 直接使用 `degrees(eom.gamma)` 接 Sensor；航迹角的水平速度正则化（epsilon=1e-3）由刚体拥有，不在模型侧重复计算。气动块只接正向上高度和 FRD 速度/角速率/风。

根仅导出 `y`、`clean_y` 两个 17 维物理观测，不拼接命令，求值函数的打包输出长度为 34。三个舵面实际位置占 `[13:16]`、有效油门占 `[16]`；控制器直接读取 `y`，命令独立记录在输入 `u`。噪声输入同为 17 维。

effects 逐项覆盖组件 out 中未被 x、y/clean_y 或已有子组件 effects 覆盖的诊断，包括质量/惯量、气动和发动机分项载荷、大气量、刚体坐标变换及加速度；状态别名和四元数导数参考项不导出。Aero 保留自己的系数/雷诺数，Wind 保留自己的稳态风/阵风，根不重复声明已有子组件诊断。可选地面诊断仍通过 effects 读取，地面开关不改变根输出布局。读取方法见[推进与记录](/guide/components/gtm/03-simulate)。

## 关键数据结构

| 结构 | 位置 | 说明 |
| --- | --- | --- |
| `GTMConfig` | `models/fr_gtm/src/fr_gtm/config.py` | 尺度、构型与全部嵌套配置 |
| `Mode`、`AeroConfig` | `models/fr_gtm/src/fr_gtm/components/aero.py` | 襟翼与起落架的四个气动构型 |
| 布局目录 | `models/fr_gtm/src/fr_gtm/layout.py` | `GTMPaths` 与 U/D/N/Y 通道索引；状态/参数维数从模型布局读取 |
| 气动表资源 | `models/fr_gtm/src/fr_gtm/data/aero_tables.npz` | 基本系数、雷诺数、构型增量与操纵导数表 |
| 运行时参数 | 块声明 | 11 个路径、33 个数值，运行时可变、免重编译 |

## 语义边界

- **坐标与姿态约定固定**：导航系 NED、机体系 FRD、四元数 scalar-first，高度是 `-pos_ned[2]`。
- **角度内外有别**：对外接口用度，块内计算用弧度；风参数公开单位是弧度，冻结参考数据里存的是度。
- **随机量都是外部输入**：Dryden 噪声、传感器噪声与参数不确定度分别从 `d`、`n`、`p` 进入，全零输入下模型确定。
- **遥测不进入反馈**：`y` 只含 17 个物理量，四个指令遥测被排除。
- **油门不是作动器通道**：`ActuatorConfig` 只描述三个舵面，油门指令直接进入 `Engine` 的可配置滞后；默认一阶滞后只有一个有效油门状态，时间常数为 1.5 s；可选 `JetCatThrottleLagConfig` 的两个内部状态是拟合坐标，不是真实双转子转速，也不能都当作油门位置。
- **滞后状态必须配平初始化**：`Root.engine.lag::lag_state` 的配平值是 `cfg.engine.lag.trim_state(throttle)`，只把输出置成目标油门而把滞后状态清零会造出人为推力暂态。
- **气动资源集中校验**：配置构造时读取 NPZ 并校验全部表轴与形状；气动块按语义资源组声明所需插值器，方向与单位转换在块内完成，顶层系统不做补正。
- **模型特有逻辑留在模型侧**：std 提供作动器、大气、风与刚体方程，GTM 特有的表格与符号处理不上升为通用能力。

## 改这一层的连带影响

改动块名或 `fr.export` 的名字会改变完整路径，`model.x.keys`、`model.y.keys`、`model.effects.keys`、记录列名与示例脚本里的路径字典都要同步。改合力或力矩的拼装会移动配平点与冻结参考，需要重新生成参考并重跑跨后端回归。改状态布局会同时影响 `GTMPaths`、`init`、`trim`、`TrimSpec` 里的路径与记录列名。切换滞后模型或修改一阶时间常数会改变状态布局/响应；改 `JetCatThrottleLagConfig` 的矩阵或 `time_scale` 会移动油门响应尺度，冻结参考里油门通道的导数定义与新模型不再逐位对应，回归要按通道分别核对。

## 怎么验证

- `just test-one tests/test_system.py`：接线与导出量的结构回归。
- `just test-one tests/test_engine_first_order.py`：默认一阶模型、滞后模型切换、解析响应与跨后端一致性。
- `just test-one tests/test_engine_lag.py`：二阶滞后对解析阶跃响应、配平初始化、指令钳位与时间尺度缩放的回归。
- `just test-one tests/test_config.py`：配置派生、组件几何注入与非法输入。
- `just test-one tests/test_reference.py`：与外部 `gtm_ad` 冻结参考的数值对比。
- `just example-inspect`：打印产物与完整路径清单，确认新增导出进入 `y.keys`。

## 相关页面

- [块方程与气动数据](/dev/components/gtm/02-aero-data)：五个块文件的实现
- [验证与扩展](/dev/components/gtm/03-verification)：命令与扩展点
- [std 实现](/dev/components/std/01-architecture)：被复用的作动器与刚体块
