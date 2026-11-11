# 组装与配置

GTM 把配置与系统分开构造：`GTMConfig` 负责按尺度与构型派生全部嵌套参数，`GTMSystem` 只把这些参数接成块。这一章说明怎么选配置、系统由哪些块组成、编译期要确定什么。

## 最小可运行

从默认配置组装并编译：

<<< @/../models/fr_gtm/examples/simulate.py#01-build-1

`GTMConfig.fullscale()` 生成全尺寸、光洁构型的配置，示例显式选择 CasADi 后端，`integrator="rk4"` 与 `dt=0.01` 是编译期设置。这段代码在 `just example-simulate` 里接着推进 100 步。

## 选择发动机响应

默认发动机使用一阶滞后，时间常数为 1.5 s。油门指令先裁切到 `[0, 100]%`，有效油门再逐渐跟随指令；转速、推力和力矩仍由原来的静态映射计算。需要调整响应速度时，在构建系统前修改配置：

```python
from fr_gtm import JetCatThrottleLagConfig, FirstOrderThrottleLagConfig, GTMConfig

cfg = GTMConfig.fullscale()
cfg.engine = cfg.engine.model_copy(
    update={"lag": FirstOrderThrottleLagConfig(time_constant_s=1.5)}
)
```

要使用原始 JetCat P70 二阶响应，将上面的 `lag` 替换为 `JetCatThrottleLagConfig(time_scale=1.0)`。选择模型和修改时间常数后需要重新构建、编译；它们不是运行时输入。

`Engine` 是组装 System：根据 `cfg.engine.lag` 选择 `FirstOrderThrottleLag` 或 `JetCatThrottleLag`，命名为 `lag`，再连接无状态的 [`EngineLoads`](/api/gtm/components/engine#engineloads)，命名为 `loads`。外部仍使用 `throttle_command`、`omega_body`、`pressure_ratio`、`qbar` 输入和 `effective_throttle`、`force`、`moment` 输出。

发动机载荷参数统一位于 `cfg.engine.loads`：转速/推力多项式、尺度增益、冲压阻力与参考面积、转子参数、左右发动机安装位置及角度、参考重心。工厂显式接收面积与重心，载荷计算不读取 `AircraftConfig`。载荷配置不可重新赋值，调整时创建替换值：

```python
loads = cfg.engine.loads.model_copy(update={"ram_drag_coeff": 0.005})
cfg.engine = cfg.engine.model_copy(update={"loads": loads})
```

需要单独接入载荷反馈时，使用 `EngineLoads("loads", cfg.engine.loads)`。由外部滞后块提供当前有效油门；该 Block 不拥有状态或油门命令，也不对有效油门作输出限幅。

一阶模型的 `Root.engine.lag::lag_state` 只有一个有效油门状态，整机状态/切空间维数为 25/24；原二阶模型有两个内部状态，维数为 26/25。两者的输入、参数和输出布局相同。初始化时把构建模型所用的同一份 `cfg` 传给 `init`，并通过命名布局读取状态，不依赖固定下标。

## 选择尺度与构型

用两个明确的工厂选择尺度，`mode` 选择四种气动构型：

```python
from fr_gtm import GTMConfig, Mode

full = GTMConfig.fullscale(mode=Mode.NORMAL)
scale = GTMConfig.subscale(mode=Mode.FULL_FLAP_GEAR_DOWN, enable_ground=True)
```

| 构型                       | 含义                         |
| -------------------------- | ---------------------------- |
| `Mode.NORMAL`              | 光洁构型，无襟翼、起落架收起 |
| `Mode.HALF_FLAP_GEAR_UP`   | 半襟翼、起落架收起           |
| `Mode.FULL_FLAP_GEAR_UP`   | 全襟翼、起落架收起           |
| `Mode.FULL_FLAP_GEAR_DOWN` | 全襟翼、起落架放下           |

两个工厂统一装配飞机、风、发动机、作动器、气动、传感器与地面参数；全尺寸长度倍率为 `1/0.055`。构型只存放在 `cfg.aero.mode`。这些都是编译期选择，改变后要重新构建、编译和配平。

## 扁平气动配置

`Aero` 只接收一个 `AeroConfig`：`S`、`b`、`Cbar`、`cg_pos`、`ref_cg`、`control_arm_ratios`、`mode`、插值选项和 `tables`，不再分成舵面、动导与力矩子配置。同一组 `b/Cbar` 同时用于动导归一化和力矩换算。

```python
cfg = GTMConfig.fullscale()
cfg.aero.interpolation_method = "cubic"
cfg.aero.control_arm_ratios = (-0.07, -0.03, 0.03, 0.07)
```

更换表格时通过工厂的 `data_path` 参数重新加载。独立构造气动块可调用 `AeroConfig.load(S=..., b=..., Cbar=..., cg_pos=..., ref_cg=..., mode=..., data_path=...)`，再传给 `Aero("aero", cfg)`；它不依赖整机或飞机配置。直接修改组件配置只影响该组件，不会隐式重派生其他组件。

## 七个块与接线

`GTMSystem.build("Root", cfg)` 创建以下块，块之间的接线由系统写在 `models/fr_gtm/src/fr_gtm/system.py`：

| 块 | 类型 | 消费与产出 |
| --- | --- | --- |
| `actuator` | `SimpleActuator` | 只负责三个舵面，消费 `Root::u[:3]`，输出升降舵、副翼与方向舵位置，增益、速率限制与上下限来自配置 |
| `wind` | `Wind` 子系统 | 消费高度、机体系速度与姿态矩阵，输出机体系风与风角速度 |
| `mass` | `Mass` | 按运行时不确定度缩放重量与惯量，输出 `mass`、`inertia`、`inertia_inv` |
| `aero` | `Aero` | 消费三个舵面位置、位置、速度、姿态、角速度与风，输出气动力、力矩、气流角与系数 |
| `engine` | `Engine` | 消费油门指令 `Root::u[3]`、角速度、压力比与动压，内部可配置滞后先给出有效油门，再输出推力、冲压阻力与陀螺力矩 |
| `eom` | `RigidBody6DOFQuaternion` | 打开 `enable_external_mass`、关闭大地坐标输出，积分刚体运动 |
| `sensor` | `Sensor` | 汇总 17 个物理反馈量，直接作为 `Root::y` 与 `Root::clean_y`，不追加命令 |
| `ground` | `GroundContact` | 仅在 `cfg.ground.enabled` 为真时挂载，输出法向力、穿透量与摩擦力分量 |

高度取 `-pos_ned[2]`，合力是气动力、发动机推力、地面力与旋转后的重力之和，合力矩只含气动力、发动机力矩与地面力矩；过载 `nz` 用非重力合力除以标称重量，符号取 FRD 下的向上为正。

油门不进作动器：`Root::u[3]` 直接进入 `Engine` 的可配置滞后，滞后输出的有效油门既决定推力，也作为第四个作动器位置通道写进观测中 `clean_y[13:17]`。气动块直接接收三个舵面实际位置，观测中的相应分量为 `clean_y[13:16]`。

## 规模

| 量 | 大小 | 组成 |
| --- | --- | --- |
| 状态 `x` | 25（原二阶为 26） | 刚体 13 + 舵面 3 + 发动机滞后 1（或 2）+ Dryden 8 |
| 离散状态 `xd` | 0 | 模型没有额外的离散状态 |
| 输入 `u` | 4 | 升降舵、副翼、方向舵（度）与油门（百分比） |
| 扰动 `d` | 4 | Dryden 白噪声 `[nu, nv, nw, np]` |
| 噪声 `n` | 17 | 物理反馈的标准化传感器噪声 |
| 参数 `p` | 33 | 风 5 + 不确定度 28 |
| 观测通道 | 17 | 13 个物理量加三个舵面与有效油门 |
| 因果反馈 `y` | 17 | 不含四个指令遥测，便于嵌套控制器 |

`layout.py` 提供 `U/D/N/Y_DIM`（4/4/17/17）和对应的 `*_INDEX` 通道索引。状态、参数和打包输出维数以编译模型布局为准，不另设固定常量。固定尺度倍率 `SCALE_FACTOR=1/0.055` 位于 `config.py`，不作为用户缩放参数。

## 这一章用到的 API

- [`fr_gtm.GTMConfig`](/api/gtm/config)：`fullscale`、`subscale` 与组件配置类
- [`fr_gtm.GTMSystem`](/api/gtm/system)：根系统的装配与 `fr.export` 的输出面
- [`fr_gtm.Mode`](/api/gtm/components/aero)：襟翼与起落架构型的四个取值
- [`fr_gtm.layout`](/api/gtm/layout)：路径目录与控制、扰动、观测通道索引
- [`fly_ruler_core.compiler`](/api/core/compiler/index)：编译选项与目标后端

## 边界与常见坑

| 边界 | 说明 |
| --- | --- |
| 尺度与构型是编译期的 | `is_fullscale` 与 `mode` 参与气动表派生，运行时改不了，要靠 `p` 加不确定性走 |
| 默认构型没有起落架 | 需要地面接触时传 `enable_ground=True`，否则没有 `Root::ground_*` 诊断 effects |
| 油门不是作动器通道 | `ActuatorConfig` 只描述三个舵面；给初值时油门交给 `init(..., command=(0, 0, 0, throttle))` 或直接写 `Root.engine.lag::lag_state` |
| 气动插值选择 | `cfg.aero.interpolation_method` 默认 `"linear"`；显式 `"cubic"` 使用统一 C² 样条，设定后重新编译 |
| 样条表外延拓 | `cfg.aero.extrapolation_method` 默认 `"polynomial"`；显式 `"linear"` 保留 cubic 表内样条，表外沿边界切线延拓；设定后重新编译 |
| 气动表来自包内资源 | 默认表在 `models/fr_gtm/src/fr_gtm/data/aero_tables.npz`，换表通过工厂的 `data_path` 参数加载并重新编译 |
| 对外用度、块内用弧度 | 舵面与气流角对外是度，稳态风与 Dryden 内部消费弧度 |
| 尺度切换要重新配平 | 缩比模型与全尺寸模型的配平点不同，切换后重新跑 `just example-trim` |

## 相关页面

- [初值、输入与参数](/guide/components/gtm/02-state-input)：默认 25 元状态与 33 维参数怎么给
- [推进与记录](/guide/components/gtm/03-simulate)：编译产物怎么推进
- [gtm 实现](/dev/components/gtm/01-architecture)：装配分层与语义边界
- [gtm 配置参考](/api/gtm/config)：`GTMConfig` 的完整字段
