# 推进与记录

编译产物提供两个求值入口：`discrete_fn` 走完整积分步，`continuous_fn` 只求一次导数与观测。这一章说明两者的签名、观测怎么按路径读回、记录怎么落盘。

## 一步推进

`model.discrete_fn(t, x, xd, u, p)` 返回 `(x, xd, y, effects)`，其中返回的 `y` 打包两个各 17 维的根端口，总长为 34。模型没有额外离散状态，`model.xd.build()` 仍是必须传入的参数：

<<< @/../models/fr_gtm/examples/simulate.py#03-simulate-1

`model.continuous_fn(t, x, xd, u, p)` 返回同形状的四元组，第一项是状态导数。它适合检查某一点的气流角与载荷，例如在给定风与不确定度下取一次输出：

<<< @/../models/fr_gtm/examples/disturbances.py#03-simulate-1

## 按路径读回输出

`model.y.read(result.y, by="path")` 只返回两个根端口：`Root::y` 与 `Root::clean_y`，各 17 维；地面接触开关不改变这两个端口或总输出长度 34。

| 观测分量  | 内容与单位                                            |
| --------- | ----------------------------------------------------- |
| `[0:5]`   | 航迹角、迎角、侧滑角（deg）、真空速（m/s）、高度（m） |
| `[5:8]`   | 欧拉角 roll/pitch/yaw（deg）                          |
| `[8:11]`  | 机体角速率 p/q/r（deg/s）                             |
| `[11:13]` | 向上为正的法向过载、动压（Pa）                        |
| `[13:17]` | 三个实际舵面位置（deg）与有效油门（%）                |

`y` 包含传感器偏置、比例与噪声，`clean_y` 是同布局真值。两个端口均不拼接命令，闭环控制直接使用 `y`；命令从输入 `Root::u` 读取和记录。分量名称与索引见 `fr_gtm.layout.Y_INDEX`。

状态从 `model.x.read(x, by="path")` 读取，例如 `Root.eom::pos_ned`、`Root.eom::vel_body`、`Root.eom::quat`。组件 out 中未被状态、物理观测或子组件已有 effects 覆盖的诊断，统一保留在 effects；不另加四元数导数或重力合力等计算项。

额外内部诊断用 `model.effects.read(result.effects, by="path")`，根和子组件各记录一份，不重复声明：

| 分组 | effects 路径 |
| --- | --- |
| 气流/大气 | `Root::VE`、`Root::mach`、`Root::pressure_ratio`、`Root::wind_omega`；总风由 `Root.wind::steady` 与 `Root.wind::gust` 相加 |
| 质量 | `Root::weight`、`Root::mass`、`Root::inertia`、`Root::inertia_inv` |
| 刚体诊断 | `Root::DCM`、`Root::vel_ned`、`Root::chi`（rad）、`Root::acceleration_body`、`Root::angular_acceleration_body` |
| 载荷 | `Root::force_aero`、`Root::force_engine`、`Root::moment_aero`、`Root::moment_engine` |
| 气动表诊断 | `Root.aero::coefficients`（六元系数）、`Root.aero::reynolds` |
| 地面（开启时） | `Root::ground_force_body`、`Root::ground_moment_body`、`Root::ground_penetrations`、`Root::ground_normal_forces`、`Root::ground_longitudinal_forces`、`Root::ground_lateral_forces`、`Root::ground_contact_count` |

```python
result = model.continuous_fn(0.0, x, xd, u, p)
from fr_gtm import GTMPaths, read_observation

paths = GTMPaths()
observation = read_observation(model, result.y)
effects = model.effects.read(result.effects, by="path")
altitude, airspeed = observation.altitude_m, observation.tas_mps
force_aero = effects[paths.effects.aero_force]
```

`acceleration_body` 是机体系速度分量的时间导数 `F/m - omega×v`（m/s²），不是去重力的加速度计比力；角加速度单位为 rad/s²。

组件 out 的覆盖关系如下；已经覆盖的量不再起别名写入 effects：

| 组件 | 从状态/观测读取 | 额外 effects |
| --- | --- | --- |
| actuator | 实际位置：`clean_y[13:16]`，或作动器状态 | 无 |
| wind | — | 根角阵风；子系统自身的稳态风/阵风，不重复总风 |
| mass | — | 重量、质量、惯量及逆矩阵 |
| aero | VTAS、alpha、beta、qbar：`clean_y` 对应分量 | 根气动力/力矩、VEAS、Mach、压力比；块自身的系数/雷诺数 |
| engine | 有效油门：`clean_y[16]` | 发动机力/力矩 |
| eom | 位置、机体速度、四元数、角速率：`x`；欧拉角和航迹角：`clean_y`（deg） | DCM、NED 速度、航向角、平动/角加速度 |
| sensor | `y`、`clean_y` 就是两个传感器输出 | 无 |

`read_observation` 默认读无误差的 `clean_y`，`measured=True` 读取包含偏置、尺度因子和噪声的 `y`。返回冻结的 `Observation`，角度和角速率分别使用 deg、deg/s；该函数在边界转为主机浮点值，不用于 JAX tracing。

`inspect_model.py` 使用 `from fly_ruler_std.runtime import graph as fr_graph`，通过 `fr_graph.serve(model, open_browser=True)` 打开 Graph WebUI，Ctrl+C 退出。`just example-inspect --no-graph` 只打印布局，`--port 18002` 可更换端口；离线示例门禁不启动交互服务。

## 记录

`Recorder(model, output_dir)` 收集每一步的状态、导数、观测、输入与参数，`save(name)` 写成 parquet：

<<< @/../models/fr_gtm/examples/simulate.py#03-simulate-2

记录列名按 `类别:完整路径[分量]` 展开，回读用 `fly_ruler_std.runtime.record.Loader`。文件结构与列名规则见[记录与回读](/guide/runtime/recording)。

## 示例与 recipe

`models/fr_gtm/examples/` 下的脚本都有对应 recipe：

| 脚本 | recipe | 内容 |
| --- | --- | --- |
| `models/fr_gtm/examples/inspect_model.py` | `just example-inspect` | 打印产物、路径清单并打开 Graph WebUI |
| `models/fr_gtm/examples/simulate.py` | `just example-simulate` | 1000 m 与 100 m/s 起做 100 步并落盘 |
| `models/fr_gtm/examples/disturbances.py` | `just example-disturbances` | 稳态风、Dryden 紊流、不确定度与噪声 |
| `models/fr_gtm/examples/ground_operations.py` | `just example-ground --scenario <scenario>` | `takeoff` 与 `touchdown`，支持四种后端 |
| `models/fr_gtm/examples/backend_compare.py` | `just example-backends` | NumPy、CasADi 与可选 JAX 的同点对比 |
| `models/fr_gtm/examples/trim_linearize.py` | `just example-trim` | 飞行配平并打印 A、B、C、D 形状 |
| `models/fr_gtm/examples/benchmark_jax_compile.py` | `just benchmark-jax-compile` | 度量 JAX 跟踪、lower 与编译耗时 |
| `models/fr_gtm/examples/fly_stick_gtm.py` | `just example-fly-stick` | 飞行手柄驱动模型并推送 Proto 遥测 |
| `models/fr_gtm/examples/fly_stick_profiles/t16000m.toml` | — | T.16000M 设备映射，供 `--profile` 加载 |

离线脚本由 `just test-examples` 串联执行，手柄脚本的 dry-run 由 `just test-examples-stick` 执行。

## 这一章用到的 API

- [`fly_ruler_core.CompiledModel`](/api/core/runtime/index)：`discrete_fn` 与 `continuous_fn` 的签名
- [`fly_ruler_std.runtime.record.Recorder`](/api/std/runtime/record/index)：记录与落盘
- [`fly_ruler_std.runtime.record.Loader`](/api/std/runtime/record/index)：按列名回读
- [`fly_ruler_core` 输出与变量视图](/api/core/index#variableview)：`model.y.read(y, by="path")` 的路径语义

## 边界与常见坑

| 边界 | 说明 |
| --- | --- |
| `xd` 必须传 | 模型没有离散状态，但 `discrete_fn` 与 `continuous_fn` 的签名要求传入 `model.xd.build()` |
| `p` 可以给默认值 | `model.p.default()` 与 `model.p.build()` 都把参数置零扰动，需要非零扰动时显式覆盖 |
| 观测是平坦向量 | `y` 只在 `model.y.read(y, by="path")` 之后才可按名取用，直接切片的列顺序不保证稳定 |
| 噪声通道是标准化的 | `Root::n` 是标准化噪声，标准差由配置决定，乘 1 不等于物理量纲 |
| 地面输出按配置出现 | `enable_ground=False` 时七个 `ground_*` 路径不存在，读它们会报错 |

## 相关页面

- [初值、输入与参数](/guide/components/gtm/02-state-input)：`x`、`u`、`p` 的构造
- [地面运行、风与扰动](/guide/components/gtm/04-scenarios)：场景化用法
- [记录与回读](/guide/runtime/recording)：列名规则与 `Loader`
- [gtm 实现](/dev/components/gtm/01-architecture)：这些输出在装配层怎么导出

## 接入手柄与协议遥测

在 GTM 目录执行 `uv sync --extra stick --extra proto`，再运行 `just example-fly-stick`。免硬件、免网络的调度检查用 `just example-fly-stick --dry-run --backend numpy --duration 0.05`。

读取 `fly_stick.PyDevicePool` 的完整快照后调用转换函数：

```python
from fr_gtm import build_input
from fr_gtm.interface.stick import StickConfig, to_command

mapping = StickConfig(control_scale=0.6, axis_deadband=0.03)
command = to_command(snapshot, description, cfg, config=mapping)
u = build_input(model, command)
```

profile 负责硬件 code 到轴别名的映射，`StickConfig` 负责轴方向、死区和舵面幅度。输出顺序是升降舵、副翼、方向舵、油门，单位 deg/deg/deg/%；正负行程分别映射作动器上下限。默认轴别名为 ABS_Y、ABS_X、ABS_RZ、ABS_THROTTLE，方向均取负；舵面死区 3%，幅度 60%。必需轴缺失时报错；油门轴缺失时使用 `nominal_throttle=40`，也可以显式将 `throttle_axis=None`。设备故障交给运行循环处理。

飞机状态和诊断遥测分开转换。先选择诊断并在建立连接时注册结构：

```python
from fly_ruler_proto_python import FlyRulerClient
from fr_gtm import GTMPaths
from fr_gtm.interface.proto import EffectsTelemetry, to_aircraft_state

paths = GTMPaths()
telemetry = EffectsTelemetry(
    model,
    paths=(paths.effects.aero_force, paths.effects.engine_force, paths.effects.inertia),
    nominal_rate_hz=20.0,
)
with FlyRulerClient(address, name, telemetry_schemas=(telemetry.schema,)) as client:
    result = model.continuous_fn(t, x, xd, u, p)
    state = to_aircraft_state(
        model, cfg, x=x, y=result.y, effects=result.effects,
        origin=(31.1434, 121.8052, 0.0), paths=paths,
    )
    client.update_state(state, timestamp=timestamp)
    client.publish_telemetry(
        telemetry.stream_id, telemetry.values(result.effects), timestamp=timestamp,
    )
```

`to_aircraft_state` 始终读取 `clean_y`，不受传感器偏置、比例或噪声影响。NED 位置、FRD 速度、四元数和角速度来自 x；实际舵面及有效油门来自 clean_y；Mach、EAS、NED 速度与 chi 来自 effects。角速率已经是 rad/s，气流角、gamma 和舵面由 deg 转 rad，油门由百分比转为协议 [0,1] 比例；左右副翼反号，襟翼比例按 NORMAL/半襟翼/全襟翼分别给 0/0.5/1。姿态为 scalar-first、BODY→NED。

`origin=(lat_deg,lon_deg,altitude_m)` 显式定义地理原点。纬经度用与 std 相同的 WGS-84 局部曲率近似，高度为原点海拔减 NED down；模型大气和配平高度仍使用模型自身的 NED 定义。`EffectsTelemetry` 的字段顺序是所选完整路径的顺序，各向量/矩阵按 C 顺序展开，字段单位与 description 来自布局。未启用 ground 时选择地面诊断会报错。`acceleration_body` 是 FRD 速度导数，作为诊断遥测发送，协议线加速度字段保留为空。

转换函数不连接设备或服务器，不管理时间戳与采样频率；实际调度、扳机按下边沿事件和资源关闭见 `fly_stick_gtm.py`。这两个可选模块按需导入，不从 `fr_gtm` 顶层加载。

- [手柄映射 API](/api/gtm/interface/stick)
- [状态投影与遥测 API](/api/gtm/interface/proto)
