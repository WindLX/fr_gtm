# 配平、线性化与后端对比

`fr_gtm.trim` 按飞行条件求三轴局部定常配平，并将结果映射回调用模型的后端与布局；线性化使用 CasADi。这一章说明目标、残差与工作点的用法。

## 配平

`trim(model, cfg, altitude=1000.0, tas=100.0)` 求完整力/力矩及内部状态平衡，不是只求作动器零导数。`cfg` 必须与编译模型一致：

<<< @/../models/fr_gtm/examples/trim_linearize.py#05-analysis-1

| 参数 | 含义 |
| --- | --- |
| `altitude`、`tas` | 必填，高度 m、真空速 m/s；TAS 必须为正 |
| `gamma` | NED 航迹角，deg，默认 0 |
| `psi` | 机头偏航角，deg，默认 0；不是地速航向 chi |
| `phi` | 滚转角，deg，默认 0 |
| `p` | 固定运行时参数向量，省略用模型默认参数；可设置风和不确定度 |
| `guess` | 可选 `(x, xd, u)` 初猜 |
| `paths` | 目标模型实例的 `GTMPaths`，默认根实例 |

`gamma/psi/phi=None` 分别释放该条件，侧滑角始终不额外固定。位置固定为 `(0,0,-altitude)`，角速率固定零，机体速度、角速率、舵面、发动机和 Dryden 状态导数为零；位置导数允许前进和爬升。舵面/指令边界来自配置，d/n 固定零。非零 gamma 是给定高度处的局部爬升条件，非零 phi 不自动建立协调转弯。

实现只在外层加标量条件观测包装，内部复用生产 `GTMSystem` 和 std `TrimSolver`；求解使用 CasADi，但返回的 `x/xd/u/p` 保留传入模型的后端类型。默认初猜由 `init` 构造，也可以复用上一次结果。

`result.success` 同时要求求解器成功和每项物理/内部状态/目标残差不超过 1e-6。`result.residuals` 使用对应物理单位（m、m/s、deg、m/s²、rad/s² 或内部状态每秒变化率）；`result.solver` 保留 std 统计，不掩盖求解失败：

<<< @/../models/fr_gtm/examples/trim_linearize.py#05-analysis-2

## 线性化

线性化的操作点直接取自配平结果，`x_op` 与 `u_op` 是路径到值的字典，`x_keys`、`u_keys`、`y_keys` 决定 A、B、C、D 的行列顺序：

<<< @/../models/fr_gtm/examples/trim_linearize.py#05-analysis-3

示例只取速度、角速度、三个舵面与发动机滞后状态这四个状态路径，加一个输入路径，`y_keys` 用 `Root::clean_y` 保持 17 通道布局。返回结果的 `A`、`B`、`C`、`D` 形状由键的数量与观测布局决定。

## 后端对比

四种后端共享同一段声明，差异来自 lowering 与浮点求和顺序。`backend_compare.py` 用同一次求值对比 NumPy、CasADi 与可选的 JAX，打印编译耗时以及相对 NumPy 的 `max|xdot|` 与 `max|y|`：

<<< @/../models/fr_gtm/examples/backend_compare.py#05-analysis-2

`backend_compare.py` 为高精度后端比较显式配置 CPU 与 float64：

<<< @/../models/fr_gtm/examples/backend_compare.py#05-analysis-1

native 后端在另外两个示例里出现：`ground_operations.py` 用 `compiler.convert(symbolic, target=fr.targets.NativeTarget())` 把 CasADi 符号结果转成 C++ 扩展，`benchmark_jax_compile.py` 单独度量 JAX 的跟踪、lower 与编译耗时，结果写进 `build/jax_compile_benchmark.json`。

产物检查用 `inspect_model.py`：`model.artifact.path` 给产物目录，`artifact.file(fr.ArtifactFile.SYSTEM_IR / BACKEND_SOURCE / LAYOUT)` 给拓扑、生成源码与布局文件，`artifact.cache_hit` 说明是否命中缓存。

## 产物与示例

| 命令 | 产物或输出 |
| --- | --- |
| `just example-trim` | 飞行配平的物理残差与 A、B、C、D 形状 |
| `just example-backends` | 三个后端的编译耗时与相对 NumPy 的偏差 |
| `just example-ground --scenario takeoff --backend native` | C++/pybind11 扩展缓存与地面场景记录 |
| `just benchmark-jax-compile` | `models/fr_gtm/build/jax_compile_benchmark.json` |
| `just example-inspect` | 产物路径、缓存命中、完整路径清单与 Graph WebUI |

## 这一章用到的 API

- [`fr_gtm.trim`](/api/gtm/trim)：飞行条件与后端布局映射
- [`fly_ruler_std.runtime.trim.TrimSolver`](/api/std/runtime/trim/index)：配平求解与统计
- [`fly_ruler_std.runtime.trim.TrimSpec`](/api/std/runtime/trim/index)：固定值、导数目标与初猜的写法
- [`fly_ruler_std.runtime.linearize.LinearizationSolver`](/api/std/runtime/linearize/index)：A、B、C、D 的求解
- [`fly_ruler_std.runtime.linearize.LinearizationSpec`](/api/std/runtime/linearize/index)：操作点与键
- [`fly_ruler_core.JaxRuntimePolicy`](/api/core/jax_runtime)：JAX 平台、预分配与 x64 开关
- [`fly_ruler_core.ArtifactFile`](/api/core/artifacts/index)：产物文件的枚举

## 边界与常见坑

| 边界 | 说明 |
| --- | --- |
| 线性化使用 CasADi | GTM trim 内部编译 CasADi 包装，结果返回原后端；LinearizationSolver 仍要求 CasADi 模型 |
| 路径写错抛 `KeyError` | `TrimSpec`、`LinearizationSpec` 里的路径先经 `model.x.keys` 与 `model.u.keys` 核对 |
| 初始猜测影响收敛 | 八个 Dryden 状态、三个舵面位置与发动机滞后状态要一起猜，离工作点太远会不收敛 |
| 键的顺序决定矩阵布局 | `x_keys`、`u_keys`、`y_keys` 的顺序与 A、B、C、D 的行列一一对应 |
| 后端偏差超出预期 | 对比时用同一初值与 `dt`；JAX 默认 float32，若参考要求 float64 则在首次导入前显式启用 x64 |
| native 首次编译慢 | 首次生成并缓存 C++ 扩展，后续运行复用缓存 |

## 相关页面

- [初值、输入与参数](/guide/components/gtm/02-state-input)：操作点的构造
- [地面运行、风与扰动](/guide/components/gtm/04-scenarios)：带扰动的场景
- [core 后端与代码生成](/dev/components/core/04-backends-codegen)：四种后端在编译层的差异
- [gtm 实现](/dev/components/gtm/03-verification)：配平与线性化的回归覆盖
