# 排障

GTM 的多数「异常」来自单位、符号与路径约定，而不是数值实现。这一章按现象给出排查顺序。

## 现象对照

| 现象 | 先查什么 |
| --- | --- |
| 姿态或高度与预期相反 | 约定是 NED 导航系、FRD 机体系与 scalar-first 四元数，高度取 `-pos_ned[2]` |
| 找不到变量路径 | 用 `model.x.keys`、`model.u.keys`、`model.y.keys` 打印实际路径；`Root.eom::`、`Root.actuator::`、`Root::u` 都是完整路径 |
| 风的方向不对 | 公开的 `wind_direction` 单位是弧度，从 NED 北向顺时针度量；冻结参考轨迹里存的是度 |
| 风完全没有作用 | `turbulence_level` 与 `wind_speed` 都是零，或 `turbulence_level` 未超过 `1e-3` 的门控阈值 |
| 后端之间数值有差异 | 先用同一初值与 `dt` 对比；JAX 默认 float32，只有高精度参考要求 float64 时才需显式启用 x64，CPU 策略见 `backend_compare.py` |
| 读了 `Root::ground_*` 报错 | 配置里没打开地面接触，重建配置时传 `enable_ground=True` |
| 配平不收敛 | 检查 `TrimSpec` 的固定状态、约束导数路径与初始猜测，三个舵面位置、发动机滞后一或两个状态与八个 Dryden 状态需要一起约束 |
| 线性化的矩阵形状不符合预期 | `x_keys`、`u_keys`、`y_keys` 的数量与顺序决定 A、B、C、D 的布局 |
| native 后端第一次运行很慢 | 首次会编译并缓存 C++/pybind11 扩展，后续运行复用缓存 |
| 在 NumPy 后端上配平报错 | GTM trim 内部使用 CasADi；线性化需用 `target=fr.targets.casadi` 编译 |
| 手柄输入无响应 | 确认 `--profile` 指向的设备文件与硬件匹配，以及 `--no-stick` 是否被打开 |
| 缩比模型数值对不上 | `is_fullscale` 与 `mode` 是编译期配置，切换后要重新编译并重新配平 |

## 先做这三件事

1. **核对约定**：高度取 `-pos_ned[2]`，四元数标量在前表示机体系相对 NED，舵面与气流角对外是度、风参数对外是弧度。
2. **打印路径清单**：`just example-inspect` 给出状态、输入与输出的完整路径，任何 `KeyError` 或路径歧义都先在这里对齐。
3. **分离随机源**：把 `Root::d` 与 `Root::n` 置零、参数保持默认，先确认确定性响应符合预期，再逐项加回风、紊流、不确定度与噪声。

## 需要更多信息时提供的信息

- 运行命令、`just --list` 里的 recipe 名与完整报错堆栈
- 配置来源：`GTMConfig.fullscale()` 或 `GTMConfig.subscale()` 的构型、资源路径及组件覆盖值
- 编译设置：`integrator`、`dt`、target 后端与是否配置过 `JaxRuntimePolicy`
- 出问题的完整路径、期望值与实测值，以及当时的初值、输入与参数
- 是否使用手柄或 Proto 脚本、设备配置文件路径与是否 dry-run

## 相关页面

- [初值、输入与参数](/guide/components/gtm/02-state-input)：路径、单位与顺序
- [配平、线性化与后端对比](/guide/components/gtm/05-analysis)：求解器与后端差异
- [gtm 实现](/dev/components/gtm/03-verification)：回归覆盖与扩展点
- [gtm 组件](/guide/components/gtm)：回到概览与教程目录
