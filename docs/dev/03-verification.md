# 验证与扩展

GTM 的回归分三层：结构回归看路径与维度，数值回归看与外部 `gtm_ad` 冻结参考的偏差，跨后端回归看四个后端在整条轨迹上的漂移。这一章给出改动到命令的映射，以及扩展时的落点。

## 扩展点

| 改什么 | 主要影响 | 先跑什么 |
| --- | --- | --- |
| `components/aero.py` 与 `data/aero_tables.npz` | 系数与载荷，配平点可能移动 | HyKal 先跑 `just test-one tests/test_hykal.py`，再 `just test-one tests/test_reference.py` 与 `just example-trim` |
| `config.py` 的字段与默认值 | 配置派生、组件几何注入、缩放与构型 | `just test-one tests/test_config.py` |
| `system.py` 的块接线与 `fr.export` | 完整路径、观测布局、状态维数 | `just test-one tests/test_system.py` |
| `components/mass.py` | 重量与惯量，配平与动导都不变但力矩尺度变 | `just test-one tests/test_reference.py` |
| `components/engine.py` 与 `config.py` 的两种发动机滞后配置 | 默认一阶/可选二阶油门滞后、推力与发动机力矩 | `just test-one tests/test_engine_first_order.py`、`just test-one tests/test_engine_lag.py`、`just test-one tests/test_reference.py` 与 `just example-backends` |
| `components/sensor.py` | 17 通道反馈与噪声通道映射 | `just test-one tests/test_system.py` |
| `components/wind.py` | 风、切变与 Dryden 门控 | `just test-one tests/test_reference.py`、`tests/test_backends.py` |
| `interface/stick.py`、`interface/proto.py` | 设备轴到命令、真值坐标/单位、遥测字段与顺序 | `just test-one tests/test_stick_input.py`、`just test-one tests/test_proto_projection.py` |
| 跨后端数值 | 四个后端一致性 | `just test-one tests/test_backends.py` |

改动后按 `just check-types`、`just check-lint`、`just check` 的顺序自查，再跑 `just pre-commit` 走完整门禁。

## 独立仓 CI

`.github/workflows/ci.yml` 只检出 `fr_gtm`，在 Ubuntu / Python 3.12 上分开执行 `just check`、逐文件回归及离线示例、wheel/sdist 构建与安装验收。NumPy、CasADi、native、CPU JAX 和 CPU PyTorch 使用真实后端，不需要主仓环境、外部 `gtm_ad` 或硬件设备。

依赖声明使用 `fly-ruler>=0.1.7`；CI 的 `--upgrade-package fly-ruler` 每次从 PyPI 解析最新稳定版，不使用锁文件中的 SDK 固定结果。其余依赖优先复用锁文件，后续命令通过 `UV_NO_SYNC=1` 使用同一环境。打包验收分别将 wheel 和 sdist 安装到新的虚拟环境，用 `python -I scripts/check_install.py` 检查导入来源、公开 SDK 来源、两尺度四构型资源和十步仿真；脚本必须由该验收环境的 Python 执行，而非开发环境。

工作流只上传构建产物，不发布 PyPI。CUDA 项在无 GPU runner 上按测试条件跳过；真实设备与网络服务仍使用单独的联调流程。

## 冻结参考

| 文件 | 生成方式 | 覆盖范围 |
| --- | --- | --- |
| `tests/data/gtm_reference.npz` | `scripts/generate_reference.py`，从外部 `gtm_ad` 导出 | 两套尺度与四种构型，另加风、不确定度与传感器三个额外用例 |
| `tests/data/gtm_backend_trajectory.npz` | `tests/generate_gtm_backend_reference.py` | 逐步轨迹基线，用于跨后端漂移比较 |

冻结参考可以重新生成，但重生成意味着接受新的数值基线：先确认改动是有意为之，再在提交说明里写清是表格、配置还是方程的差异。

HyKal 已恢复 NASA 原始精确零判据与 R 覆盖 P 的顺序，旧实现的 $10^{-10}$ 死区与 P 优先不再是验收合同；两份冻结 NPZ 保留，不以重生成或放宽容差消除差异。边界规则由 `tests/test_hykal.py` 的独立标量转录对照；近零方向分母会放大度/弧度换算的一个 ulp 差异，因此该测试读取各后端三角值，独立计算分支、剩余量和尺度。重新使用该模型的 true/pred 或其他实例时，必须从修正后的同一源声明重新编译；旧缓存/归档仍包含旧规则。

两份参考都把油门建成一阶作动器状态，原二阶模型用例显式选择 `JetCatThrottleLagConfig`，因此 `tests/test_reference.py` 逐通道比对时跳过参考布局里的油门导数槽，改用配平反解得到的解析式核对；`tests/test_backends.py` 的轨迹用例油门指令恒为 40%，默认一阶滞后停在配平状态，整条轨迹仍与参考一致。新增的 `tests/test_engine_lag.py` 承担该通道自身的解析回归。

冻结参考的有效油门观测通道不施加噪声（参考生成源 `gtm_ad/core/noise/sensor.py` 固定该通道 noise_std=0）。当前 `SensorConfig.throttle` 的默认标准差为 1 个油门百分比点，因此参考夹具显式采用零油门噪声标定；`test_current_throttle_noise_calibration` 独立验证当前默认值。两种标定不混为同一数值合同，冻结 NPZ 保持原值。

## 回归覆盖

| 测试 | 覆盖内容 |
| --- | --- |
| `tests/test_hykal.py` | NASA 独立规则对照；±1e−10 邻域及等号、斜缝两侧、普通/零速率、微小非零分母、零分母、R 覆盖与重构；NumPy、CasADi SX/MX、native、JAX 及零分母符号 Jacobian |
| `tests/test_aero.py` | 独立气动块、单一配置、显式几何、载荷换算与元数据 |
| `tests/test_config.py` | `Mode` 四种构型乘两套尺度都能装载，扁平配置与几何注入与非法输入 |
| `tests/test_model_tools.py` | 根/嵌套路径、一阶/二阶与两尺度初值、有风来流、参数、Gaussian 噪声、观测解析、Graph 服务 |
| `tests/test_init_jax.py` | 一阶/二阶原生初始化、jit/vmap、正反向导数与有限差分、有风初值及 PRNG 输入 |
| `tests/test_trim_tools.py` | 三轴物理平衡、目标释放、风/参数/初猜、一阶/二阶及后端布局映射 |
| `tests/test_system.py` | 状态、输入、输出路径与维度，`PyCompiler` 产物的结构一致性 |
| `tests/test_reference.py` | 与冻结参考的数值对比，油门导数槽按解析式单独核对 |
| `tests/test_engine_first_order.py` | 默认配置、滞后模型切换、一阶解析响应与五后端一致性 |
| `tests/test_engine_structure.py` | Block/配置类型对应、原始系数、独立载荷解析式及嵌套 组件几何注入 |
| `tests/test_engine_lag.py` | 二阶油门滞后的解析阶跃响应、配平初始化、指令钳位与时间尺度缩放 |
| `tests/test_stick_input.py` | 无硬件快照、轴方向/别名、死区、不对称行程、油门区间与缺失必需轴 |
| `tests/test_proto_projection.py` | 四种构型、NumPy/CasADi 真值、传感器误差隔离、WGS-84 投影与遥测 schema/取值顺序 |
| `tests/test_backends.py` | NumPy、CasADi、JAX 与 native 在轨迹上的偏差 |

示例脚本承担端到端检查：`just test-examples` 串联六个离线脚本，`just test-examples-stick` 跑手柄 dry-run，`just example-ground --scenario takeoff` 与 `just example-ground --scenario touchdown` 覆盖地面场景。

## 外部接口的数值边界

`interface/stick.py` 的快照接口用 Protocol 描述 fly_stick 已有的 `get_alias_axes`，可以用明确类型的无硬件快照测试映射；实际运行仍读取 DevicePool 完整快照。中心死区内输出零，其余归一化为 `sign(a) max(|a|-d,0)/(1-d)`，再按符号分别乘正/负舵面行程及 control_scale。油门不施加中心死区，线性映射发动机指令边界。三舵面缺失会报错，油门缺失是显式支持的固定指令模式。

`interface/proto.py` 是主机数值边界，不参与模型图或自动微分。x 给出 NED/FRD 状态，clean_y 提供无传感器误差的实际量，effects 补充气动与航迹诊断。`EffectsTelemetry` 在构造时绑定有序完整路径，schema 的字段 ID 使用 `完整路径[多维下标]`；标量省略下标。取值和字段生成均使用 C 顺序，矩阵不会被隐式转置。

地理投影与 std `ned_position_to_geodetic` 使用同一个局部曲率近似。WGS-84 长半轴 `a=6378137 m`、第一偏心率平方 `e²=6.6943799901413165e-3`，在原点纬度 φ₀ 处令 `q=1-e² sin²φ₀`，子午圈半径 `Rm=a(1-e²)/q^(3/2)`、卯酉圈半径 `Rn=a/sqrt(q)`。NED 位移 `(n,e,d)` 转为 `Δφ=n/(Rm+h₀)`、`Δλ=e/((Rn+h₀)cosφ₀)`（弧度），高度为 `h₀-d`；接口最后把纬经度转为协议需要的度。该近似用于原点附近，经纬度不是长距离地球导航解算。

两个模块只在显式导入时加载各自的可选依赖。模型顶层不重导出这些入口；设备重连、网络会话、实际频率与时间戳属于调用方。实时示例将仿真状态保存在带后端类型的闭包中，按 input→dynamics→output 调度；记录使用步末时间，扳机事件只在按下边沿触发。

## 与 std、core 的边界

- std 的作动器、六自由度、大气、风与地面接触块有自己的回归，模型侧只验证接线与参数传递，不去重复 std 的数值断言。
- core 的编译、后端与产物门禁由内核仓库负责，GTM 侧验证的是路径、维度与数值差异。
- GTM trim 在 CasADi 条件包装中求解，并按路径返回目标后端类型；线性化仍使用 CasADi 模型。

## 相关页面

- [装配与语义边界](/dev/components/gtm/01-architecture)：配置与接线
- [块方程与气动数据](/dev/components/gtm/02-aero-data)：方程与数据契约
- [gtm 组件](/guide/components/gtm/06-troubleshooting)：面向使用的排障
