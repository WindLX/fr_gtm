# fr_gtm

GTM 是[独立 Git 仓库](https://github.com/WindLX/fr_gtm)，使用自己的 `uv.lock` 与 `.venv`，依赖 PyPI 的 `fly-ruler>=0.2.0`。可单独克隆，也可作为 FlyRuler 工作区的 `models/fr_gtm` 子模块使用；不依赖父仓源码或虚拟环境。

当前源码使用 `compiler.prepare` / `compiler.compile` / `runtime.bind` 接口，需要公开发行的 `fly-ruler>=0.2.0`。安装和验证均从本仓解析 PyPI 依赖，不需要父工作区源码或本地候选 SDK。

```bash
git clone git@github.com:WindLX/fr_gtm.git
cd fr_gtm
```

在仓库根目录运行：

```bash
just setup        # 安装独立环境与开发依赖
just check        # 静态检查
just test         # 模型测试
just build        # 构建 wheel 和 sdist
```

CPU JAX 使用 `uv sync --extra jax`；CUDA 后端使用 `jax-cuda12` 或 `jax-cuda13` extra。根工作区的 setup、fmt、check-models、test-models 不管理 GTM。

手柄集成入口 `just setup-stick` / `just example-fly-stick` 通过可选依赖从 PyPI 安装 `fly-stick` 与 `fly-ruler-proto-python`，无需检出它们的源码。

模型章节正文位于本仓的 [用户手册](docs/guide/01-build.md)与[实现说明](docs/dev/01-architecture.md)，由 FlyRuler 文档站引用并生成包含源码片段和 API 的页面。

## 持续集成

`.github/workflows/ci.yml` 在推送 main、Pull Request 和手动触发时运行 Ubuntu / Python 3.12 检查，仅检出本仓。

- **Lint 与类型**：`just check` 覆盖源码、测试、示例和发行安装验收脚本。
- **回归**：`just test` 逐文件独立进程运行，覆盖冻结参考、NumPy/CasADi/JAX/native；随后运行离线示例和手柄 dry-run。CUDA、真实手柄和 Proto 服务联调不在这个 CPU 作业范围内。
- **打包**：构建 wheel/sdist，检查元数据；两种文件分别装进干净环境，从已安装包加载两尺度、四构型气动资源并运行十步仿真。产物保留为 Actions artifact，不自动发布。

CI 用 `uv sync --extra jax --extra stick --extra proto --upgrade-package fly-ruler` 主动解析 PyPI 的最新稳定 SDK；`>=0.2.0` 是当前依赖声明，不是版本锁定。签入的锁文件固定本次验收的 0.2.0。`UV_NO_SYNC=1` 让后续 just 命令复用本作业已解析的环境。安装验收也独立从 PyPI 解析 SDK，不引用父仓目录。

本地复现同样的 SDK 升级与静态检查：

```bash
uv sync --extra jax --extra stick --extra proto --upgrade-package fly-ruler
UV_NO_SYNC=1 just check
UV_NO_SYNC=1 just test
```

本地默认 `uv sync --locked` 可复现签入的锁文件；CI 的临时解析不会回写仓库。

## 发动机配置与状态

`components.engine.Engine` 是组装 System，内部由 `lag` 和 `loads` 两个 Block 组成。默认 `FirstOrderThrottleLagConfig(time_constant_s=1.5)`；原始二阶使用 `JetCatThrottleLagConfig()`。载荷参数集中在 `cfg.engine.loads`，滞后配置位于 `cfg.engine.lag`。多项式与尺度系数、指令限幅及有效油门不作输出限幅的约定保持原模型定义。

载荷配置不依赖 `AircraftConfig`，面积与重心在外部构造时显式传入：

```python
loads = EngineLoadsConfig.fullscale(S=aircraft.S, cg_pos=aircraft.cg_pos0)
engine = EngineConfig(loads=loads)
```

缩比使用 `EngineLoadsConfig.subscale(S=..., cg_pos=...)`。`GTMConfig.fullscale()` 与 `GTMConfig.subscale()` 负责传入当前面积和重心；发动机安装位置及安装角由载荷配置自身拥有。JetCat 使用 `A @ lag_state + B * command` 计算状态导数、`C @ lag_state` 计算有效油门，载荷使用整向量运算。

初始化、配平和读数使用 `Root.engine.lag::lag_state`，值由 `cfg.engine.lag.trim_state(throttle)` 给出；载荷输出在 `Root.engine.loads::force`、`Root.engine.loads::moment`，系统对外导出仍为 `engine.force`、`engine.moment` 和 `engine.effective_throttle`。旧编译产物需要从声明重新构建，不手工修改缓存。

## 整机与气动配置

```python
from fr_gtm import GTMConfig, Mode

full = GTMConfig.fullscale()
scale = GTMConfig.subscale(mode=Mode.FULL_FLAP_GEAR_DOWN, enable_ground=True)
```

两个工厂统一构造所有组件配置；`mode` 只存放在 `cfg.aero.mode`。`components/aero.py` 同时定义 `Aero` 与扁平的 `AeroConfig`：参考面积 `S`、共享的 `b/Cbar`、`cg_pos/ref_cg`、四个无量纲 `control_arm_ratios`、构型、插值选项与资源。气动配置与块实现位于同一模块。

气动块使用 `Aero("aero", cfg.aero)`；独立加载通过 `AeroConfig.load(S=..., b=..., Cbar=..., cg_pos=..., ref_cg=..., data_path=...)`，不依赖整机配置。表格不会写入配置序列化结果。更换整机气动表时向两个工厂传 `data_path`；直接修改组件字段不会触发其他组件的隐式重派生。

## 观测与诊断

`GTMSystem` 只导出 `y`、`clean_y`，各 17 维，只含物理反馈，不追加命令；控制器直接使用 `plant.y`，命令从输入 `u` 读取。根输入 `n` 为 17 路标准化传感器噪声。

```python
result = model.continuous_fn(0.0, x, xd, u, p)
from fr_gtm import GTMPaths, read_observation

paths = GTMPaths()
observation = read_observation(model, result.y)
effects = model.effects.read(result.effects, by="path")
altitude, airspeed = observation.altitude_m, observation.tas_mps
force_aero = effects[paths.effects.aero_force]
```

状态从 `model.x` 读取；组件 out 中未被状态/观测覆盖的质量、载荷、大气、刚体和地面诊断通过 effects 读取，气动系数与雷诺数分别在 `Root.aero::coefficients`、`Root.aero::reynolds`，不再作为根输出重复导出。

## 初始化、配平与查看模型

```python
from fr_gtm import init, trim

x, xd, u, p = init(
    model,
    cfg,
    position=(0, 0, -1000),
    airdata=(3, 0, 100),
    euler=(0, 3, 0),
    command=(-2, 0, 0, 40),
    wind=(12, 35, 2),
    uncertainty={"weight": 0.03},
)
result = trim(model, cfg, altitude=1000, tas=100, gamma=0, psi=0, phi=None, p=p)
if not result.success:
    raise RuntimeError(result.residuals)
x, xd, u, p = result.x, result.xd, result.u, result.p
```

位置用 NED 米，来流为 `(alpha_deg,beta_deg,TAS_mps)`，欧拉角用度，风为 `(速度_mps,风向_deg,强度)`；init 会补偿稳态风并初始化发动机平衡状态。配平的 gamma/psi/phi 可分别传 None 释放条件，侧滑角自由，不隐式执行协调转弯。

`build_input` 按命令和可选 d/n 生成后端输入；`sample_disturbance(rng)` 为 4 路标准差 π 的高斯样本，`sample_sensor_noise(rng)` 为 17 路标准差 1 的样本。`read_observation` 默认读 clean_y，`measured=True` 读测量输出。

`GTMPaths()` 提供状态、输入、参数、观测与 effects 的字符串目录；嵌套实例可指定 root，父系统接出 I/O 时另指定 io_root。U/D/Y/N 通道索引与维数位于 layout，固定 SCALE_FACTOR 位于 config，不保留 constants 模块。

`just example-inspect` 使用 `fr_graph`（`fly_ruler_std.runtime.graph`）打开 Graph WebUI；Ctrl+C 退出。只打印布局用 `just example-inspect --no-graph`，全部离线示例用 `just test-examples`，不启动手柄、Proto 或交互图服务。

## 手柄与 Proto 接口

```bash
uv sync --extra stick --extra proto
just example-fly-stick
just example-fly-stick --dry-run --backend numpy --duration 0.05
```

`fr_gtm.interface.stick.to_command(snapshot, description, cfg, config=StickConfig())` 接收 `fly_stick` 完整快照，输出 `[elevator_deg, aileron_deg, rudder_deg, throttle_percent]`。默认三舵面轴与油门轴均负向映射，舵面死区 3%、行程比例 60%；正负行程分别取作动器边界。必需轴缺失时报错，油门轴缺失时保持 40%。设备 profile、连接和失联处置由调用方管理。

`fr_gtm.interface.proto.to_aircraft_state(model, cfg, x=x, y=result.y, effects=result.effects, origin=(lat_deg, lon_deg, altitude_m))` 只使用 x、clean_y 与 effects 真值。状态和观测必须来自同一时刻；欧拉角/气流角和舵面按协议转为 rad，实际有效油门转为 [0,1]，不以 command 代替实际操纵面。地理原点必须显式给出；使用 WGS-84 局部曲率近似，适用于原点附近飞行。协议海拔为原点海拔减 NED down；模型大气与配平高度仍按模型自身的 NED 定义。

```python
from fr_gtm import GTMPaths
from fr_gtm.interface.proto import EffectsTelemetry
from fly_ruler_proto_python import FlyRulerClient

paths = GTMPaths()
telemetry = EffectsTelemetry(
    model,
    paths=(paths.effects.aero_force, paths.effects.engine_force, paths.effects.inertia),
    nominal_rate_hz=20.0,
)
with FlyRulerClient(address, name, telemetry_schemas=(telemetry.schema,)) as client:
    client.publish_telemetry(telemetry.stream_id, telemetry.values(result.effects))
```

遥测只发送显式选择的完整 effects 路径；字段的单位和说明来自编译布局，向量、矩阵按 C 顺序展开。连接前注册 schema，之后以相同顺序取值发送。`acceleration_body` 为 FRD 速度导数，可选作诊断遥测；协议线加速度字段保留为空。网络连接、时间戳、发送频率和按钮事件均由示例负责。接口模块不会从包顶层自动加载，未安装 extras 时仍可正常使用模型、init 与 trim。

## JAX 可微初始化

`init` 保留主机输入校验；JAX 批量构造和求导使用预编译的 `Initializer(model, cfg)`，在变换外创建一次，然后套 `jax.jit`、`jax.vmap`、`jax.jacfwd/jacrev`。初值公式统一复用 std `Quaternion.from_euler` 和 `aero_to_body_vel`；状态、稳态风补偿和参数均保留后端数组，不回到 NumPy。`build_input` 的 JAX 分支也保留原生数组。JAX 噪声由调用方拆分 PRNG key，用 `jax.random.normal(key, (4,))*jnp.pi` 与 `jax.random.normal(key, (17,))` 构造；`read_observation` 放在变换外用于主机读数。
