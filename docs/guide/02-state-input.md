# 初值、输入与参数

用 `init` 一次生成目标后端的 `(x, xd, u, p)`，不用自己拼接状态下标。位置为 NED 米，`airdata=(alpha, beta, VTAS)` 使用度、度、m/s，`euler=(phi, theta, psi)` 使用度，`command` 为三个舵面角度及油门百分比。

## 构造物理初值

<<< @/../models/fr_gtm/examples/simulate.py#02-state-input-1

`init` 默认位置 `(0,0,-1000)`、来流 `(0,0,100)`、欧拉角零、命令 `(0,0,0,40)`；默认风和不确定度为零。可选 `rates=(p,q,r)` 使用 deg/s。它把欧拉角转换为 scalar-first 四元数，按来流构造相对空气速度，再加上模型自身计算的稳态风，得到地速对应的 FRD 速度；因此有风时指定的 TAS、alpha、beta 仍成立。舵面状态等于命令，发动机状态通过 `cfg.engine.lag.trim_state(throttle)` 初始化，兼容一阶和二阶；Dryden 状态为零。这只是初值，不保证飞机已配平。

地面场景先用 std `solve_static_ground_pose` 求停机姿态，再将所需 NED 速度转到 FRD，换算为来流后传给同一个 `init`；完整流程见 `ground_operations.py`，不另设地面初始化入口。

## JAX 原生初值、批量与微分

普通 `init` 是带有限值与非负 TAS 校验的主机入口。要在 JAX 中对初值求导或批量构造，在 `jit` 外预先创建 `Initializer`；它把共享的 std `Quaternion.from_euler`、`aero_to_body_vel` 和发动机初值映射编译到模型后端，调用时不经过 NumPy、Python float 或设备到主机的转换。有风时同样读取生产模型的稳态风，再在后端完成速度相加。

```python
import jax
import jax.numpy as jnp
from fr_gtm import Initializer, build_input

initialize = Initializer(model, cfg)  # model 是已编译的 JAX 模型
initialize_jit = jax.jit(initialize)
x, xd, u, p = initialize_jit(
    airdata=jnp.array([3.0, 0.0, 100.0]),
    euler=jnp.array([0.0, 3.0, 0.0]),
    wind=jnp.array([12.0, 35.0, 2.0]),
)
batch = jax.vmap(lambda v: initialize(airdata=v))(
    jnp.array([[3.0, 0.0, 100.0], [4.0, 1.0, 110.0]])
)
jacobian = jax.jacfwd(lambda v: initialize(airdata=v)[0])(jnp.array([3.0, 0.0, 100.0]))
```

配置和路径绑定在构造器上，原生调用接受后端数组或普通序列；输入形状与不确定度名称会检查，动态值由调用方保证有限、TAS 非负。正向与反向微分都保留姿态、来流、风和参数的计算依赖。`read_observation` 仍是主机读数工具，放在 JAX 变换之外。

JAX 随机数使用显式 PRNG key；NumPy 的 `sample_disturbance`、`sample_sensor_noise` 保留原用法。d 标准差为 π、n 标准差为 1，不额外按 dt 缩放：

```python
d_key, n_key = jax.random.split(key)
d = jax.random.normal(d_key, (4,)) * jnp.pi
n = jax.random.normal(n_key, (17,))
u = jax.jit(lambda d, n: build_input(model, command, d=d, n=n))(d, n)
```

## 用路径目录覆盖状态

```python
from fr_gtm import GTMPaths

paths = GTMPaths()
values = {
    key: model.to_numpy(value) for key, value in model.x.read(x, by="path").items()
}
values[paths.x.pos_ned] = np.array([0.0, 0.0, -800.0])
x = model.x.build(values)
```

上例需要 `import numpy as np`。也可直接给 `model.x.build` 传主机数组。`GTMPaths` 的属性是普通字符串，可用于 build、read、trim 与线性化。嵌套实例用 `GTMPaths("Root.aircraft")`；若父系统把输入输出接到根接口，则用 `GTMPaths("Root.aircraft", io_root="Root")`，状态/参数/诊断前缀与根 I/O 前缀分别指定。

状态路径与维度：

| 路径 | 维度 | 说明 |
| --- | --- | --- |
| `Root.eom::pos_ned` | `(3,)` | NED 位置，高度取 `-pos_ned[2]` |
| `Root.eom::vel_body` | `(3,)` | 机体系速度 |
| `Root.eom::quat` | `(4,)` | scalar-first 四元数，声明为 SO3 状态 |
| `Root.eom::omega_body` | `(3,)` | 机体系角速度 |
| `Root.actuator::position` | `(3,)` | 升降舵、副翼与方向舵位置，单位度 |
| `Root.engine.lag::lag_state` | `(1,)` 或 `(2,)` | 默认有效油门状态，或原二阶模型的内部状态 |
| `Root.wind.dryden::u_gw` | `(1,)` | 纵向 Dryden 成形滤波器状态 |
| `Root.wind.dryden::v_gw` | `(2,)` | 横向 Dryden 状态 |
| `Root.wind.dryden::w_gw` | `(2,)` | 垂向 Dryden 状态 |
| `Root.wind.dryden::p_gw` | `(1,)` | 滚转 Dryden 状态 |
| `Root.wind.dryden::q_gw` | `(1,)` | 俯仰 Dryden 状态 |
| `Root.wind.dryden::r_gw` | `(1,)` | 偏航 Dryden 状态 |

## 输入

| 路径 | 维度 | 说明 |
| --- | --- | --- |
| `Root::u` | `(4,)` | `[升降舵_deg, 副翼_deg, 方向舵_deg, 油门指令_percent]`；油门指令先按 `[0,100]` 钳位，再经所选滞后成为有效油门 |
| `Root::d` | `(4,)` | 四路 Dryden 白噪声 `[nu, nv, nw, np]` |
| `Root::n` | `(17,)` | 17 个物理反馈通道的标准化传感器噪声，命令不属于观测，不占噪声通道 |

`Root::u[3]` 是油门指令，`clean_y[16]` 是所选滞后的等效输出。只有指令裁切到 `[0,100]%`；有效油门不裁切；默认一阶模型从范围内初值出发时保持在指令范围内，原二阶模型在阶跃开始时可能短暂低于 0%。它直接进入发动机静态映射，最终推力再裁切到非负。

`build_input(model, command, d=..., n=...)` 生成完整输入，省略 d/n 时填零，不隐式采样。`sample_disturbance(rng)` 返回 4 路 N(0,π²)，`sample_sensor_noise(rng)` 返回 17 路 N(0,1)，与 gtm_ad 的高斯激励约定一致；调用者拥有 `np.random.Generator`，可以固定种子。

所有随机量都从外部输入进入系统，因此全零输入下模型是确定性的：给定同一初值、输入与参数，四个后端给出相同结果。

## 运行时参数

参数共 11 个路径、33 个数值，用 `model.p.build({...})` 按路径覆盖，覆盖只需写要改的端口：

| 路径 | 维度 | 单位 | 说明 |
| --- | --- | --- | --- |
| `Root.wind.params::wind_speed` | `(1,)` | m/s | 稳态风大小 |
| `Root.wind.params::wind_direction` | `(1,)` | rad | 从 NED 北向顺时针度量的风向 |
| `Root.wind.params::turbulence_level` | `(1,)` | — | 紊流强度，超过 `1e-3` 才开启激励与输出 |
| `Root.wind.dryden::wind_speed_low` | `(1,)` | m/s | 低空风速 |
| `Root.wind.dryden::scale_length_high` | `(1,)` | m | 高空尺度长度 |
| `Root.mass::weight_unc` | `(1,)` | — | 重量相对不确定度 |
| `Root.mass::inertia_unc` | `(6,)` | — | 惯量相对不确定度 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]` |
| `Root.aero::c_static_unc` | `(6,)` | — | 静态系数相对不确定度 |
| `Root.aero::c_e_unc` | `(3,)` | — | 升降舵系数相对不确定度 |
| `Root.aero::c_a_unc` | `(6,)` | — | 副翼系数相对不确定度 |
| `Root.aero::c_r_unc` | `(6,)` | — | 方向舵系数相对不确定度 |

`init` 的 `uncertainty` 接受 `weight` 标量、`inertia` 六元、`c_static` 六元、`c_e` 三元、`c_a/c_r` 六元相对扰动。风、惯量与气动不确定度一起给：

<<< @/../models/fr_gtm/examples/disturbances.py#02-state-input-1

惯量的六个分量顺序是 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]`，与外部参考数据一致；不确定度都按相对量给出，零扰动复现标称参考数据。

## 这一章用到的 API

- [`fr_gtm.init`](/api/gtm/init)：物理初值、输入、噪声与观测工具
- [`fr_gtm.GTMPaths`](/api/gtm/layout)：显式路径目录与通道索引
- [`fr_gtm.GTMConfig`](/api/gtm/config)：构造模型与初始化共用的配置

## 边界与常见坑

| 边界 | 说明 |
| --- | --- |
| 状态是平坦向量 | `Root.eom::quat` 是 4 维 SO3 块，`pos_ned` 从索引 0 开始，位置与单位由 `model.x.info` 给出 |
| 高度是 `-pos_ned[2]` | NED 向下为正，写高度要取负 |
| 风向的转换边界 | `init(wind=(speed,direction,level))` 接受度，底层 `p` 路径存弧度 |
| 惯量顺序固定 | `inertia_unc` 是 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]`，与对角分量的直觉顺序不同 |
| 路径写错直接报错 | `model.x.build`、`model.u.build`、`model.p.build` 收到未知路径时抛错，先用 `model.x.keys` 与 `model.p.keys` 打印实际清单 |
| 地面初值需要开地面 | 先传 `enable_ground=True`，再求静态停机姿态并调用 `init` |
| 滞后状态要用配平初值 | `Root.engine.lag::lag_state` 零初始对应 0% 有效油门；指定油门时要写 `cfg.engine.lag.trim_state(throttle)`，否则仿真起点会出现一段人为推力爬升 |

## 相关页面

- [组装与配置](/guide/components/gtm/01-build)：`cfg.ground` 从哪儿来
- [推进与记录](/guide/components/gtm/03-simulate)：这些初值与参数怎么进入一步推进
- [地面运行、风与扰动](/guide/components/gtm/04-scenarios)：参数在场景里的实际取法
- [gtm 实现](/dev/components/gtm/02-aero-data)：气动表与参数化契约
