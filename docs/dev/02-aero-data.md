# 块方程与气动数据

组件位于 `models/fr_gtm/src/fr_gtm/components/`：`aero.py`、`engine.py`、`mass.py`、`sensor.py` 与 `wind.py`。这一章说明每块实现什么方程、数据往哪儿放、以及改数据时必须守住的约定。

## 气动流水线

`Aero` 的输入是 `altitude (1,)`、`surfaces (3,)`、`velocity (3,)`、`omega_body (3,)`、`wind_uvw (3,)` 与 `wind_omega (3,)`。高度/速度用 m、m/s，角速率用 rad/s，舵面与查表气流角用 deg。流水线分四步：

1. **来流与高度边界**：系统从刚体位置构造正向上高度 `-pos_ned[2]`；气动块接收该高度，以及 FRD 速度/角速率/风。相对气流速度为 `air_velocity = velocity - wind_uvw`，相对角速率为 `air_omega = omega_body - wind_omega`。姿态变换和 NED 航迹角由系统层负责，不在 Aero 重复计算。
2. **气流角与状态量**：`body_vel_to_aero_regularized(air_velocity, speed_epsilon=1e-3)` 给出真空速与弧度制的迎角、侧滑角，块边界立刻换算成度；`atmos76(altitude)` 给出密度、声速与压力比，于是 `qbar = 0.5*rho*tas**2`、`mach = tas/a`、`VE = tas*sqrt_density_ratio`。
3. **系数合成**：静态项、操纵项与动导项相加，再乘 `qbar` 与参考面积得到载荷。
4. **载荷到重心**：力取前三个通道；力矩系数乘参考长度 `[b, Cbar, b]`，再用 `cross(arm, force)` 把载荷从气动参考点平移到配置重心，其中 `arm = ref_cg - cg_pos`。

## 气动数据契约

| 约定 | 内容 |
| --- | --- |
| 表资源 | `models/fr_gtm/src/fr_gtm/data/aero_tables.npz`，由 `resources.py` 读取并校验，存放在 `cfg.aero.tables`，以 `Field(exclude=True)` 排除序列化 |
| 系数顺序 | `[CX, CY, CZ, Cl, Cm, Cn]`，前三通道是力系数，后三通道是力矩系数 |
| 雷诺数表方向 | 原始数据先 `flatten(order="F")` 再 reshape 成 `(len(altitude_ft), len(mach_bp))`，沿用 gtm_ad 的 MATLAB 展开 |
| 静态系数 | 三维插值器（默认 linear），网格 `[alpha_c2, beta_c2, reynolds_c2]`，`n_out=2` 一次给出 CX 与 CZ |
| 升降舵 | `cfg.aero.control_arm_ratios` 的四个无量纲展向力臂比值 y/b 累加 `contribution = eye(6)` 并写 `contribution[3,2] = arm`、`contribution[5,0] = -arm` |
| 构型增量 | `flap = [-0.0282,0,-0.2989,0,0.0728,0]*2`、`gear = [-0.016,0,0.014,0,-0.008,0]*2`；半襟翼乘 0.5，全襟翼加 `flap`，全襟翼加起落架再加 `gear`，`normal` 全零 |
| 操纵导数 | 升降舵用四维网格 `[alpha_ele, beta_ele, stab_ele, elev_bp]`（`n_out=3`），`elevator_delta = 0.25*(table(alpha,beta,0,elevator) - stabilizer)`；副翼用 `table(alpha,-beta,-aileron)*[1,-1,1,-1,1,-1] + table(alpha,beta,aileron)`；方向舵用 `where(rudder>=0, -(table(alpha,-beta,-rudder)-table(alpha,-beta,0)), table(alpha,beta,rudder)-table(alpha,beta,0))` |
| 方向舵零点 | 上面的单侧写法在零点保持单侧导数；旧写法在 CasADi/JAX 下会因 `sign(0)=0` 误报零舵效 |
| 动导系数 | `_hykal(alpha, beta, tas, pqr, span, chord)` 做 Kalviste 速率拆分，分母带 `abs(cacb)>1e-10`、`abs(p)>1e-10` 等安全分支 |
| 不确定度顺序 | 运行时扰动按 gtm_ad 的 31 参数顺序排列：`c_static_unc(6)`、`c_e_unc(3)`、`c_a_unc(6)`、`c_r_unc(6)`，落在风与质量参数之后 |

`resources.py` 用有类型的资源组保存雷诺数、基本机体、操纵面和动导表。加载时检查轴有限、严格递增，表值有限且形状与各轴和输出通道数一致；数组从 `cfg.aero.tables.reynolds`、`basic_airframe`、`control_surfaces`、`dynamic_derivatives` 读取。单一的 `AeroConfig` 同时保存构型、面积、共享的 `b/Cbar`、重心、气动参考点和操纵面力臂；`Aero` 只消费该配置。整机通过 `GTMConfig.fullscale()` 或 `GTMConfig.subscale()` 注入几何。修改资源内容不改变源表轴序或单位转换。

`control_arm_ratios` 是无量纲的展向力臂比值 $r=y/b$，不是米制安装位置。由 $F_z=qS C_Z$、$L=yF_z$、$C_l=L/(qSb)$，得到 $\Delta C_l=r\,\Delta C_Z$；同理由 $N=-yF_x$ 得到 $\Delta C_n=-r\,\Delta C_X$。全尺寸下 $y$ 与 $b$ 同比缩放，所以比值不变；载荷换算时再乘实际翼展 $b$。默认四个比值成对对称、总和为零，各段使用相同升降舵输入时相应滚转和偏航项抵消。

`cfg.aero.interpolation_method` 是编译期选择，默认 `"linear"`；显式设为 `"cubic"` 后，雷诺数、静态系数、操纵面及动导资源统一采用 core 的 not-a-knot C² 样条及首末片延拓。所有轴至少四点。切换需要重新编译；默认冻结轨迹仍对应 linear。气动镜像、分支与限幅仍须独立检查，表的 C² 性质不意味着整个 GTM 闭环全局光滑。

### 样条表外使用边界切线

恢复运动可能把归一化角速率推到动导表之外。三次样条默认延续首末三次多项式，远离数据域时可能快速放大气动力矩。需要限制这种外推阶数时，显式设置：

```python
cfg = GTMConfig.fullscale()
cfg.aero.interpolation_method = "cubic"
cfg.aero.extrapolation_method = "linear"
```

此选项保留表内原 not-a-knot 样条，表外逐轴沿边界切线延拓，不是夹住气动力值或清零梯度。令 $\bar q=\mathrm{clip}(q,\ell,u)$、$\delta=q-\bar q$，张量积延拓为 $F(q)=\sum_{S}(\partial_S f)(\bar q)\prod_{i\in S}\delta_i$，子集 $S$ 对每个轴至多求一次导数；表内所有增量为零，恢复原样条。多轴同时越界时保留混合项，使各边界面及其交线保持值与一阶导数连续。域内仍为 C²，跨数据边界为 C¹。

实现只在 core 准备阶段增加每轴两个线性端胞、复制端系数，既有后端继续计算同一局部基函数，没有运行时模型切换或有限差分。默认 `extrapolation_method="polynomial"` 不变；core 的直接资源入口是 `fr.resource.interpolator(..., method="cubic", extrapolation="linear")`。此选择不约束气动力大小，也不保证整机动力学在任意状态稳定。

## Kalviste 角速率分解与动导合成

先把度制迎角/侧滑角转为弧度，构造机体系来流单位向量 $e_V=[\cos\alpha\cos\beta,\sin\beta,\sin\alpha\cos\beta]^T$。HyKal 按 $[p,q,r]^T=\Omega e_V+\omega_{osc}$ 分离流向轴稳转与剩余各轴运动，输入角速率均为扣除角阵风后的 rad/s；这不是正交投影，剩余分量不要求垂直于来流轴。

优先令 $\Omega=p/e_{V,x}$、$p_{osc}=0$，且要求剩余 $r_{osc}$ 与原 $r$ 同号、幅值不增大。若不满足，再尝试 $\Omega=r/e_{V,z}$、$r_{osc}=0$，要求剩余 $p_{osc}$ 同号且不增大。两个候选互斥并有明确优先级；均不满足时令 $\Omega=0$，保留原角速率。每个有效候选都从三个轴扣除 $\Omega e_V$，其中 $q_{osc}=q-\Omega\sin\beta$。

以 $D=\operatorname{diag}(b,\bar c,b)/(2V)$ 将剩余角速率无量纲化，得到 $[\hat p,\hat q,\hat r]^T=D\omega_{osc}$；稳转坐标为 $\hat\Omega=\Omega b/(2V)$。仅这条归一化路径将速度限为 1～10000 m/s，并将稳转查表坐标限为 ±0.5；真实动压和 Mach 不使用限幅后的速度。

p/r 表输出 `[ΔCY,ΔCl,ΔCn]`，q 表输出 `[ΔCX,ΔCZ,ΔCm]`，w 表输出完整六元系数，其中 w 指流向轴稳转坐标，不是垂向速度。设 `E_lon`、`E_lat` 分别把纵向/横向三元系数嵌入统一六元顺序，则 `ΔC = E_lon @ q_delta + E_lat @ (p_delta + r_delta) + w_delta`。表值已经是无量纲系数增量，不再额外乘对应角速率；最终合成载荷时才乘动压、面积和力矩参考长度。

## 插值成本与复现

完整的 linear/cubic 大表、舵效嵌套 AD、GTM 连续仿真及 JAX CPU/GPU 基准见[插值基准](../core/13-interpolation-benchmarks.md)。该页记录测量条件、优化前后成本、FP32/FP64 精度与复现命令；原始表与默认 linear 选择保持不变。

## 块方程

| 块 | 方程要点 | 输出 |
| --- | --- | --- |
| `Engine` | 油门指令先按 `[command_lower, command_upper]` 钳位，再经所选滞后得有效油门，见下一节；有效油门经 `polyval(throttle_coefficients, effective_throttle)` 得参考转速；`ram_drag = ram_drag_coeff*qbar*S`；`thrust = polyval(thrust_coefficients, rpm)*T_gain*pressure_ratio - ram_drag`，再 `max(...,0)`；轴动量由轴径、轴长、密度与角速度换算，两台发动机角动量按 `_engine_direction(angles_deg) = [cos(a1)cos(a2), sin(a2), -sin(a1)cos(a2)]` 合成；`moment = cross(omega_body, angular_momentum) + cross(arm_l, thrust_l) + cross(arm_r, thrust_r)`，力臂是发动机位置减重心 | `force (3,)`、`moment (3,)` |
| `Mass` | `weight = w0*(1+weight_unc[0])`、`mass = weight/gravity_acc`；六个惯量按 `nominal*(1+unc)` 缩放，顺序 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]`；惯量矩阵按符号约定填入，行列式用闭式表达并 `max(det, 1e-8)` 兜底，逆由代数余子式除以行列式 | `weight (1,)`、`mass (1,)`、`inertia (3,3)`、`inertia_inv (3,3)` |
| `Wind` | 三个公开参数 `wind_speed`、`wind_direction`（弧度）、`turbulence_level`；风切变由 `WindShear(runway_altitude=...)` 提供，稳态风由 `SteadyWind(enable_wind_shear=True)` 组合，紊流由 `DrydenTurbulence` 以 `wing_span=cfg.aircraft.b`、最小空速 0.1 与最小尺度 0.1 配置；空速取 `smooth_norm(velocity, epsilon=1e-3)`；门控 `turbulence_level > 1e-3` 时透传白噪声，否则送零向量 | `uvw`、`omega`、`steady`、`gust` |
| `Sensor` | 17 个物理通道顺序固定：gamma、alpha、beta、V、altitude、`degrees(euler)` 的 phi/theta/psi、`degrees(omega_body)` 的 p/q/r、nz、qbar，以及 `[elevator, aileron, rudder, effective_throttle]`；`feedback = bias + scale*clean_feedback + noise*noise_std`，噪声输入 17 维；第 17 通道 bias 为 0、scale 为 1、noise_std 为 0，等价于有效油门原样透传 | `clean_feedback (17,)`、`feedback (17,)` |

## 发动机油门滞后

`EngineConfig.lag` 是以 `model` 判别的一阶/二阶配置联合。声明 `Engine` 时选择状态维数和动力学，各后端编译同一表达式，无运行时模型选择分支。

`components/engine.py` 中，`EngineConfig` 只组合 `lag` 与 `loads`；两个滞后 Block 分别关联自己的配置，`EngineLoads` 关联 `EngineLoadsConfig`，`Engine` System 只负责选择及接线。载荷配置持有计算所需的几何快照，`config.py` 的工厂负责尺度派生；多项式系数只在载荷配置中定义，不在工厂重复覆盖。子块输出与导数分离，固定当前状态时，改变指令仅改变滞后导数，不直接改变当前有效油门和载荷。

默认 `FirstOrderThrottleLagConfig` 使用 `τ δ̇ = δ_c − δ`，其中 `δ_c` 是限幅后的指令，`δ` 是有效油门，`τ = 1.5 s`。积分因子解给出恒定指令下 `δ(t) = δ_c + (δ(0) − δ_c) exp(−t/τ)`，因此一倍时间常数完成约 63.2% 的变化。配平条件直接给出 `trim_state(δ_★) = (δ_★,)`。全尺寸与缩比共用这个时间常数，这是一阶近似的配置选择。

选择 `JetCatThrottleLagConfig(model="second_order")` 时使用原始 GTM 缩比模型的二阶状态空间。`JetCatThrottleLagConfig` 给出四个矩阵分量：

```
A = [[state_matrix[0], state_matrix[1]],   B = [[input_matrix[0]],
     [state_matrix[2], state_matrix[3]]]        [input_matrix[1]]]
C = [output_matrix[0], output_matrix[1]],  D = 0
```

默认值 `A = [[-1.336, -0.7314], [1, 0]]`、`B = [[1], [0]]`、`C = [-0.1474, 0.7314]` 对应传递函数

```
G(s) = C (sI - A)^{-1} B
     = (-0.1474 s + 0.7314) / (s^2 + 1.336 s + 0.7314)
```

直流增益为 1，极点约为 `-0.668 ± 0.535j`，右半平面零点在 `4.962 s^-1`，因此大油门阶跃开头会有一点反向响应。块内按 `xdot = (A x + B u) / time_scale`、`y = C x` 推进，`u` 是钳位后的油门指令、`y` 是有效油门，两者都保持百分比单位，所以 `C` 与配平状态都不随 `time_scale` 变化。

限幅的位置要区分：输入指令按默认 `[0,100]%` 钳位；滞后输出 `C x` 不钳位，直接进入参考转速多项式；扣除冲压阻力后的最终推力用 `max(thrust,0)` 保持非负。有效油门是滞后模型的等效量，不是有机械止挡的舵面位置，正向阶跃初期可以短暂小于 0%。在它上面再加位置限幅会改变原始二阶模型的反向响应。

配平初始化由直流条件反解：`x_★ = -A^{-1} B δ_★` 同时使 `A x_★ + B δ_★ = 0` 与 `C x_★ = δ_★` 成立，展开成 `trim_state(δ_★) = (m1 δ_★, m2 δ_★)`，`m1 = (a12 b2 - a22 b1)/det = 0`、`m2 = (a21 b1 - a11 b2)/det = 1/0.7314`，`det = a11 a22 - a12 a21`。`init`、`trim` 与示例脚本都用这个映射写 `Root.engine.lag::lag_state`，避免零初始把有效油门钉在 0% 造成人为推力暂态。

时间尺度系数 `time_scale = c` 把 `A` 与 `B` 同时除以 `c`，得到 `G_c(s) = G(cs)`：只拉伸时间轴，不改变直流增益与稳态值。原始缩比 GTM 的 `c = 1`，几何扩比因子是 `1/0.055`，严格重力相似对应 `c = sqrt(1/0.055) ≈ 4.264`；本模型两侧都取 `c = 1`，即扩比后沿用原始二阶响应尺度，而不是把一阶时间常数乘上 `sqrt(scale)`。阶跃响应首次达到 63.2% 与 95% 的时间分别是 2.346 s 与 4.022 s，`c = 4.264` 时按比例放大到 10.003 s 与 17.150 s。

## 运行时参数面

`model.p` 暴露 11 个路径、33 个数值，全部可运行时改写而免重编译：

| 路径 | 维度 | 单位 | 作用 |
| --- | --- | --- | --- |
| `Root.aero::c_static_unc` | 6 | — | 静态系数相对增量 |
| `Root.aero::c_e_unc` | 3 | — | 升降舵系数相对增量 |
| `Root.aero::c_a_unc` | 6 | — | 副翼系数相对增量 |
| `Root.aero::c_r_unc` | 6 | — | 方向舵系数相对增量 |
| `Root.mass::weight_unc` | 1 | — | 重量相对增量 |
| `Root.mass::inertia_unc` | 6 | — | 惯量相对增量，顺序 `[Ixx, Iyy, Izz, Ixz, Iyz, Ixy]` |
| `Root.wind.params::wind_speed` | 1 | m/s | 稳态风速 |
| `Root.wind.params::wind_direction` | 1 | rad | 风来向 |
| `Root.wind.params::turbulence_level` | 1 | — | 紊流强度，兼作 Dryden 门控 |
| `Root.wind.dryden::wind_speed_low` | 1 | m/s | Dryden 低频风速 |
| `Root.wind.dryden::scale_length_high` | 1 | m | Dryden 高频尺度长度 |

## 改这里的连带影响

改 `state_matrix`、`input_matrix`、`output_matrix` 或 `time_scale` 会改变油门响应尺度与配平状态，`tests/test_engine_first_order.py`、`tests/test_engine_lag.py` 与配平示例都要一起看；冻结 `gtm_ad` 参考把油门建成一阶作动器状态，因此油门通道的导数不能与新模型逐位比对，只能比有效油门的取值与其余通道；改系数顺序或参考长度会让力矩失配，跨后端比较仍可能一致，只有冻结参考能发现；改不确定度向量顺序会同时影响 模型参数布局与参数路径文档；改 17 通道顺序会移动 `y` 的每一列，控制器与 Proto 投影都要同步；改构型增量或含襟翼构型会移动配平点。

## 怎么验证

- `just test-one tests/test_reference.py`：与外部 `gtm_ad` 冻结参考的系数与载荷对比。
- `just test-one tests/test_backends.py`：四个后端在逐步轨迹上的偏差。
- `just test-one tests/test_engine_lag.py`：油门二阶滞后与解析响应的对比。
- `just example-trim`：配平点是否仍落在同一姿态与油门附近。
- `just example-inspect`：`n_p` 与 `model.p.keys` 是否与参数面一致。

## 相关页面

- [装配与语义边界](/dev/components/gtm/01-architecture)：配置派生与系统接线
- [验证与扩展](/dev/components/gtm/03-verification)：命令与扩展点
- [gtm 配置参考](/api/gtm/config)：字段与默认值
