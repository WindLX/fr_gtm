"""GTM 顶层接线，使用 NED 导航系与 FRD 机体系。

根系统只连接语义化的作动器、风、质量、气动、发动机、刚体与传感器组件；
GTM 特有的表格与符号约定留在各自的 Block 内，标准库负责作动器与 6 自由度积分。
"""

from __future__ import annotations

import fly_ruler_core as fr
from fly_ruler_std.blocks.actuator import SimpleActuator, SimpleActuatorConfig
from fly_ruler_std.blocks.physics import (
    GroundContact,
    RigidBody6DOFConfig,
    RigidBody6DOFQuaternion,
)

from .components.aero import Aero
from .components.engine import Engine
from .components.mass import Mass
from .components.sensor import Sensor
from .components.wind import Wind
from .config import GTMConfig

_DEFAULT_CONFIG = GTMConfig.fullscale()


@fr.system(config=GTMConfig)
def GTMSystem(cfg: GTMConfig = _DEFAULT_CONFIG) -> None:
    """组装默认 25 状态 GTM 被控对象与固定 17 通道物理观测。

    Args:
        cfg: GTM 配置，给出几何、发动机、作动器、传感器、气动表与地面接触
            参数；`is_fullscale` 与 `aero.mode` 在编译期决定派生数据。

    Note:
        根输入 `u(4,)` 为 `[elevator_deg, aileron_deg, rudder_deg,
        throttle_percent]`，`d(4,)` 驱动四个 Dryden 滤波器，`n(17,)` 提供
        标准化传感器噪声。所有随机量都由外部输入，因此零输入下系统是
        确定性的。根输出仅为 y/clean_y，各含 17 个物理反馈通道，不追加命令。
        闭环控制可直接读取 plant.y；命令从输入 u 读取，独立诊断放在 effects。

        前三个指令通道进入三通道一阶作动器，第四通道直接进入 `Engine` 的
        可配置油门滞后；油门不再经过作动器，因此 `Root::u[3]` 是命令油门，
        y/clean_y 的第 16 通道是滞后后的有效油门。
    """
    u = fr.input(
        "u",
        (4,),
        description=("指令：[升降舵deg, 副翼deg, 方向舵deg, 油门%]"),
    )
    d = fr.input("d", (4,), description="Dryden 四路标准化白噪声 [nu, nv, nw, np]")
    n = fr.input(
        "n", (17,), description="17 个物理传感器通道的标准化噪声，不含指令遥测噪声"
    )

    actuator = SimpleActuator(
        "actuator",
        SimpleActuatorConfig(
            adim=3,
            gain=cfg.actuator.gain,
            rate_limit=cfg.actuator.rate_limit,
            position_lower=cfg.actuator.position_lower,
            position_upper=cfg.actuator.position_upper,
            expose_params=False,
        ),
    )
    wind = Wind("wind", cfg.wind)
    mass = Mass("mass", cfg.aircraft)
    aero = Aero("aero", cfg.aero)
    engine = Engine("engine", cfg.engine)
    eom = RigidBody6DOFQuaternion(
        "eom",
        RigidBody6DOFConfig(
            enable_external_mass=True,
            enable_geodetic_output=False,
        ),
    )
    sensor = Sensor("sensor", cfg.sensor)

    # 舵面、油门、风激励与传感器噪声是独立输入；模型内部不生成随机数。
    # 仅前三个舵面指令经过作动器的增益、速率与位置限幅，单位保持 deg。
    u[:3] >> actuator.command

    # 油门指令（%）直接进入发动机自有滞后；载荷使用滞后后的状态，不由命令直通。
    # 发动机读取刚体角速率计算转子陀螺力矩，读取气动块的大气压力比和动压计算载荷。
    u[3] >> engine.throttle_command
    eom.omega_body_out >> engine.omega_body
    aero.pressure_ratio >> engine.pressure_ratio
    aero.qbar >> engine.qbar

    # 气动只接三个舵面实际位置；油门不参与气动查表。
    surface_position = actuator.position_out
    # 物理观测的最后四项为 [升降舵deg, 副翼deg, 方向舵deg, 有效油门%]，
    # 与末尾追加的四个“指令”通道不同；两者不应互换。
    actuator_position = fr.math.concatenate(
        [surface_position, engine.effective_throttle]
    )

    # NED 的位置第三分量向下为正，大气/风/传感器共享其负值作为正向上高度。
    altitude = -eom.pos_ned_out[2]

    # 风模块用刚体速度模值驱动 Dryden 时间常数，不用已扣风的真空速，避免气动反馈环。
    # DCM 为 FRD→NED；风模块内部负责把稳态风和突风转换到 FRD 后输出。
    altitude >> wind.altitude
    eom.vel_body_out >> wind.velocity_body
    eom.DCM >> wind.DCM
    d >> wind.noise

    # 气动块内部再扣除风的平动/角运动，并执行度制查表与 Kalviste 角速率分解。
    # 这里连接的是实际声明的 velocity 端口，而不是 Python 局部变量 velocity_body。
    altitude >> aero.altitude
    surface_position >> aero.surfaces
    eom.vel_body_out >> aero.velocity
    eom.omega_body_out >> aero.omega_body
    wind.uvw >> aero.wind_uvw
    wind.omega >> aero.wind_omega

    # 质量块统一提供运行时扰动后的质量、惯量及其逆矩阵，刚体开启外部质量模式。
    # 重力在 NED 中为 [0,0,W]；DCM.T 将其旋转到 FRD。各载荷块不再重复加重力。
    mass.mass >> eom.mass
    mass.inertia >> eom.inertia
    mass.inertia_inv >> eom.inertia_inv

    gravity_body = eom.DCM.T @ fr.math.vector([0.0, 0.0, mass.weight[0]])
    # 跑道接触是独立的编译期开关；与 aero.mode 中起落架造成的气动增量不是同一开关。
    if cfg.ground.enabled:
        ground = GroundContact("ground", cfg.ground)
        eom.pos_ned_out >> ground.position
        eom.vel_body_out >> ground.velocity_body
        eom.DCM >> ground.dcm
        eom.omega_body_out >> ground.omega_body
        ground_force_body = ground.force_body
        ground_moment_body = ground.moment_body
        force_body = aero.force + engine.force + ground_force_body + gravity_body
        moment_body = aero.moment + engine.moment + ground_moment_body
        non_gravitational_force = aero.force + engine.force + ground_force_body
    else:
        force_body = aero.force + engine.force + gravity_body
        moment_body = aero.moment + engine.moment
        non_gravitational_force = aero.force + engine.force
    force_body >> eom.force_body
    moment_body >> eom.moment_body

    # 法向过载 nz=-F_non_gravity,z/W：FRD 的 +Z 向下，因此向上的非重力载荷为正。
    # 地面开启时计入接触力，但重力始终排除；用同一份扰动后重量 W 归一化。
    load_factor = non_gravitational_force / (-mass.weight)
    nz = load_factor[2]

    # 刚体按 NED 速度给出航迹角（rad），在 Sensor 的 deg 输入边界转换一次。
    fr.math.degrees(eom.gamma) >> sensor.gamma
    # 气流角为 deg；刚体欧拉角/角速率为 rad、rad/s，Sensor 内部统一转换为度制观测。
    aero.alpha >> sensor.alpha
    aero.beta >> sensor.beta
    aero.VTAS >> sensor.V
    altitude >> sensor.altitude
    eom.euler_angles >> sensor.euler
    eom.omega_body_out >> sensor.omega_body
    nz >> sensor.nz
    aero.qbar >> sensor.qbar
    actuator_position >> sensor.act_pos
    n >> sensor.noise

    # 观测只包含 Sensor 的 17 个物理通道，命令保留在输入 u 中，不重复输出。
    fr.export(y=sensor.output, clean_y=sensor.clean_output)

    # 逐项保留各组件 out 中未被 x、y/clean_y 或子组件 effects 覆盖的诊断。
    # 作动器实际位置、气流角/空速/动压、发动机有效油门、姿态欧拉角/航迹角
    # 已在观测中；刚体位置/机体速度/四元数/角速率直接从 x 读取，不重复保存。
    # Aero 系数/雷诺数、Wind 稳态风/阵风已有各自 effects，不在根再声明。
    fr.effects(
        # 当前运行时质量特性，包含参数扰动。
        weight=mass.weight,
        mass=mass.mass,
        inertia=mass.inertia,
        inertia_inv=mass.inertia_inv,
        # 观测未单列的大气/空速输出。
        VE=aero.VEAS,
        mach=aero.mach,
        pressure_ratio=aero.pressure_ratio,
        # 刚体状态之外的坐标变换、航向和导数输出。
        DCM=eom.DCM,
        vel_ned=eom.velocity_ned,
        chi=eom.chi,
        acceleration_body=eom.acceleration_body,
        angular_acceleration_body=eom.angular_acceleration_body,
        # 尚未由其他 effects 提供的角阵风和分项载荷。
        wind_omega=wind.omega,
        force_aero=aero.force,
        force_engine=engine.force,
        moment_aero=aero.moment,
        moment_engine=engine.moment,
    )
    if cfg.ground.enabled:
        fr.effects(
            ground_force_body=ground_force_body,
            ground_moment_body=ground_moment_body,
            ground_penetrations=ground.penetrations,
            ground_normal_forces=ground.normal_forces,
            ground_longitudinal_forces=ground.longitudinal_forces,
            ground_lateral_forces=ground.lateral_forces,
            ground_contact_count=ground.contact_count,
        )
