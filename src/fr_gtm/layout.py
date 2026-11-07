"""GTM 通道索引与可重定位的完整变量路径。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

U_INDEX: Final[dict[str, int]] = {
    "elevator": 0,
    "aileron": 1,
    "rudder": 2,
    "throttle": 3,
}
"""根输入通道名称到下标的映射。"""

U_DIM: Final = len(U_INDEX)
"""根输入通道数。"""


D_INDEX: Final[dict[str, int]] = {
    "dryden_nu": 0,
    "dryden_nv": 1,
    "dryden_nw": 2,
    "dryden_np": 3,
}
"""Dryden 湍流噪声通道名称到下标的映射。
"""

D_DIM: Final = len(D_INDEX)
"""Dryden 湍流噪声通道数。"""


Y_INDEX: Final[dict[str, int]] = {
    "gamma": 0,
    "alpha": 1,
    "beta": 2,
    "V": 3,
    "altitude": 4,
    "phi": 5,
    "theta": 6,
    "psi": 7,
    "p": 8,
    "q": 9,
    "r": 10,
    "nz": 11,
    "qbar": 12,
    "elevator": 13,
    "aileron": 14,
    "rudder": 15,
    "throttle": 16,
}
"""物理传感器通道名称到下标的映射。"""

Y_DIM: Final = len(Y_INDEX)
"""物理传感器通道数。"""

N_INDEX: Final[dict[str, int]] = Y_INDEX
"""噪声传感器通道名称到下标的映射。"""

N_DIM: Final = len(N_INDEX)
"""噪声传感器通道数。"""


@dataclass(frozen=True)
class _StatePaths:
    """StatePaths 空间的完整路径。"""

    pos_ned: str
    vel_body: str
    quat: str
    omega_body: str
    actuator_position: str
    engine_lag: str
    dryden_u: str
    dryden_v: str
    dryden_w: str
    dryden_p: str
    dryden_q: str
    dryden_r: str


@dataclass(frozen=True)
class _InputPaths:
    """InputPaths 空间的完整路径。"""

    command: str
    disturbance: str
    noise: str


@dataclass(frozen=True)
class _OutputPaths:
    """OutputPaths 空间的完整路径。"""

    measurement: str
    clean: str


@dataclass(frozen=True)
class _ParameterPaths:
    """ParameterPaths 空间的完整路径。"""

    wind_speed: str
    wind_direction: str
    turbulence_level: str
    wind_speed_low: str
    scale_length_high: str
    weight: str
    inertia: str
    c_static: str
    c_e: str
    c_a: str
    c_r: str
    ground_down: str


@dataclass(frozen=True)
class _EffectPaths:
    """EffectPaths 空间的完整路径。"""

    weight: str
    mass: str
    inertia: str
    inertia_inv: str
    equivalent_airspeed: str
    mach: str
    pressure_ratio: str
    dcm: str
    vel_ned: str
    chi: str
    acceleration_body: str
    angular_acceleration_body: str
    wind_omega: str
    aero_force: str
    engine_force: str
    aero_moment: str
    engine_moment: str
    aero_coefficients: str
    reynolds: str
    wind_steady: str
    wind_gust: str
    ground_force_body: str
    ground_moment_body: str
    ground_penetrations: str
    ground_normal_forces: str
    ground_longitudinal_forces: str
    ground_lateral_forces: str
    ground_contact_count: str


@dataclass(frozen=True)
class GTMPaths:
    """GTM 完整路径目录，属性均为普通字符串，不缓存偏移或状态维数。

    Args:
        root: 默认 Root；嵌套实例例如 Root.aircraft。
        io_root: 父系统公开接口前缀；省略时与 root 相同。

    Note:
        当前没有离散状态路径；xd 直接使用 model.xd.build()。
    """

    root: str = "Root"
    """生产 GTM 实例的完整路径前缀。"""
    io_root: str | None = None
    """父系统转接了标准 u/d/n/y/clean_y 时，可显式指定该接口前缀。"""

    @property
    def x(self) -> _StatePaths:
        """x 空间的路径目录。"""
        return _StatePaths(
            pos_ned=self.root + ".eom::pos_ned",
            vel_body=self.root + ".eom::vel_body",
            quat=self.root + ".eom::quat",
            omega_body=self.root + ".eom::omega_body",
            actuator_position=self.root + ".actuator::position",
            engine_lag=self.root + ".engine.lag::lag_state",
            dryden_u=self.root + ".wind.dryden::u_gw",
            dryden_v=self.root + ".wind.dryden::v_gw",
            dryden_w=self.root + ".wind.dryden::w_gw",
            dryden_p=self.root + ".wind.dryden::p_gw",
            dryden_q=self.root + ".wind.dryden::q_gw",
            dryden_r=self.root + ".wind.dryden::r_gw",
        )

    @property
    def u(self) -> _InputPaths:
        """u 空间的路径目录。"""
        return _InputPaths(
            command=(self.root if self.io_root is None else self.io_root) + "::u",
            disturbance=(self.root if self.io_root is None else self.io_root) + "::d",
            noise=(self.root if self.io_root is None else self.io_root) + "::n",
        )

    @property
    def y(self) -> _OutputPaths:
        """y 空间的路径目录。"""
        return _OutputPaths(
            measurement=(self.root if self.io_root is None else self.io_root) + "::y",
            clean=(self.root if self.io_root is None else self.io_root) + "::clean_y",
        )

    @property
    def p(self) -> _ParameterPaths:
        """p 空间的路径目录。"""
        return _ParameterPaths(
            wind_speed=self.root + ".wind.params::wind_speed",
            wind_direction=self.root + ".wind.params::wind_direction",
            turbulence_level=self.root + ".wind.params::turbulence_level",
            wind_speed_low=self.root + ".wind.dryden::wind_speed_low",
            scale_length_high=self.root + ".wind.dryden::scale_length_high",
            weight=self.root + ".mass::weight_unc",
            inertia=self.root + ".mass::inertia_unc",
            c_static=self.root + ".aero::c_static_unc",
            c_e=self.root + ".aero::c_e_unc",
            c_a=self.root + ".aero::c_a_unc",
            c_r=self.root + ".aero::c_r_unc",
            ground_down=self.root + ".ground.params::ground_down",
        )

    @property
    def effects(self) -> _EffectPaths:
        """effects 空间的路径目录。"""
        return _EffectPaths(
            weight=self.root + "::weight",
            mass=self.root + "::mass",
            inertia=self.root + "::inertia",
            inertia_inv=self.root + "::inertia_inv",
            equivalent_airspeed=self.root + "::VE",
            mach=self.root + "::mach",
            pressure_ratio=self.root + "::pressure_ratio",
            dcm=self.root + "::DCM",
            vel_ned=self.root + "::vel_ned",
            chi=self.root + "::chi",
            acceleration_body=self.root + "::acceleration_body",
            angular_acceleration_body=self.root + "::angular_acceleration_body",
            wind_omega=self.root + "::wind_omega",
            aero_force=self.root + "::force_aero",
            engine_force=self.root + "::force_engine",
            aero_moment=self.root + "::moment_aero",
            engine_moment=self.root + "::moment_engine",
            aero_coefficients=self.root + ".aero::coefficients",
            reynolds=self.root + ".aero::reynolds",
            wind_steady=self.root + ".wind::steady",
            wind_gust=self.root + ".wind::gust",
            ground_force_body=self.root + "::ground_force_body",
            ground_moment_body=self.root + "::ground_moment_body",
            ground_penetrations=self.root + "::ground_penetrations",
            ground_normal_forces=self.root + "::ground_normal_forces",
            ground_longitudinal_forces=self.root + "::ground_longitudinal_forces",
            ground_lateral_forces=self.root + "::ground_lateral_forces",
            ground_contact_count=self.root + "::ground_contact_count",
        )


ROOT_PATHS: Final = GTMPaths()
"""独立 Root 模型的默认路径。"""


__all__ = [
    "GTMPaths",
    "ROOT_PATHS",
    "U_INDEX",
    "U_DIM",
    "D_INDEX",
    "D_DIM",
    "Y_INDEX",
    "Y_DIM",
    "N_INDEX",
    "N_DIM",
]
