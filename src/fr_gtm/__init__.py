"""NASA Generic Transport Model（GTM）的 FlyRuler 实现。

本包导出 15 个 pydantic 配置类、状态与控制维度常量、空速角投影助手，以及
25 状态的 `GTMSystem` 根系统和两个初始状态构造函数。角度对外使用度（deg），
内部动力学使用弧度（rad）；`is_fullscale` 与 `mode` 是编译期配置。
"""

from .config import (
    ActuatorConfig,
    AeroConfig,
    AeroForceMomentConfig,
    AircraftConfig,
    ControlSurfacesConfig,
    DynamicDerivativesConfig,
    EngineConfig,
    GTMConfig,
    Mode,
    SensorConfig,
    SensorSingleConfig,
    UncertaintyConfig,
    WindConfig,
)
from .constants import DDIM, NDIM, PDIM, UDIM, XDIM, YDIM
from .proto_projection import airdata_angles_to_proto_radians
from .system import GTMSystem, build_ground_initial_state, build_initial_state

__all__ = [
    "ActuatorConfig",
    "AeroConfig",
    "AeroForceMomentConfig",
    "AircraftConfig",
    "ControlSurfacesConfig",
    "DDIM",
    "DynamicDerivativesConfig",
    "EngineConfig",
    "GTMConfig",
    "GTMSystem",
    "Mode",
    "NDIM",
    "PDIM",
    "SensorConfig",
    "SensorSingleConfig",
    "UDIM",
    "UncertaintyConfig",
    "WindConfig",
    "XDIM",
    "YDIM",
    "airdata_angles_to_proto_radians",
    "build_ground_initial_state",
    "build_initial_state",
]
