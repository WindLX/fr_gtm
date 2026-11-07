"""NASA Generic Transport Model（GTM）的 FlyRuler 实现。

本包提供组件配置、GTMSystem、路径目录、直接初值构造、观测解析与配平工具。角度对外使用度（deg），
内部动力学使用弧度（rad）；`is_fullscale` 与 `mode` 是编译期配置。
"""

from .config import (
    ActuatorConfig,
    AeroConfig,
    AircraftConfig,
    EngineConfig,
    EngineLoadsConfig,
    FirstOrderThrottleLagConfig,
    GTMConfig,
    JetCatThrottleLagConfig,
    Mode,
    SensorConfig,
    SensorSingleConfig,
    WindConfig,
)
from .init import (
    Initializer,
    Observation,
    Uncertainty,
    build_input,
    init,
    read_observation,
    sample_disturbance,
    sample_sensor_noise,
)
from .layout import (
    D_DIM,
    D_INDEX,
    N_DIM,
    N_INDEX,
    U_DIM,
    U_INDEX,
    Y_DIM,
    Y_INDEX,
    GTMPaths,
)
from .system import GTMSystem
from .trim import GTMTrimResult, trim

__all__ = [
    "ActuatorConfig",
    "AeroConfig",
    "AircraftConfig",
    "EngineConfig",
    "EngineLoadsConfig",
    "JetCatThrottleLagConfig",
    "FirstOrderThrottleLagConfig",
    "GTMConfig",
    "GTMSystem",
    "trim",
    "GTMTrimResult",
    "init",
    "Initializer",
    "build_input",
    "sample_disturbance",
    "sample_sensor_noise",
    "read_observation",
    "Observation",
    "Uncertainty",
    "GTMPaths",
    "U_INDEX",
    "U_DIM",
    "D_INDEX",
    "D_DIM",
    "Y_INDEX",
    "Y_DIM",
    "N_INDEX",
    "N_DIM",
    "Mode",
    "SensorConfig",
    "SensorSingleConfig",
    "WindConfig",
]
