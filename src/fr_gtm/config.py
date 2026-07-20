"""Configuration objects for the FlyRuler GTM model.

The source ``gtm_ad`` package builds one large runtime configuration object.
This module keeps the same default parameter derivation, but exposes the data
as FlyRuler ``BaseModel`` objects so each subsystem can receive only the
configuration it owns.
"""

from __future__ import annotations

from enum import Enum
from importlib import resources
from math import pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import rtoml
from fly_ruler_std.blocks.physics import (
    GroundContactConfig,
    GroundContactPointConfig,
)
from fly_ruler_std.libs.units import Force, Inertia, Length, Pressure, Speed
from pydantic import BaseModel, Field, field_validator, model_validator


class Mode(str, Enum):
    """GTM flap and landing-gear aerodynamic configuration."""

    NORMAL = "normal"
    HALF_FLAP_GEAR_UP = "0.5_flap_gear_up"
    FULL_FLAP_GEAR_UP = "1_flap_gear_up"
    FULL_FLAP_GEAR_DOWN = "1_flap_gear_down"


class UncertaintyConfig(BaseModel):
    """Nominal full/scale mass properties used by runtime perturbations.

    Products of inertia retain the source-model sign convention. The Mass block
    converts this six-element representation into a symmetric FRD tensor.
    """

    w0: float = Field(default=Force.lbf_to_n(185_000.0), gt=0.0)  # Weight [N].
    Ixx0: float = Field(default=Inertia.slug_ft2_to_kg_m2(1_770_000.0), gt=0.0)
    Iyy0: float = Field(default=Inertia.slug_ft2_to_kg_m2(5_680_000.0), gt=0.0)
    Izz0: float = Field(default=Inertia.slug_ft2_to_kg_m2(7_270_000.0), gt=0.0)
    Ixz0: float = Inertia.slug_ft2_to_kg_m2(160_000.0)  # Ixz (kg*m^2).
    Iyz0: float = 0.0  # Iyz (kg*m^2).
    Ixy0: float = 0.0  # Ixy (kg*m^2).
    gravity_acc: float = Field(default=9.80665, gt=0.0)  # m/s^2


class EngineConfig(BaseModel):
    """GTM twin-engine thrust and gyroscopic moment configuration."""

    T_gain: float = 1.0  # Thrust gain converting model thrust to Newtons.
    h_gain: float = 1.0  # Angular momentum gain (kg*m^2)/(slug*ft^2).
    RPMref_gain: float = 1.0  # Reference RPM scaling gain.
    ram_drag_coeff: float = 0.01  # Ram drag coefficient.
    S: float = 1.0  # Reference wing area used for ram drag (m^2).
    shaft_diameter: float = 0.75 / 12.0  # Shaft diameter (ft, per source model).
    shaft_length: float = 6.25 / 12.0  # Shaft length (ft, per source model).
    density: float = 15.2  # Shaft material density (slug/ft^3).
    thrust_coefficients: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.41266]
    )  # Polynomial coefficients for thrust from reference RPM.
    throttle_coefficients: list[float] = Field(
        default_factory=lambda: [0.0, -0.00835, 1.19113, 31.519]
    )  # Polynomial coefficients for reference RPM from throttle.
    engl_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    engr_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    engl_ang: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    engr_ang: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    cg_pos: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @field_validator(
        "engl_pos",
        "engr_pos",
        "engl_ang",
        "engr_ang",
        "cg_pos",
    )
    @classmethod
    def validate_vector3(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or not np.all(np.isfinite(value)):
            raise ValueError(
                "engine position/alignment vectors must contain 3 finite values"
            )
        return value

    @field_validator("thrust_coefficients", "throttle_coefficients")
    @classmethod
    def validate_polynomial(cls, value: list[float]) -> list[float]:
        if len(value) != 4 or not np.all(np.isfinite(value)):
            raise ValueError(
                "engine polynomial coefficients must contain 4 finite values"
            )
        return value


class SensorSingleConfig(BaseModel):
    """Bias, scale, and white-noise standard deviation for one sensor channel."""

    bias: float = 0.0
    scale: float = 1.0
    noise_std: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> "SensorSingleConfig":
        if not np.isfinite([self.bias, self.scale, self.noise_std]).all():
            raise ValueError("sensor bias, scale and noise_std must be finite")
        return self


class SensorConfig(BaseModel):
    """GTM sensor error configuration for the sensor output vector."""

    gamma: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    alpha: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    beta: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    V: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    altitude: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=3.0)
    )
    phi: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    theta: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    psi: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.125)
    )
    p: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    q: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    r: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.5)
    )
    nz: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.015)
    )
    qbar: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(
            noise_std=Pressure.psi_to_pa(0.00028)
        )
    )
    elevator: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.01)
    )
    aileron: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.01)
    )
    rudder: SensorSingleConfig = Field(
        default_factory=lambda: SensorSingleConfig(noise_std=0.01)
    )


class ReynoldsConfig(BaseModel):
    """Reynolds lookup with source altitude in feet and dimensionless Mach.

    The stored MATLAB table is flattened in Fortran order by the aerodynamic
    block before interpolation; changing that orientation changes all tables
    that use Reynolds as an axis.
    """

    altitude_ft: Any = Field(default=None, exclude=True)  # Altitude breakpoints (ft).
    mach_bp: Any = Field(default=None, exclude=True)  # Mach breakpoints.
    reynolds_values: Any = Field(default=None, exclude=True)  # Reynolds table values.


class BasicAirframeConfig(BaseModel):
    """Baseline force/moment coefficient tables.

    C2 data returns ``[CX, CZ]`` over alpha/beta/Reynolds. C4 returns
    ``[CY, Cl, Cm, Cn]`` over alpha/beta. Angles are source-table degrees.
    """

    alpha_c2: Any = Field(default=None, exclude=True)  # C2 alpha breakpoints (deg).
    beta_c2: Any = Field(default=None, exclude=True)  # C2 beta breakpoints (deg).
    reynolds_c2: Any = Field(default=None, exclude=True)  # C2 Reynolds breakpoints.
    data_cx_cz: Any = Field(default=None, exclude=True)  # [CX, CZ] table.
    alpha_c4: Any = Field(default=None, exclude=True)  # C4 alpha breakpoints (deg).
    beta_c4: Any = Field(default=None, exclude=True)  # C4 beta breakpoints (deg).
    data_cy_cl_cm_cn: Any = Field(default=None, exclude=True)  # [CY, Cl, Cm, Cn].


class ControlSurfacesConfig(BaseModel):
    """Elevator/aileron/rudder increments and spanwise moment arms.

    Each table retains gtm_ad axis order and degree-valued actuator breakpoints.
    The four arm values shift local surface forces into roll/yaw moments.
    """

    alpha_ele: Any = Field(default=None, exclude=True)
    beta_ele: Any = Field(default=None, exclude=True)
    stab_ele: Any = Field(default=None, exclude=True)
    elev_bp: Any = Field(default=None, exclude=True)
    data_ele: Any = Field(default=None, exclude=True)
    alpha_ail: Any = Field(default=None, exclude=True)
    beta_ail: Any = Field(default=None, exclude=True)
    ail_bp: Any = Field(default=None, exclude=True)
    data_ail: Any = Field(default=None, exclude=True)
    alpha_rud: Any = Field(default=None, exclude=True)
    beta_rud: Any = Field(default=None, exclude=True)
    rud_bp: Any = Field(default=None, exclude=True)
    data_rud: Any = Field(default=None, exclude=True)
    mode: str = Mode.NORMAL.value
    arm_left_out: float = -0.07
    arm_left_in: float = -0.03
    arm_right_in: float = 0.03
    arm_right_out: float = 0.07


class DynamicDerivativesConfig(BaseModel):
    """Kalviste rate-decomposition tables and reference lengths.

    Normalized ``p/q/r`` contributions use wing span ``b`` or mean chord
    ``Cbar``. The w-axis table carries coupled alpha/beta/rate coefficients.
    """

    b: float = 6.0  # Wing span (m).
    Cbar: float = 1.0  # Mean aerodynamic chord (m).
    alpha_p: Any = Field(default=None, exclude=True)
    phat_bp: Any = Field(default=None, exclude=True)
    data_p: Any = Field(default=None, exclude=True)
    alpha_q: Any = Field(default=None, exclude=True)
    qhat_bp: Any = Field(default=None, exclude=True)
    data_q: Any = Field(default=None, exclude=True)
    alpha_r: Any = Field(default=None, exclude=True)
    rhat_bp: Any = Field(default=None, exclude=True)
    data_r: Any = Field(default=None, exclude=True)
    alpha_w: Any = Field(default=None, exclude=True)
    what_bp: Any = Field(default=None, exclude=True)
    beta_w: Any = Field(default=None, exclude=True)
    data_w: Any = Field(default=None, exclude=True)


class AeroForceMomentConfig(BaseModel):
    """Reference area/lengths and aerodynamic-center offset in SI units."""

    Sref: float = 1.0  # Reference wing area (m^2).
    b: float = 1.0  # Wing span (m).
    Cbar: float = 1.0  # Mean aerodynamic chord (m).
    arm_x: float = 0.0  # CP - CG offset x (m).
    arm_y: float = 0.0  # CP - CG offset y (m).
    arm_z: float = 0.0  # CP - CG offset z (m).


class AircraftConfig(BaseModel):
    """Nominal aircraft geometry and mass properties.

    Values are SI unless a field name states otherwise. Defaults are generated
    from the ``gtm_ad`` scale-model data and full-scale scaling rules.
    """

    w0: float = Force.lbf_to_n(185_000.0)  # Nominal aircraft weight (N).
    inertia0: list[float] = Field(
        default_factory=lambda: (
            Inertia.slug_ft2_to_kg_m2(
                np.array(
                    [
                        1_770_000.0,
                        5_680_000.0,
                        7_270_000.0,
                        160_000.0,
                        0.0,
                        0.0,
                    ]
                )
            )
        ).tolist()
    )  # [Ixx, Iyy, Izz, Ixz, Iyz, Ixy] in kg*m^2.
    S: float = 5.9018 * (1.0 / 0.055) ** 2 * Length.ft_to_m(1.0) ** 2
    Cbar: float = Length.ft_to_m(0.9153 * (1.0 / 0.055))  # Mean aero chord (m).
    b: float = Length.ft_to_m(6.8488 * (1.0 / 0.055))  # Wing span (m).
    le_mac: float = Length.ft_to_m(4.5462 * (1.0 / 0.055))  # MAC LE from nose (m).
    cg_pos0: list[float] = Field(default_factory=list)  # CG from nose (m).
    ref_cg: list[float] = Field(
        default_factory=list
    )  # Aero reference CG from nose (m).
    engl_pos: list[float] = Field(default_factory=list)  # Left engine position (m).
    engr_pos: list[float] = Field(default_factory=list)  # Right engine position (m).
    engl_ang: list[float] = Field(
        default_factory=lambda: [0.0, 1.98, 2.22]
    )  # Left engine [roll, pitch, yaw] alignment (deg).
    engr_ang: list[float] = Field(
        default_factory=lambda: [0.0, 1.95, -1.18]
    )  # Right engine [roll, pitch, yaw] alignment (deg).

    @model_validator(mode="after")
    def validate_geometry_and_inertia(self) -> "AircraftConfig":
        if self.w0 <= 0.0 or self.S <= 0.0 or self.Cbar <= 0.0 or self.b <= 0.0:
            raise ValueError("aircraft weight and reference geometry must be positive")
        inertia = np.asarray(self.inertia0, dtype=float)
        if inertia.shape != (6,) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia0 must contain 6 finite values")
        tensor = np.array(
            [
                [inertia[0], -inertia[5], -inertia[3]],
                [-inertia[5], inertia[1], -inertia[4]],
                [-inertia[3], -inertia[4], inertia[2]],
            ]
        )
        if np.min(np.linalg.eigvalsh(tensor)) <= 0.0:
            raise ValueError("inertia0 must define a positive-definite tensor")
        for name, value in (
            ("cg_pos0", self.cg_pos0),
            ("ref_cg", self.ref_cg),
            ("engl_pos", self.engl_pos),
            ("engr_pos", self.engr_pos),
            ("engl_ang", self.engl_ang),
            ("engr_ang", self.engr_ang),
        ):
            if value and (len(value) != 3 or not np.all(np.isfinite(value))):
                raise ValueError(f"{name} must contain 3 finite values")
        return self


class WindConfig(BaseModel):
    """Compile-time wind-shear and Dryden calibration values.

    Mean wind speed, direction and turbulence level remain runtime parameters;
    direction uses radians throughout the public model.
    """

    runway_altitude: float = 0.0  # Runway altitude used by wind shear (m).
    wind_speed_low: float = Field(default=15.0, gt=0.0)  # Low-altitude intensity [m/s].
    scale_length_high: float = Field(default=533.4, gt=0.0)  # High scale [m].
    noise_sample_time: float = Field(default=0.1, gt=0.0)  # Noise sample time [s].


class ActuatorConfig(BaseModel):
    """Actuator dynamics limits for [elevator, aileron L/R, rudder/throttle]."""

    gain: list[float] = Field(default_factory=list)  # First-order actuator gains.
    rate_limit: list[float] = Field(default_factory=list)  # Rate limits (deg/s or %/s).
    position_lower: list[float] = Field(
        default_factory=lambda: [-30.0, -20.0, -30.0, 0.0]
    )  # Lower position limits [deg, deg, deg, %].
    position_upper: list[float] = Field(
        default_factory=lambda: [20.0, 20.0, 30.0, 100.0]
    )  # Upper position limits [deg, deg, deg, %].

    @model_validator(mode="after")
    def validate_channels(self) -> "ActuatorConfig":
        vectors = [self.gain, self.rate_limit, self.position_lower, self.position_upper]
        for value in vectors:
            if value and (len(value) != 4 or not np.all(np.isfinite(value))):
                raise ValueError("actuator vectors must contain 4 finite values")
        if (
            self.position_lower
            and self.position_upper
            and np.any(
                np.asarray(self.position_lower) >= np.asarray(self.position_upper)
            )
        ):
            raise ValueError("actuator lower limits must be below upper limits")
        if self.rate_limit and np.any(np.asarray(self.rate_limit) <= 0.0):
            raise ValueError("actuator rate limits must be positive")
        return self


class AeroConfig(BaseModel):
    """All GTM aerodynamic tables and force/moment geometry.

    Arrays are loaded from the packaged ``aero_tables.npz`` resource and kept
    out of Pydantic serialization so system JSON contains resource references,
    not duplicated table payloads.
    """

    # Optional aero_tables.npz path or containing directory.
    data_path: str | None = None
    reynolds: ReynoldsConfig = Field(default_factory=ReynoldsConfig)
    basic_airframe: BasicAirframeConfig = Field(default_factory=BasicAirframeConfig)
    control_surfaces: ControlSurfacesConfig = Field(
        default_factory=ControlSurfacesConfig
    )
    dynamic_derivatives: DynamicDerivativesConfig = Field(
        default_factory=DynamicDerivativesConfig
    )
    force_moment: AeroForceMomentConfig = Field(default_factory=AeroForceMomentConfig)


class GTMConfig(BaseModel):
    """Top-level configuration derived consistently for scale/mode choices.

    Call :meth:`default` rather than independently constructing nested configs
    when switching full-scale/scale model or flap/gear mode; geometry, engine,
    actuators, sensors and aerodynamic resources are coupled by that choice.
    """

    is_fullscale: bool = True
    mode: Mode = Mode.NORMAL
    aircraft: AircraftConfig = Field(
        default_factory=lambda: _build_aircraft_config(True)
    )
    wind: WindConfig = Field(default_factory=WindConfig)
    ground: GroundContactConfig = Field(
        default_factory=lambda: _build_ground_config(True, _build_aircraft_config(True))
    )
    uncertainty: UncertaintyConfig = Field(
        default_factory=lambda: _build_uncertainty_config(_build_aircraft_config(True))
    )
    engine: EngineConfig = Field(
        default_factory=lambda: _build_engine_config(True, _build_aircraft_config(True))
    )
    actuator: ActuatorConfig = Field(
        default_factory=lambda: _build_actuator_config(True)
    )
    aero: AeroConfig = Field(
        default_factory=lambda: _build_aero_config(
            _build_aircraft_config(True), Mode.NORMAL
        )
    )
    sensor: SensorConfig = Field(default_factory=lambda: _build_sensor_config(True))

    @classmethod
    def default(
        cls,
        is_fullscale: bool = True,
        mode: Mode | str = Mode.NORMAL,
        *,
        data_path: str | Path | None = None,
        enable_ground: bool = False,
    ) -> "GTMConfig":
        """Create the derived default GTM configuration.

        Args:
            is_fullscale: Use full-scale aircraft parameters when true.
            mode: Flap and landing-gear aerodynamic mode.
            data_path: Optional ``aero_tables.npz`` path or containing directory.
            enable_ground: Instantiate the fixed-wing runway-contact subsystem.
        """
        mode_value = _parse_mode(mode)
        aircraft = _build_aircraft_config(is_fullscale)
        return cls(
            is_fullscale=is_fullscale,
            mode=mode_value,
            aircraft=aircraft,
            wind=WindConfig(),
            ground=_build_ground_config(
                is_fullscale,
                aircraft,
                enabled=enable_ground,
            ),
            uncertainty=_build_uncertainty_config(aircraft),
            engine=_build_engine_config(is_fullscale, aircraft),
            actuator=_build_actuator_config(is_fullscale),
            aero=_build_aero_config(aircraft, mode_value, data_path=data_path),
            sensor=_build_sensor_config(is_fullscale),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "GTMConfig":
        """Load a TOML file as default GTM config plus sparse overrides."""
        config_path = Path(path)
        raw = rtoml.loads(config_path.read_text(encoding="utf-8"))
        gtm_raw = dict(raw.get("gtm", {}))
        if "trim_model_type" in gtm_raw:
            raise ValueError(
                "trim_model_type has been removed from fr_gtm; remove it from TOML."
            )
        mode = _parse_mode(gtm_raw.get("mode", Mode.NORMAL))
        is_fullscale = bool(gtm_raw.get("is_fullscale", True))

        aero_raw = dict(raw.get("aero", {}))
        data_path = aero_raw.pop("data_path", None)
        if data_path is not None:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                data_path = config_path.parent / data_path

        cfg = cls.default(
            is_fullscale=is_fullscale,
            mode=mode,
            data_path=data_path,
        )

        if "aircraft" in raw:
            cfg.aircraft = _merge_model(cfg.aircraft, raw["aircraft"])
            cfg.uncertainty = _build_uncertainty_config(cfg.aircraft)
            cfg.engine = _build_engine_config(cfg.is_fullscale, cfg.aircraft)
            cfg.aero = _build_aero_config(
                cfg.aircraft,
                cfg.mode,
                data_path=cfg.aero.data_path,
            )

        if "wind" in raw:
            cfg.wind = _merge_model(cfg.wind, raw["wind"])
        if "ground" in raw:
            cfg.ground = _merge_model(cfg.ground, raw["ground"])
        if "uncertainty" in raw:
            cfg.uncertainty = _merge_model(cfg.uncertainty, raw["uncertainty"])
        if "engine" in raw:
            cfg.engine = _merge_model(cfg.engine, raw["engine"])
        if "actuator" in raw:
            cfg.actuator = _merge_model(cfg.actuator, raw["actuator"])
        if aero_raw:
            cfg.aero = _merge_model(cfg.aero, aero_raw)
        if "sensor" in raw:
            cfg.sensor = _merge_model(cfg.sensor, raw["sensor"])

        return cfg


def _scale_factor(is_fullscale: bool) -> float:
    return 1.0 / 0.055 if is_fullscale else 1.0


def _build_aircraft_config(is_fullscale: bool) -> AircraftConfig:
    scale = _scale_factor(is_fullscale)

    if is_fullscale:
        w0 = Force.lbf_to_n(185_000.0)
        inertia0 = Inertia.slug_ft2_to_kg_m2(
            np.array([1_770_000.0, 5_680_000.0, 7_270_000.0, 160_000.0, 0.0, 0.0])
        )
    else:
        w0 = Force.lbf_to_n(57.75)
        inertia0 = Inertia.slug_ft2_to_kg_m2(
            np.array([1.221, 4.655, 5.587, 0.274, 0.0, 0.006])
        )

    S = 5.9018 * scale**2 * Length.ft_to_m(1.0) ** 2
    Cbar = Length.ft_to_m(0.9153 * scale)
    b = Length.ft_to_m(6.8488 * scale)
    le_mac = Length.ft_to_m(4.5462 * scale)

    cg_pos0 = np.array(
        [
            -(le_mac + Cbar * 0.2199),
            Length.ft_to_m(-0.1416 / 12.0 * scale),
            Length.ft_to_m(-0.9761 * scale),
        ]
    )
    ref_cg = np.array([-(le_mac + Cbar * 0.25), 0.0, Length.ft_to_m(-0.9401 * scale)])
    engl_pos = np.array(
        [
            Length.ft_to_m(-51.903 / 12.0 * scale),
            Length.ft_to_m(-14.20 / 12.0 * scale),
            Length.ft_to_m(-7.71 / 12.0 * scale),
        ]
    )
    engr_pos = np.array(
        [
            Length.ft_to_m(-51.903 / 12.0 * scale),
            Length.ft_to_m(14.20 / 12.0 * scale),
            Length.ft_to_m(-7.71 / 12.0 * scale),
        ]
    )

    return AircraftConfig(
        w0=w0,
        inertia0=inertia0.tolist(),
        S=S,
        Cbar=Cbar,
        b=b,
        le_mac=le_mac,
        cg_pos0=cg_pos0.tolist(),
        ref_cg=ref_cg.tolist(),
        engl_pos=engl_pos.tolist(),
        engr_pos=engr_pos.tolist(),
        engl_ang=[0.0, 1.98, 2.22],
        engr_ang=[0.0, 1.95, -1.18],
    )


def _build_uncertainty_config(aircraft: AircraftConfig) -> UncertaintyConfig:
    inertia = aircraft.inertia0
    return UncertaintyConfig(
        w0=aircraft.w0,
        Ixx0=inertia[0],
        Iyy0=inertia[1],
        Izz0=inertia[2],
        Ixz0=inertia[3],
        Iyz0=inertia[4],
        Ixy0=inertia[5],
    )


def _build_ground_config(
    is_fullscale: bool,
    aircraft: AircraftConfig,
    *,
    enabled: bool = False,
) -> GroundContactConfig:
    """Derive FRD/SI wheel data from the public NASA GTM T2 parameters."""

    length_ratio = _scale_factor(is_fullscale)
    weight_ratio = aircraft.w0 / Force.lbf_to_n(57.75)
    source_cg_ft = np.array(
        [
            -(4.5462 + 0.9153 * 0.2199),
            -0.1416 / 12.0,
            -0.9761,
        ]
    )
    source_points_ft = np.array(
        [
            [-22.24 / 12.0, 0.0, -1.7 / 12.0],
            [-61.31 / 12.0, -8.0 / 12.0, -2.64 / 12.0],
            [-61.31 / 12.0, 7.8 / 12.0, -2.64 / 12.0],
        ]
    )
    points_body = Length.ft_to_m((source_points_ft - source_cg_ft) * length_ratio)
    force_per_length = Force.lbf_to_n(1.0) / Length.ft_to_m(1.0)
    stiffness_ratio = weight_ratio / length_ratio
    mass_ratio = weight_ratio
    stiffness = np.array([25.0, 41.6, 42.4]) * 12.0 * force_per_length * stiffness_ratio
    damping = np.full(3, 7.0) * force_per_length * np.sqrt(stiffness_ratio * mass_ratio)
    support_matrix = np.vstack([np.ones(3), points_body[:, 0], points_body[:, 1]])
    static_loads = np.linalg.solve(support_matrix, np.array([aircraft.w0, 0.0, 0.0]))
    names = ("nose", "main_left", "main_right")
    contacts = [
        GroundContactPointConfig(
            name=name,
            point_body=point.tolist(),
            normal_stiffness=float(wheel_stiffness),
            normal_damping=float(wheel_damping),
            rolling_resistance_coefficient=0.02,
            lateral_friction_coefficient=0.5,
            max_normal_force=float(10.0 * static_load),
        )
        for name, point, wheel_stiffness, wheel_damping, static_load in zip(
            names, points_body, stiffness, damping, static_loads, strict=True
        )
    ]
    return GroundContactConfig(enabled=enabled, contacts=contacts)


def _build_engine_config(
    is_fullscale: bool,
    aircraft: AircraftConfig,
) -> EngineConfig:
    scale = _scale_factor(is_fullscale)
    if is_fullscale:
        T_gain = Force.lbf_to_n(40000.0 / 14.0)
        h_gain = scale**5 * Inertia.slug_ft2_to_kg_m2(1.0)
        RPMref_gain = 20000.0 / 123000.0
        ram_drag_coeff = 0.004
    else:
        T_gain = Force.lbf_to_n(1.0)
        h_gain = Inertia.slug_ft2_to_kg_m2(1.0)
        RPMref_gain = 1.0
        ram_drag_coeff = 0.01

    return EngineConfig(
        T_gain=T_gain,
        h_gain=h_gain,
        RPMref_gain=RPMref_gain,
        ram_drag_coeff=ram_drag_coeff,
        S=aircraft.S,
        shaft_diameter=0.75 / 12.0,
        shaft_length=6.25 / 12.0,
        density=15.2,
        thrust_coefficients=[0.000014847593013, 0.0, 0.0, 0.412662356277121],
        throttle_coefficients=[
            0.000033035538475,
            -0.008354385314499,
            1.191126728328792,
            31.518959847101669,
        ],
        engl_pos=aircraft.engl_pos,
        engr_pos=aircraft.engr_pos,
        engl_ang=aircraft.engl_ang,
        engr_ang=aircraft.engr_ang,
        cg_pos=aircraft.cg_pos0,
    )


def _build_actuator_config(is_fullscale: bool) -> ActuatorConfig:
    scale = _scale_factor(is_fullscale)
    if is_fullscale:
        return ActuatorConfig(
            gain=[
                5.0 * 2.0 * pi / sqrt(scale),
                5.0 * 2.0 * pi / sqrt(scale),
                5.0 * 2.0 * pi / sqrt(scale),
                1.0 / 2.36 / sqrt(scale),
            ],
            rate_limit=[
                300.0 / sqrt(scale),
                300.0 / sqrt(scale),
                300.0 / sqrt(scale),
                33.3,
            ],
        )
    return ActuatorConfig(
        gain=[5.0 * 2.0 * pi, 5.0 * 2.0 * pi, 5.0 * 2.0 * pi, 1.0 / 2.36],
        rate_limit=[300.0, 300.0, 300.0, 100.0],
    )


def _build_sensor_config(is_fullscale: bool) -> SensorConfig:
    if is_fullscale:
        return SensorConfig(
            gamma=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            alpha=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            beta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            V=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            altitude=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=3.0),
            phi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            theta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            psi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.125),
            p=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            q=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            r=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.5),
            nz=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.015),
            qbar=SensorSingleConfig(
                bias=0.0,
                scale=1.0,
                noise_std=Pressure.psi_to_pa(0.00028),
            ),
            elevator=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
            aileron=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
            rudder=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
        )

    return SensorConfig(
        gamma=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.033),
        alpha=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.031),
        beta=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.033),
        V=SensorSingleConfig(
            bias=Speed.kts_to_mps(0.07),
            scale=1.0,
            noise_std=Speed.kts_to_mps(0.06),
        ),
        altitude=SensorSingleConfig(
            bias=0.0,
            scale=1.0,
            noise_std=Length.ft_to_m(2.36),
        ),
        phi=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.02),
        theta=SensorSingleConfig(bias=-0.09, scale=1.0, noise_std=0.02),
        psi=SensorSingleConfig(bias=-0.04, scale=1.0, noise_std=0.02),
        p=SensorSingleConfig(bias=-0.0057, scale=1.0, noise_std=0.5350),
        q=SensorSingleConfig(bias=-0.0115, scale=1.0, noise_std=0.4737),
        r=SensorSingleConfig(bias=-0.0286, scale=1.0, noise_std=0.3397),
        nz=SensorSingleConfig(bias=0.00086, scale=1.0, noise_std=0.0082),
        qbar=SensorSingleConfig(
            bias=0.0,
            scale=1.0,
            noise_std=Pressure.psi_to_pa(0.00028),
        ),
        elevator=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
        aileron=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
        rudder=SensorSingleConfig(bias=0.0, scale=1.0, noise_std=0.01),
    )


def _build_aero_config(
    aircraft: AircraftConfig,
    mode: Mode,
    *,
    data_path: str | Path | None = None,
) -> AeroConfig:
    tables, resolved_data_path = _load_aero_tables(data_path)
    arm = np.asarray(aircraft.ref_cg) - np.asarray(aircraft.cg_pos0)

    return AeroConfig(
        data_path=str(resolved_data_path) if data_path is not None else None,
        reynolds=ReynoldsConfig(
            altitude_ft=tables["reynolds"]["altitude_ft"],
            mach_bp=tables["reynolds"]["mach"],
            reynolds_values=tables["reynolds"]["reynolds"],
        ),
        basic_airframe=BasicAirframeConfig(
            alpha_c2=tables["C2_bas_Re_dep"]["alpha"],
            beta_c2=tables["C2_bas_Re_dep"]["beta"],
            reynolds_c2=tables["C2_bas_Re_dep"]["reynolds"],
            data_cx_cz=tables["C2_bas_Re_dep"]["data"],
            alpha_c4=tables["C4_bas_Re_indep"]["alpha"],
            beta_c4=tables["C4_bas_Re_indep"]["beta"],
            data_cy_cl_cm_cn=tables["C4_bas_Re_indep"]["data"],
        ),
        control_surfaces=ControlSurfacesConfig(
            alpha_ele=tables["dC3_ele"]["alpha"],
            beta_ele=tables["dC3_ele"]["beta"],
            stab_ele=tables["dC3_ele"]["stab"],
            elev_bp=tables["dC3_ele"]["elev"],
            data_ele=tables["dC3_ele"]["data"],
            alpha_ail=tables["dC6_ail"]["alpha"],
            beta_ail=tables["dC6_ail"]["beta"],
            ail_bp=tables["dC6_ail"]["ail"],
            data_ail=tables["dC6_ail"]["data"],
            alpha_rud=tables["dC6_rud"]["alpha"],
            beta_rud=tables["dC6_rud"]["beta"],
            rud_bp=tables["dC6_rud"]["rud"],
            data_rud=tables["dC6_rud"]["data"],
            mode=mode.value,
        ),
        dynamic_derivatives=DynamicDerivativesConfig(
            b=aircraft.b,
            Cbar=aircraft.Cbar,
            alpha_p=tables["dC3_p"]["alpha"],
            phat_bp=tables["dC3_p"]["phat"],
            data_p=tables["dC3_p"]["data"],
            alpha_q=tables["dC3_q"]["alpha"],
            qhat_bp=tables["dC3_q"]["qhat"],
            data_q=tables["dC3_q"]["data"],
            alpha_r=tables["dC3_r"]["alpha"],
            rhat_bp=tables["dC3_r"]["rhat"],
            data_r=tables["dC3_r"]["data"],
            alpha_w=tables["dC6_w"]["alpha"],
            what_bp=tables["dC6_w"]["what"],
            beta_w=tables["dC6_w"]["beta"],
            data_w=tables["dC6_w"]["data"],
        ),
        force_moment=AeroForceMomentConfig(
            Sref=aircraft.S,
            b=aircraft.b,
            Cbar=aircraft.Cbar,
            arm_x=float(arm[0]),
            arm_y=float(arm[1]),
            arm_z=float(arm[2]),
        ),
    )


def _load_aero_tables(data_path: str | Path | None) -> tuple[dict[str, Any], Path]:
    if data_path is None:
        resource = resources.files("fr_gtm") / "data" / "aero_tables.npz"
        with resources.as_file(resource) as package_data_path:
            return _read_aero_tables(package_data_path), package_data_path

    resolved = Path(data_path)
    if resolved.is_dir():
        resolved /= "aero_tables.npz"
    return _read_aero_tables(resolved), resolved


def _read_aero_tables(data_path: Path) -> dict[str, Any]:
    groups = {
        "C2_bas_Re_dep": ("data", "alpha", "beta", "reynolds"),
        "C4_bas_Re_indep": ("data", "alpha", "beta"),
        "reynolds": ("reynolds", "altitude_ft", "mach"),
        "dC3_ele": ("data", "alpha", "beta", "stab", "elev"),
        "dC6_ail": ("data", "alpha", "beta", "ail"),
        "dC6_rud": ("data", "alpha", "beta", "rud"),
        "dC3_p": ("data", "alpha", "phat"),
        "dC3_q": ("data", "alpha", "qhat"),
        "dC3_r": ("data", "alpha", "rhat"),
        "dC6_w": ("data", "alpha", "what", "beta"),
    }
    with np.load(data_path, allow_pickle=False) as archive:
        return {
            group: {key: archive[f"{group}__{key}"] for key in keys}
            for group, keys in groups.items()
        }


def _parse_mode(mode: Mode | str) -> Mode:
    if isinstance(mode, Mode):
        return mode
    return Mode(mode)


def _merge_model[T: BaseModel](model: T, updates: dict[str, Any]) -> T:
    # Use model_fields to get all fields including those with exclude=True,
    # since pydantic v2 Field(exclude=True) permanently excludes from model_dump.
    data = {name: getattr(model, name) for name in type(model).model_fields}
    _deep_update(data, updates)
    return type(model)(**data)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


__all__ = [
    "ActuatorConfig",
    "AeroConfig",
    "AeroForceMomentConfig",
    "AircraftConfig",
    "BasicAirframeConfig",
    "ControlSurfacesConfig",
    "DynamicDerivativesConfig",
    "EngineConfig",
    "GTMConfig",
    "Mode",
    "ReynoldsConfig",
    "SensorConfig",
    "SensorSingleConfig",
    "UncertaintyConfig",
    "WindConfig",
]
