"""地面测试通过公共 init 与 std 静态接触姿态组合构造场景。"""

import numpy as np
from fly_ruler_core.runtime.instance import RuntimeInstance
from fly_ruler_std.blocks.physics import solve_static_ground_pose

from fr_gtm import GTMConfig, init


def ground_state[T](
    model: RuntimeInstance[T],
    cfg: GTMConfig,
    *,
    forward_speed_mps: float = 0.0,
    altitude_agl_m: float = 0.0,
    sink_rate_mps: float = 0.0,
) -> T:
    down, roll, pitch = solve_static_ground_pose(cfg.ground, cfg.aircraft.w0)
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    dcm = np.array([[cp, sr * sp, cr * sp], [0.0, cr, -sr], [-sp, sr * cp, cr * cp]])
    velocity = dcm.T @ [forward_speed_mps, 0.0, sink_rate_mps]
    alpha = np.rad2deg(np.arctan2(velocity[2], velocity[0]))
    beta = np.rad2deg(np.arctan2(velocity[1], np.hypot(velocity[0], velocity[2])))
    x, _, _, _ = init(
        model,
        cfg,
        position=(0.0, 0.0, down - altitude_agl_m),
        airdata=(alpha, beta, float(np.linalg.norm(velocity))),
        euler=(float(np.rad2deg(roll)), float(np.rad2deg(pitch)), 0.0),
        command=(0.0, 0.0, 0.0, 0.0),
    )
    return x
