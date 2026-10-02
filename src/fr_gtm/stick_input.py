"""手柄轴输入归一化助手。"""

from dataclasses import dataclass

STICK_VISUAL_MODEL_ID = "boeing_737_800"
"""三维视图中使用的机型标识，与可视化资产目录同名。"""


@dataclass(frozen=True)
class StickAxes:
    """归一化手柄轴输入。

    Attributes:
        roll: 滚转轴，范围 [-1, 1]。
        pitch: 俯仰轴，范围 [-1, 1]。
        yaw: 偏航轴，范围 [-1, 1]。
        throttle: 油门轴；为 `None` 时表示该轴未提供。
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float | None = None
