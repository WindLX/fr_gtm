from fr_gtm.stick_input import STICK_VISUAL_MODEL_ID


def test_stick_examples_select_the_737_visual_profile() -> None:
    """手柄示例使用 737-800 的可视化配置。"""
    assert STICK_VISUAL_MODEL_ID == "boeing_737_800"
