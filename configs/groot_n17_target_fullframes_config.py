from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


groot_n17_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_head",
        ],
    ),
    # 비전 attention supervision용 대상 물체 마스크 (video와 같은 프레임)
    "mask": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_head",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left",
            "arm_right",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left",
            "arm_right",
        ],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF, format=ActionFormat.DEFAULT) for _ in range(2)],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["sub_task"],
    ),
}


register_modality_config(groot_n17_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
