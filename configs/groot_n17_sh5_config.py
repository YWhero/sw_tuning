import os

ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON", "16"))
if not 2 <= ACTION_HORIZON <= 40:
    raise ValueError("ACTION_HORIZON must be between 2 and 40")

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
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "tactile_left",
            "tactile_right",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF, format=ActionFormat.DEFAULT) for _ in range(4)],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}


register_modality_config(groot_n17_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
