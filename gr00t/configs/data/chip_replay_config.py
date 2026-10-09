"""Real replay: measured 37-D FK state and paired absolute 37-D reference actions."""

from gr00t.configs.data.chip_native_config import CHIP_ACTION_HORIZON, chip_native_modality_config
from gr00t.data.chip_native import REFERENCE_FIELDS
from gr00t.data.types import ModalityConfig


REPLAY_SCHEMA = "chip_real_replay_multimodal_v001"


def chip_replay_modality_config(*, yaw_canonical=False):
    config = chip_native_modality_config()
    config["state"] = ModalityConfig(
        delta_indices=[-6, -3, 0], modality_keys=list(REFERENCE_FIELDS)
    )
    config["action"].modality_keys = [
        "reference_yaw_canonical" if yaw_canonical else "reference_absolute"
    ]
    # Each state is pre-action: row t predicts the action paired with row t.
    config["action"].delta_indices = list(range(CHIP_ACTION_HORIZON))
    return config
