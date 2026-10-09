"""The 30 Hz CHIP contract with a 24-step prediction horizon."""

from gr00t.data.chip_native import REFERENCE_FIELDS
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


CHIP_ACTION_HORIZON = 24


def chip_native_modality_config():
    return {
        "video": ModalityConfig(delta_indices=[-6, 0], modality_keys=["camera0_rgb"]),
        "state": ModalityConfig(delta_indices=[-6, -3, 0], modality_keys=list(REFERENCE_FIELDS)),
        "action": ModalityConfig(
            delta_indices=list(range(1, CHIP_ACTION_HORIZON + 1)),
            modality_keys=["native_relative"],
            # The dataset already encodes quaternion-composed relative targets.
            # ABSOLUTE here means pass through GR00T's automatic conversion;
            # the physical output remains RELATIVE, including at inference.
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
