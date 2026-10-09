"""Bridge the existing CHIP history contract to a local or remote GR00T policy."""

import numpy as np

from gr00t.configs.data.chip_native_config import CHIP_ACTION_HORIZON
from gr00t.data.chip_native import (
    OBS_FIELD_DIMS,
    pack_chip_reference,
    reference_relative_to_current,
    relative_reference_to_absolute,
    unpack_chip_reference,
)


def build_chip_observation(
    images: np.ndarray,
    native_history: np.ndarray,
    task_description: str = "Perform the demonstrated task.",
):
    """Build one request from RGB uint8 [2,224,400,3] and canonical [3,37].

    History must already use the episode/A-time yaw canonical frame, exactly
    as HIL's observation builder does. Do not canonicalize each sliding window.
    """
    images = np.asarray(images)
    history = np.asarray(native_history, dtype=np.float32)
    if images.shape != (2, 224, 400, 3) or images.dtype != np.uint8:
        raise ValueError("images must be RGB uint8 with shape (2,224,400,3)")
    if history.shape != (3, 37) or not np.isfinite(history).all():
        raise ValueError("native_history must be finite with shape (3,37)")
    if not task_description.strip():
        raise ValueError("task_description must be nonempty")
    fields = unpack_chip_reference(history)
    for key, value in fields.items():
        if "quat" in key and not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-4):
            raise ValueError(f"{key} must contain unit wxyz quaternions")
    return {
        "video": {"camera0_rgb": images[None]},
        "state": {key: value.reshape(1, 3, OBS_FIELD_DIMS[key]) for key, value in fields.items()},
        "language": {"task": [[task_description]]},
    }


def predict_chip_relative(
    policy,
    images,
    native_history,
    task_description="Perform the demonstrated task.",
    *,
    rtc_prefix=None,
):
    """Return [24,37] relative actions from Gr00tPolicy or PolicyClient.

    Decode against the captured request-time observation using HIL's existing
    decoder. This function deliberately returns the relative model prediction.
    """
    observation = build_chip_observation(images, native_history, task_description)
    if rtc_prefix is None:
        action, _ = policy.get_action(observation)
    else:
        prefix = np.asarray(rtc_prefix, dtype=np.float32)
        if (
            prefix.ndim != 2
            or prefix.shape[1] != 37
            or not 0 < len(prefix) < CHIP_ACTION_HORIZON
            or not np.isfinite(prefix).all()
        ):
            raise ValueError(f"rtc_prefix must be finite [d,37] with 0 < d < {CHIP_ACTION_HORIZON}")
        action, _ = policy.get_action(
            observation,
            options={
                "rtc_mode": "trained",
                "rtc_prefix": {"native_relative": prefix[None]},
            },
        )
    relative = np.asarray(action["native_relative"], dtype=np.float32)
    expected_shape = (1, CHIP_ACTION_HORIZON, 37)
    if relative.shape != expected_shape or not np.isfinite(relative).all():
        raise ValueError(f"Invalid CHIP prediction: {relative.shape}, expected {expected_shape}")
    return relative[0]


def rebase_chip_prefix(
    previous_chunk, previous_reference, current_reference, elapsed_steps, delay_steps
):
    """Align the previous chunk to a new request and re-encode its committed prefix.

    Both reference rows must be [37] in the SAME episode/A-time canonical frame.
    elapsed_steps counts 30 Hz rows since the previous request (not 10 Hz replans
    or 50 Hz CHIP ticks). The returned prefix spans new t+1 through new t+d.
    """
    chunk = np.asarray(previous_chunk, dtype=np.float32)
    old = np.asarray(previous_reference, dtype=np.float32)
    new = np.asarray(current_reference, dtype=np.float32)
    if chunk.shape != (CHIP_ACTION_HORIZON, 37) or old.shape != (37,) or new.shape != (37,):
        raise ValueError(f"Expected previous chunk [{CHIP_ACTION_HORIZON},37] and reference rows [37]")
    if not all(np.isfinite(v).all() for v in [chunk, old, new]):
        raise ValueError("CHIP references and actions must be finite")
    if (
        not isinstance(elapsed_steps, int)
        or not isinstance(delay_steps, int)
        or elapsed_steps < 0
        or not 0 < delay_steps < CHIP_ACTION_HORIZON
        or elapsed_steps + delay_steps > len(chunk)
    ):
        raise ValueError("Previous chunk does not cover the requested RTC prefix")
    absolute = relative_reference_to_absolute(
        unpack_chip_reference(chunk[elapsed_steps : elapsed_steps + delay_steps]),
        unpack_chip_reference(old),
    )
    return pack_chip_reference(reference_relative_to_current(absolute, unpack_chip_reference(new)))
