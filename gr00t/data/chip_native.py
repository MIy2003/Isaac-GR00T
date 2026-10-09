"""CHIP-native geometry, matching HIL_GentleHumanoid's chip_native_dataset.py.

Scalar-first quaternions; points ordered left_wrist, right_wrist, torso.
These functions intentionally preserve the source training/deployment convention.
"""

from typing import Dict, Mapping

import numpy as np


CHIP_SCHEMA = "g1_chip_episode_folder_v001"
CHIP_ACTION_DIM = 37
CHIP_POINT_NAMES = ("left_wrist", "right_wrist", "torso")
REFERENCE_FIELDS = (
    "lower_joint_pos",
    "anchor_quat_wxyz",
    "vr_3point_pos_anchor",
    "vr_3point_quat_anchor_wxyz",
)
FIELD_SHAPES = {
    "lower_joint_pos": (12,),
    "anchor_quat_wxyz": (4,),
    "vr_3point_pos_anchor": (3, 3),
    "vr_3point_quat_anchor_wxyz": (3, 4),
}
OBS_FIELD_DIMS = {
    "lower_joint_pos": 12,
    "anchor_quat_wxyz": 4,
    "vr_3point_pos_anchor": 9,
    "vr_3point_quat_anchor_wxyz": 12,
}


def _normalize_quaternion_wxyz(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != 4 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must end in four finite values")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm < 1e-7):
        raise ValueError(f"{name} contains a near-zero quaternion")
    return (array / norm).astype(np.float32)


def _quat_conjugate_wxyz(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32).copy()
    output[..., 1:] *= -1.0
    return output


def _quat_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Hamilton product with NumPy broadcasting and scalar-first ordering."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    w1, v1 = first[..., :1], first[..., 1:]
    w2, v2 = second[..., :1], second[..., 1:]
    return np.concatenate(
        [w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True), w1 * v2 + w2 * v1 + np.cross(v1, v2)],
        axis=-1,
    ).astype(np.float32)


def _shortest_hemisphere_wxyz(value: np.ndarray) -> np.ndarray:
    value = _normalize_quaternion_wxyz(value, "relative quaternion")
    return np.where(value[..., :1] < 0.0, -value, value).astype(np.float32)


def _yaw_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    """Extract the Z-up yaw component from one scalar-first quaternion."""
    quat = _normalize_quaternion_wxyz(value, "anchor quaternion")
    w, x, y, z = [quat[..., axis] for axis in range(4)]
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    output = np.zeros(quat.shape, dtype=np.float32)
    output[..., 0] = np.cos(yaw / 2.0)
    output[..., 3] = np.sin(yaw / 2.0)
    return output


def canonicalize_anchor_by_episode_yaw(
    fields: Mapping[str, np.ndarray],
    episode_first_anchor_wxyz: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Remove one episode's initial global heading from its anchor rows."""
    output = {name: np.asarray(value, dtype=np.float32) for name, value in fields.items()}
    first_yaw = _yaw_quaternion_wxyz(episode_first_anchor_wxyz)
    inverse_yaw = _quat_conjugate_wxyz(first_yaw)
    output["anchor_quat_wxyz"] = _normalize_quaternion_wxyz(
        _quat_multiply_wxyz(
            inverse_yaw,
            output["anchor_quat_wxyz"],
        ),
        "yaw-canonical anchor_quat_wxyz",
    )
    return output


def pack_chip_reference(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """Pack named native fields into the exact 37-D DP action layout."""
    missing = [name for name in REFERENCE_FIELDS if name not in fields]
    if missing:
        raise KeyError(f"CHIP reference is missing fields: {missing}")
    leading_shape = np.asarray(fields["lower_joint_pos"]).shape[:-1]
    parts = []
    for name in REFERENCE_FIELDS:
        value = np.asarray(fields[name], dtype=np.float32)
        expected = leading_shape + FIELD_SHAPES[name]
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        parts.append(value.reshape(leading_shape + (-1,)))
    packed = np.concatenate(parts, axis=-1).astype(np.float32)
    if packed.shape != leading_shape + (CHIP_ACTION_DIM,):
        raise AssertionError(f"internal CHIP action shape error: {packed.shape}")
    return packed


def unpack_chip_reference(action: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a flat 37-D action without changing quaternion ordering."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape[-1] != CHIP_ACTION_DIM:
        raise ValueError(f"CHIP action must end in {CHIP_ACTION_DIM} values, got {value.shape}")
    leading = value.shape[:-1]
    return {
        "lower_joint_pos": value[..., 0:12],
        "anchor_quat_wxyz": value[..., 12:16],
        "vr_3point_pos_anchor": value[..., 16:25].reshape(leading + (3, 3)),
        "vr_3point_quat_anchor_wxyz": value[..., 25:37].reshape(leading + (3, 4)),
    }


def reference_relative_to_current(
    fields: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Encode native references relative to one current native reference."""
    anchor_current = _normalize_quaternion_wxyz(
        current["anchor_quat_wxyz"], "current anchor_quat_wxyz"
    )
    point_current = _normalize_quaternion_wxyz(
        current["vr_3point_quat_anchor_wxyz"],
        "current vr_3point_quat_anchor_wxyz",
    )
    return {
        "lower_joint_pos": (
            np.asarray(fields["lower_joint_pos"], dtype=np.float32)
            - np.asarray(current["lower_joint_pos"], dtype=np.float32)
        ).astype(np.float32),
        "anchor_quat_wxyz": _shortest_hemisphere_wxyz(
            _quat_multiply_wxyz(
                _quat_conjugate_wxyz(anchor_current),
                _normalize_quaternion_wxyz(fields["anchor_quat_wxyz"], "anchor_quat_wxyz"),
            )
        ),
        "vr_3point_pos_anchor": (
            np.asarray(fields["vr_3point_pos_anchor"], dtype=np.float32)
            - np.asarray(current["vr_3point_pos_anchor"], dtype=np.float32)
        ).astype(np.float32),
        "vr_3point_quat_anchor_wxyz": _shortest_hemisphere_wxyz(
            _quat_multiply_wxyz(
                _quat_conjugate_wxyz(point_current),
                _normalize_quaternion_wxyz(
                    fields["vr_3point_quat_anchor_wxyz"],
                    "vr_3point_quat_anchor_wxyz",
                ),
            )
        ),
    }


def relative_reference_to_absolute(
    relative: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Invert :func:`reference_relative_to_current` without FK or IK."""
    return {
        "lower_joint_pos": (
            np.asarray(current["lower_joint_pos"], dtype=np.float32)
            + np.asarray(relative["lower_joint_pos"], dtype=np.float32)
        ).astype(np.float32),
        "anchor_quat_wxyz": _normalize_quaternion_wxyz(
            _quat_multiply_wxyz(
                _normalize_quaternion_wxyz(current["anchor_quat_wxyz"], "current anchor_quat_wxyz"),
                _normalize_quaternion_wxyz(
                    relative["anchor_quat_wxyz"], "relative anchor_quat_wxyz"
                ),
            ),
            "decoded anchor_quat_wxyz",
        ),
        "vr_3point_pos_anchor": (
            np.asarray(current["vr_3point_pos_anchor"], dtype=np.float32)
            + np.asarray(relative["vr_3point_pos_anchor"], dtype=np.float32)
        ).astype(np.float32),
        "vr_3point_quat_anchor_wxyz": _normalize_quaternion_wxyz(
            _quat_multiply_wxyz(
                _normalize_quaternion_wxyz(
                    current["vr_3point_quat_anchor_wxyz"],
                    "current vr_3point_quat_anchor_wxyz",
                ),
                _normalize_quaternion_wxyz(
                    relative["vr_3point_quat_anchor_wxyz"],
                    "relative vr_3point_quat_anchor_wxyz",
                ),
            ),
            "decoded vr_3point_quat_anchor_wxyz",
        ),
    }
