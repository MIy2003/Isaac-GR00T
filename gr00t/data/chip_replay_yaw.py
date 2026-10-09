"""Absolute replay targets in the source episode's initial-heading frame."""

import numpy as np

from gr00t.data.chip_native import canonicalize_anchor_by_episode_yaw


def canonicalize_replay_actions(actions):
    """Remove reference yaw at t=0, retaining roll/pitch and later heading changes.

    Measured state has its own initial IMU heading. Subtracting that heading
    here would mix the independent measured and source-reference world frames.
    Local three-point geometry and joint commands remain exactly unchanged.
    """
    result = np.asarray(actions, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 37 or not len(result):
        raise ValueError("Expected nonempty replay actions [T,37]")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite replay actions")
    anchor = canonicalize_anchor_by_episode_yaw(
        {"anchor_quat_wxyz": result[:, 12:16]}, result[0, 12:16]
    )["anchor_quat_wxyz"]
    # Choose one continuous quaternion branch, just as measured state FK does.
    if anchor[0, 0] < 0:
        anchor[0] *= -1
    for i in range(1, len(anchor)):
        if np.dot(anchor[i - 1], anchor[i]) < 0:
            anchor[i] *= -1
    result[:, 12:16] = anchor
    return result
