"""Measured replay joints -> CHIP 37-D geometry, using the source robot's FK tree.

The packaged tree is exported from the replay dataset's MimicLite scene. Only
kinematics are retained; meshes and a MuJoCo runtime are unnecessary for training.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t.data.chip_native import canonicalize_anchor_by_episode_yaw


POINT_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
LOWER_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)


def measured_chip_state(q_real, imu_wxyz, joint_names, point_offsets, *, canonicalize=True):
    q = np.asarray(q_real, dtype=np.float64)
    imu = np.asarray(imu_wxyz, dtype=np.float64)
    offsets = np.asarray(point_offsets, dtype=np.float64)
    names = list(map(str, joint_names))
    if q.ndim != 2 or q.shape[1] != 29 or imu.shape != (len(q), 4):
        raise ValueError("Expected measured joints [T,29] and IMU wxyz [T,4]")
    if len(names) != 29 or len(set(names)) != 29 or offsets.shape != (3, 3):
        raise ValueError("Invalid joint names or point offsets")
    if not (np.isfinite(q).all() and np.isfinite(imu).all() and np.isfinite(offsets).all()):
        raise ValueError("Nonfinite measured FK inputs")
    if np.any(np.linalg.norm(imu, axis=-1) < 1e-7):
        raise ValueError("Zero measured IMU quaternion")
    tree = json.loads(Path(__file__).with_name("chip_replay_kinematics.json").read_text())
    index = {name: i for i, name in enumerate(names)}
    batch = len(q)
    rotations = [np.broadcast_to(np.eye(3), (batch, 3, 3))]
    positions = [np.zeros((batch, 3))]
    body_ids = {}
    for body in tree["bodies"]:
        body_ids[body["name"]] = len(rotations)
        parent = body["parent"]
        local_r = Rotation.from_quat(np.asarray(body["quat_wxyz"])[[1, 2, 3, 0]]).as_matrix()
        r = rotations[parent] @ local_r
        p = positions[parent] + np.einsum("tij,j->ti", rotations[parent], body["pos"])
        # Work in the pelvis frame. The measured root rotation is supplied
        # separately as anchor_quat_wxyz; local three-point geometry is invariant.
        if body["name"] == "pelvis":
            r = rotations[0].copy()
            p = positions[0].copy()
        for joint in body["joints"]:
            jr = Rotation.from_rotvec(q[:, index[joint["name"]], None] * joint["axis"]).as_matrix()
            jp = np.asarray(joint["pos"])
            p = p + np.einsum("tij,tj->ti", r, jp - np.einsum("tij,j->ti", jr, jp))
            r = r @ jr
        rotations.append(r)
        positions.append(p)
    point_pos, point_quat = [], []
    for i, name in enumerate(POINT_BODIES):
        b = body_ids[name]
        point_pos.append(positions[b] + np.einsum("tij,j->ti", rotations[b], offsets[i]))
        quat = Rotation.from_matrix(rotations[b]).as_quat()[:, [3, 0, 1, 2]]
        point_quat.append(np.where(quat[:, :1] < 0, -quat, quat))
    fields = {
        "lower_joint_pos": q[:, [index[n] for n in LOWER_JOINT_NAMES]].astype(np.float32),
        "anchor_quat_wxyz": (imu / np.linalg.norm(imu, axis=-1, keepdims=True)).astype(np.float32),
        "vr_3point_pos_anchor": np.stack(point_pos, axis=1).astype(np.float32),
        "vr_3point_quat_anchor_wxyz": np.stack(point_quat, axis=1).astype(np.float32),
    }
    # Same initial-heading removal as the original CHIP state history.
    if canonicalize:
        fields = canonicalize_anchor_by_episode_yaw(fields, fields["anchor_quat_wxyz"][0])
    # q and -q represent the same orientation; maintain temporal sign continuity.
    for key in ("anchor_quat_wxyz", "vr_3point_quat_anchor_wxyz"):
        value = fields[key]
        for t in range(1, batch):
            value[t] = np.where(
                np.sum(value[t - 1] * value[t], axis=-1, keepdims=True) < 0, -value[t], value[t]
            )
    return fields
