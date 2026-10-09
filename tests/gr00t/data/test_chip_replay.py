"""Replay contract regression tests; runnable with unittest (no GPU required)."""

import json
from pathlib import Path
import tempfile
import unittest

from gr00t.configs.data.chip_replay_config import chip_replay_modality_config
from gr00t.data.chip_replay_fk import LOWER_JOINT_NAMES, measured_chip_state
from gr00t.data.dataset.chip_replay_dataset import ChipReplayDataset
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from gr00t.data.chip_replay_yaw import canonicalize_replay_actions


class ReplayContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "episode"
        (self.path / "images").mkdir(parents=True)
        tree = json.loads(
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("gr00t/data/chip_replay_kinematics.json")
            .read_text()
        )
        self.names = [j["name"] for b in tree["bodies"] for j in b["joints"]]
        state = np.zeros((40, 65), np.float32)
        state[:, :29] = np.arange(40)[:, None] * 0.001
        state[:, 58] = 1
        self.data = dict(
            state=state,
            action=np.arange(40 * 37, dtype=np.float32).reshape(40, 37) / 100,
            joint_names=self.names,
            image_index=np.arange(40),
            image_file=[f"images/frame_{i:06d}.jpg" for i in range(40)],
            **{
                k: np.ones(40, bool)
                for k in ("valid", "state_valid", "action_valid", "state_action_timing_valid")
            },
        )
        for i in range(40):
            Image.new("RGB", (400, 224), (i, i, i)).save(self.path / self.data["image_file"][i])
        (self.root / "dataset_manifest.json").write_text(
            json.dumps(
                dict(
                    schema_version="chip_real_replay_multimodal_v001",
                    nominal_hz=30,
                    state_dim=65,
                    action_dim=37,
                    episode_names=["episode"],
                    source_image_frames=40,
                )
            )
        )
        (self.path / "chip_manifest.json").write_text(
            json.dumps(
                dict(
                    point_bodies=["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
                    point_offsets_m=[[0.18, -0.025, 0], [0.0719, -0.003, 0], [0, 0, 0.35]],
                )
            )
        )

    def dataset(self):
        np.savez(self.path / "state_action_30hz.npz", **self.data)
        return ChipReplayDataset(self.root, chip_replay_modality_config(), val_ratio=0)

    def test_independent_state_action_and_current_row_alignment(self):
        ds = self.dataset()
        self.assertEqual(ds.samples, [(0, t) for t in range(0, 17, 3)])
        state, action = ds.state_action(3)
        np.testing.assert_array_equal(action, self.data["action"][9:33])
        lower_ids = [self.names.index(n) for n in LOWER_JOINT_NAMES]
        np.testing.assert_array_equal(
            state["lower_joint_pos"], self.data["state"][[3, 6, 9]][:, lower_ids]
        )
        self.data["action"] += 50
        changed = self.dataset()
        new_state, new_action = changed.state_action(3)
        for key in state:
            np.testing.assert_array_equal(new_state[key], state[key])
        np.testing.assert_allclose(new_action, action + 50)
        step = changed.get_step(3)
        self.assertEqual(np.concatenate(list(step.states.values()), axis=-1).shape, (3, 37))
        self.assertEqual(step.actions["reference_absolute"].shape, (24, 37))

    def test_invalid_windows_are_excluded(self):
        self.data["state_valid"][0] = False
        ds = self.dataset()
        self.assertEqual(ds.samples, [(0, 9), (0, 12), (0, 15)])
        self.data["state_action_timing_valid"][25] = False
        with self.assertRaisesRegex(ValueError, "No valid"):
            self.dataset()

    def test_ideal_state_uses_previous_action_and_ignores_measured_pose(self):
        for start in (12, 25, 29, 33):
            self.data["action"][:, start:start + 4] = [1, 0, 0, 0]

        def load():
            np.savez(self.path / "state_action_30hz.npz", **self.data)
            return ChipReplayDataset(
                self.root, chip_replay_modality_config(yaw_canonical=True),
                val_ratio=0, yaw_canonical=True, ideal_state=True,
            )

        ds = load()
        state, action = ds.state_action(3)  # t=9: history action indices 2,5,8
        packed = np.concatenate(list(state.values()), axis=-1)
        np.testing.assert_array_equal(packed, self.data["action"][[2, 5, 8]])
        np.testing.assert_array_equal(action, self.data["action"][9:33])
        initial, _ = ds.state_action(0)
        np.testing.assert_array_equal(
            np.concatenate(list(initial.values()), axis=-1),
            self.data["action"][[0, 0, 0]],
        )
        self.data["state"][:, :29] += 0.2
        changed, _ = load().state_action(3)
        np.testing.assert_array_equal(np.concatenate(list(changed.values()), axis=-1), packed)

    def test_fk_ignores_joint_array_order_and_world_heading(self):
        rng = np.random.default_rng(3)
        q = rng.normal(0, 0.1, (8, 29))
        imu = np.tile([1.0, 0, 0, 0], (8, 1))
        offsets = np.zeros((3, 3))
        a = measured_chip_state(q, imu, self.names, offsets)
        permutation = rng.permutation(29)
        imu[:] = [np.cos(0.4), 0, 0, np.sin(0.4)]
        b = measured_chip_state(q[:, permutation], imu, np.array(self.names)[permutation], offsets)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-6)

    def test_canonical_actions_remove_only_fixed_initial_heading(self):
        angles = np.column_stack([np.linspace(-72, -64, 40), np.full(40, 8), np.full(40, -4)])
        source = Rotation.from_euler("ZYX", angles, degrees=True)
        self.data["action"][:, 12:16] = source.as_quat()[:, [3, 0, 1, 2]]
        original = self.data["action"].copy()
        canonical = canonicalize_replay_actions(original)
        result = Rotation.from_quat(canonical[:, [13, 14, 15, 12]])
        expected = angles.copy()
        expected[:, 0] += 72
        np.testing.assert_allclose(result.as_euler("ZYX", degrees=True), expected, atol=2e-5)
        np.testing.assert_array_equal(canonical[:, :12], original[:, :12])
        np.testing.assert_array_equal(canonical[:, 16:], original[:, 16:])
        np.testing.assert_array_equal(original, self.data["action"])
        # Changing only the source world's heading or quaternion signs must not
        # change the labels; rotations between frames must still be preserved.
        rotated = original.copy()
        rotated[:, 12:16] = (Rotation.from_euler("z", 120, degrees=True) * source).as_quat()[:, [3, 0, 1, 2]]
        rotated[::2, 12:16] *= -1
        np.testing.assert_allclose(canonicalize_replay_actions(rotated), canonical, atol=2e-6)
        np.testing.assert_allclose((result[:-1].inv() * result[1:]).as_matrix(),
                                   (source[:-1].inv() * source[1:]).as_matrix(), atol=2e-6)

    def test_yaw_dataset_uses_separate_state_and_action_initial_headings(self):
        state_yaw = np.linspace(35, 39, 40)
        action_yaw = np.linspace(-72, -64, 40)
        self.data["state"][:, 58:62] = Rotation.from_euler("z", state_yaw, degrees=True).as_quat()[:, [3, 0, 1, 2]]
        self.data["action"][:, 12:16] = Rotation.from_euler("z", action_yaw, degrees=True).as_quat()[:, [3, 0, 1, 2]]
        np.savez(self.path / "state_action_30hz.npz", **self.data)
        ds = ChipReplayDataset(self.root, chip_replay_modality_config(yaw_canonical=True),
                               val_ratio=0, yaw_canonical=True)
        state, action = ds.state_action(3)
        sy = Rotation.from_quat(state["anchor_quat_wxyz"][:, [1, 2, 3, 0]]).as_euler("ZYX", degrees=True)[:, 0]
        ay = Rotation.from_quat(action[:, [13, 14, 15, 12]]).as_euler("ZYX", degrees=True)[:, 0]
        np.testing.assert_allclose(sy, state_yaw[[3, 6, 9]] - 35, atol=1e-5)
        np.testing.assert_allclose(ay, action_yaw[9:33] + 72, atol=1e-5)
        self.assertEqual(ds.action_key, "reference_yaw_canonical")
        self.assertIn(ds.action_key, ds.get_step(3).actions)
        self.assertIn(ds.action_key, ds.get_dataset_statistics()["action"])


if __name__ == "__main__":
    unittest.main()
