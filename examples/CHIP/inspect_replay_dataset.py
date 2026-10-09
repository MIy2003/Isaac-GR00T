"""Verify replay source parity, episode split and normalization before training."""

import argparse
import json

from gr00t.configs.data.chip_replay_config import chip_replay_modality_config
from gr00t.data.chip_native import pack_chip_reference
from gr00t.data.chip_replay_fk import measured_chip_state
from gr00t.data.dataset.chip_replay_dataset import ChipReplayDataset
from gr00t.data.state_action.state_action_processor import StateActionProcessor
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--yaw-canonical", action="store_true")
    parser.add_argument("--ideal-state", action="store_true")
    args = parser.parse_args()
    config = chip_replay_modality_config(yaw_canonical=args.yaw_canonical)
    dataset = ChipReplayDataset(args.dataset_path, config, yaw_canonical=args.yaw_canonical, ideal_state=args.ideal_state)
    val = ChipReplayDataset(args.dataset_path, config, split="val", yaw_canonical=args.yaw_canonical, ideal_state=args.ideal_state)
    action_yaws = []
    assert set(dataset.episode_names).isdisjoint(val.episode_names)
    tag = dataset.embodiment_tag.value
    processor = StateActionProcessor(
        {tag: config},
        {tag: dataset.get_dataset_statistics()},
        use_percentiles=False,
        use_relative_action=False,
    )
    # Check every sample against the source arrays, plus decode RGB/normalize
    # at both ends of every episode. This catches accidental reuse of action as state.
    for ds in (dataset, val):
        for ep, name in enumerate(ds.episode_names):
            indices = [i for i, (episode, _) in enumerate(ds.samples) if episode == ep]
            with np.load(ds.root / name / "state_action_30hz.npz") as source:
                metadata = json.loads((ds.root / name / "chip_manifest.json").read_text())
                measured = pack_chip_reference(
                    measured_chip_state(
                        source["state"][:, :29],
                        source["state"][:, 58:62],
                        source["joint_names"],
                        metadata["point_offsets_m"],
                    )
                )
                expected = source["action"].copy()
                if args.yaw_canonical:
                    rotations = Rotation.from_quat(expected[:, [13, 14, 15, 12]])
                    initial_yaw = rotations[0].as_euler("ZYX")[0]
                    canonical = Rotation.from_euler("z", -initial_yaw) * rotations
                    quaternion = canonical.as_quat()[:, [3, 0, 1, 2]]
                    if quaternion[0, 0] < 0:
                        quaternion[0] *= -1
                    for j in range(1, len(quaternion)):
                        if np.dot(quaternion[j - 1], quaternion[j]) < 0:
                            quaternion[j] *= -1
                    expected[:, 12:16] = quaternion
                    assert abs(canonical[0].as_euler("ZYX")[0]) < 1e-6
                    if ds is dataset:
                        action_yaws.extend(np.rad2deg(canonical.as_euler("ZYX")[:, 0]).tolist())
                if args.ideal_state:
                    measured = expected[np.maximum(np.arange(len(expected)) - 1, 0)].copy()
                for index in indices:
                    _, t = ds.samples[index]
                    states, actions = ds.state_action(index)
                    np.testing.assert_allclose(
                        np.concatenate(list(states.values()), axis=-1),
                        measured[np.maximum(t + np.array([-6, -3, 0]), 0)],
                        atol=2e-6 if args.ideal_state else 0,
                        rtol=0,
                    )
                    np.testing.assert_allclose(actions, expected[t : t + 24], atol=2e-6)
                    np.testing.assert_array_equal(actions[:, :12], source["action"][t : t + 24, :12])
                    np.testing.assert_array_equal(actions[:, 16:], source["action"][t : t + 24, 16:])
                for index in indices[:1] + indices[-1:]:
                    step = ds.get_step(index)
                    normalized = processor.apply_action(step.actions, tag, step.states)
                    decoded = processor.unapply_action(normalized, tag, step.states)
                    np.testing.assert_allclose(
                        decoded[ds.action_key],
                        step.actions[ds.action_key],
                        atol=1e-5,
                    )
    print(
        json.dumps(
            {
                "passed": True,
                "train_episodes": len(dataset.episode_names),
                "val_episodes": len(val.episode_names),
                "train_samples": len(dataset.samples),
                "val_samples": len(val.samples),
                "state_shape": [3, 37],
                "state_source": "previous_action_30hz" if args.ideal_state else "measured_fk",
                "initial_history": "hold_action_0" if args.ideal_state else "hold_state_0",
                "action_shape": [24, 37],
                "action_representation": dataset.action_key,
                "action_yaw_degrees_min_p01_median_p99_max": (
                    np.percentile(action_yaws, [0, 1, 50, 99, 100]).tolist() if action_yaws else None
                ),
                "first_action_offset": 0,
                "all_samples_match_source": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
