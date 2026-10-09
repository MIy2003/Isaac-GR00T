"""Load independent measured states and source actions from merged replay NPZs."""

import json
from pathlib import Path

import numpy as np

from gr00t.configs.data.chip_native_config import CHIP_ACTION_HORIZON
from gr00t.configs.data.chip_replay_config import REPLAY_SCHEMA, chip_replay_modality_config
from gr00t.data.chip_native import OBS_FIELD_DIMS, REFERENCE_FIELDS
from gr00t.data.chip_replay_fk import measured_chip_state
from gr00t.data.dataset.chip_native_dataset import ChipNativeDataset
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag


class ChipReplayDataset(ChipNativeDataset):
    state_keys = REFERENCE_FIELDS
    action_key = "reference_absolute"

    def __init__(
        self,
        dataset_path,
        modality_configs,
        *,
        shard_size=128,
        seed=42,
        task_description="Wipe the car.",
        val_ratio=0.05,
        split="train",
        yaw_canonical=False,
        ideal_state=False,
    ):
        ShardedDataset.__init__(self, dataset_path)
        self.yaw_canonical = yaw_canonical
        self.ideal_state = ideal_state
        if ideal_state and not yaw_canonical:
            raise ValueError("Ideal action-derived state requires yaw canonical actions")
        self.action_key = "reference_yaw_canonical" if yaw_canonical else "reference_absolute"
        if modality_configs != chip_replay_modality_config(yaw_canonical=yaw_canonical):
            raise ValueError("chip_replay requires its own modality config")
        if shard_size < 1 or not 0 <= val_ratio < 1 or split not in {"train", "val"}:
            raise ValueError("Invalid shard_size, val_ratio or split")
        if not task_description.strip():
            raise ValueError("task_description must be nonempty")
        self.root = Path(dataset_path).expanduser().resolve()
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.task_description = task_description
        self.processor = None
        manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        if manifest.get("schema_version") != REPLAY_SCHEMA or manifest.get("nominal_hz") != 30:
            raise ValueError(f"Expected {REPLAY_SCHEMA} at 30 Hz")
        if (manifest.get("state_dim"), manifest.get("action_dim")) != (65, 37):
            raise ValueError("Expected independent 65-D state and 37-D action")
        names = manifest.get("episode_names", [])
        if not names or len(set(names)) != len(names):
            raise ValueError("Manifest must contain unique episode_names")
        n_val = min(max(1, round(len(names) * val_ratio)), len(names) - 1) if val_ratio else 0
        val_ids = set(np.random.default_rng(seed).choice(len(names), n_val, replace=False))
        self.episodes, self.samples, self.episode_names = [], [], []
        total_frames = 0
        for ep_id, name in enumerate(names):
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Episode outside dataset root: {name}")
            with np.load(path / "state_action_30hz.npz", allow_pickle=False) as data:
                state = np.asarray(data["state"], dtype=np.float32)
                action = np.asarray(data["action"], dtype=np.float32)
                joint_names = data["joint_names"].copy()
                length = len(state)
                if state.shape != (length, 65) or action.shape != (length, 37) or not length:
                    raise ValueError(f"{name}: invalid state/action shape")
                valid = np.ones(length, dtype=bool)
                for key in ("valid", "state_valid", "action_valid", "state_action_timing_valid"):
                    mask = data[key]
                    if mask.shape != (length,) or mask.dtype != np.bool_:
                        raise ValueError(f"{name}: invalid {key}")
                    valid &= mask
                valid &= np.isfinite(state).all(axis=1) & np.isfinite(action).all(axis=1)
                if not np.array_equal(data["image_index"], np.arange(length)):
                    raise ValueError(f"{name}: image row alignment mismatch")
                image_paths = [(path / str(p)).resolve() for p in data["image_file"]]
                if len(image_paths) != length or any(
                    not p.is_relative_to(path) for p in image_paths
                ):
                    raise ValueError(f"{name}: invalid image paths")
            total_frames += length
            if (ep_id in val_ids) != (split == "val"):
                continue
            for image in image_paths:
                if not image.is_file():
                    raise FileNotFoundError(image)
            metadata = json.loads((path / "chip_manifest.json").read_text())
            if metadata["point_bodies"] != [
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "torso_link",
            ]:
                raise ValueError(f"{name}: unsupported point bodies")
            # Missing/invalid state rows are excluded from windows. Use a valid
            # row only to make FK evaluable there; it never becomes a sample.
            finite = np.isfinite(state).all(axis=1) & (
                np.linalg.norm(state[:, 58:62], axis=1) > 1e-7
            )
            valid &= finite
            if not finite.any():
                continue
            fk_state = state.copy()
            fk_state[~finite] = fk_state[np.flatnonzero(finite)[0]]
            fields = measured_chip_state(
                fk_state[:, :29], fk_state[:, 58:62], joint_names, metadata["point_offsets_m"]
            )
            if yaw_canonical:
                from gr00t.data.chip_replay_yaw import canonicalize_replay_actions

                # A missing source first heading cannot define this episode's frame.
                if not np.isfinite(action[0]).all() or np.linalg.norm(action[0, 12:16]) < 1e-7:
                    raise ValueError(f"{name}: invalid initial reference heading")
                finite_action = np.isfinite(action).all(axis=1) & (
                    np.linalg.norm(action[:, 12:16], axis=1) > 1e-7
                )
                valid &= finite_action
                action[~finite_action] = action[0]
                action = canonicalize_replay_actions(action)
            if ideal_state:
                from gr00t.data.chip_native import unpack_chip_reference

                # Perfect tracking of the PREVIOUS 30-Hz action. Initial history
                # is held at action[0]; never shift across episode boundaries.
                previous = np.maximum(np.arange(length) - 1, 0)
                fields = unpack_chip_reference(action[previous].copy())
                valid &= valid[previous]
            index = len(self.episodes)
            self.episodes.append(({"state": fields, "action": action}, image_paths))
            self.episode_names.append(name)
            for t in range(0, length - CHIP_ACTION_HORIZON + 1, 3):
                # Reject windows containing invalid rows, including history intervals.
                if valid[max(0, t - 6) : t + CHIP_ACTION_HORIZON].all():
                    self.samples.append((index, t))
        if total_frames != manifest.get("source_image_frames"):
            raise ValueError("Manifest source_image_frames does not match NPZ lengths")
        if not self.samples:
            raise ValueError(f"No valid {split} action windows")
        order = np.random.default_rng(seed).permutation(len(self.samples))
        self.shards = [order[i : i + shard_size] for i in range(0, len(order), shard_size)]
        self._statistics = None

    def state_action(self, index):
        episode, t = self.samples[index]
        arrays, _ = self.episodes[episode]
        history = np.maximum(t + np.array([-6, -3, 0]), 0)
        states = {
            key: arrays["state"][key][history].reshape(3, OBS_FIELD_DIMS[key])
            for key in self.state_keys
        }
        return states, arrays["action"][t : t + CHIP_ACTION_HORIZON].copy()
