"""Read existing CHIP NPZ/JPEG episodes without LeRobot conversion or DP dependencies."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gr00t.configs.data.chip_native_config import CHIP_ACTION_HORIZON, chip_native_modality_config
from gr00t.data.chip_native import (
    CHIP_POINT_NAMES,
    CHIP_SCHEMA,
    FIELD_SHAPES,
    OBS_FIELD_DIMS,
    REFERENCE_FIELDS,
    canonicalize_anchor_by_episode_yaw,
    pack_chip_reference,
    reference_relative_to_current,
)
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, VLAStepData


class ChipNativeDataset(ShardedDataset):
    state_keys = REFERENCE_FIELDS
    action_key = "native_relative"

    def __init__(
        self,
        dataset_path,
        modality_configs,
        *,
        shard_size=128,
        seed=42,
        task_description="Perform the demonstrated task.",
        val_ratio=0.05,
        split="train",
    ):
        super().__init__(dataset_path)
        if modality_configs != chip_native_modality_config():
            raise ValueError("chip_native requires the fixed CHIP modality config")
        if shard_size < 1 or not 0 <= val_ratio < 1 or split not in {"train", "val"}:
            raise ValueError("Invalid shard_size, val_ratio or split")
        if not task_description.strip():
            raise ValueError("task_description must be nonempty")
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.task_description = task_description
        self.processor = None
        self.root = Path(dataset_path).expanduser().resolve()
        manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        if manifest.get("schema_version") != CHIP_SCHEMA or manifest.get("source_hz") != 30:
            raise ValueError(f"Expected {CHIP_SCHEMA} at 30 Hz")
        names = manifest.get("episode_names", [])
        if not names or len(set(names)) != len(names):
            raise ValueError("Manifest must contain unique episode_names")
        # Same episode split as diffusion_policy.common.sampler.get_val_mask.
        n_val = min(max(1, round(len(names) * val_ratio)), len(names) - 1) if val_ratio else 0
        val_ids = set(np.random.default_rng(seed).choice(len(names), n_val, replace=False))
        self.episodes = []
        self.samples = []
        self.episode_names = []
        total_frames = 0
        for ep_id, name in enumerate(names):
            episode_path = (self.root / name).resolve()
            if not episode_path.is_relative_to(self.root):
                raise ValueError(f"Episode outside dataset root: {name}")
            with np.load(episode_path / "chip_reference_30hz.npz", allow_pickle=False) as data:
                fields = {k: np.asarray(data[k], dtype=np.float32) for k in REFERENCE_FIELDS}
                length = len(fields["lower_joint_pos"])
                if length == 0:
                    raise ValueError(f"Empty episode: {name}")
                if not np.array_equal(data["vr_3point_names"], CHIP_POINT_NAMES):
                    raise ValueError(f"{name}: expected point order {CHIP_POINT_NAMES}")
                for key, value in fields.items():
                    if value.shape != (length, *FIELD_SHAPES[key]) or not np.isfinite(value).all():
                        raise ValueError(f"{name}: invalid {key} shape or nonfinite values")
                    if "quat" in key and np.max(abs(np.linalg.norm(value, axis=-1) - 1)) > 1e-4:
                        raise ValueError(f"{name}: {key} must contain unit quaternions")
            total_frames += length
            if (ep_id in val_ids) != (split == "val"):
                continue
            image_paths = [episode_path / "images" / f"frame_{i:06d}.jpg" for i in range(length)]
            for path in image_paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            fields = canonicalize_anchor_by_episode_yaw(fields, fields["anchor_quat_wxyz"][0])
            index = len(self.episodes)
            self.episodes.append((fields, image_paths))
            self.episode_names.append(name)
            # Pad history at episode start, never pad future targets or cross episodes.
            self.samples.extend((index, t) for t in range(0, length - CHIP_ACTION_HORIZON, 3))
        if total_frames != manifest.get("source_frames"):
            raise ValueError("Manifest source_frames does not match NPZ lengths")
        if not self.samples:
            raise ValueError(f"No valid {split} samples with {CHIP_ACTION_HORIZON} future frames")
        order = np.random.default_rng(seed).permutation(len(self.samples))
        self.shards = [order[i : i + shard_size] for i in range(0, len(order), shard_size)]
        self._statistics = None

    def state_action(self, index):
        episode, t = self.samples[index]
        fields, _ = self.episodes[episode]
        history = np.maximum(t + np.array([-6, -3, 0]), 0)
        states = {k: v[history].reshape(3, OBS_FIELD_DIMS[k]) for k, v in fields.items()}
        future = {k: v[t + 1 : t + CHIP_ACTION_HORIZON + 1] for k, v in fields.items()}
        current = {k: v[t] for k, v in fields.items()}
        action = pack_chip_reference(reference_relative_to_current(future, current))
        return states, action

    def get_step(self, index):
        states, action = self.state_action(index)
        episode, t = self.samples[index]
        _, paths = self.episodes[episode]
        images = []
        for i in [max(0, t - 6), t]:
            with Image.open(paths[i]) as image:
                if image.size != (400, 224):
                    raise ValueError(f"{paths[i]}: expected 400x224 RGB source image")
                images.append(np.asarray(image.convert("RGB")))
        return VLAStepData(
            images={"camera0_rgb": images},
            states=states,
            actions={self.action_key: action},
            text=self.task_description,
            embodiment=self.embodiment_tag,
        )

    def __len__(self):
        return len(self.shards)

    def get_shard_length(self, idx):
        return len(self.shards[idx])

    def get_shard(self, idx):
        if self.processor is None:
            raise RuntimeError("Set processor before loading shards")
        return [
            self.processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": self.get_step(int(i))}]
            )
            for i in self.shards[idx]
        ]

    def get_dataset_statistics(self):
        if self._statistics is None:
            # Compute on train samples only, after canonicalization/relative encoding.
            # Use GR00T's per-dimension normalization (shared across timesteps).
            state_parts = {k: [] for k in self.state_keys}
            actions = []
            for i in range(len(self.samples)):
                states, action = self.state_action(i)
                for key in self.state_keys:
                    state_parts[key].append(states[key])
                actions.append(action)

            def stats(parts):
                value = np.concatenate(parts).astype(np.float64)
                return {
                    k: v.tolist()
                    for k, v in {
                        "min": value.min(0),
                        "max": value.max(0),
                        "mean": value.mean(0),
                        "std": value.std(0),
                        "q01": np.quantile(value, 0.01, axis=0),
                        "q99": np.quantile(value, 0.99, axis=0),
                    }.items()
                }

            self._statistics = {
                "state": {k: stats(v) for k, v in state_parts.items()},
                "action": {self.action_key: stats(actions)},
            }
        return self._statistics
