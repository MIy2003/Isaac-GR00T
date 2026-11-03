"""Independent TA Force adapter for CHIP replay, with explicit source-time pairing.

Replay state and original force are NOT simultaneous physical measurements.
No Force MoT dataset, contexts, normalization or scheduler is used here.
"""
import csv
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from gr00t.configs.data.chip_replay_config import chip_replay_modality_config
from gr00t.data.dataset.chip_replay_dataset import ChipReplayDataset
from gr00t.data.types import MessageType


CHANNELS = ("Fx(N)", "Fy(N)", "Fz(N)")
PAIRING = "replay_state_and_original_wrench_by_source_time_NOT_simultaneous"


def read_force(episode):
    path = Path(episode) / "source_raw"
    with (path / "ft_data.csv").open() as stream:
        rows = {int(r["frame_idx"]): r for r in csv.DictReader(stream)}
    times, values, seen = [], [], set()
    for line in (path / "sync_frames.jsonl").read_text().splitlines():
        row = json.loads(line)
        ft = row["sync_diagnostics"].get("ft", {})
        identity, arrival = ft.get("detail"), ft.get("arrival_server_ns")
        idx = int(row["frame_idx"])
        if row.get("status", {}).get("ft") != "OK" or not identity or identity in seen or arrival is None or idx not in rows:
            continue
        value = np.array([float(rows[idx][c]) for c in CHANNELS], dtype=np.float32)
        if not np.isfinite(value).all():
            continue
        seen.add(identity)
        times.append(max(int(row["server_time_ns"]), int(arrival)))
        values.append(value)
    if not values:
        raise ValueError(f"No valid force measurements in {path}")
    order = np.argsort(times, kind="stable")
    return np.asarray(times, dtype=np.int64)[order], np.asarray(values)[order]


def force_window(times, values, query, history, future_times, max_age_ns):
    """Causal history; future supervision at t+1..t+H, masked outside coverage."""
    end = int(np.searchsorted(times, query, side="right"))
    start = max(0, end - history)
    n = end - start
    force = np.zeros((history, values.shape[-1]), dtype=np.float32)
    valid = np.zeros(history, dtype=bool)
    if n:
        force[-n:] = values[start:end]
        valid[-n:] = query - times[start:end] <= max_age_ns
        force[~valid] = 0
    ids = np.searchsorted(times, future_times, side="right") - 1
    safe = np.maximum(ids, 0)
    fv = ((ids >= 0) & (future_times <= times[-1])
          & (future_times - times[safe] <= max_age_ns))
    future = np.where(fv[:, None], values[safe], 0).astype(np.float32)
    return dict(force=force, force_valid=valid, future_force=future, future_force_valid=fv)


class TAForceReplayDataset(Dataset):
    def __init__(self, root, processor, *, split="train", seed=42, history=10, max_age_ms=500, task_description="Wipe the car."):
        if history < 1 or max_age_ms <= 0:
            raise ValueError("Invalid force history/maximum age")
        self.processor, self.history = processor, history
        self.max_age_ns = int(max_age_ms * 1e6)
        self.replay = ChipReplayDataset(root, chip_replay_modality_config(yaw_canonical=True),
                                       split=split, seed=seed, yaw_canonical=True, ideal_state=False, action_stride=1,
                                       task_description=task_description)
        if processor.get_modality_configs()["new_embodiment"] != chip_replay_modality_config(yaw_canonical=True):
            raise ValueError("TA Force replay requires the canonical 30 Hz CHIP processor")
        self.episode_names = self.replay.episode_names
        self.force, self.timeline = [], []
        for name in self.episode_names:
            path = Path(root) / name
            self.force.append(read_force(path))
            with np.load(path / "state_action_30hz.npz", allow_pickle=False) as data:
                timeline = data["source_target_time_ns"].copy()
            if np.any(np.diff(timeline) <= 0):
                raise ValueError("Source timeline must be increasing")
            self.timeline.append(timeline)

    def __len__(self):
        return len(self.replay.samples)

    def force_statistics(self):
        values = np.concatenate([v for _, v in self.force]).astype(np.float64)
        return dict(mean=values.mean(0).tolist(), std=values.std(0).clip(1e-6).tolist(),
                    count=len(values), episodes=self.episode_names)

    def __getitem__(self, index):
        ep, start = self.replay.samples[index]
        step = self.replay.get_step(index)
        processed = self.processor([dict(type=MessageType.EPISODE_STEP.value, content=step)])
        timeline = self.timeline[ep]
        horizon = processed["action"].shape[0]
        # Targets start one step after the current observation, as in TA-VLA.
        ids = np.arange(start + 1, start + 1 + horizon)
        future_times = timeline[np.minimum(ids, len(timeline) - 1)]
        result = force_window(*self.force[ep], timeline[start], self.history,
                              future_times, self.max_age_ns)
        result["future_force_valid"] &= (ids < len(timeline)) & np.asarray(processed["action_mask"]).astype(bool).any(-1)
        result["future_force"][~result["future_force_valid"]] = 0
        processed.update(result)
        return processed


class TAForceShardedReplayDataset(ChipReplayDataset):
    """Base replay sampling/normalization unchanged; append only XYZ force fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ideal_state or not self.yaw_canonical or self.action_stride != 1:
            raise ValueError("TA Force requires measured 30 Hz replay state")
        self.history = 10
        self.max_age_ns = 500_000_000
        self.force, self.timeline = [], []
        for name in self.episode_names:
            path = self.root / name
            meta = json.loads((path / 'merged_manifest.json').read_text())
            if meta['modality_origin']['state'] != 'real replay LowState, pre-action':
                raise ValueError('Expected actual replay measured state')
            self.force.append(read_force(path))
            with np.load(path / 'state_action_30hz.npz', allow_pickle=False) as data:
                self.timeline.append(data['source_target_time_ns'].copy())
            if np.any(np.diff(self.timeline[-1]) <= 0):
                raise ValueError('Source timeline must be increasing')

    def force_statistics(self):
        return TAForceReplayDataset.force_statistics(self)

    def get_shard(self, idx):
        samples = super().get_shard(idx)
        for sample, index in zip(samples, self.shards[idx]):
            ep, start = self.samples[int(index)]
            timeline = self.timeline[ep]
            ids = np.arange(start + 1, start + 1 + sample['action'].shape[0])
            force = force_window(*self.force[ep], timeline[start], self.history,
                                 timeline[np.minimum(ids, len(timeline) - 1)], self.max_age_ns)
            force['future_force_valid'] &= (ids < len(timeline)) & np.asarray(sample['action_mask']).astype(bool).any(-1)
            force['future_force'][~force['future_force_valid']] = 0
            sample.update(force)
        return samples
