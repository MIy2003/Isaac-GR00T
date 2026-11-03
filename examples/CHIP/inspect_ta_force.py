"""Validate task text, XYZ force coverage and processor shapes before TA Force training."""
import argparse
import json
from pathlib import Path
import hashlib

import numpy as np
from transformers import AutoProcessor
import gr00t.model  # noqa: F401
from gr00t.data.dataset.ta_force_dataset import TAForceReplayDataset, force_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--base-model-path', required=True)
    parser.add_argument('--task-description', required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.base_model_path, local_files_only=True)
    processor.eval()
    dataset = TAForceReplayDataset(args.dataset_path, processor, task_description=args.task_description)
    from gr00t.data.chip_replay_fk import measured_chip_state
    from gr00t.data.chip_native import OBS_FIELD_DIMS
    config = json.loads((Path(args.base_model_path) / 'config.json').read_text())
    assert config['state_history_length'] == 3
    assert not dataset.replay.ideal_state
    audited_rows = 0
    for ep, name in enumerate(dataset.episode_names):
        path = Path(args.dataset_path) / name
        merged = json.loads((path / 'merged_manifest.json').read_text())
        assert merged['modality_origin']['state'] == 'real replay LowState, pre-action'
        geometry = json.loads((path / 'chip_manifest.json').read_text())
        with np.load(path / 'state_action_30hz.npz', allow_pickle=False) as raw:
            expected = measured_chip_state(raw['state'][:, :29], raw['state'][:, 58:62],
                                           raw['joint_names'], geometry['point_offsets_m'])
        stored = dataset.replay.episodes[ep][0]['state']
        for key in expected:
            np.testing.assert_allclose(stored[key], expected[key], rtol=0, atol=0)
        # Check every sampled history against independently recomputed measured FK.
        for index, (episode, start) in enumerate(dataset.replay.samples):
            if episode != ep:
                continue
            state, _ = dataset.replay.state_action(index)
            ids = np.maximum(start + np.array([-6, -3, 0]), 0)
            for key in state:
                np.testing.assert_allclose(state[key], expected[key][ids].reshape(3, OBS_FIELD_DIMS[key]), rtol=0, atol=0)
            audited_rows += 1
    stats = dataset.force_statistics()
    assert len(stats['mean']) == len(stats['std']) == 3
    first = dataset[0]
    assert first['force'].shape == (10, 3)
    assert first['future_force'].shape[-1] == 3
    assert dataset.replay.get_step(0).text == args.task_description
    batch = processor.collator([first])['inputs']
    counts = dict(history_valid=0, history_total=0, future_valid=0, future_total=0)
    for ep, start in dataset.replay.samples:
        timeline = dataset.timeline[ep]
        ids = np.arange(start + 1, min(start + 25, len(timeline)))
        window = force_window(*dataset.force[ep], timeline[start], 10, timeline[ids], dataset.max_age_ns)
        counts['history_valid'] += int(window['force_valid'].sum())
        counts['history_total'] += 10
        counts['future_valid'] += int(window['future_force_valid'].sum())
        counts['future_total'] += 24
    if not counts['history_valid'] or not counts['future_valid']:
        raise ValueError('No usable historical/future force supervision')
    print(json.dumps(dict(task=args.task_description, samples=len(dataset), statistics=stats,
                         coverage=counts, state_audit=dict(source='real replay LowState, pre-action',
                         representation='measured FK 37D, own initial IMU yaw removed',
                         ideal_state=False, history_offsets=[-6,-3,0], audited_samples=audited_rows,
                         checkpoint_state_history_length=config['state_history_length'],
                         statistics_sha256=hashlib.sha256((Path(args.base_model_path)/'statistics.json').read_bytes()).hexdigest()), shapes={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}), indent=2))


if __name__ == '__main__':
    main()
