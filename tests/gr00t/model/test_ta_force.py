"""Actual small GR00T DiT forward/backward and sampling for TA Force."""
import copy

import numpy as np
import pytest
import torch

from gr00t.data.dataset.ta_force_dataset import force_window
from gr00t.model.modules.ta_force import TAForceConfig
from tests.gr00t.model.test_training_rtc import head_class, inputs, modules, small_config  # noqa: F401


@pytest.mark.parametrize("alternate", [False, True])
def test_train_sample_and_reload(head_class, alternate):
    config = small_config(alternate=alternate, delay=0)
    config.ta_force = dict(history_samples=3)
    head = head_class(config)
    bi, ai = inputs(batch=2)
    ai.update(force=torch.randn(2, 3, 3), force_valid=torch.ones(2, 3, dtype=torch.bool),
              future_force=torch.randn(2, 40, 3), future_force_valid=torch.ones(2, 40, dtype=torch.bool))
    result = head(copy.deepcopy(bi), copy.deepcopy(ai))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    branch = head.ta_force
    assert branch.future_decoder.weight.grad.abs().sum() > 0
    assert branch.projector[0].weight.grad.abs().sum() > 0
    head.eval()
    ai.pop("action")
    ai.pop("future_force")
    ai.pop("future_force_valid")  # Inference never requires future ground truth.
    torch.manual_seed(7)
    output = head.get_action(copy.deepcopy(bi), copy.deepcopy(ai))
    assert output.action_pred.shape == (2, 40, 40)
    assert output.force_pred.shape == (2, 40, 3)
    restored = head_class(config).eval()
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.manual_seed(7)
    second = restored.get_action(copy.deepcopy(bi), copy.deepcopy(ai))
    torch.testing.assert_close(output.action_pred, second.action_pred)


def test_causal_window_and_future_mask():
    times = np.array([10, 20, 30, 40])
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    result = force_window(times, values, 20, 3, np.array([30, 40, 50]), 100)
    np.testing.assert_array_equal(result["force_valid"], [False, True, True])
    np.testing.assert_array_equal(result["force"][-1], values[1])
    np.testing.assert_array_equal(result["future_force_valid"], [True, True, False])
    changed = values.copy()
    changed[2:] = 999
    other = force_window(times, changed, 20, 3, np.array([30, 40, 50]), 100)
    np.testing.assert_array_equal(result["force"], other["force"])


def test_invalid_and_masked_targets(head_class):
    config = small_config(delay=0)
    config.ta_force = dict(mode="EXPERT_HIS_C_FUT", history_samples=3)
    head = head_class(config)
    bi, ai = inputs(batch=2)
    ai.update(force=torch.zeros(2, 3, 3), force_valid=torch.zeros(2, 3, dtype=torch.bool),
              future_force=torch.full((2, 40, 3), float("nan")),
              future_force_valid=torch.zeros(2, 40, dtype=torch.bool))
    result = head(bi, ai)
    assert torch.isfinite(result["loss"])
    assert result["force_loss"] == 0
    ai.force_valid[:] = True
    ai.force[:] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        head.ta_force.tokens(ai)


def test_incompatible_config(head_class):
    config = small_config(delay=0)
    config.ta_force = {}
    config.force_mot = {}
    with pytest.raises(ValueError, match="without Force MoT"):
        head_class(config)
    with pytest.raises(ValueError, match="only supports EXPERT_HIS_C_FUT"):
        TAForceConfig(mode="LLM")


@pytest.mark.parametrize("mode", ["EXPERT", "EXPERT_HIS_C", "EXPERT_HIS_T", "EXPERT_FUT", "EXPERT_HIS_C_L_FUT"])
def test_reject_removed_modes(mode):
    with pytest.raises(ValueError, match="only supports EXPERT_HIS_C_FUT"):
        TAForceConfig(mode=mode)


def test_torque_is_not_read_or_used(tmp_path):
    import json
    from gr00t.data.dataset.ta_force_dataset import read_force

    raw = tmp_path / "source_raw"
    raw.mkdir()
    csv = raw / "ft_data.csv"
    csv.write_text("frame_idx,Fx(N),Fy(N),Fz(N),Tx(Nm),Ty(Nm),Tz(Nm)\n0,1,2,3,nan,inf,garbage\n")
    record = dict(frame_idx=0, server_time_ns=10, status={"ft": "OK"},
                  sync_diagnostics={"ft": {"detail": "sample0", "arrival_server_ns": 9}})
    (raw / "sync_frames.jsonl").write_text(json.dumps(record))
    times, values = read_force(tmp_path)
    np.testing.assert_array_equal(times, [10])
    np.testing.assert_array_equal(values, [[1, 2, 3]])
    csv.write_text("frame_idx,Fx(N),Fy(N),Fz(N)\n0,1,2,3\n")
    np.testing.assert_array_equal(read_force(tmp_path)[1], values)


@pytest.mark.parametrize('delay', [0, 6])
@pytest.mark.parametrize('alternate', [False, True])
def test_rtc_train_and_sample(head_class, delay, alternate):
    from unittest.mock import patch
    config = small_config(alternate=alternate, delay=6)
    config.ta_force = dict(history_samples=3)
    head = head_class(config)
    bi, ai = inputs(batch=2)
    ai.update(force=torch.randn(2, 3, 3), force_valid=torch.ones(2, 3, dtype=torch.bool),
              future_force=torch.randn(2, 40, 3), future_force_valid=torch.ones(2, 40, dtype=torch.bool))
    prefix_mask = torch.zeros(2, 40, dtype=torch.bool)
    prefix_mask[:, :delay] = True
    with patch('gr00t.model.modules.rtc.sample_prefix_mask', return_value=prefix_mask):
        torch.manual_seed(22)
        first = head(copy.deepcopy(bi), copy.deepcopy(ai))
        changed = copy.deepcopy(ai)
        changed.future_force[:, :delay] += 1000
        torch.manual_seed(22)
        second = head(copy.deepcopy(bi), changed)
    # Future labels on committed actions cannot enter the network or either loss.
    torch.testing.assert_close(first['loss'], second['loss'])
    first['loss'].backward()
    assert head.ta_force.projector[0].weight.grad.abs().sum() > 0
    inference = copy.deepcopy(ai)
    inference.pop('future_force')
    inference.pop('future_force_valid')
    head.eval()
    result = head.get_action(copy.deepcopy(bi), inference,
                             options=dict(rtc_mode='trained', rtc_prefix_length=delay))
    torch.testing.assert_close(result.action_pred[:, :delay], ai.action[:, :delay])
    assert not result.force_pred_valid[:, :delay].any()
    assert result.force_pred_valid[:, delay:].all()
    with pytest.raises(ValueError, match='exceeds'):
        head.get_action(copy.deepcopy(bi), inference,
                        options=dict(rtc_mode='trained', rtc_prefix_length=7))


def test_sharded_adapter_preserves_base_data():
    import csv
    import json
    from tests.gr00t.data.test_chip_replay import ReplayContractTest
    from gr00t.data.dataset.chip_replay_dataset import ChipReplayDataset
    from gr00t.data.dataset.ta_force_dataset import TAForceShardedReplayDataset
    from gr00t.configs.data.chip_replay_config import chip_replay_modality_config
    fixture = ReplayContractTest()
    fixture.setUp()
    try:
        for s in (12, 25, 29, 33):
            fixture.data['action'][:, s:s+4] = [1, 0, 0, 0]
        fixture.data['source_target_time_ns'] = np.arange(40, dtype=np.int64) * 33333333
        fixture.dataset()
        path = fixture.path
        (path/'merged_manifest.json').write_text(json.dumps(dict(modality_origin={'state':'real replay LowState, pre-action'})))
        raw = path/'source_raw'
        raw.mkdir()
        with (raw/'ft_data.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx','Fx(N)','Fy(N)','Fz(N)'])
            writer.writerows([[i, i, 2, 3] for i in range(40)])
        records = [dict(frame_idx=i, server_time_ns=int(t), status={'ft':'OK'},
                        sync_diagnostics={'ft':{'detail':str(i),'arrival_server_ns':int(t)}})
                   for i,t in enumerate(fixture.data['source_target_time_ns'])]
        (raw/'sync_frames.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
        args = (fixture.root, chip_replay_modality_config(yaw_canonical=True))
        kwargs = dict(yaw_canonical=True, val_ratio=0, task_description='Pivot the box.')
        base = ChipReplayDataset(*args, **kwargs)
        force = TAForceShardedReplayDataset(*args, **kwargs)
        assert force.samples == base.samples
        for a,b in zip(base.shards, force.shards):
            np.testing.assert_array_equal(a,b)
        assert force.get_dataset_statistics() == base.get_dataset_statistics()
        for i in range(len(base.samples)):
            bs, ba = base.state_action(i)
            fs, fa = force.state_action(i)
            for key in bs:
                np.testing.assert_array_equal(bs[key], fs[key])
            np.testing.assert_array_equal(ba, fa)
        assert force.get_step(0).text == 'Pivot the box.'
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize('task', ['dough', 'pivot'])
def test_base_config_only_adds_force(task, tmp_path):
    from pathlib import Path
    import importlib.util
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('ta_force_entry', root / 'examples/CHIP/train_ta_force.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare_config = module.prepare_config
    cfg, diff = prepare_config(root / f'examples/CHIP/assets/ta_force_xyz_rtc_base/{task}_base_config.yaml', tmp_path)
    cfg.validate()
    assert set(diff) == {'model.ta_force', 'training.output_dir'}
    assert cfg.training.global_batch_size == 160
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.lr_scheduler_type == 'cosine'
    assert cfg.training.warmup_ratio == 0.05
    assert cfg.model.rtc_training_max_delay == 6
    assert 'models/gr00t' in cfg.training.start_from_checkpoint
