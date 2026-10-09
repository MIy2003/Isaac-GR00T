"""Generate measured MuJoCo states using Robofarm's real 862 BatchPolicy.

No DDS or robot controller is initialized. Original RGB/action/image-to-50Hz-row
mapping is retained. Each episode uses an independent nominal cold start.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(name, source_root, output_root, deploy_root, config_path):
    source_root, output_root, deploy_root = map(Path, (source_root, output_root, deploy_root))
    sys.path.insert(0, str(deploy_root))
    from tools.batch_chip_real_replay import load_tracks, make_mock
    from tools.probe_batch_replay_startup import ProbePolicy
    from tools import replay_mimic_6d_aligned_sim as sim

    cfg = yaml.safe_load(Path(config_path).read_text())
    bundle = Path(cfg['onnx_model_path']).parent
    contract = json.loads((bundle / 'contract.json').read_text())
    expected_hash = 'fd0a332c99f3e44143ab38b9b00b8849db14fd996a3936a4ec96045fd04a36e4'
    assert contract['checkpoint_sha256'] == expected_hash
    expected_compliance = np.zeros((3, 6))
    expected_compliance[1, :3] = .005
    np.testing.assert_allclose(cfg['compliance_6d'], expected_compliance, atol=1e-9)
    assert cfg['max_command_speed_rad_s'] == 8 and cfg['full_body_slew']
    assert cfg['startup_transition_s'] == 0
    source_episode = source_root / name
    original_meta = json.loads((source_episode / 'merged_manifest.json').read_text())
    reference_root = Path(original_meta['source_chip_episode']).parent
    source_contract = json.loads((deploy_root.parent / 'HIL_Dagger/models/mimic_lite_chip_6d_xyz_rot_scalar_977_24500/contract.json').read_text())
    tracks = load_tracks(reference_root, contract, 'chip_motion_50hz.npz',
                         episode_names=[name], reference_contract=source_contract)
    assert len(tracks) == 1
    captured = []

    class CapturePolicy(ProbePolicy):
        def compute_action(self):
            target = super().compute_action()
            if self.flow.engine.action_tick:
                c = self.controller
                captured.append(dict(
                    row=int(self.flow.engine.row_index), tick=int(c.policy_step),
                    state=np.concatenate([c.qj_real, c.dqj_real, c.quat, c.gyro]).astype(np.float32),
                    actor_command=self.policy_input['command'][0].copy(),
                    applied_reference_action=np.concatenate([self.lower_pos, self.anchor_quat_wxyz,
                        self.point_pos_anchor.ravel(), self.point_quat_anchor_wxyz.ravel()]).astype(np.float32),
                    target_q_real=(target + c.default_qpos_real).astype(np.float32),
                ))
            return target

    CapturePolicy.tracks = tracks
    CapturePolicy.sink = None
    controller_config = deploy_root / 'init_pose_bundle/controller_variants/controller_v9_symmetric_v0_mild_wrist_lower_10cm.yaml'

    def prepare(args, episode, fraction):
        c = make_mock(controller_config)
        policy = CapturePolicy('sim-state', NS(**cfg), c)
        c.current_policy = policy
        np.testing.assert_allclose(policy.wrist_6d.physical, expected_compliance, atol=1e-9)
        return policy, c

    sim.prepare = prepare
    dest = output_root / name
    dest.mkdir(exist_ok=False)
    with (dest / 'simulation.log').open('w') as log, redirect_stdout(log):
        result = sim.simulate(NS(bundle=bundle, output=dest, viewer=False),
                              reference_root / name, .25)
    if result['status'] != 'complete':
        raise RuntimeError(f'{name}: {result}')
    indices = np.array([r['row'] for r in captured])
    np.testing.assert_array_equal(indices, np.arange(tracks[0]['source_rows']))
    values = {k: np.asarray([r[k] for r in captured]) for k in captured[0]}
    assert np.isfinite(values['state']).all()
    # Independent check against the simulator's pre-step MuJoCo record.
    with np.load(Path(result['output']) / 'samples.npz') as samples:
        active = samples['action_tick']
        np.testing.assert_array_equal(samples['q'][active].astype(np.float32), values['state'][:, :29])
        np.testing.assert_allclose(samples['command'][active], values['actor_command'], atol=1e-7)
        np.testing.assert_allclose(samples['sent'][active], values['target_q_real'], atol=1e-6)
    np.savez_compressed(dest / 'sim_state_action_50hz.npz', **values,
                        source_action=tracks[0]['source_action'])
    with np.load(source_episode / 'state_action_30hz.npz') as src:
        # Explicit selection excludes real-world replay timestamps, torques and
        # next-control-state aliases, which would be stale in a sim dataset.
        keep = ['fps', 'action', 'action_valid', 'valid', 'state_definition', 'action_definition',
                'joint_names', 'image_index', 'image_file', 'source_image_file', 'source_target_time_ns',
                'source_reference_row_index', 'reference_minus_image_target_ms', 'reference_endpoint_clamped',
                'image_age_ms', 'image_is_duplicate', 'image_recorded_valid', 'gopro_status',
                'source_reference_action_30hz', 'source_measured_q_real', 'ft_raw_6d', 'ft_processed_6d',
                'ft_valid', 'ft_source_ati_frame_idx']
        paired = {k: src[k].copy() for k in keep if k in src}
        mapping = paired['source_reference_row_index']
        assert mapping.min() >= 0 and mapping.max() < len(values['state'])
        # Replaying source 50Hz rows must reproduce the exact original labels.
        np.testing.assert_array_equal(paired['action'], tracks[0]['source_action'][mapping])
        state = values['state'][mapping]
        paired.update(schema_version=np.array(['chip_real_replay_multimodal_v001']), state=state,
                      state_valid=np.isfinite(state).all(1), state_action_timing_valid=np.ones(len(state), bool),
                      replay_state_available=np.ones(len(state), bool), record_row_index=mapping.copy(),
                      policy_step=values['tick'][mapping], sim_time_s=values['tick'][mapping]*.02,
                      q_real=state[:, :29], dq_real=state[:, 29:58], imu_quat_wxyz=state[:, 58:62],
                      imu_gyro=state[:, 62:65], state_origin=np.array(['mujoco_pre_action']),
                      source_reference_action=paired['action'].copy())
        for key in ['actor_command', 'applied_reference_action', 'target_q_real']:
            paired[key] = values[key][mapping]
        paired['reference_action'] = paired['applied_reference_action'].copy()
        paired['valid'] &= paired['state_valid']
        assert paired['valid'].all(), 'Do not silently alter the controlled comparison split'
        difference = state[:, :29] - src['state'][:, :29]
    np.savez_compressed(dest / 'state_action_30hz.npz', **paired)
    # The dataset loader requires resolved image paths inside this episode.
    # Hard links reuse immutable RGB bytes without outside-directory symlinks.
    shutil.copytree(source_episode / 'images', dest / 'images', copy_function=os.link)
    for file in ['chip_motion_50hz.npz', 'chip_reference_30hz.npz', 'frames.jsonl']:
        (dest / file).symlink_to(os.path.relpath(source_episode / file, dest))
    shutil.copy2(source_episode / 'chip_manifest.json', dest / 'chip_manifest.json')
    meta = dict(original_meta)
    meta.update(state_source=str(dest / 'sim_state_action_50hz.npz'), selected_replay_attempt='sim_cold_start',
                replay_status='completed', replay_complete=True, state_source_kind='mujoco',
                attempted_replays=[], original_real_replay_state_source=original_meta['state_source'])
    meta['modality_origin'] = dict(original_meta['modality_origin'],
        state='nominal MuJoCo BatchPolicy measured q/dq/IMU/gyro, pre-action',
        applied_reference_action='sim BatchPolicy applied reference including heading alignment')
    meta['timing_note'] = 'Preserved source image-to-50Hz-row mapping; simulated state captured before that row is applied.'
    meta['simulation'] = dict(result, physics_dt=.005, control_dt=.02,
        physical_compliance=expected_compliance.tolist(), cold_start_per_episode=True,
        initial_hold_s=3, lead_in_s=4, servo_delay_s=.005, servo_filter_alpha=.95,
        external_forces=False, car_contact=False,
        real_vs_sim_joint_rmse_rad=float(np.sqrt(np.mean(difference**2))))
    (dest / 'merged_manifest.json').write_text(json.dumps(meta, indent=2))
    return dict(episode=name, frames=len(state), rows_50hz=len(indices), status=result['status'],
                min_height=result['minimum_height_m'], max_tilt=result['maximum_tilt_deg'],
                real_vs_sim_joint_rmse_rad=meta['simulation']['real_vs_sim_joint_rmse_rad'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deploy-root', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episode', help='Smoke test one source episode before full generation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.source / 'dataset_manifest.json').read_text())
    names = [args.episode] if args.episode else manifest['episode_names']
    assert set(names) <= set(manifest['episode_names'])
    script_names = ['batch_chip_real_replay.py', 'probe_batch_replay_startup.py',
                    'replay_mimic_6d_aligned_sim.py', 'chip_replay_slew.py']
    cfg = yaml.safe_load(args.policy_config.read_text())
    protocol = dict(source_dataset=str(args.source), deploy_root=str(args.deploy_root),
        actor_onnx_sha256=sha(cfg['onnx_model_path']),
        static_reference_sha256=sha(cfg['reference_motion_path']),
        policy_config=str(args.policy_config), policy_config_sha256=sha(args.policy_config),
        generator_sha256=sha(__file__), tool_sha256={n:sha(args.deploy_root/'tools'/n) for n in script_names},
        matched_real_state_experiment=1458, no_dds=True, no_robot_commands=True,
        comparison='same RGB, actions, source row mapping and episode order; replace measured state only',
        limitation='Nominal MuJoCo, independent cold starts, no car contact/external force; not identical to continuous real batch dynamics.')
    (args.output / 'generation_protocol.json').write_text(json.dumps(protocol, indent=2))
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate, n, str(args.source), str(args.output),
                               str(args.deploy_root), str(args.policy_config)) for n in names]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps(dict(completed=len(reports), total=len(names), **report)), flush=True)
    order = {n:i for i,n in enumerate(names)}
    reports.sort(key=lambda x:order[x['episode']])
    manifest.update(episode_names=names, episodes=len(names), training_episodes=len(names),
                    complete_replay_episodes=len(names), source_image_frames=sum(r['frames'] for r in reports),
                    paired_frames=sum(r['frames'] for r in reports), valid_frames=sum(r['frames'] for r in reports),
                    state_source_kind='mujoco', generator_protocol='generation_protocol.json')
    manifest.pop('copied_files_verified', None)
    (args.output / 'simulation_summary.json').write_text(json.dumps(reports, indent=2))
    # Publish only after every selected source episode has succeeded.
    (args.output / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(passed=True, episodes=len(names), frames=manifest['source_image_frames'])), flush=True)


if __name__ == '__main__':
    main()
