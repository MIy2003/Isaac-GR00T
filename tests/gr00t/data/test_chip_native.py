"""Contract tests: episode boundaries, rotations, normalization and policy I/O."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from gr00t.configs.data.chip_native_config import chip_native_modality_config
from gr00t.data.chip_native import (
    CHIP_POINT_NAMES,
    CHIP_SCHEMA,
    OBS_FIELD_DIMS,
    canonicalize_anchor_by_episode_yaw,
    pack_chip_reference,
    reference_relative_to_current,
    relative_reference_to_absolute,
    unpack_chip_reference,
)
from gr00t.data.dataset.chip_native_dataset import ChipNativeDataset
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.policy.chip_native_policy import build_chip_observation, predict_chip_relative
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation
import torch


def load_model_module(name):
    # gr00t.model.__init__ eagerly imports the entire VLM training stack.
    # These CPU unit tests only need the two independent torch modules.
    path = Path(__file__).resolve().parents[3] / "gr00t/model/modules" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quaternion(euler):
    return Rotation.from_euler("xyz", euler).as_quat()[..., [3, 0, 1, 2]].astype(np.float32)


@pytest.fixture
def chip_root(tmp_path):
    length = 40
    names = ["episode_a", "episode_b", "episode_c"]
    for ep, name in enumerate(names):
        path = tmp_path / name
        (path / "images").mkdir(parents=True)
        t = np.arange(length, dtype=np.float32)
        angles = np.stack((0.2 + 0.01 * t, -0.3 + 0.002 * t, 1.1 + 0.03 * t), axis=-1)
        fields = {
            "lower_joint_pos": np.broadcast_to(t[:, None] * 0.1 + ep, (length, 12)).copy(),
            "anchor_quat_wxyz": quaternion(angles),
            "vr_3point_pos_anchor": np.broadcast_to(t[:, None, None] * 0.01, (length, 3, 3)).copy(),
            "vr_3point_quat_anchor_wxyz": np.stack(
                [quaternion(angles * s) for s in (0.5, 1, 1.5)], axis=1
            ),
        }
        np.savez(path / "chip_reference_30hz.npz", **fields, vr_3point_names=CHIP_POINT_NAMES)
        for i in range(length):
            Image.fromarray(np.full((224, 400, 3), i, dtype=np.uint8)).save(
                path / "images" / f"frame_{i:06d}.jpg"
            )
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHIP_SCHEMA,
                "source_hz": 30,
                "source_frames": length * len(names),
                "episode_names": names,
            }
        )
    )
    return tmp_path


def test_sampling_and_boundaries(chip_root):
    ds = ChipNativeDataset(chip_root, chip_native_modality_config(), val_ratio=0)
    assert ds.samples[:6] == [(0, t) for t in range(0, 16, 3)]
    assert len(ds.samples) == 18
    for index, expected_history, expected_images in [
        (0, [0, 0, 0], [0, 0]),
        (3, [3, 6, 9], [3, 9]),
        (5, [9, 12, 15], [9, 15]),
    ]:
        step = ds.get_step(index)
        np.testing.assert_allclose(
            step.states["lower_joint_pos"][:, 0], np.array(expected_history) * 0.1
        )
        assert [int(im[0, 0, 0]) for im in step.images["camera0_rgb"]] == expected_images
        assert step.actions["native_relative"].shape == (24, 37)
        np.testing.assert_allclose(
            step.actions["native_relative"][:, 0], np.arange(1, 25) * 0.1, atol=1e-6
        )
    assert ds.samples[6] == (1, 0)
    np.testing.assert_allclose(ds.get_step(6).states["lower_joint_pos"][:, 0], 1)


def test_yaw_and_noncommutative_quaternion_roundtrip(chip_root):
    with np.load(chip_root / "episode_a/chip_reference_30hz.npz") as data:
        fields = {key: data[key] for key in OBS_FIELD_DIMS}
    canonical = canonicalize_anchor_by_episode_yaw(fields, fields["anchor_quat_wxyz"][0])
    euler = Rotation.from_quat(canonical["anchor_quat_wxyz"][0][[1, 2, 3, 0]]).as_euler("xyz")
    np.testing.assert_allclose(euler, [0.2, -0.3, 0], atol=1e-6)
    current = {k: v[6] for k, v in fields.items()}
    future = {k: v[7:31] for k, v in fields.items()}
    relative = reference_relative_to_current(future, current)
    decoded = relative_reference_to_absolute(relative, current)
    np.testing.assert_allclose(pack_chip_reference(decoded), pack_chip_reference(future), atol=1e-6)
    canonical_relative = reference_relative_to_current(
        {k: v[7:31] for k, v in canonical.items()},
        {k: v[6] for k, v in canonical.items()},
    )
    np.testing.assert_allclose(
        pack_chip_reference(relative), pack_chip_reference(canonical_relative), atol=1e-6
    )
    # Independently verify Hamilton product direction with SciPy rotations.
    expected = Rotation.from_quat(
        current["anchor_quat_wxyz"][[1, 2, 3, 0]]
    ).inv() * Rotation.from_quat(future["anchor_quat_wxyz"][:, [1, 2, 3, 0]])
    actual = Rotation.from_quat(relative["anchor_quat_wxyz"][:, [1, 2, 3, 0]])
    np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-6)


def test_split_stats_and_no_double_relative_conversion(chip_root):
    config = chip_native_modality_config()
    train = ChipNativeDataset(chip_root, config)
    val = ChipNativeDataset(chip_root, config, split="val")
    assert set(train.episode_names).isdisjoint(val.episode_names)
    assert len(train.episode_names) + len(val.episode_names) == 3
    tag = train.embodiment_tag.value
    stats = train.get_dataset_statistics()
    states, action = train.state_action(3)
    # Even with generic GR00T relative conversion enabled, the pre-encoded
    # native_relative key must remain untouched except for normalization.
    processor = StateActionProcessor({tag: config}, {tag: stats}, use_relative_action=True)
    normalized = processor.apply_action({"native_relative": action}, tag, states)
    decoded = processor.unapply_action(normalized, tag, states)
    np.testing.assert_allclose(decoded["native_relative"], action, atol=1e-6)
    assert np.asarray(stats["action"]["native_relative"]["min"]).shape == (37,)


def test_policy_contract(chip_root):
    ds = ChipNativeDataset(chip_root, chip_native_modality_config())
    step = ds.get_step(3)
    states = np.concatenate(list(step.states.values()), axis=-1)
    images = np.stack(step.images["camera0_rgb"])
    obs = build_chip_observation(images, states, "Flip the object.")
    assert obs["state"]["anchor_quat_wxyz"].shape == (1, 3, 4)
    assert obs["video"]["camera0_rgb"].shape == (1, 2, 224, 400, 3)
    assert obs["language"]["task"] == [["Flip the object."]]
    expected = step.actions["native_relative"]
    policy = SimpleNamespace(
        get_action=lambda observation: ({"native_relative": expected[None]}, {})
    )
    np.testing.assert_array_equal(predict_chip_relative(policy, images, states), expected)
    with pytest.raises(ValueError, match="RGB uint8"):
        build_chip_observation(images.astype(np.float32) / 255, states)


def test_state_history_preserves_pretraining_and_learns_past():
    CategorySpecificMLP = load_model_module("embodiment_conditioned_mlp").CategorySpecificMLP
    expand_state_history = load_model_module("state_history").expand_state_history
    encoder = CategorySpecificMLP(2, 5, 8, 7)
    config = SimpleNamespace(state_history_length=1, max_state_dim=5)
    model = SimpleNamespace(
        config=config, action_head=SimpleNamespace(config=config, state_encoder=encoder)
    )
    states = torch.randn(2, 3, 5)
    ids = torch.tensor([0, 1])
    expected = encoder(states[:, -1:], ids).detach()
    expand_state_history(model, 3)
    result = encoder(states.reshape(2, 1, 15), ids)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.count_nonzero(encoder.layer1.W.grad[:, :10]) > 0
    assert model.config.state_history_length == 3
    before = encoder.layer1.W
    expand_state_history(model, 3)
    assert encoder.layer1.W is before


def test_invalid_schema_and_missing_images(chip_root):
    image = chip_root / "episode_a/images/frame_000003.jpg"
    image.unlink()
    with pytest.raises(FileNotFoundError):
        ChipNativeDataset(chip_root, chip_native_modality_config(), val_ratio=0)
    manifest_path = chip_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_hz"] = 50
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="30 Hz"):
        ChipNativeDataset(chip_root, chip_native_modality_config())


@pytest.mark.skipif(
    not os.environ.get("CHIP_DP_DATASET_SOURCE"), reason="optional original DP source parity check"
)
def test_parity_with_original_diffusion_policy(chip_root):
    spec = importlib.util.spec_from_file_location(
        "chip_dp_reference", os.environ["CHIP_DP_DATASET_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 224, 400],
                "type": "rgb",
                "horizon": 2,
                "down_sample_steps": 6,
                "latency_steps": 0,
            },
            **{
                key: {
                    "shape": [dim],
                    "type": "low_dim",
                    "horizon": 3,
                    "down_sample_steps": 3,
                    "latency_steps": 0,
                }
                for key, dim in OBS_FIELD_DIMS.items()
            },
        },
        "action": {"shape": [37], "horizon": 24, "down_sample_steps": 1, "latency_steps": 0},
    }
    original = module.ChipNativeDataset(
        shape_meta,
        str(chip_root),
        action_representation="relative",
        anchor_history_representation="episode_yaw_canonical",
    )
    adapted = ChipNativeDataset(chip_root, chip_native_modality_config())
    assert len(original) == len(adapted.samples)
    for i in range(len(original)):
        obs, action, _ = original._state_action_numpy(i)
        new_obs, new_action = adapted.state_action(i)
        for key in obs:
            np.testing.assert_allclose(new_obs[key], obs[key], atol=1e-6)
        np.testing.assert_allclose(new_action, action, atol=1e-6)
        # Both encoders must preserve the exact 37-D pack/unpack layout.
        np.testing.assert_array_equal(
            pack_chip_reference(unpack_chip_reference(new_action)), action
        )
