"""CPU tests of real action-head forward/backward and trained RTC denoising.

Load the independent modules and the actual action-head class without importing
gr00t.model.__init__, which eagerly requires the entire VLM training environment.
No backbone weights, downloads, or substituted attention layers are used.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

from gr00t.configs.data.chip_native_config import chip_native_modality_config
from gr00t.data.chip_native import (
    pack_chip_reference,
    reference_relative_to_current,
    relative_reference_to_absolute,
    unpack_chip_reference,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.policy.chip_native_policy import rebase_chip_prefix
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


ROOT = Path(__file__).resolve().parents[3]


def load_module(name):
    path = ROOT / "gr00t/model/modules" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rtc_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return SimpleNamespace(
        rtc=load_module("rtc"),
        dit=load_module("dit"),
        mlp=load_module("embodiment_conditioned_mlp"),
    )


@pytest.fixture(scope="module")
def head_class(modules):
    path = ROOT / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
    tree = ast.parse(path.read_text())
    definition = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gr00tN1d7ActionHead"
    )
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "Beta": torch.distributions.Beta,
        "BatchFeature": BatchFeature,
        "logger": logging.getLogger(__name__),
        "DiT": modules.dit.DiT,
        "AlternateVLDiT": modules.dit.AlternateVLDiT,
        "SelfAttentionTransformer": modules.dit.SelfAttentionTransformer,
        "CategorySpecificMLP": modules.mlp.CategorySpecificMLP,
        "MultiEmbodimentActionEncoder": modules.mlp.MultiEmbodimentActionEncoder,
        "condition_training_prefix": modules.rtc.condition_training_prefix,
        "prepend_state_time": modules.rtc.prepend_state_time,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, definition], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Gr00tN1d7ActionHead"]


def small_config(alternate=False, delay=6):
    return SimpleNamespace(
        hidden_size=32,
        input_embedding_dim=32,
        backbone_embedding_dim=32,
        max_state_dim=40,
        max_action_dim=40,
        action_horizon=40,
        state_history_length=3,
        max_num_embodiments=2,
        num_inference_timesteps=3,
        add_pos_embed=True,
        max_seq_len=64,
        use_vlln=True,
        use_alternate_vl_dit=alternate,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        rtc_training_max_delay=delay,
        diffusion_model_cfg={
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": True,
        },
    )


def inputs(batch=4):
    backbone = BatchFeature(
        {
            "backbone_features": torch.randn(batch, 6, 32),
            "backbone_attention_mask": torch.ones(batch, 6, dtype=torch.bool),
            "image_mask": torch.tensor([[True, True, True, False, False, False]]).expand(batch, -1),
        }
    )
    mask = torch.zeros(batch, 40, 40)
    mask[:, :24, :37] = 1
    action = BatchFeature(
        {
            "state": torch.randn(batch, 3, 40),
            "action": torch.randn(batch, 40, 40),
            "action_mask": mask,
            "embodiment_id": torch.zeros(batch, dtype=torch.long),
        }
    )
    return backbone, action


def test_delay_sampling_uses_valid_rows_and_keeps_postfix(modules):
    mask = torch.zeros(1024, 40, 40)
    mask[:, :24, :37] = 1
    torch.manual_seed(2)
    prefix = modules.rtc.sample_prefix_mask(mask, 6)
    assert set(prefix.sum(1).tolist()) == set(range(7))
    assert not prefix[:, 24:].any()
    mask[:4, 1:] = 0
    assert not modules.rtc.sample_prefix_mask(mask, 100)[:4].any()
    assert (modules.rtc.sample_prefix_mask(mask, 100)[4:].sum(1) < 24).all()
    state = torch.get_rng_state()
    assert not modules.rtc.sample_prefix_mask(mask, 0).any()
    assert torch.equal(state, torch.get_rng_state())


def test_clean_prefix_time_and_loss_mask(modules):
    torch.manual_seed(5)
    actions = torch.randn(64, 40, 40)
    noise = torch.randn_like(actions)
    time = torch.full((64, 1, 1), 0.25)
    noisy = (1 - time) * noise + time * actions
    mask = torch.zeros_like(actions)
    mask[:, :24, :37] = 1
    trajectory, times, loss_mask = modules.rtc.condition_training_prefix(
        actions, noisy, time, mask, 6
    )
    prefix = times == 1
    assert prefix.any()
    torch.testing.assert_close(trajectory[prefix], actions[prefix])
    torch.testing.assert_close(trajectory[~prefix], noisy[~prefix])
    assert not loss_mask[prefix].any()
    assert not loss_mask[:, 24:].any()
    assert not loss_mask[:, :, 37:].any()
    assert (loss_mask.sum((1, 2)) > 0).all()
    prediction = torch.randn_like(actions, requires_grad=True)
    (((prediction - (actions - noise)) ** 2 * loss_mask).sum() / loss_mask.sum()).backward()
    assert not prediction.grad[prefix].any()
    assert prediction.grad[loss_mask.bool()].abs().sum() > 0


@pytest.mark.parametrize("alternate", [False, True])
def test_scalar_time_matches_repeated_token_times(modules, alternate):
    torch.manual_seed(3)
    model_type = modules.dit.AlternateVLDiT if alternate else modules.dit.DiT
    model = model_type(**small_config().diffusion_model_cfg, cross_attention_dim=32).eval()
    hidden, vl = torch.randn(2, 9, 32), torch.randn(2, 6, 32)
    t = torch.tensor([234, 678])
    kwargs = {}
    if alternate:
        kwargs = {
            "image_mask": torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool).expand(2, -1),
            "backbone_attention_mask": torch.ones(2, 6, dtype=torch.bool),
        }
    expected = model(hidden, vl, timestep=t, **kwargs)
    actual = model(hidden, vl, timestep=t[:, None].expand(-1, 9), **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert model.timestep_encoder.timestep_embedder.linear_1.weight.grad is not None


def test_action_encoder_accepts_token_time_without_new_parameters(modules):
    encoder = modules.mlp.MultiEmbodimentActionEncoder(7, 32, 2)
    state_dict = {k: v.clone() for k, v in encoder.state_dict().items()}
    actions, times, ids = torch.randn(2, 8, 7), torch.tensor([123, 456]), torch.tensor([0, 1])
    expected = encoder(actions, times, ids)
    repeated = times[:, None].expand(-1, 8).clone()
    torch.testing.assert_close(encoder(actions, repeated, ids), expected)
    repeated[:, :3] = 1000
    changed = encoder(actions, repeated, ids)
    assert not torch.allclose(changed[:, :3], expected[:, :3])
    torch.testing.assert_close(changed[:, 3:], expected[:, 3:])
    encoder.load_state_dict(state_dict, strict=True)


@pytest.mark.parametrize("alternate", [False, True])
def test_actual_head_training_forward_backward(head_class, alternate):
    torch.manual_seed(11)
    head = head_class(small_config(alternate)).train()
    backbone, action = inputs(batch=8)
    captured = []
    hook = head.action_encoder.register_forward_pre_hook(lambda _, args: captured.append(args))
    original_mask = action.action_mask.clone()
    out = head(backbone, action)
    hook.remove()
    trajectory, times, _ = captured[0]
    prefix = times == 1000
    assert prefix.any()
    torch.testing.assert_close(trajectory[prefix], action.action[prefix])
    assert not out["action_mask"][prefix].any()
    torch.testing.assert_close(action.action_mask, original_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert head.state_encoder.layer1.W.grad[:, :80].abs().sum() > 0
    assert head.model.timestep_encoder.timestep_embedder.linear_1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("alternate", [False, True])
def test_trained_inference_clamps_prefix_every_iteration(head_class, alternate):
    head = head_class(small_config(alternate)).eval()
    backbone, action = inputs(batch=2)
    original = action.action[:, :4].clone()
    captured = []
    hook = head.action_encoder.register_forward_pre_hook(
        lambda _, args: captured.append((args[0].clone(), args[1].clone()))
    )
    out = head.get_action(backbone, action, {"rtc_mode": "trained", "rtc_prefix_length": 4})
    hook.remove()
    assert len(captured) == 3
    for trajectory, times in captured:
        torch.testing.assert_close(trajectory[:, :4], original, rtol=0, atol=0)
        assert (times[:, :4] == 1000).all()
        assert (times[:, 4:] < 1000).all()
    torch.testing.assert_close(out.action_pred[:, :4], original, rtol=0, atol=0)
    assert not out.action_pred.requires_grad
    assert torch.isfinite(out.action_pred).all()


def test_old_checkpoints_and_zero_prefix_inference(head_class):
    old = head_class(small_config(delay=0)).eval()
    new = head_class(small_config(delay=6)).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    backbone, action = inputs(batch=2)
    del action["action"]
    # process_backbone_output writes normalized features, so use separate copies.
    backbone2 = BatchFeature({k: v.clone() for k, v in backbone.items()})
    torch.manual_seed(19)
    ordinary = old.get_action(backbone2, action).action_pred
    torch.manual_seed(19)
    trained = new.get_action(
        backbone, action, {"rtc_mode": "trained", "rtc_prefix_length": 0}
    ).action_pred
    torch.testing.assert_close(trained, ordinary, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="checkpoint"):
        old.get_action(backbone, action, {"rtc_mode": "trained", "rtc_prefix_length": 1})
    with pytest.raises(ValueError, match="prefix length"):
        new.get_action(backbone, action, {"rtc_mode": "trained", "rtc_prefix_length": 7})


def test_legacy_overlap_ramp_path_remains_available(head_class):
    head = head_class(small_config(delay=0)).eval()
    backbone, action = inputs(batch=2)
    expected_prefix = action.action[:, 12:14].clone()
    out = head.get_action(
        backbone,
        action,
        {
            "action_horizon": 16,
            "rtc_overlap_steps": 4,
            "rtc_frozen_steps": 2,
            "rtc_ramp_rate": 5.0,
        },
    )
    torch.testing.assert_close(out.action_pred[:, :2], expected_prefix, rtol=0, atol=0)
    assert torch.isfinite(out.action_pred).all()


def test_chip_prefix_changes_reference_and_time_alignment():
    def reference(angle, position):
        row = np.zeros(37, dtype=np.float32)
        q = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0], dtype=np.float32)
        row[:12] = position
        row[12:16] = q
        row[16:25] = position
        row[25:] = np.tile(q, 3)
        return row

    old, current = reference(0.7, 0.2), reference(-0.4, 0.5)
    future = np.stack([reference(0.8 + i * 0.03, 0.3 + i * 0.02) for i in range(24)])
    previous = pack_chip_reference(
        reference_relative_to_current(unpack_chip_reference(future), unpack_chip_reference(old))
    )
    prefix = rebase_chip_prefix(previous, old, current, elapsed_steps=3, delay_steps=6)
    reconstructed = pack_chip_reference(
        relative_reference_to_absolute(
            unpack_chip_reference(prefix), unpack_chip_reference(current)
        )
    )
    np.testing.assert_allclose(reconstructed, future[3:9], atol=1e-6)
    assert not np.allclose(prefix, previous[3:9])
    with pytest.raises(ValueError, match="cover"):
        rebase_chip_prefix(previous, old, current, 22, 3)


def make_policy():
    """Use the real policy/normalizer with lightweight VLM and model boundaries."""
    tag = EmbodimentTag.NEW_EMBODIMENT
    config = chip_native_modality_config()
    stats = {
        tag.value: {
            "action": {
                "native_relative": {
                    "min": [-1.0] * 37,
                    "max": [1.0] * 37,
                    "mean": [0.0] * 37,
                    "std": [1.0] * 37,
                    "q01": [-1.0] * 37,
                    "q99": [1.0] * 37,
                }
            }
        }
    }
    sap = StateActionProcessor({tag.value: config}, stats, clip_outliers=True)

    class Processor:
        state_action_processor = sap
        max_action_horizon = 40
        max_action_dim = 40

        def __call__(self, messages):
            return {"state": torch.zeros(3, 40)}

        def decode_action(self, values, embodiment, states):
            return sap.unapply_action({"native_relative": values[:, :24, :37]}, tag.value, states)

    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.modality_configs, policy.embodiment_tag = config, tag
    policy.language_key = "task"
    policy.processor = Processor()
    captured = []

    def inference(inputs, options=None):
        captured.append((inputs, options))
        return {"action_pred": inputs["action"]}

    policy.model = SimpleNamespace(
        config=SimpleNamespace(rtc_training_max_delay=6), get_action=inference
    )
    policy.collate_fn = lambda items: {
        "inputs": {k: torch.stack([item[k] for item in items]) for k in items[0]}
    }
    return policy, captured


def test_policy_passes_normalized_prefix_without_clipping_and_preserves_output():
    policy, captured = make_policy()
    prefix = np.full((1, 3, 37), 2.1234567, dtype=np.float32)  # outside training stats
    observation = {
        "video": {"camera0_rgb": np.zeros((1, 2, 224, 400, 3), dtype=np.uint8)},
        "state": {
            k: np.zeros((1, 3, d), dtype=np.float32)
            for k, d in zip(policy.modality_configs["state"].modality_keys, [12, 4, 9, 12])
        },
        "language": {"task": [["test"]]},
    }
    actions, _ = policy._get_action(
        observation, {"rtc_mode": "trained", "rtc_prefix": {"native_relative": prefix}}
    )
    assert captured[0][1] == {"rtc_mode": "trained", "rtc_prefix_length": 3}
    assert (captured[0][0]["action"][:, :3, :37] > 2).all()
    assert policy.processor.state_action_processor.clip_outliers
    np.testing.assert_array_equal(actions["native_relative"][:, :3], prefix)
    with pytest.raises(ValueError, match="trained delay"):
        policy._prepare_rtc_prefix(
            {"rtc_mode": "trained", "rtc_prefix": {"native_relative": np.zeros((1, 7, 37))}}, 1
        )
