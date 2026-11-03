"""TA-VLA action-expert force adaptation (independent of Force MoT).

Reference: ZZongzheng0918/TA-VLA, revision 149d4be, pi0.py.
Uses GR00T's noise -> data flow convention and retains pretrained action layers.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from gr00t.model.modules.rtc import condition_training_prefix


@dataclass(frozen=True)
class TAForceConfig:
    mode: str = "EXPERT_HIS_C_FUT"
    force_dim: int = 3
    history_samples: int = 10
    loss_weight: float = 0.1

    def __post_init__(self):
        if self.mode != "EXPERT_HIS_C_FUT":
            raise ValueError("TA Force only supports EXPERT_HIS_C_FUT (history + future objective)")
        if self.force_dim < 1 or self.history_samples < 1 or self.loss_weight < 0:
            raise ValueError("Invalid TA Force dimensions/loss weight")



class TAForce(nn.Module):
    def __init__(self, config, specification):
        super().__init__()
        self.spec = TAForceConfig(**specification)
        s = self.spec
        width = config.input_embedding_dim
        size = s.force_dim * s.history_samples
        self.projector = nn.Sequential(nn.Linear(size, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
        # Preserve pretrained action weights and add force input/output projections.
        self.future_encoder = nn.Linear(s.force_dim, width, bias=False)
        self.future_decoder = nn.Linear(config.hidden_size, s.force_dim)
        self.register_buffer("force_mean", torch.zeros(s.force_dim))
        self.register_buffer("force_std", torch.ones(s.force_dim))

    def normalize(self, value, valid):
        if value.shape[-1] != self.spec.force_dim or valid.shape != value.shape[:-1]:
            raise ValueError("Force/valid shape mismatch")
        valid = valid.bool()
        if not torch.isfinite(value[valid]).all():
            raise ValueError("Nonfinite valid force")
        value = torch.where(valid[..., None], value, self.force_mean)
        return (value - self.force_mean) / self.force_std.clamp_min(1e-6)

    def tokens(self, inputs):
        value = inputs["force"]
        if value.ndim != 3 or value.shape[1] != self.spec.history_samples:
            raise ValueError("Expected force [B, saved history_samples, force_dim], oldest first")
        value = self.normalize(value, inputs["force_valid"])
        value = value.flatten(1)[:, None]
        return self.projector(value.to(self.projector[0].weight.dtype))

    def velocity(self, head, features, backbone, inputs, action, future, time, tokens, action_time=None):
        # Inject into the first action projection, before timestep fusion.
        encoder = head.action_encoder
        projected = encoder.W1(action, inputs.embodiment_id) + self.future_encoder(future)
        at = time[:, None].expand(-1, action.shape[1]) if action_time is None else action_time
        time_features = encoder.pos_encoding(at).to(projected.dtype)
        model_time = time if action_time is None else torch.cat((time[:, None].expand(-1, 2), at), 1)
        hidden = torch.nn.functional.silu(encoder.W2(torch.cat((projected, time_features), -1), inputs.embodiment_id))
        action_features = encoder.W3(hidden, inputs.embodiment_id)
        if head.config.add_pos_embed:
            action_features = action_features + head.position_embedding(
                torch.arange(action.shape[1], device=action.device))[None]
        parts = [tokens, features.state_features, action_features]
        kwargs = dict(hidden_states=torch.cat(parts, 1),
                      encoder_hidden_states=features.backbone_features,
                      timestep=model_time,
                      encoder_attention_mask=backbone.backbone_attention_mask)
        if head.config.use_alternate_vl_dit:
            kwargs.update(image_mask=backbone.image_mask,
                          backbone_attention_mask=backbone.backbone_attention_mask)
        output = head.model(**kwargs)
        start = 1 + tokens.shape[1]
        output = output[:, start:start + head.action_horizon]
        return (head.action_decoder(output, inputs.embodiment_id),
                self.future_decoder(output))

    def training_forward(self, head, backbone, inputs):
        head.set_frozen_modules_to_eval_mode()
        features = head._encode_features(backbone, inputs)
        if head.training and head.state_dropout_prob:
            keep = torch.rand(features.state_features.shape[0], device=inputs.state.device) >= head.state_dropout_prob
            features.state_features = features.state_features * keep[:, None, None]
        target = inputs.action
        noise = torch.randn_like(target)
        t = head.sample_time(target.shape[0], target.device, target.dtype)
        blend = t[:, None, None]
        action = (1 - blend) * noise + blend * target
        force_target = self.normalize(inputs["future_force"], inputs["future_force_valid"])
        if force_target.shape[:2] != target.shape[:2]:
            raise ValueError("Future force must match the padded action horizon")
        force_noise = torch.randn_like(force_target)
        force = (1 - blend) * force_noise + blend * force_target
        action_mask = inputs.action_mask
        action_time = None
        if head.training and head.config.rtc_training_max_delay > 0:
            action, token_time, action_mask = condition_training_prefix(
                target, action, blend, action_mask, head.config.rtc_training_max_delay)
            action_time = (token_time * head.num_timestep_buckets).long()
            committed = (token_time == 1) & inputs.action_mask.bool().any(-1)
            # Committed actions do not imply known future force. Do not leak its labels.
            force = force.masked_fill(committed[..., None], 0)
        prediction, fp = self.velocity(head, features, backbone, inputs, action, force,
                                       (t * head.num_timestep_buckets).long(), self.tokens(inputs), action_time)
        mask = action_mask.bool()
        action_loss = (prediction.float() - (target - noise).float()).square().masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)
        fm = inputs.future_force_valid.bool() & mask.any(-1)
        error = (fp.float() - (force_target - force_noise).float()).square()
        force_loss = error.masked_fill(~fm[..., None], 0).sum() / (fm.sum() * self.spec.force_dim).clamp_min(1)
        return {"loss": action_loss + self.spec.loss_weight * force_loss,
                "action_loss": action_loss.detach(), "force_loss": force_loss.detach()}

    @torch.no_grad()
    def sample(self, head, features, backbone, inputs, options):
        options = options or {}
        if options and options.get("rtc_mode") != "trained":
            raise ValueError("Use the base policy trained RTC interface")
        trained = options.get("rtc_mode") == "trained"
        length = options.get("rtc_prefix_length", 0) if trained else 0
        if trained and head.config.rtc_training_max_delay <= 0:
            raise ValueError("Checkpoint was not trained with RTC")
        if not isinstance(length, int) or not 0 <= length <= head.config.rtc_training_max_delay or length >= head.action_horizon:
            raise ValueError("RTC prefix length exceeds the trained delay or action horizon")
        if not trained and "action" in inputs:
            raise ValueError("Action prefix requires rtc_mode='trained'")
        b = features.state_features.shape[0]
        ref = features.state_features
        action = torch.randn(b, head.action_horizon, head.action_dim, device=ref.device, dtype=ref.dtype)
        force = (torch.randn(b, head.action_horizon, self.spec.force_dim, device=ref.device,
                             dtype=ref.dtype))
        if length:
            prefix = inputs.get("action")
            if prefix is None or prefix.shape[0] != b or prefix.shape[1] < length or prefix.shape[2] != head.action_dim:
                raise ValueError("RTC action prefix shape does not match the model")
            if not torch.isfinite(prefix[:, :length]).all():
                raise ValueError("Nonfinite RTC prefix")
            action[:, :length] = prefix[:, :length]
            force[:, :length] = 0
        tokens = self.tokens(inputs)
        for i in range(head.num_inference_timesteps):
            time = torch.full((b,), int(i / head.num_inference_timesteps * head.num_timestep_buckets), device=ref.device)
            action_time = None
            if trained:
                action_time = time[:, None].expand(-1, head.action_horizon).clone()
                action_time[:, :length] = head.num_timestep_buckets
            av, fv = self.velocity(head, features, backbone, inputs, action, force, time, tokens, action_time)
            action = action + av / head.num_inference_timesteps
            force = force + fv / head.num_inference_timesteps
            if length:
                action[:, :length] = prefix[:, :length]
                force[:, :length] = 0
        result = {"action_pred": action}
        result["force_pred"] = force * self.force_std + self.force_mean
        result["force_pred_valid"] = torch.arange(head.action_horizon, device=ref.device)[None].expand(b, -1) >= length
        return BatchFeature(result)


def enable_ta_force(model, specification=None, mean=None, std=None):
    """Attach only after loading an ordinary GR00T checkpoint."""
    if getattr(model.config, "force_mot", None) is not None:
        raise ValueError("TA Force and Force MoT are separate model paths")
    if hasattr(model.action_head, "ta_force"):
        raise ValueError("TA Force already enabled")
    spec = asdict(TAForceConfig(**(specification or {})))
    model.config.ta_force = spec
    branch = TAForce(model.config, spec).to(device=model.device, dtype=model.action_head.dtype)
    for name, value in (("force_mean", mean), ("force_std", std)):
        if value is not None:
            value = torch.as_tensor(value).flatten()
            if value.shape != (branch.spec.force_dim,) or not torch.isfinite(value).all():
                raise ValueError("Invalid force statistics")
            if name == "force_std" and (value <= 0).any():
                raise ValueError("Force std must be positive")
            getattr(branch, name).copy_(value)
    model.action_head.ta_force = branch
    return model
