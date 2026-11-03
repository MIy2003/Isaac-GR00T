"""Ordinary GR00T inference with explicit TA Force history; no MoT scheduler."""
from copy import copy
import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy


class TAForcePolicy(Gr00tPolicy):
    """Add force [B,H,D] and force_valid [B,H] to the usual observation.

    Values are physical sensor-frame force (Fx, Fy, Fz for new checkpoints), oldest first. The caller supplies
    causal history and resets it on episode boundaries. Future predictions are
    diagnostic observations, never robot force commands.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.model.action_head, "ta_force"):
            raise ValueError("TAForcePolicy requires a TA Force checkpoint")

    def _get_action(self, observation, options=None):
        spec = self.model.action_head.ta_force.spec
        observations = self._unbatch_observation(observation)
        batch = len(observations)
        prefix, model_options = self._prepare_rtc_prefix(options, batch)
        if prefix is not None:
            prefix_processor = copy(self.processor.state_action_processor)
            prefix_processor.clip_outliers = False
        force = np.asarray(observation["force"], dtype=np.float32)
        valid = np.asarray(observation["force_valid"], dtype=bool)
        if force.shape != (batch, spec.history_samples, spec.force_dim) or valid.shape != force.shape[:-1]:
            raise ValueError("Force history shape disagrees with checkpoint")
        processed, states = [], []
        for i, obs in enumerate(observations):
            step = self._to_vla_step_data(obs)
            states.append(step.states)
            item = self.processor([dict(type=MessageType.EPISODE_STEP.value, content=step)])
            item.update(force=force[i], force_valid=valid[i])
            if prefix is not None:
                normalized = prefix_processor.apply_action(
                    {k: v[i] for k, v in prefix.items()}, self.embodiment_tag.value, step.states)
                normalized = np.concatenate([normalized[k] for k in self.modality_configs["action"].modality_keys], -1)
                padded = np.zeros((self.processor.max_action_horizon, self.processor.max_action_dim), np.float32)
                padded[:len(normalized), :normalized.shape[-1]] = normalized
                item["action"] = padded
            processed.append(item)
        with torch.inference_mode():
            result = self.model.get_action(**self.collate_fn(processed), options=model_options)
        state = {k: np.stack([s[k] for s in states]) for k in self.modality_configs["state"].modality_keys}
        action = self.processor.decode_action(result["action_pred"].float().cpu().numpy(), self.embodiment_tag, state)
        if prefix is not None:
            for key, value in prefix.items():
                action[key][:, :value.shape[1]] = value
        info = {}
        if "force_pred" in result:
            # Return only real action rows, excluding the model's padded horizon.
            horizon = len(self.modality_configs["action"].delta_indices)
            info["force_pred_valid"] = result["force_pred_valid"][:, :horizon].cpu().numpy()
            info["force_pred"] = result["force_pred"][:, :horizon].float().cpu().numpy()
        return {k: v.astype(np.float32) for k, v in action.items()}, info
