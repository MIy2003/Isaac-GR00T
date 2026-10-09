"""Training-time action-prefix conditioning (arXiv:2512.05964).

GR00T's flow convention is noise at t=0 and clean actions at t=1.
No learnable parameters are added; existing time embeddings become tokenwise.
"""

import torch


def sample_prefix_mask(action_mask: torch.Tensor, max_delay: int) -> torch.Tensor:
    """Sample d uniformly from 0..min(max_delay, valid_horizon-1), per example.

    Only leading valid rows may be conditioned. Padding cannot become a prefix,
    and at least one valid action remains supervised for every nonempty sample.
    """
    if max_delay < 0:
        raise ValueError("rtc_training_max_delay must be nonnegative")
    valid = action_mask.bool().any(dim=-1)
    if max_delay == 0:
        return torch.zeros_like(valid)
    leading_valid = valid.long().cumprod(dim=1).sum(dim=1)
    limit = (leading_valid - 1).clamp(min=0, max=max_delay)
    delay = (torch.rand(len(limit), device=action_mask.device) * (limit + 1)).long()
    return torch.arange(valid.shape[1], device=action_mask.device)[None, :] < delay[:, None]


def condition_training_prefix(actions, noisy_trajectory, flow_time, action_mask, max_delay):
    """Keep clean ground-truth prefixes, mark their time as 1, mask their loss."""
    prefix = sample_prefix_mask(action_mask, max_delay)
    trajectory = torch.where(prefix[..., None], actions, noisy_trajectory)
    token_time = flow_time.reshape(-1, 1).expand(-1, actions.shape[1])
    token_time = torch.where(prefix, torch.ones_like(token_time), token_time)
    loss_mask = action_mask * (~prefix[..., None]).to(action_mask.dtype)
    return trajectory, token_time, loss_mask


def prepend_state_time(action_time, sample_time):
    """The state token retains the original global flow time; actions are tokenwise."""
    return torch.cat((sample_time.reshape(-1, 1), action_time), dim=1)
