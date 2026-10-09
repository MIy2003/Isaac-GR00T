"""Extend a pretrained state encoder without discarding its current-state weights."""

import torch


def expand_state_history(model, history_length: int):
    old_length = model.config.state_history_length
    if history_length == old_length:
        return
    if old_length != 1 or history_length < 1:
        raise ValueError(f"Cannot expand state history from {old_length} to {history_length}")
    layer = model.action_head.state_encoder.layer1
    old = layer.W
    state_dim = model.config.max_state_dim
    if old.shape[1] != state_dim:
        raise ValueError("State encoder weight does not match checkpoint max_state_dim")
    # History is oldest -> newest. Initially reproduce the pretrained response
    # to the latest state exactly; earlier states acquire weights during tuning.
    weight = old.new_zeros(old.shape[0], state_dim * history_length, old.shape[2])
    with torch.no_grad():
        weight[:, -state_dim:] = old
    layer.W = torch.nn.Parameter(weight, requires_grad=old.requires_grad)
    model.config.state_history_length = history_length
    model.action_head.config.state_history_length = history_length
