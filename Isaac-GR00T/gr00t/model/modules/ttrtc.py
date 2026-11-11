"""Weight-free helpers for training-time RTC in normalized action coordinates."""

import math
from numbers import Integral

import torch


def validate_ttrtc_config(config):
    """Only validate the opt-in path; legacy checkpoint defaults remain inert."""
    if not getattr(config, "ttrtc_enabled", False):
        return
    policy_hz = config.ttrtc_policy_hz
    if not math.isfinite(policy_hz) or policy_hz <= 0:
        raise ValueError("ttrtc_policy_hz must be finite and positive")
    maximum = config.ttrtc_max_delay_steps
    if isinstance(maximum, bool) or not isinstance(maximum, Integral):
        raise ValueError("ttrtc_max_delay_steps must be an integer")
    if not 0 <= maximum < config.action_horizon:
        raise ValueError("ttrtc_max_delay_steps must satisfy 0 <= max_delay < action_horizon")
    if not 0 < config.ttrtc_zero_delay_probability <= 1:
        raise ValueError("ttrtc_zero_delay_probability must be in (0, 1]")


def sample_delay_steps(action_mask, max_delay_steps, zero_delay_probability):
    """Sample delays, retaining at least the last valid action token as suffix.

    Positive delays are uniform up to the per-example valid bound, mixed with
    an explicit zero-delay probability. Padded time/action dimensions are never
    promoted into the loss. All-padding samples get delay zero and zero loss.
    """
    batch, horizon, _ = action_mask.shape
    positions = torch.arange(horizon, device=action_mask.device)[None, :]
    last_valid = torch.where(action_mask.bool().any(-1), positions, 0).amax(-1)
    bound = last_valid.clamp(max=max_delay_steps)
    positive = (torch.rand(batch, device=action_mask.device) * bound).long() + 1
    zeros = torch.rand(batch, device=action_mask.device) < zero_delay_probability
    return torch.where(zeros | (bound == 0), 0, positive)


def prefix_mask_from_delays(delays, horizon):
    return torch.arange(horizon, device=delays.device)[None, :, None] < delays[:, None, None]


def training_trajectory(actions, noise, time, action_mask, delays, num_timestep_buckets):
    """Build exact clean prefix, per-action time IDs, and independent loss mask."""
    prefix_mask = prefix_mask_from_delays(delays, actions.shape[1])
    noisy = torch.where(prefix_mask, actions, (1 - time) * noise + time * actions)
    times = time[:, 0, 0, None].expand(-1, actions.shape[1])
    times = torch.where(prefix_mask.squeeze(-1), torch.ones_like(times), times)
    token_times = (times * num_timestep_buckets).long()
    loss_mask = action_mask * (~prefix_mask).to(action_mask.dtype)
    return noisy, token_times, loss_mask, prefix_mask


def inference_prefix(options, config, batch_size, device, dtype):
    """Validate normalized prefix options, or return None for legacy inference.

    Accept integer scalar or integer [B] delays, and [B,max(d),D] / [B,H,D]
    floating tensors. No denormalization, clipping, slicing, or cursor adjustment
    belongs here. The caller consumes the returned full horizon after its prefix.
    """
    options = options or {}
    keys = ("ttrtc_prefix", "ttrtc_delay_steps")
    if not any(key in options for key in keys):
        return None
    if not getattr(config, "ttrtc_enabled", False):
        raise ValueError("TTRTC requires a checkpoint trained with config.ttrtc_enabled=True")
    validate_ttrtc_config(config)
    if not all(key in options for key in keys):
        raise ValueError("TTRTC requires both ttrtc_prefix and ttrtc_delay_steps")
    if any(key.startswith("rtc_") for key in options):
        raise ValueError("TTRTC cannot be combined with legacy RTC options")
    delays = torch.as_tensor(options["ttrtc_delay_steps"], device=device)
    if delays.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("ttrtc_delay_steps must be integer scalar or integer [B]")
    if delays.ndim == 0:
        delays = delays.expand(batch_size)
    if delays.shape != (batch_size,):
        raise ValueError("ttrtc_delay_steps must have shape [B]")
    if ((delays < 0) | (delays > config.ttrtc_max_delay_steps) | (delays >= config.action_horizon)).any():
        raise ValueError("TTRTC delay must be within trained max delay and below action horizon")
    delays = delays.long()
    prefix = options["ttrtc_prefix"]
    if not torch.is_tensor(prefix) or not torch.is_floating_point(prefix):
        raise ValueError("ttrtc_prefix must be a floating normalized action tensor")
    maximum = int(delays.max().item())
    expected_lengths = (maximum, config.action_horizon)
    if (prefix.ndim != 3 or prefix.shape[0] != batch_size
            or prefix.shape[1] not in expected_lengths or prefix.shape[2] != config.max_action_dim):
        raise ValueError("ttrtc_prefix must have shape [B,max(d),D] or [B,H,D], D=max_action_dim")
    if not torch.isfinite(prefix).all():
        raise ValueError("ttrtc_prefix contains non-finite values")
    prefix = prefix.to(device=device, dtype=dtype)
    if not torch.isfinite(prefix).all():
        raise ValueError("ttrtc_prefix overflows model dtype")
    clean = torch.zeros(batch_size, config.action_horizon, config.max_action_dim, device=device, dtype=dtype)
    clean[:, :prefix.shape[1]] = prefix
    return clean, prefix_mask_from_delays(delays, config.action_horizon), delays
