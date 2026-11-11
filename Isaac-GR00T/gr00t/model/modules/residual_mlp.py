"""A supervised, bounded residual MLP after a frozen GR00T/TTRTC policy.

All public action/state tensors contain decoded physical joint positions. The
same train-split action affine transform normalizes every MLP input and loss;
the GR00T processor's relative-action/clipping transform is deliberately not
used here. No optimizer or gradient is attached to the base policy.
"""
from dataclasses import asdict, dataclass
import math
from numbers import Integral

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ResidualMLPConfig:
    action_dim: int = 16
    hidden_dim: int = 256
    correction_dims: tuple = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
    max_residual: float = 0.05
    reference_dt: float = 1 / 15
    center: tuple = ()
    scale: tuple = ()
    action_lower: tuple = ()
    action_upper: tuple = ()

    def __post_init__(self):
        if isinstance(self.action_dim, bool) or not isinstance(self.action_dim, Integral) or self.action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if isinstance(self.hidden_dim, bool) or self.hidden_dim != 256:
            raise ValueError("the residual MLP requires exactly two 256-unit hidden layers")
        dims = tuple(self.correction_dims)
        if (not dims or len(set(dims)) != len(dims) or any(
                isinstance(d, bool) or not isinstance(d, Integral) or not 0 <= d < self.action_dim
                for d in dims)):
            raise ValueError("correction_dims must contain unique valid action indices")
        object.__setattr__(self, "correction_dims", dims)
        for name in ("max_residual", "reference_dt"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, default in (("center", 0.0), ("scale", 1.0)):
            values = tuple(getattr(self, name)) or (default,) * self.action_dim
            if len(values) != self.action_dim or any(not math.isfinite(v) for v in values):
                raise ValueError(f"{name} must have action_dim finite values")
            if name == "scale" and any(v <= 0 for v in values):
                raise ValueError("scale values must be positive")
            object.__setattr__(self, name, values)
        lower, upper = tuple(self.action_lower), tuple(self.action_upper)
        if bool(lower) != bool(upper):
            raise ValueError("action_lower and action_upper must both be provided or both be empty")
        if lower:
            if (len(lower) != self.action_dim or len(upper) != self.action_dim
                    or any(not math.isfinite(v) for v in lower + upper)
                    or any(a > b for a, b in zip(lower, upper))):
                raise ValueError("action bounds must be finite action_dim values with lower <= upper")
        object.__setattr__(self, "action_lower", lower)
        object.__setattr__(self, "action_upper", upper)

    def to_dict(self):
        result = asdict(self)
        for key in ("correction_dims", "center", "scale", "action_lower", "action_upper"):
            result[key] = list(result[key])
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class ResidualLossConfig:
    reconstruction_weight: float = 1.0
    boundary_weight: float = 0.5
    acceleration_weight: float = 0.05
    jerk_weight: float = 0.01
    residual_weight: float = 0.001
    huber_delta: float = 0.1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.reconstruction_weight <= 0 or self.huber_delta <= 0:
            raise ValueError("reconstruction_weight and huber_delta must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def _floating_tensor(value, shape, name, device):
    if (not torch.is_tensor(value) or not torch.is_floating_point(value)
            or tuple(value.shape) != tuple(shape)):
        raise ValueError(f"{name} must be a floating point tensor with shape {tuple(shape)}")
    if value.device != device or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and on the nominal action device")
    return value.detach()


def _bool_tensor(value, shape, name, device):
    value = torch.as_tensor(value, device=device)
    if tuple(value.shape) != tuple(shape) or not ((value == 0) | (value == 1)).all():
        raise ValueError(f"{name} must be a binary tensor with shape {tuple(shape)}")
    return value.bool()


def _timing(value, shape, name, reference):
    result = torch.as_tensor(value, device=reference.device, dtype=torch.float32).detach()
    if result.ndim == 0:
        result = result.expand(shape)
    elif len(shape) == 2 and tuple(result.shape) == (shape[0],):
        result = result[:, None].expand(shape)
    if tuple(result.shape) != tuple(shape) or not torch.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f"{name} must be positive finite seconds, scalar or shape {tuple(shape)}")
    return result


def _context(nominal, history, config, prefix_lengths, action_mask,
             dt_seconds, history_dt_seconds, history_valid):
    if (not torch.is_tensor(nominal) or nominal.ndim != 3 or not torch.is_floating_point(nominal)
            or nominal.shape[-1] != config.action_dim or min(nominal.shape[:2]) < 1):
        raise ValueError("nominal must have shape [B,T,action_dim] with B,T > 0")
    batch, horizon, dim = nominal.shape
    nominal = _floating_tensor(nominal, nominal.shape, "nominal", nominal.device)
    history = _floating_tensor(history, (batch, 2, dim), "history", nominal.device)
    lengths = torch.as_tensor(prefix_lengths, device=nominal.device)
    if lengths.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("prefix_lengths must be an integer scalar or [B]")
    if lengths.ndim == 0:
        lengths = lengths.expand(batch)
    if lengths.shape != (batch,) or ((lengths < 0) | (lengths > horizon)).any():
        raise ValueError("prefix_lengths must be in [0,T] with shape [B]")
    valid = (torch.ones_like(nominal, dtype=torch.bool) if action_mask is None else
             _bool_tensor(action_mask, nominal.shape, "action_mask", nominal.device))
    known = (torch.ones(batch, 2, dtype=torch.bool, device=nominal.device) if history_valid is None else
             _bool_tensor(history_valid, (batch, 2), "history_valid", nominal.device))
    # An older command cannot be known while the immediately preceding one is absent.
    if (known[:, 0] & ~known[:, 1]).any():
        raise ValueError("history_valid must be chronological: [false,false], [false,true], or [true,true]")
    dims = torch.zeros(dim, dtype=torch.bool, device=nominal.device)
    dims[list(config.correction_dims)] = True
    suffix = torch.arange(horizon, device=nominal.device)[None, :, None] >= lengths[:, None, None]
    editable = valid & dims[None, None, :] & suffix
    dt = _timing(config.reference_dt if dt_seconds is None else dt_seconds,
                 (batch, horizon), "dt_seconds", nominal)
    history_dt = _timing(config.reference_dt if history_dt_seconds is None else history_dt_seconds,
                         (batch,), "history_dt_seconds", nominal)
    return nominal, history, valid, known, editable, dt, history_dt, dims


class ResidualMLP(nn.Module):
    """Two 256-unit hidden layers, deterministic autoregressive chunk correction.

    ``history[:, 0]`` is the older and ``history[:, 1]`` the latest committed
    action before the chunk. ``dt_seconds[:, t]`` is the interval from the
    preceding command to nominal token t; ``history_dt_seconds`` is the interval
    between the two history commands. Values are seconds, including at variable
    control rates. Missing bootstrap history is replaced by measured state.

    The nominal chunk must already contain the TTRTC committed prefix. Prefix,
    gripper/excluded dimensions, and invalid padded elements pass through
    exactly. State/history/nominal are detached; only MLP parameters train.

    Optional physical action bounds describe the frozen processor's representable
    domain. The learned head projects its editable proposed targets into that
    domain in both training and inference. Bounds are widened only to include
    the nominal value so zero initialization remains exact even when the base
    itself is out of bounds; the residual can move that value inward, not farther
    outward. This is part of the head, not a temporal filter or runtime gate.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ResidualMLPConfig()
        config = self.config
        self.register_buffer("center", torch.tensor(config.center, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(config.scale, dtype=torch.float32))
        self.register_buffer("action_lower", torch.tensor(config.action_lower, dtype=torch.float32)
                             if config.action_lower else None)
        self.register_buffer("action_upper", torch.tensor(config.action_upper, dtype=torch.float32)
                             if config.action_upper else None)
        self.network = nn.Sequential(
            nn.Linear(5 * config.action_dim + 3, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, config.action_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, nominal, state, history, *, prefix_lengths=0,
                action_mask=None, dt_seconds=None, history_dt_seconds=None, history_valid=None):
        nominal, history, valid, known, editable, dt, history_dt, _ = _context(
            nominal, history, self.config, prefix_lengths, action_mask,
            dt_seconds, history_dt_seconds, history_valid)
        batch, horizon, dim = nominal.shape
        state = _floating_tensor(state, (batch, dim), "state", nominal.device)
        if self.center.device != nominal.device:
            raise ValueError("model and actions must be on the same device")
        dtype = self.network[0].weight.dtype
        center, scale = self.center.to(dtype), self.scale.to(dtype)
        normalize = lambda values: (values.to(dtype) - center) / scale
        normalized = normalize(nominal)
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        present_dims = valid.any(dim=1)
        normalized_state = torch.where(present_dims, normalize(state), torch.zeros_like(state, dtype=dtype))
        previous = [torch.where(known[:, i, None] & present_dims, normalize(history[:, i]),
                                normalized_state) for i in range(2)]
        previous_dt = history_dt
        output = []
        for step in range(horizon):
            current = normalized[:, step]
            lookahead = normalized[:, min(step + 1, horizon - 1)]
            if step + 1 < horizon:
                lookahead = torch.where(valid[:, step + 1], lookahead, current)
            phase = nominal.new_full((batch, 1), step / max(1, horizon - 1)).to(dtype)
            features = torch.cat((current, lookahead, normalized_state, previous[0], previous[1],
                                  phase, (dt[:, step:step + 1] / self.config.reference_dt).to(dtype),
                                  (previous_dt[:, None] / self.config.reference_dt).to(dtype)), dim=-1)
            residual = self.config.max_residual * torch.tanh(self.network(features))
            proposed = nominal[:, step] + residual.to(nominal.dtype)
            if self.action_lower is not None:
                lower = torch.minimum(self.action_lower.to(nominal.dtype), nominal[:, step])
                upper = torch.maximum(self.action_upper.to(nominal.dtype), nominal[:, step])
                proposed = torch.clamp(proposed, min=lower, max=upper)
            corrected = torch.where(editable[:, step], proposed, nominal[:, step])
            output.append(corrected)
            # The same corrected/committed trajectory is used in training and runtime.
            latest = torch.where(valid[:, step], normalize(corrected), previous[1])
            previous = [previous[1], latest]
            previous_dt = dt[:, step]
        return torch.stack(output, dim=1)


def _masked_mean(value, mask):
    # torch.where avoids allowing invalid entries to influence denominators or gradients.
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def residual_mlp_loss(corrected, target, nominal, history, *, config,
                      prefix_lengths=0, action_mask=None, dt_seconds=None,
                      history_dt_seconds=None, history_valid=None, loss_config=None):
    """Supervised reconstruction plus GT boundary velocity/acceleration/jerk matching.

    Huber(delta=.1) reconstructs GT on valid editable joint elements; each term
    is independently divided by its valid-element count. Second/third physical
    time derivatives of normalized actions are multiplied by reference_dt**2
    and reference_dt**3 before squared matching. Thus all terms have normalized
    action units, and changing the sample rate does not introduce fps**6 loss
    growth. Smoothness matches demonstrated motion, never a zero-motion target.

    Boundary velocity matching is averaged separately over valid joint elements
    of the first editable token so its contribution is not diluted by the rest
    of the chunk. Its prediction starts at the actual committed predecessor;
    its GT velocity uses the recorded predecessor. For zero-delay chunks the
    latest known history command anchors both. Missing bootstrap history and
    invalid predecessor/current pairs do not enter this term.

    Derivative stencils can cross the immutable prefix/suffix boundary, but
    must be fully valid and touch an editable suffix element. Both reference
    and prediction use the actual committed prefix/history at that boundary.
    Prefix/history/nominal/target never receive gradients. The residual L2 term
    lightly favors the frozen policy when reconstruction does not require change.
    """
    loss_config = loss_config or ResidualLossConfig()
    nominal, history, valid, known, editable, dt, history_dt, dims = _context(
        nominal, history, config, prefix_lengths, action_mask,
        dt_seconds, history_dt_seconds, history_valid)
    target = _floating_tensor(target, nominal.shape, "target", nominal.device)
    # Validate without detaching the trainable output.
    _floating_tensor(corrected, nominal.shape, "corrected", nominal.device)
    scale = torch.tensor(config.scale, device=nominal.device, dtype=torch.float32)
    prediction = torch.where(editable, corrected.float(), nominal.float())
    reconstruction = _masked_mean(F.huber_loss(prediction / scale, target.float() / scale,
                                               reduction="none", delta=loss_config.huber_delta), editable)
    residual = _masked_mean(((prediction - nominal.float()) / scale).square(), editable)
    reference = torch.where(editable, target.float(), nominal.float())
    first = editable.any(dim=-1).to(torch.int64).argmax(dim=1)
    row = torch.arange(nominal.shape[0], device=nominal.device)
    preceding = (first - 1).clamp_min(0)
    has_prefix = first[:, None] > 0
    previous_prediction = torch.where(has_prefix, nominal[row, preceding].float(), history[:, 1].float())
    previous_target = torch.where(has_prefix, target[row, preceding].float(), history[:, 1].float())
    predecessor_valid = torch.where(has_prefix, valid[row, preceding], known[:, 1, None])
    boundary_mask = editable[row, first] & predecessor_valid
    predicted_step = prediction[row, first] - previous_prediction
    target_step = target[row, first].float() - previous_target
    boundary_error = (predicted_step - target_step) / scale * (config.reference_dt / dt[row, first, None])
    boundary_velocity = _masked_mean(boundary_error.square(), boundary_mask)
    predicted_path = torch.cat((history.float(), prediction), dim=1) / scale
    target_path = torch.cat((history.float(), reference), dim=1) / scale
    history_mask = known[:, :, None] & dims[None, None, :]
    stencil_valid = torch.cat((history_mask, valid & dims[None, None, :]), dim=1)
    touches_suffix = torch.cat((torch.zeros_like(history_mask), editable), dim=1)
    times = torch.cat(((-dt[:, :1] - history_dt[:, None]), -dt[:, :1],
                       dt.cumsum(dim=1) - dt[:, :1]), dim=1)
    derivatives = {}
    for order in range(1, 4):
        interval = (times[:, 1:] - times[:, :-1])[:, :, None]
        predicted_path = (predicted_path[:, 1:] - predicted_path[:, :-1]) / interval
        target_path = (target_path[:, 1:] - target_path[:, :-1]) / interval
        stencil_valid = stencil_valid[:, 1:] & stencil_valid[:, :-1]
        touches_suffix = touches_suffix[:, 1:] | touches_suffix[:, :-1]
        times = (times[:, 1:] + times[:, :-1]) / 2
        if order >= 2:
            error = (predicted_path - target_path) * config.reference_dt ** order
            derivatives[order] = _masked_mean(error.square(), stencil_valid & touches_suffix)
    total = (loss_config.reconstruction_weight * reconstruction
             + loss_config.boundary_weight * boundary_velocity
             + loss_config.acceleration_weight * derivatives[2]
             + loss_config.jerk_weight * derivatives[3]
             + loss_config.residual_weight * residual)
    return {"loss": total, "reconstruction": reconstruction, "boundary_velocity": boundary_velocity,
            "acceleration": derivatives[2], "jerk": derivatives[3],
            "residual_l2": residual, "valid_elements": editable.sum().detach()}
