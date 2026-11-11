"""TASK90000 section 7 loss, applied to a selected decoder's hidden features."""
import logging
import torch
import torch.nn.functional as F


def configure_vlm_last2(model, fp32=True):
    """Only the last two visual blocks and last two retained decoder blocks."""
    core = model.backbone.model.model
    if len(core.visual.blocks) < 2 or len(core.language_model.layers) < 2:
        raise ValueError("VLM last-two mode needs at least two blocks in each tower")
    model.requires_grad_(False)
    for block in [*core.visual.blocks[-2:], *core.language_model.layers[-2:]]:
        block.requires_grad_(True)
        if fp32:
            block.float()


def final_vlm_features(backbone, inputs):
    """Keep derivatives through frozen merger/LLM; skip language and action heads."""
    keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    return backbone.model.model(
        **{k: inputs[k] for k in keys if k in inputs},
        output_hidden_states=False, use_cache=False, return_dict=True,
    ).last_hidden_state


class _FeaturesCaptured(Exception):
    pass


def capture_layer_features(backbone, inputs, layer_index=11, first_trainable=None):
    """Frozen prefix; autograd only across selected contiguous language layers."""
    language = backbone.model.model.language_model
    last = layer_index % len(language.layers)
    first = last if first_trainable is None else first_trainable % len(language.layers)
    if first > last:
        raise ValueError("Invalid feature capture layer range")
    originals = [(language.layers[i], language.layers[i].forward) for i in range(first, last + 1)]
    # Transformers output_hidden_states wraps every decoder forward. Early exit
    # bypasses its normal cleanup, so restore the whole language subtree.
    restore = [(module, module.forward) for module in language.modules()]
    captured = []
    def wrap(original, final):
        def forward(*args, **kwargs):
            with torch.enable_grad():
                result = original(*args, **kwargs)
                if final:
                    hidden = result[0] if isinstance(result, tuple) else result
                    captured.append(language.norm(hidden))
            if final:
                raise _FeaturesCaptured()
            return result
        return forward
    for i, (layer, original) in enumerate(originals):
        layer.forward = wrap(original, i == len(originals) - 1)
    try:
        try:
            with torch.no_grad():
                backbone(inputs)
        except _FeaturesCaptured:
            pass
    finally:
        for layer, original in restore:
            layer.forward = original
    if len(captured) != 1:
        raise RuntimeError("Selected decoder layer did not produce features")
    return captured[0]


def instruction_indices(ids, valid, vision_end_id, message_end_id):
    """Single-image user prompt: instruction after vision_end and before im_end."""
    ends = ((ids == vision_end_id) & valid).nonzero().flatten()
    if len(ends) != 1:
        raise ValueError("Feature segmentation expects one image per prompt")
    start = int(ends[0]) + 1
    stops = ((ids == message_end_id) & valid).nonzero().flatten()
    stops = stops[stops >= start]
    if not len(stops):
        raise ValueError("Instruction closing token missing")
    positions = torch.arange(start, int(stops[0]), device=ids.device)
    positions = positions[valid[positions]]
    if not len(positions):
        raise ValueError("Empty instruction span")
    return positions


def hybrid_terms(similarity, coverage, mode="threshold"):
    """One camera, section 7.2--7.8. Return None for invalid nonempty masks."""
    s, m = similarity.float(), coverage.float()
    if s.shape != m.shape or not s.numel():
        raise ValueError("Feature/mask shape mismatch")
    if mode not in ("threshold", "overlap"):
        raise ValueError(mode)
    z = 10 * s
    zero = s.sum() * 0
    fg = m >= .5 if mode == "threshold" else m > 0
    bg = ~fg
    nfg, nbg = int(fg.sum()), int(bg.sum())
    if not bool(m.sum() > 0):
        bce = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
        peak = z.sigmoid().topk(max(1, round(.1 * len(z)))).values.mean()
        return dict(bce=bce, spatial_ce=zero, margin=zero, background=peak,
                    segmentation=bce + .2 * peak)
    if nfg < 1 or nbg < 1:
        return None
    target = m if mode == "threshold" else fg.float()
    ratio = min(max(nbg, 1) / max(nfg, 1), 20.)
    weight = torch.where(fg, ratio, 1.)
    bce = F.binary_cross_entropy_with_logits(z, target, weight=weight)
    spatial = -(m / m.sum() * F.log_softmax(z / .5, dim=-1)).sum()
    margin = F.relu(s[bg].mean() - s[fg].mean() + .1) - .1 * s[fg].mean()
    peak = z[bg].sigmoid().topk(max(1, round(.1 * nbg))).values.mean()
    return dict(bce=bce, spatial_ce=spatial, margin=margin, background=peak,
                segmentation=bce + .5 * spatial + .2 * margin + .2 * peak)


def upper_background_attention(attention_rows, targets, grids, merge_size):
    """Penalize upper background mass; any positive target overlap is exempt.

    Merged image tokens are row-major. Position weights decrease linearly from
    1 at the top row to 0 at the bottom, in the actual cropped/resized image grid.
    Do not renormalize attention over background: this is weighted image mass.
    """
    if len(attention_rows) != len(targets) or len(grids) != len(targets):
        raise ValueError("Attention/target/grid batch mismatch")
    terms = []
    for attention, target, grid in zip(attention_rows, targets, grids):
        t, gh, gw = map(int, grid)
        if t != 1 or gh % merge_size or gw % merge_size:
            raise ValueError("Invalid image grid")
        height, width = gh // merge_size, gw // merge_size
        coverage = target[target >= 0].to(attention.device)
        if attention.numel() != height * width or coverage.numel() != attention.numel():
            raise ValueError("Attention/target/grid size mismatch")
        weights = torch.linspace(1., 0., height, device=attention.device).repeat_interleave(width)
        terms.append((attention.float() * weights * (coverage == 0)).sum())
    if not terms:
        raise ValueError("Empty attention batch")
    return torch.stack(terms).mean()


def feature_hybrid_loss(model, inputs, targets):
    supervisor = model.attn_supervisor
    try:
        supervisor.begin(inputs)  # Validate image token grid and identify image indices.
        keys = supervisor.keys
    finally:
        supervisor.end()  # Do not compute an additional Q/K attention map.
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not hasattr(model, "_feature_anchor"):
        # Capture after checkpoint loading, not in __init__ before weights are restored.
        model._feature_anchor = {n: p.detach().float().clone() for n, p in params.items()}
    last2 = getattr(model.config, "language_last2_feature_only", False)
    upper_weight = getattr(model.config, "upper_background_attention_weight", 0.0)
    if upper_weight:
        # Capture Q/K from the SAME differentiable VLM forward as hybrid features.
        supervisor.begin(inputs)
    try:
        if getattr(model.config, "vlm_last2_feature_only", False) or getattr(model.config, "tune_visual", False):
            hidden = final_vlm_features(model.backbone, inputs)
        else:
            hidden = capture_layer_features(model.backbone, inputs, -1 if last2 else 11, -2 if last2 else None)
        if upper_weight:
            if len(supervisor.layers) != 1 or set(supervisor.received) != set(supervisor.layers):
                raise RuntimeError("Upper-background supervision requires one captured attention layer")
            upper_loss = upper_background_attention(
                supervisor.received[supervisor.layers[0]], targets,
                inputs["image_grid_thw"], supervisor.merge_size)
    finally:
        if upper_weight:
            supervisor.end()
            supervisor.received = {}
    tokenizer = model.collator.processor.tokenizer
    end_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    end_message = tokenizer.convert_tokens_to_ids('<|im_end|>')
    terms = []
    for b, image_keys in enumerate(keys):
        text = instruction_indices(inputs['input_ids'][b], inputs['attention_mask'][b].bool(), end_image, end_message)
        image = F.normalize(hidden[b, image_keys].float(), dim=-1)
        instruction = F.normalize(hidden[b, text].float().mean(0), dim=-1)
        coverage = targets[b][targets[b] >= 0].to(hidden.device)
        values = hybrid_terms(image @ instruction, coverage, mode="threshold")
        if values is not None:
            terms.append(values)
    if not terms:
        # Section 7.8: invalid cameras contribute no segmentation objective.
        # Keep a zero autograd path so other accumulated microbatches retain
        # their gradients; never reinterpret a small positive mask as background.
        zero = hidden.float().sum() * 0
        means = dict.fromkeys(("bce", "spatial_ce", "margin", "background", "segmentation"), zero)
        model._invalid_feature_batches = getattr(model, "_invalid_feature_batches", 0) + 1
        if model._invalid_feature_batches <= 3 or model._invalid_feature_batches % 100 == 0:
            valid = targets >= 0
            logging.getLogger(__name__).warning(
                "Skipping segmentation for all-invalid microbatch #%d: foreground_counts=%s coverage_max=%s",
                model._invalid_feature_batches,
                ((targets >= .5) & valid).sum(-1).tolist(),
                targets.masked_fill(~valid, 0).max(-1).values.tolist())
    else:
        means = {k: torch.stack([v[k] for v in terms]).mean() for k in terms[0]}
    anchor = torch.stack([(p.float() - model._feature_anchor[n]).square().mean() for n, p in params.items()]).mean()
    loss = means['segmentation'] + 1e-4 * anchor
    if upper_weight:
        loss = loss + upper_weight * upper_loss
    result = {'loss': loss, 'loss_total': loss, 'loss_bce': means['bce'],
            'loss_spatial_ce': means['spatial_ce'], 'loss_margin': means['margin'],
            'loss_background': means['background'], 'loss_anchor': anchor}
    if upper_weight:
        result['loss_upper_background_attention'] = upper_loss
    return result
