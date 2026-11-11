"""Strict, complete visual overlay shared by training and rendering."""

from safetensors.torch import load_file


PREFIX = "backbone.model.model.visual."


def load_visual_overlay(visual, path):
    weights = load_file(str(path))
    if not weights or any(not k.startswith(PREFIX) for k in weights):
        raise ValueError("Expected visual-only safetensors with canonical GR00T prefixes")
    state = {k[len(PREFIX) :]: v for k, v in weights.items()}
    reference = visual.state_dict()
    if set(state) != set(reference):
        raise ValueError(
            f"Overlay key mismatch: missing={set(reference) - set(state)}, extra={set(state) - set(reference)}"
        )
    for key, value in state.items():
        if value.shape != reference[key].shape:
            raise ValueError(f"Overlay shape mismatch: {key}")
    visual.load_state_dict(state, strict=True)
    return len(state)


def load_vlm_overlay(backbone, path):
    """Validate every backbone tensor before loading; never load action weights."""
    prefix = "backbone."
    weights = load_file(str(path))
    if not weights or any(not k.startswith(prefix) for k in weights):
        raise ValueError("Expected full VLM safetensors with backbone.* keys")
    state = {k[len(prefix):]: v for k, v in weights.items()}
    reference = backbone.state_dict()
    # HF safetensors omits duplicate tied parameters. Restore ONLY aliases that
    # are the very same Parameter in the loaded model, never arbitrary missing keys.
    groups = {}
    for key, parameter in backbone.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(key)
    for aliases in groups.values():
        present = [key for key in aliases if key in state]
        if not present:
            continue
        value = state[present[0]]
        for key in present[1:]:
            import torch
            if not torch.equal(value, state[key]):
                raise ValueError(f"Conflicting tied VLM weights: {present}")
        for key in aliases:
            if key not in state:
                state[key] = value
    if set(state) != set(reference):
        raise ValueError(f"VLM overlay key mismatch: missing={set(reference)-set(state)}, extra={set(state)-set(reference)}")
    for key, value in state.items():
        if value.shape != reference[key].shape:
            raise ValueError(f"VLM overlay shape mismatch: {key}")
    backbone.load_state_dict(state, strict=True)
    return len(state)
