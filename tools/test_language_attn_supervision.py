"""CPU-safe contract tests; real Qwen3-VL hooks, no downloaded weights."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Isaac-GR00T"))
from gr00t.model.modules.attn_supervision import (
    LanguageAttnSupervisor,
    last_valid_queries,
    mask_to_targets,
)
from gr00t.model.modules.vision_overlay import PREFIX, load_visual_overlay
import pytest
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


def model():
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3VLTextModel(config)


def inputs():
    return {
        "input_ids": torch.tensor([[0, 0, 1, 7, 7, 7, 7, 2, 3], [1, 7, 7, 7, 7, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1], [1] * 9]),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
    }


def test_targets_token_order_and_temporal_rejection():
    masks = torch.zeros(2, 8, 8)
    masks[0, :4, :4] = 1
    masks[1, 4:, 4:] = 1
    assert torch.equal(
        mask_to_targets(masks, torch.tensor([[1, 4, 4]] * 2)),
        torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]),
    )
    with pytest.raises(ValueError):
        mask_to_targets(masks[:1], torch.tensor([[2, 4, 4]]))
    with pytest.raises(ValueError):
        mask_to_targets(masks[:1], torch.tensor([[1, 3, 4]]))


def test_last_query_left_and_right_padding():
    assert last_valid_queries(torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])).tolist() == [3, 1]


def test_hook_matches_eager_head_conditionals_and_gradient():
    torch.manual_seed(5)
    llm = model().eval()
    llm.requires_grad_(False)
    data = inputs()
    sup = LanguageAttnSupervisor(llm, [-1], 7, 2)
    embedded = llm.embed_tokens(data["input_ids"]).detach().requires_grad_(True)
    sup.begin(data)
    result = llm(
        inputs_embeds=embedded,
        attention_mask=data["attention_mask"],
        output_attentions=True,
        use_cache=False,
    )
    sup.end()
    for b, keys in enumerate(sup.keys):
        eager = result.attentions[-1][b, :, sup.queries[b], keys]
        expected = (eager / eager.sum(-1, keepdim=True)).mean(0)
        torch.testing.assert_close(sup.received[sup.layers[0]][b], expected, atol=2e-6, rtol=2e-5)
    targets = torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 0]])
    loss = sup.loss(targets)
    loss.backward()
    assert embedded.grad is not None and embedded.grad.abs().sum() > 0
    assert all(p.grad is None for p in llm.parameters())
    with pytest.raises(ValueError):
        sup.loss(torch.zeros(2, 4))
    sup.close()


def test_strict_overlay_roundtrip(tmp_path):
    from safetensors.torch import save_file

    visual = torch.nn.Linear(4, 3)
    path = tmp_path / "visual.safetensors"
    save_file({PREFIX + k: v for k, v in visual.state_dict().items()}, str(path))
    copy = torch.nn.Linear(4, 3)
    load_visual_overlay(copy, path)
    torch.testing.assert_close(copy.weight, visual.weight)
    save_file({PREFIX + "weight": visual.weight}, str(path))
    with pytest.raises(ValueError):
        load_visual_overlay(copy, path)


def test_geometric_mask_alignment():
    from gr00t.model.gr00t_n1d7.image_augmentations import (
        apply_with_replay,
        build_image_transformations_albumentations,
    )
    import numpy as np
    from PIL import Image

    _, transform = build_image_transformations_albumentations(
        None, None, None, None, 256, 1.0, letter_box_transform=True
    )
    mask = np.zeros((60, 100), np.uint8)
    mask[5:20, 5:20] = 1
    rgb = np.repeat((mask * 255)[..., None], 3, axis=2)
    images, _, masks = apply_with_replay(
        transform, [Image.fromarray(rgb)], [mask], return_masks=True
    )
    assert images[0].shape[1:] == masks[0].shape == (256, 256)
    assert np.mean((images[0][0].numpy() > 127) == (masks[0] > 0)) > 0.99
    assert not masks[0][:30].any()



def test_export_contains_only_visual_weights_and_loads_strictly(tmp_path):
    import json
    from safetensors.torch import save_file, load_file
    from extract_vision import extract

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    visual = torch.nn.Linear(4, 3)
    state = {PREFIX + k: v for k, v in visual.state_dict().items()}
    state["action_head.weight"] = torch.zeros(2, 2)
    state["backbone.language_model.weight"] = torch.zeros(3, 3)
    save_file(state, str(checkpoint / "model.safetensors"))
    output = tmp_path / "vision/run-vision.safetensors"
    extract(checkpoint, output)
    assert set(load_file(output)) == {PREFIX + "weight", PREFIX + "bias"}
    restored = torch.nn.Linear(4, 3)
    load_visual_overlay(restored, output)
    torch.testing.assert_close(restored.weight, visual.weight)
    torch.testing.assert_close(restored.bias, visual.bias)
    assert json.loads(output.with_suffix('.json').read_text())["tensors"] == 2
    assert (checkpoint / "model.safetensors").exists()
    with pytest.raises(FileExistsError):
        extract(checkpoint, output)
