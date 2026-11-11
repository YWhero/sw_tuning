"""KL-only must not evaluate the action head or require action targets."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from torch import nn
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config


def fake_model(kl_only=True):
    model = Gr00tN1d7.__new__(Gr00tN1d7)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_kl_only=kl_only, attn_supervision_loss='kl',
                                   attn_supervision_margin=.9, attn_supervision_weight=1.)
    model.prepare_input = Mock(return_value=({}, {}))
    model.backbone = Mock(return_value={})
    model.action_head = Mock(return_value={'loss': torch.tensor(3., requires_grad=True)})
    model.attn_supervisor = Mock()
    model.attn_supervisor.loss.return_value = torch.tensor(2., requires_grad=True)
    return model


def test_kl_only_skips_action_head_and_backpropagates_kl():
    model = fake_model()
    result = model({'attn_targets': torch.ones(1, 4)})
    model.action_head.assert_not_called()
    assert isinstance(result, dict)  # transformers.Trainer indexes dict losses by name.
    assert 'loss_action' not in result
    assert result['loss'] is result['loss_kl']
    result['loss'].backward()
    assert model.attn_supervisor.loss.return_value.grad.item() == 1


def test_kl_only_requires_masks_without_action_fallback():
    model = fake_model()
    with pytest.raises(ValueError, match='mask|target|supervision'):
        model({})
    model.action_head.assert_not_called()


def test_default_keeps_combined_objective():
    model = fake_model(False)
    result = model({'attn_targets': torch.ones(1, 4)})
    model.action_head.assert_called_once()
    assert result['loss'].item() == 5


def test_kl_only_setting_is_serialized():
    config = Gr00tN1d7Config(vision_kl_only=True)
    assert config.to_dict()['vision_kl_only'] is True


@pytest.mark.parametrize("layer", [11, 15, -1])
def test_visual_kl_layer_selection_preserves_frozen_llm_gradient_path(monkeypatch, layer):
    """Run GR00T's constructor/forward with a tiny, real 16-block Qwen on CPU."""
    import importlib
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(7)
    text_config = Qwen3VLTextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=16,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    text_config._attn_implementation = "eager"

    class TinyBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs["tune_visual"] and not kwargs["tune_llm"]
            assert kwargs["select_layer"] == 16
            self.model = nn.Module()
            self.model.config = SimpleNamespace(image_token_id=5)
            self.model.model = nn.Module()
            self.model.model.visual = nn.Sequential(nn.Linear(8, 32), nn.Linear(32, 32))
            self.model.model.visual.spatial_merge_size = 2
            self.model.model.language_model = Qwen3VLTextModel(text_config).eval()
            self.model.model.language_model.requires_grad_(False)

        def forward(self, inputs):
            core = self.model.model
            embeddings = core.language_model.embed_tokens(inputs["input_ids"])
            image_features = core.visual(inputs["pixels"])
            embeddings = embeddings.masked_scatter(
                (inputs["input_ids"] == 5).unsqueeze(-1), image_features
            )
            return core.language_model(
                inputs_embeds=embeddings, attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

    model_module = importlib.import_module("gr00t.model.gr00t_n1d7.gr00t_n1d7")
    processor_module = importlib.import_module("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
    monkeypatch.setattr(model_module, "get_backbone_cls", lambda config: TinyBackbone)
    monkeypatch.setattr(model_module, "Gr00tN1d7ActionHead",
                        lambda config: nn.Linear(1, 1).requires_grad_(False))
    monkeypatch.setattr(processor_module, "Gr00tN1d7DataCollator", lambda **kwargs: None)
    config = Gr00tN1d7Config(
        select_layer=16, tune_visual=True, tune_llm=False, tune_projector=False,
        tune_diffusion_model=False, tune_vlln=False, vision_kl_only=True,
        attn_supervision_layers=[layer], attn_supervision_loss="kl",
        attn_supervision_weight=1.0,
    )
    model = Gr00tN1d7(config)
    model.prepare_input = lambda inputs: (inputs, {})
    def reject_action(*args):
        raise AssertionError("KL-only must not run the action head")
    handle = model.action_head.register_forward_pre_hook(reject_action)
    try:
        result = model({
            "input_ids": torch.tensor([[1, 5, 5, 5, 5, 2, 3]]),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "pixels": torch.randn(4, 8),
            "attn_targets": torch.tensor([[1., 0., 0., 0.]]),
        })
        assert "loss_action" not in result
        result["loss"].backward()
        assert model.attn_supervisor.layers == [layer if layer >= 0 else 15]
        assert len(model.backbone.model.model.language_model.layers) == 16
        assert config.to_dict()["attn_supervision_layers"] == [layer]
        for name, parameter in model.named_parameters():
            if name.startswith("backbone.model.model.visual."):
                assert parameter.requires_grad
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
            else:
                assert not parameter.requires_grad and parameter.grad is None
    finally:
        handle.remove()
        model.attn_supervisor.close()
