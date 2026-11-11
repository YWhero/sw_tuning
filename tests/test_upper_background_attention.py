import pytest
import torch
from gr00t.model.modules.feature_segmentation import upper_background_attention


def test_upper_background_penalty_and_mask_exemption():
    # A rectangular 3x2 merged grid: upper / middle / lower rows.
    logits = torch.zeros(6, requires_grad=True)
    coverage = torch.tensor([0., .01, 0., 1., 0., 0.])
    loss = upper_background_attention([logits.softmax(0)], coverage[None],
                                      torch.tensor([[1, 6, 4]]), 2)
    torch.testing.assert_close(loss, torch.tensor(1.5 / 6))
    loss.backward()
    assert logits.grad[0] > 0  # Gradient descent suppresses upper background.
    assert logits.grad[1] < 0  # Any target overlap is exempt, even at the top.
    assert logits.grad[-1] < 0  # Bottom row has no location penalty.


def test_full_target_is_zero_and_padding_is_ignored():
    a = torch.full((6,), 1/6, requires_grad=True)
    targets = torch.tensor([[1., 1., 1., 1., 1., 1., -1.]])
    loss = upper_background_attention([a], targets, torch.tensor([[1, 6, 4]]), 2)
    loss.backward()
    assert loss == 0 and a.grad.count_nonzero() == 0


def test_rejects_geometry_mismatch():
    with pytest.raises(ValueError):
        upper_background_attention([torch.ones(4)/4], torch.zeros(1,4),
                                   torch.tensor([[1,6,4]]), 2)


def test_penalty_uses_same_forward_and_backpropagates_through_frozen_llm(monkeypatch):
    from types import SimpleNamespace
    from test_qwen_last_qk_kl import build
    from gr00t.model.modules.attn_supervision import LanguageAttnSupervisor
    from gr00t.model.modules import feature_segmentation as module
    language, old_sup, _, _ = build(layers=16)
    old_sup.close()
    language.requires_grad_(False)
    sup = LanguageAttnSupervisor(language, [15], image_token_id=5, merge_size=2)
    model = torch.nn.Module()
    model.visual = torch.nn.Linear(32,32)
    model.language = language
    model.attn_supervisor = sup
    model.config = SimpleNamespace(tune_visual=True, upper_background_attention_weight=.1)
    model.backbone = None
    model.collator = SimpleNamespace(processor=SimpleNamespace(tokenizer=SimpleNamespace(
        convert_tokens_to_ids=lambda x: 2 if x=='<|vision_end|>' else 3)))
    inputs = {'input_ids':torch.tensor([[1,5,5,5,5,2,9,3]]),
              'attention_mask':torch.ones(1,8,dtype=torch.long),
              'image_grid_thw':torch.tensor([[1,4,4]])}
    embedded = language.embed_tokens(inputs['input_ids']).detach()
    calls = []
    def features(backbone, inputs):
        calls.append(1)
        return language(inputs_embeds=model.visual(embedded),
                        attention_mask=inputs['attention_mask'], use_cache=False).last_hidden_state
    monkeypatch.setattr(module, 'final_vlm_features', features)
    result = module.feature_hybrid_loss(model, inputs, torch.tensor([[0.,1.,0.,0.]]))
    assert len(calls)==1 and not sup.active and not sup.received
    base = (result['loss_bce'] + .5*result['loss_spatial_ce'] + .2*result['loss_margin']
            + .2*result['loss_background'] + 1e-4*result['loss_anchor'])
    torch.testing.assert_close(result['loss'], base+.1*result['loss_upper_background_attention'])
    result['loss_upper_background_attention'].backward()
    assert model.visual.weight.grad.norm() > 0
    assert all(p.grad is None for p in language.parameters())
    sup.close()
