import types
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from gr00t.model.modules.attn_supervision import LanguageAttnSupervisor


def build(layers=12):
    torch.manual_seed(13)
    cfg=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=layers,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2]})
    cfg._attn_implementation='eager'
    model=Qwen3VLTextModel(cfg).eval().requires_grad_(False)
    for name in ['q_proj','k_proj','q_norm','k_norm']:
        getattr(model.layers[11].self_attn,name).requires_grad_(True)
    supervisor=LanguageAttnSupervisor(model,[11],image_token_id=5,merge_size=2)
    inputs={'input_ids':torch.tensor([[1,5,5,5,5,2,3]]),'attention_mask':torch.ones(1,7,dtype=torch.long),'image_grid_thw':torch.tensor([[1,4,4]])}
    def forward(x):
        return model(input_ids=x['input_ids'],attention_mask=x['attention_mask'],use_cache=False)
    return model,supervisor,inputs,forward


def test_fast_path_matches_full_attention_and_skips_downstream():
    model,sup,inputs,forward=build()
    sup.begin(inputs)
    forward(inputs)
    sup.end()
    expected=sup.received[11][0].detach().clone()
    calls=[]
    def forbidden(*args,**kwargs):
        raise AssertionError('Unused downstream computation ran')
    handles=[model.layers[11].self_attn.v_proj.register_forward_pre_hook(forbidden),model.layers[11].mlp.register_forward_pre_hook(forbidden)]
    handles.append(model.layers[0].register_forward_pre_hook(lambda *a:calls.append(torch.is_grad_enabled())))
    sup.forward_qk_only(forward,inputs)
    torch.testing.assert_close(sup.received[11][0],expected)
    assert calls==[False]
    target=torch.tensor([[1.,0.,0.,0.]])
    loss=sup.loss(target);loss.backward()
    changed=[]
    for name,p in model.named_parameters():
        if p.grad is not None:
            assert name.startswith('layers.11.self_attn.')
            assert any('.'+part+'.' in name for part in ['q_proj','k_proj','q_norm','k_norm'])
            assert torch.isfinite(p.grad).all()
            changed.append(name)
    assert len(changed)==4
    assert not sup.active and not sup.qk_only
    for h in handles:h.remove()
    sup.close()


def test_optimizer_reduces_kl_and_frozen_weights_stay_identical():
    model,sup,inputs,forward=build()
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optim=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
    values=[]
    for _ in range(12):
        optim.zero_grad();sup.forward_qk_only(forward,inputs);loss=sup.loss(torch.tensor([[1.,0.,0.,0.]]));values.append(loss.item());loss.backward();optim.step()
    assert values[-1]<values[0]*.5
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n])
    sup.close()


def test_failed_forward_resets_capture_mode():
    model,sup,inputs,forward=build()
    import pytest
    with pytest.raises(RuntimeError,match='test failure'):
        sup.forward_qk_only(lambda _: (_ for _ in ()).throw(RuntimeError('test failure')),inputs)
    assert not sup.active and not sup.qk_only
    sup.forward_qk_only(forward,inputs)
    assert sup.loss(torch.tensor([[1.,0.,0.,0.]])).requires_grad
    sup.close()


def test_groot_forward_never_prepares_or_runs_action():
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    model,sup,inputs,forward=build()
    class Backbone:
        def eval(self):return self
        def __call__(self,x):return forward(x)
    class Action:
        def eval(self):return self
        def prepare_input(self,*a):raise AssertionError('action input preparation')
        def __call__(self,*a):raise AssertionError('action forward')
    def prepare(x,include_action=True):
        assert include_action is False
        return x,{}
    fake=types.SimpleNamespace(training=True,config=types.SimpleNamespace(qwen_layer11_qk_kl_only=True,attn_supervision_loss="kl"),attn_supervisor=sup,backbone=Backbone(),action_head=Action(),prepare_input=prepare)
    result=Gr00tN1d7.forward(fake,{**inputs,'attn_targets':torch.tensor([[1.,0.,0.,0.]])})
    assert 'loss_action' not in result
    result['loss'].backward()
    sup.close()


def test_layer11_early_exit_preserves_16_layer_model():
    model,sup,inputs,forward=build(layers=16)
    def forbidden(*args):raise AssertionError('layer 12 must not run')
    h=model.layers[12].register_forward_pre_hook(forbidden)
    sup.forward_qk_only(forward,inputs)
    assert len(model.layers)==16
    sup.loss(torch.tensor([[1.,0.,0.,0.]])).backward()
    h.remove();sup.close()


def test_kl_early_exit_restores_hidden_state_capture_wrappers():
    model,sup,inputs,_=build(layers=16)
    original={n:m.forward for n,m in model.named_modules()}
    def forward(x):
        return model(input_ids=x['input_ids'],attention_mask=x['attention_mask'],use_cache=False,output_hidden_states=True)
    try:
        for _ in range(4):
            sup.forward_qk_only(forward,inputs)
            for n,m in model.named_modules():assert m.forward==original[n],n
    finally:sup.close()
