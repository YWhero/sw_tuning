import torch
import torch.nn.functional as F
from gr00t.model.modules.feature_segmentation import hybrid_terms, instruction_indices, capture_layer_features
from test_qwen_last_qk_kl import build
from types import SimpleNamespace


def test_section7_terms_match_independent_formula():
    s=torch.tensor([.2,-.1,.5,-.3],requires_grad=True);m=torch.tensor([.8,.1,1.,0.]);z=10*s
    t=hybrid_terms(s,m);fg=m>=.5;bg=~fg
    bce=-(torch.where(fg,1.,1.)*(m*F.logsigmoid(z)+(1-m)*F.logsigmoid(-z))).mean()
    ce=-(m/m.sum()*F.log_softmax(20*s,dim=0)).sum()
    margin=torch.relu(s[bg].mean()-s[fg].mean()+.1)-.1*s[fg].mean()
    peak=torch.sigmoid(z[bg]).max()
    torch.testing.assert_close(t['segmentation'],bce+.5*ce+.2*margin+.2*peak)
    t['segmentation'].backward();assert torch.isfinite(s.grad).all()


def test_balancing_and_empty_overlap():
    s=torch.linspace(-.4,.4,30);m=torch.zeros(30);m[0]=.7
    t=hybrid_terms(s,m);weight=torch.ones(30);weight[0]=20
    torch.testing.assert_close(t['bce'],F.binary_cross_entropy_with_logits(10*s,m,weight=weight))
    m[0]=.1;t=hybrid_terms(s,m,'overlap');target=(m>0).float()
    torch.testing.assert_close(t['bce'],F.binary_cross_entropy_with_logits(10*s,target,weight=weight))
    e=hybrid_terms(s,torch.zeros_like(s));assert e['spatial_ce']==0 and e['margin']==0
    torch.testing.assert_close(e['segmentation'],F.softplus(10*s).mean()+.2*(10*s).sigmoid().topk(3).values.mean())
    assert hybrid_terms(s,torch.ones_like(s)) is None


def test_instruction_span_excludes_image_and_control_tokens():
    ids=torch.tensor([1,5,5,7,9,10,8,0]);valid=ids!=0
    assert instruction_indices(ids,valid,7,8).tolist()==[4,5]


def test_feature_capture_gradients_and_early_exit():
    model,sup,inputs,forward=build(layers=16)
    class Backbone:
        def __init__(self):self.model=SimpleNamespace(model=SimpleNamespace(language_model=model))
        def __call__(self,x):return forward(x)
    def forbidden(*a):raise AssertionError('layer 12 ran')
    h=model.layers[12].register_forward_pre_hook(forbidden)
    output=capture_layer_features(Backbone(),inputs)
    s=F.cosine_similarity(output[0,1:5].float(),output[0,5:7].float().mean(0),dim=-1)
    loss=hybrid_terms(s,torch.tensor([1.,0.,0.,0.]))['segmentation'];loss.backward()
    grads=[n for n,p in model.named_parameters() if p.grad is not None]
    assert len(grads)==4 and all('layers.11.self_attn.' in n for n in grads)
    h.remove();sup.close()


def test_named_mask_array(tmp_path):
    import numpy as np
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    p=tmp_path/'m.npz';np.savez(p,arr_0=np.ones((2,2,2)),object_1=np.zeros((2,2,2)))
    masks=LeRobotEpisodeLoader._load_mask_file(None,p,np.array([0]),array_key='object_1')
    assert masks.sum()==0


def test_two_full_layers_receive_gradients_and_prefix_is_frozen():
    model,sup,inputs,forward=build(layers=16)
    model.requires_grad_(False)
    for layer in model.layers[-2:]:layer.requires_grad_(True)
    class Backbone:
        def __init__(self):self.model=SimpleNamespace(model=SimpleNamespace(language_model=model))
        def __call__(self,x):return forward(x)
    output=capture_layer_features(Backbone(),inputs,-1,-2)
    s=F.cosine_similarity(output[0,1:5].float(),output[0,5:7].float().mean(0),dim=-1)
    hybrid_terms(s,torch.tensor([1.,0.,0.,0.]))['segmentation'].backward()
    for n,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),n
        else:assert p.grad is None,n
    sup.close()


def test_all_invalid_batch_has_zero_segmentation_contribution(monkeypatch):
    from gr00t.model.modules import feature_segmentation as module
    hidden=torch.randn(1,7,8,requires_grad=True)
    class Supervisor:
        keys=[torch.tensor([1,2,3,4])]
        def begin(self,inputs):pass
        def end(self):pass
    model=torch.nn.Linear(8,8)
    model.attn_supervisor=Supervisor()
    model.config=SimpleNamespace(language_last2_feature_only=False)
    model.backbone=None
    model.collator=SimpleNamespace(processor=SimpleNamespace(tokenizer=SimpleNamespace(
        convert_tokens_to_ids=lambda token:7 if token=='<|vision_end|>' else 8)))
    monkeypatch.setattr(module,'capture_layer_features',lambda *args:hidden)
    inputs={'input_ids':torch.tensor([[5,5,5,5,7,9,8]]),'attention_mask':torch.ones(1,7)}
    result=module.feature_hybrid_loss(model,inputs,torch.full((1,4),.2))
    assert result['loss']==0
    result['loss'].backward()
    assert hidden.grad is not None and hidden.grad.count_nonzero()==0


def test_capture_restores_transformers_hidden_state_wrappers():
    model,sup,inputs,_=build(layers=16)
    class Backbone:
        def __init__(self):self.model=SimpleNamespace(model=SimpleNamespace(language_model=model))
        def __call__(self,x):return model(input_ids=x['input_ids'],attention_mask=x['attention_mask'],use_cache=False,output_hidden_states=True)
    originals={n:m.forward for n,m in model.named_modules()}
    try:
        for _ in range(400):
            capture_layer_features(Backbone(),inputs)
            for n,m in model.named_modules():
                assert m.forward==originals[n],n
    finally:sup.close()
