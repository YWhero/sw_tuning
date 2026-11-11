from types import SimpleNamespace
import torch
import pytest
from gr00t.model.modules.feature_segmentation import hybrid_terms


@pytest.mark.parametrize("whole_visual", [False, True])
def test_vision_gradient_crosses_frozen_bridge(whole_visual):
    from gr00t.model.modules.feature_segmentation import configure_vlm_last2, final_vlm_features
    class VLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual=torch.nn.Module()
            self.visual.blocks=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(24)])
            self.visual.merger=torch.nn.Linear(8,8)
            self.language_model=torch.nn.Module()
            self.language_model.layers=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(16)])
        def forward(self,pixel_values,**kwargs):
            x=pixel_values
            for layer in self.visual.blocks:x=x+torch.tanh(layer(x))
            x=self.visual.merger(x)
            for layer in self.language_model.layers:x=x+torch.tanh(layer(x))
            return SimpleNamespace(last_hidden_state=x)
    torch.manual_seed(2)
    vlm=VLM();model=torch.nn.Module();model.backbone=SimpleNamespace(model=SimpleNamespace(model=vlm))
    model.vlm=vlm
    if whole_visual:
        model.requires_grad_(False)
        vlm.visual.requires_grad_(True)
    else:
        configure_vlm_last2(model,fp32=True)
    originals={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    output=final_vlm_features(model.backbone,{'pixel_values':torch.randn(1,7,8)})
    s=torch.nn.functional.cosine_similarity(output[0,:4],output[0,4:].mean(0),dim=-1)
    hybrid_terms(s,torch.tensor([1.,0.,0.,0.]))['segmentation'].backward()
    for name,p in model.named_parameters():
        expected=any(k in name for k in ['visual.blocks.22.','visual.blocks.23.','language_model.layers.14.','language_model.layers.15.'])
        if whole_visual:expected=name.startswith("vlm.visual.")
        assert p.requires_grad==expected,name
        if expected:assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
        else:assert p.grad is None,name
    opt.step()
    for n,p in model.named_parameters():
        if n in originals:assert torch.equal(p,originals[n])


def test_visual_received_attention_matches_matrix_and_raster_order():
    from gr00t.model.modules.vision_attention_maps import received_attention
    from gr00t.model.modules.attn_supervision import vision_token_order
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    torch.manual_seed(7)
    attn=SimpleNamespace(qkv=torch.nn.Linear(8,24),num_heads=2,scaling=.5)
    hidden=torch.randn(32,8);cos=torch.randn(32,4).cos();sin=torch.randn(32,4).sin()
    grids=torch.tensor([[1,4,4],[1,4,4]])
    maps=received_attention(attn,hidden,torch.tensor([0,16,32]),(cos,sin),grids,2)
    q,k,_=attn.qkv(hidden).reshape(32,3,2,4).permute(1,0,2,3).unbind(0)
    q,k=apply_rotary_pos_emb_vision(q,k,cos,sin)
    for b,m in enumerate(maps):
        sl=slice(16*b,16*(b+1))
        a=((q[sl].transpose(0,1)@k[sl].transpose(0,1).transpose(-1,-2))*.5).softmax(-1).mean((0,1))
        expected=torch.empty(16);expected[vision_token_order(4,4,2)]=a
        torch.testing.assert_close(m,expected.reshape(4,4))
        torch.testing.assert_close(m.sum(),torch.tensor(1.))
    modified=hidden.clone();modified[16:]*=100
    unchanged=received_attention(attn,modified,torch.tensor([0,16,32]),(cos,sin),grids,2)[0]
    torch.testing.assert_close(maps[0],unchanged)


def test_real_qwen_vision_capture_is_repeatable():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from gr00t.model.modules.vision_attention_maps import extract_vision_attention
    cfg=Qwen3VLVisionConfig(depth=24,hidden_size=16,intermediate_size=32,num_heads=2,patch_size=2,temporal_patch_size=1,spatial_merge_size=2,out_hidden_size=16,num_position_embeddings=16,deepstack_visual_indexes=[5,11,17])
    cfg._attn_implementation='eager'
    visual=Qwen3VLVisionModel(cfg).eval()
    pixels=torch.randn(16,12);grid=torch.tensor([[1,4,4]])
    with torch.no_grad():
        first=extract_vision_attention(visual,pixels,grid,23)[0]
        second=extract_vision_attention(visual,pixels,grid,23)[0]
    torch.testing.assert_close(first,second)
    assert first.shape==(4,4) and torch.isfinite(first).all()
    assert not visual.blocks[23].attn._forward_pre_hooks


def test_visual_hybrid_routes_to_features_without_action(monkeypatch):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    import gr00t.model.modules.feature_segmentation as feature
    targets=torch.ones(1,4)
    called=[]
    def loss(model,inputs,received):
        assert received is targets
        called.append(True)
        return {'loss':torch.tensor(1.)}
    monkeypatch.setattr(feature,'feature_hybrid_loss',loss)
    def prepare(inputs,include_action=True):
        assert include_action is False
        return {},None
    model=SimpleNamespace(training=True,config=SimpleNamespace(tune_visual=True,attn_supervision_loss='feature_hybrid',vision_kl_only=False),attn_supervisor=object(),prepare_input=prepare,backbone=SimpleNamespace(eval=lambda:None),action_head=SimpleNamespace(eval=lambda:None))
    result=Gr00tN1d7.forward(model,{'attn_targets':targets})
    assert called and result['loss']==1
