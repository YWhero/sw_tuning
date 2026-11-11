import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file, load_file
from extract_vlm import extract
from gr00t.model.modules.vision_overlay import load_vlm_overlay


def backbone():
    b=torch.nn.Module(); b.model=torch.nn.Module(); b.model.model=torch.nn.Module()
    b.model.model.visual=torch.nn.Linear(3,3)
    b.model.model.language_model=torch.nn.Linear(3,3)
    b.model.lm_head=torch.nn.Linear(3,2)
    return b


@pytest.mark.parametrize('sharded',[False,True])
def test_full_vlm_roundtrip_and_frozen_action_training(tmp_path,sharded):
    source=backbone(); ckpt=tmp_path/'ckpt';ckpt.mkdir()
    state={'backbone.'+k:v for k,v in source.state_dict().items()}
    state['action_head.weight']=torch.zeros(2,3)
    if sharded:
        files={};keys=list(state)
        for i,subset in enumerate([keys[:3],keys[3:]]):
            name=f'model-{i}.safetensors';save_file({k:state[k] for k in subset},ckpt/name)
            files.update({k:name for k in subset})
        (ckpt/'model.safetensors.index.json').write_text(json.dumps({'weight_map':files}))
    else:save_file(state,ckpt/'model.safetensors')
    output=tmp_path/'run-vlm.safetensors';extract(ckpt,output)
    assert set(load_file(output))==set(state)-{'action_head.weight'}
    target=backbone().requires_grad_(False);load_vlm_overlay(target,output)
    for k,v in source.state_dict().items():torch.testing.assert_close(target.state_dict()[k],v)
    before={k:v.clone() for k,v in target.state_dict().items()}
    action=torch.nn.Linear(3,1);optim=torch.optim.SGD(action.parameters(),lr=.1)
    old=action.weight.detach().clone()
    action(target.model.model.language_model(target.model.model.visual(torch.ones(2,3)))).square().mean().backward();optim.step()
    assert not torch.equal(old,action.weight)
    assert all(p.grad is None for p in target.parameters())
    for k,v in before.items():torch.testing.assert_close(target.state_dict()[k],v,rtol=0,atol=0)
    with pytest.raises(FileExistsError):extract(ckpt,output)


def test_partial_or_wrong_shape_rejected_before_mutation(tmp_path):
    target=backbone();before={k:v.clone() for k,v in target.state_dict().items()}
    weights={'backbone.'+k:v.clone() for k,v in target.state_dict().items()}
    for mode in ['visual_only','shape','action']:
        bad=dict(weights)
        if mode=='visual_only':bad={k:v for k,v in bad.items() if '.visual.' in k}
        elif mode=='shape':bad['backbone.model.model.language_model.weight']=torch.ones(2,2)
        else:bad['action_head.weight']=torch.ones(2,2)
        path=tmp_path/f'{mode}.safetensors';save_file(bad,path)
        with pytest.raises(ValueError):load_vlm_overlay(target,path)
        for k,v in before.items():torch.testing.assert_close(target.state_dict()[k],v,rtol=0,atol=0)


def test_only_real_tied_aliases_may_be_omitted(tmp_path):
    b=backbone()
    b.model.lm_head=b.model.model.language_model
    weights={'backbone.'+k:v.clone() for k,v in b.state_dict().items() if not k.startswith('model.lm_head.')}
    file=tmp_path/'tied.safetensors';save_file(weights,file)
    load_vlm_overlay(b,file)
    with pytest.raises(ValueError):load_vlm_overlay(backbone(),file)
    weights['backbone.model.lm_head.weight']=torch.zeros_like(b.model.lm_head.weight)+99
    save_file(weights,file)
    with pytest.raises(ValueError,match='Conflicting tied'):load_vlm_overlay(b,file)
