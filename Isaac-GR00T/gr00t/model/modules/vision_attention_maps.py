"""Actual visual self-attention, averaged over heads and all image queries."""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
from .attn_supervision import vision_token_order


def received_attention(attention, hidden, cu_seqlens, position_embeddings, grids, merge_size):
    q,k,_=attention.qkv(hidden).reshape(len(hidden),3,attention.num_heads,-1).permute(1,0,2,3).unbind(0)
    q,k=apply_rotary_pos_emb_vision(q,k,*position_embeddings)
    boundaries=cu_seqlens.tolist()
    if len(boundaries)!=len(grids)+1:
        raise ValueError('Vision attention rendering requires one temporal frame per image')
    result=[]
    for i,grid in enumerate(grids.tolist()):
        t,h,w=map(int,grid)
        start,end=boundaries[i:i+2]
        if t!=1 or end-start!=h*w:
            raise ValueError('Vision token grid mismatch')
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            scores=q[start:end].transpose(0,1).float()@k[start:end].transpose(0,1).float().transpose(-1,-2)
            weights=(scores*attention.scaling).softmax(-1).mean((0,1))
        raster=torch.empty_like(weights)
        raster[vision_token_order(h,w,merge_size).to(weights.device)]=weights
        result.append(raster.reshape(h,w))
    return result


def extract_vision_attention(visual, pixel_values, grids, layer=-1):
    collected=[]
    def capture(module,args,kwargs):
        hidden=args[0] if args else kwargs['hidden_states']
        collected.extend(received_attention(module,hidden,kwargs['cu_seqlens'],kwargs['position_embeddings'],grids,visual.spatial_merge_size))
    handle=visual.blocks[layer].attn.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        visual(pixel_values,grid_thw=grids)
    finally:
        handle.remove()
    if len(collected)!=len(grids):
        raise RuntimeError('Visual attention capture missing')
    return collected
