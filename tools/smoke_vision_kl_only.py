"""One bounded real-GR00T step; does not save/modify user checkpoints."""

import json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from transformers import AutoModel
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.modules.attn_supervision import mask_to_targets

base = "nvidia/GR00T-N1.7-3B"
model, loading = AutoModel.from_pretrained(
    base,
    local_files_only=True,
    tune_visual=True,
    tune_llm=False,
    tune_projector=False,
    tune_diffusion_model=False,
    tune_vlln=False,
    attn_supervision_layers=[15],
    vision_kl_only=True,
    attn_supervision_weight=1.0,
    attn_supervision_loss="kl",
    state_dropout_prob=0,
    load_bf16=True,
    torch_dtype=torch.bfloat16,
    backbone_trainable_params_fp32=True,
    output_loading_info=True,
)
assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), (
    loading
)
model = model.cuda().train()
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
assert trainable and all(n.startswith("backbone.model.model.visual.") for n, p in trainable)
print(
    "Trainable tensors:",
    len(trainable),
    "LLM layers:",
    len(model.backbone.model.model.language_model.layers),
    flush=True,
)
image = Image.fromarray(np.full((256, 256, 3), 100, np.uint8))
processor = model.collator.processor
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "pick up the red object"},
        ],
    }
]
prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
batch = dict(processor(text=[prompt], images=[image], return_tensors="pt", do_resize=False))
mask = torch.zeros(1, 256, 256)
mask[:, 70:160, 80:160] = 1
batch.update(
    state=torch.zeros(1, 1, model.config.max_state_dim),
    action=torch.zeros(1, model.config.action_horizon, model.config.max_action_dim),
    action_mask=torch.ones(1, model.config.action_horizon, model.config.max_action_dim),
    embodiment_id=torch.zeros(1, dtype=torch.long),
    attn_targets=mask_to_targets(
        mask, batch["image_grid_thw"], model.backbone.model.model.visual.spatial_merge_size
    ),
)
optimizer = torch.optim.AdamW([p for n, p in trainable], lr=1e-5)
before = trainable[0][1].detach().clone()
frozen = model.backbone.model.model.language_model.layers[-1].self_attn.q_proj.weight
frozen_before = frozen.detach().clone()
def reject_action_forward(*args, **kwargs):
    raise AssertionError("KL-only training called action head")
action_guard = model.action_head.register_forward_pre_hook(reject_action_forward)
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(dict(batch))
action_guard.remove()
assert "loss_action" not in outputs and outputs["loss"] is outputs["loss_kl"]
# Each loss independently reaches visual parameters through frozen modules.
for key in ("loss_kl",):
    gradients = torch.autograd.grad(
        outputs[key], [p for n, p in trainable], retain_graph=True, allow_unused=True
    )
    magnitude = sum(g.detach().float().abs().sum().item() for g in gradients if g is not None)
    assert magnitude > 0, (key, magnitude)
    print(key, float(outputs[key].detach()), "visual gradient L1", magnitude, flush=True)
outputs["loss"].backward()
assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
grad_norm = torch.nn.utils.clip_grad_norm_([p for n, p in trainable], 1.0)
optimizer.step()
assert not torch.equal(before, trainable[0][1])
assert torch.equal(frozen_before, frozen)
assert not model.backbone.model.model.language_model.training
assert not model.action_head.model.training
result = {
    "trainable_tensors": len(trainable),
    "action_head_called": False,
    "loss_kl": outputs["loss_kl"].item(),
    "loss_total": outputs["loss"].item(),
    "grad_norm": float(grad_norm),
    "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
}
path = Path(__file__).resolve().parents[1] / "results/validation/vision_kl_only_smoke.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2))
print(result, flush=True)

# Save/reload the real visual tower and verify identical deterministic attention.
import tempfile
from safetensors.torch import save_file
from gr00t.model.modules.vision_overlay import load_visual_overlay, PREFIX

model.eval()
inputs, _ = model.prepare_input(dict(batch))
supervisor = model.attn_supervisor
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    supervisor.begin(inputs)
    model.backbone(inputs)
    supervisor.end()
    original_attention = supervisor.received[supervisor.layers[0]][0].clone()
with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
    overlay = Path(temporary) / "visual.safetensors"
    visual = model.backbone.model.model.visual
    save_file(
        {
            PREFIX + key: value.detach().cpu().contiguous()
            for key, value in visual.state_dict().items()
        },
        str(overlay),
    )
    with torch.no_grad():
        next(visual.parameters()).zero_()
    load_visual_overlay(visual, overlay)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        supervisor.begin(inputs)
        model.backbone(inputs)
        supervisor.end()
        restored_attention = supervisor.received[supervisor.layers[0]][0]
    torch.testing.assert_close(original_attention, restored_attention, atol=1e-5, rtol=1e-4)
print("Real visual overlay attention roundtrip: PASS", flush=True)
