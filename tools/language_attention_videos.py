"""Render language attention, final-feature cosine, or native vision self-attention."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

import cv2
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401 (AutoModel registration)
from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.modules.attn_supervision import LanguageAttnSupervisor
from gr00t.model.modules.feature_segmentation import capture_layer_features, instruction_indices
from gr00t.model.modules.vision_overlay import load_visual_overlay, load_vlm_overlay
from labeling_storage import Dataset, atomic_json, read_json
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
from transformers.feature_extraction_utils import BatchFeature


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def extract_many(rgbs, instructions, backbone, processor, supervisor, formalize=True):
    images = [Image.fromarray(rgb) for rgb in rgbs]
    prompts = []
    for image, instruction in zip(images, instructions):
        if formalize:
            instruction = re.sub(r"[^\w\s]", "", instruction.lower())
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        prompts.append(
            processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        )
    inputs = processor(
        text=prompts, images=images, padding=True, return_tensors="pt",
        do_resize=any(rgb.shape[0] % (processor.image_processor.patch_size * processor.image_processor.merge_size)
                      or rgb.shape[1] % (processor.image_processor.patch_size * processor.image_processor.merge_size)
                      for rgb in rgbs)
    ).to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    if getattr(supervisor, 'map_type', 'attention') == 'vision_attention':
        from gr00t.model.modules.vision_attention_maps import extract_vision_attention
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            maps = extract_vision_attention(backbone.model.model.visual, inputs['pixel_values'], inputs['image_grid_thw'], supervisor.vision_layer)
        result = [m.float().cpu().numpy() for m in maps]
        if any(not np.isfinite(m).all() or not np.isclose(m.sum(),1,atol=1e-4) for m in result):
            raise ValueError('Invalid visual attention map')
        return result
    if getattr(supervisor, 'map_type', 'attention') == 'feature_cosine':
        supervisor.begin(inputs)
        keys = supervisor.keys
        supervisor.end()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = capture_layer_features(backbone, BatchFeature(dict(inputs)), supervisor.layers[0])
            result = []
            tokenizer = processor.tokenizer
            for b, image_keys in enumerate(keys):
                text = instruction_indices(inputs['input_ids'][b], inputs['attention_mask'][b].bool(),
                    tokenizer.convert_tokens_to_ids('<|vision_end|>'), tokenizer.convert_tokens_to_ids('<|im_end|>'))
                image = torch.nn.functional.normalize(hidden[b, image_keys].float(), dim=-1)
                instruction = torch.nn.functional.normalize(hidden[b, text].float().mean(0), dim=-1)
                values = (10 * (image @ instruction)).sigmoid()
                t, gh, gw = map(int, inputs['image_grid_thw'][b].tolist())
                if t != 1 or not torch.isfinite(values).all():
                    raise ValueError('Invalid cosine feature map')
                result.append(values.float().cpu().numpy().reshape(gh // supervisor.merge_size, gw // supervisor.merge_size))
        return result
    supervisor.begin(inputs)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            backbone(BatchFeature(dict(inputs)))
    finally:
        supervisor.end()
    result = []
    for row, grid in zip(supervisor.received[supervisor.layers[0]], inputs["image_grid_thw"]):
        t, gh, gw = map(int, grid.tolist())
        attention = row.float().cpu().numpy()
        if (
            t != 1
            or not np.isfinite(attention).all()
            or not np.isclose(attention.sum(), 1, atol=1e-4)
        ):
            raise ValueError("Invalid attention probability map")
        result.append(attention.reshape(gh // supervisor.merge_size, gw // supervisor.merge_size))
    return result


def extract(rgb, instruction, backbone, processor, supervisor, formalize=True):
    return extract_many([rgb], [instruction], backbone, processor, supervisor, formalize)[0]


def panel(rgb, grid, episode, frame, instruction, map_type="attention"):
    # Preserve the input aspect ratio, including unpadded rectangular crops.
    size = (512, max(2, round(512 * rgb.shape[0] / rgb.shape[1] / 2) * 2))
    display = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size)
    density = grid if map_type == "feature_cosine" else np.clip(grid * grid.size / 4.0, 0, 1)
    heat = cv2.applyColorMap(
        (cv2.resize(density, size, interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    overlay = cv2.addWeighted(display, 0.5, heat, 0.5, 0)
    image = cv2.copyMakeBorder(
        np.concatenate([display, overlay], axis=1),
        56,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.putText(
        image,
        f"Episode {episode:06d} | Frame {frame + 1} | {instruction}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Input | Feature sigmoid(10*cosine), fixed [0,1]" if map_type == "feature_cosine" else ("Input | Vision self-attention: all queries / heads (red >= 4x uniform)" if map_type == "vision_attention" else "Input | LLM attention (red >= 4x uniform)"),
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return image


def review_panel(rgb, grid, mask, episode, frame, map_type="feature_cosine"):
    display = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (512,512))
    mask = cv2.resize(mask.astype(np.uint8), (512,512), interpolation=cv2.INTER_NEAREST)>0
    gt = display.copy()
    gt[mask] = (gt[mask]*.4 + np.array([40,230,40])*.6).astype(np.uint8)
    density = grid if map_type == "feature_cosine" else np.clip(grid*grid.size/4,0,1)
    heat = cv2.applyColorMap((cv2.resize(density,(512,512),interpolation=cv2.INTER_NEAREST)*255).clip(0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    overlay=cv2.addWeighted(display,.5,heat,.5,0)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(255,255,255),1)
    for panel_,title in zip([display,gt,heat,overlay],['RGB','GT target (object_1)',('Final hidden: sigmoid(10*cosine)' if map_type == 'feature_cosine' else 'Vision L23: mean over queries / heads'),'Score overlay / white = GT boundary']):
        cv2.rectangle(panel_,(0,0),(512,30),(15,15,15),-1)
        cv2.putText(panel_,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    canvas=np.concatenate([np.concatenate([display,gt],axis=1),np.concatenate([heat,overlay],axis=1)],axis=0)
    footer=np.zeros((64,1024,3),dtype=np.uint8)
    bar=cv2.applyColorMap(np.linspace(0,255,500,dtype=np.uint8)[None,:].repeat(16,axis=0),cv2.COLORMAP_TURBO)
    footer[35:51,15:515]=bar
    cv2.putText(footer,f'Episode {episode} | frame {frame} | '+('score: 0 to 1' if map_type == 'feature_cosine' else 'attention density: 0 to 4x uniform'),(15,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.putText(footer,f'Native token grid: {grid.shape[1]} x {grid.shape[0]}',(540,47),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return np.concatenate([canvas,footer],axis=0)


def render(
    dataset,
    episode,
    out,
    backbone,
    processor,
    transform,
    supervisor,
    formalize,
    batch_size=1,
    verify_batch=False,
):
    review_masks = None
    if getattr(supervisor, "review_target", False):
        if getattr(transform, 'mask_transforms', None):
            raise ValueError('Review must not alter RGB using masks')
        with np.load(dataset.dataset_mask_path(episode)) as archive:
            review_masks = archive['object_1']
        if len(review_masks) != dataset.episode(episode)['length']:
            raise ValueError('Mask/video frame count mismatch')
    video = dataset.video(episode)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video: {video}")
    path = out / f"episode_{episode:06d}_language_attention.mp4"
    encoder = None
    grids = []
    count = 0
    try:
        verified = False
        while True:
            rgbs, instructions, frame_masks = [], [], []
            for _ in range(batch_size):
                ok, bgr = cap.read()
                if not ok:
                    break
                instruction = dataset.instruction(episode, count + len(rgbs))
                source = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if review_masks is None:
                    tensors, _ = apply_with_replay(transform, [source])
                else:
                    tensors, _, transformed = apply_with_replay(transform, [source], masks=[review_masks[count+len(rgbs)]], return_masks=True)
                    frame_masks.append(transformed[0])
                rgb = tensors[0].permute(1, 2, 0).numpy()
                rgbs.append(rgb)
                instructions.append(instruction)
            if not rgbs:
                break
            maps = extract_many(rgbs, instructions, backbone, processor, supervisor, formalize)
            if verify_batch and not verified and len(rgbs) > 1:
                delta = max(
                    float(
                        np.max(
                            np.abs(
                                grid
                                - extract(
                                    rgb, instruction, backbone, processor, supervisor, formalize
                                )
                            )
                        )
                    )
                    for rgb, instruction, grid in zip(rgbs, instructions, maps)
                )
                if delta > 0.005:
                    raise ValueError(f"Batched/single-frame attention differs: {delta}")
                print(f"Batch verification max probability delta: {delta:.6f}", flush=True)
                verified = True
            for batch_index, (rgb, instruction, grid) in enumerate(zip(rgbs, instructions, maps)):
                image = panel(rgb, grid, episode, count, instruction, getattr(supervisor, "map_type", "attention"))
                if review_masks is not None:
                    image = review_panel(rgb, grid, frame_masks[batch_index], episode, count, getattr(supervisor, "map_type", "feature_cosine"))
                if encoder is None:
                    h, w = image.shape[:2]
                    encoder = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-n",
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            "bgr24",
                            "-s",
                            f"{w}x{h}",
                            "-r",
                            str(fps),
                            "-i",
                            "pipe:0",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-crf",
                            "20",
                            "-pix_fmt",
                            "yuv420p",
                            "-threads",
                            "2",
                            "-movflags",
                            "+faststart",
                            str(path),
                        ],
                        stdin=subprocess.PIPE,
                    )
                    cv2.imwrite(str(path.with_suffix(".png")), image)
                encoder.stdin.write(np.ascontiguousarray(image).tobytes())
                grids.append(grid)
                count += 1
                if count % 20 == 0:
                    print(
                        f"Episode {episode}: {count}/{dataset.episode(episode)['length']}",
                        flush=True,
                    )
        if not encoder or count != dataset.episode(episode)["length"]:
            raise ValueError("Incomplete source video")
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError("FFmpeg failed")
        stream = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,codec_name,pix_fmt,duration",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["streams"][0]
        if (
            int(stream["nb_read_frames"]) != count
            or stream["codec_name"] != "h264"
            or stream["pix_fmt"] != "yuv420p"
        ):
            raise ValueError("Encoded video validation failed")
        np.savez_compressed(path.with_suffix(".npz"), attention=np.stack(grids))
        return {
            "episode": episode,
            "frames": count,
            "fps": fps,
            "video": path.name,
            "model_input_hw": list(rgb.shape[:2]),
            "attention_grid_hw": list(grid.shape),
            "segments": dataset.episode(episode).get("sub_tasks", []),
            "tasks": dataset.episode(episode).get("tasks", []),
        }
    except Exception:
        for artifact in (path, path.with_suffix(".png"), path.with_suffix(".npz")):
            artifact.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-target", action="store_true", help="Four-panel target-mask review for final feature cosine or vision attention")
    parser.add_argument("--map-type", choices=["attention", "feature_cosine", "vision_attention"], default="attention")
    parser.add_argument("--vision-layer", type=int, default=23)
    parser.add_argument("--base", required=True)
    parser.add_argument("--vision-weights")
    parser.add_argument("--vlm-weights", help="Full backbone safetensors, including vision and LLM")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attention-layer", type=int, default=None,
                        help="Decoder index; defaults to 11 for Q/K-only checkpoints, otherwise last layer")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verify-batch", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    dataset = Dataset(args.dataset)
    model_path = args.checkpoint or args.base
    model, loading = AutoModel.from_pretrained(
        model_path,
        tune_visual=False,
        tune_llm=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
        attn_supervision_layers=None,
        load_bf16=True,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise RuntimeError(f"Full GR00T checkpoint mismatch: {loading}")
    if args.vlm_weights and args.vision_weights:
        raise ValueError("Choose VLM or legacy visual weights, not both")
    if args.vlm_weights:
        load_vlm_overlay(model.backbone, args.vlm_weights)
        args.vision_weights = args.vlm_weights  # Shared provenance handling below.
    elif args.vision_weights:
        load_visual_overlay(model.backbone.model.model.visual, args.vision_weights)
    backbone = model.backbone.to("cuda").eval()
    # Geometry follows the vision phase; no GT masks enter this forward.
    provenance = (
        read_json(Path(args.vision_weights).with_suffix(".json"), {}) if args.vision_weights else {}
    )
    geometry = provenance.get("processor", {}).get("processor_kwargs", {})
    if args.vision_weights and not provenance:
        raise ValueError(
            "Visual overlay provenance is missing; extract it from its full checkpoint first"
        )
    recorded_base = provenance.get("run", {}).get("base")
    if recorded_base and recorded_base != args.base:
        raise ValueError(f"Overlay base mismatch: recorded {recorded_base}, selected {args.base}")
    if geometry:
        allowed = (
            "image_crop_size",
            "image_target_size",
            "shortest_image_edge",
            "crop_fraction",
            "letter_box_transform",
            "use_albumentations",
            "formalize_language",
        )
        overrides = {key: geometry[key] for key in allowed if key in geometry}
    elif args.checkpoint:
        overrides = {}
    else:
        overrides = dict(
            use_albumentations=True,
            shortest_image_edge=256,
            crop_fraction=0.9,
            image_crop_size=None,
            image_target_size=None,
            letter_box_transform=False,
        )
    data_processor = Gr00tN1d7Processor.from_pretrained(
        model_path, emit_attn_targets=False, **overrides
    )
    processor = data_processor.collator.processor
    attention_layer = args.attention_layer
    if attention_layer is None:
        attention_layer = 11 if getattr(model.config, "qwen_layer11_qk_kl_only", False) else -1
    supervisor = LanguageAttnSupervisor(
        backbone.model.model.language_model,
        [attention_layer],
        backbone.model.config.image_token_id,
        backbone.model.model.visual.spatial_merge_size,
    )
    if args.review_target and not (args.map_type == "vision_attention" or (args.map_type == "feature_cosine" and attention_layer == -1)):
        raise ValueError("Target review requires vision_attention or final feature_cosine")
    supervisor.vision_layer = args.vision_layer
    supervisor.review_target = args.review_target
    supervisor.map_type = args.map_type
    out.mkdir(parents=True)
    records = []
    try:
        for episode in args.episodes:
            records.append(
                render(
                    dataset,
                    episode,
                    out,
                    backbone,
                    processor,
                    data_processor.eval_image_transform,
                    supervisor,
                    data_processor.formalize_language,
                    args.batch_size,
                    args.verify_batch,
                )
            )
    finally:
        supervisor.close()
    local_model = Path(model_path)
    if not local_model.is_dir():
        from transformers.utils import cached_file

        index_file = cached_file(model_path, "model.safetensors.index.json", local_files_only=True)
        local_model = Path(index_file).parent
    weights = (
        {p.name: sha256(p) for p in sorted(local_model.glob("*.safetensors"))}
        if local_model.is_dir()
        else {}
    )
    manifest = {
        "base": args.base,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": weights,
        "overlay": args.vision_weights,
        "overlay_sha256": sha256(args.vision_weights) if args.vision_weights else None,
        "processor": {
            "shortest_image_edge": data_processor.shortest_image_edge,
            "crop_fraction": data_processor.crop_fraction,
            "letter_box": data_processor.letter_box_transform,
            "formalize_language": data_processor.formalize_language,
        },
        "instruction_override": False,
        "review_target": args.review_target,
        "mask_array": "object_1" if args.review_target else None,
        "inference_batch_size": args.batch_size,
        "attention_module": "vision" if args.map_type == "vision_attention" else "language",
        "attention_layer": args.vision_layer if args.map_type == "vision_attention" else supervisor.layers[0],
        "map_type": args.map_type,
        "color_scale": "sigmoid(10*cosine), fixed [0,1]" if args.map_type == "feature_cosine" else "clip(attention * token_count / 4, 0, 1)",
        "attention": "normalized image feature dot normalized mean instruction feature" if args.map_type == "feature_cosine" else ("all visual queries and heads averaged, per-image noncausal softmax; native patch grid" if args.map_type == "vision_attention" else "last valid prompt query; per-head image-conditional softmax then head mean"),
        "episodes": records,
    }
    atomic_json(out / "manifest.json", manifest)
    page = '<!doctype html><meta charset="utf-8"><title>Language attention</title><style>body{background:#12181e;color:white;font-family:system-ui}video{max-width:100%}</style>'
    for record in records:
        page += f'<h2>Episode {record["episode"]:06d}</h2><video controls preload="metadata" src="{html.escape(record["video"])}"></video>'
    (out / "index.html").write_text(page)
    print(f"Validated videos: {out}")


if __name__ == "__main__":
    main()
