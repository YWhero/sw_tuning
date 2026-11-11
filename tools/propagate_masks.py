"""SAM2 point propagation from arbitrary anchor, review artifacts on CPU."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import cv2
from labeling_storage import atomic_json, atomic_npz, read_json, object_prompts, object_signature
import numpy as np


def check_seed_points(seed, objects):
    mismatches = []
    for obj in seed['objects']:
        for prompt in object_prompts(seed, obj):
            mask = objects[str(obj['id'])][prompt['frame']]
            for i, ((x, y), expected) in enumerate(zip(prompt['points'], prompt['labels'])):
                actual = int(mask[int(y), int(x)])
                if actual != expected:
                    mismatches.append({'object_id':obj['id'], 'name':obj['name'], 'frame':prompt['frame'],
                                       'point_index':i, 'point':[x,y], 'expected':expected, 'actual':actual})
    warning = f"예시 포인트 {len(mismatches)}개 불일치 · 마스크 검토/포인트 수정 필요" if mismatches else ''
    return {'frame':seed['frame'], 'mismatches':mismatches, 'warning':warning}


def load_predictor(runtime):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime.get("gpu", 0))
    source, dependencies = runtime.get("source"), runtime.get("dependencies")
    for path in (source, dependencies):
        if path:
            sys.path.insert(0, path)
    from sam2.build_sam import build_sam2_video_predictor
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    free = torch.cuda.mem_get_info()[0] // 1024**2
    if free < runtime.get("min_free_gpu_mib", 10000):
        raise RuntimeError(f"Insufficient free GPU memory: {free} MiB")
    if runtime.get("checkpoint_sha256"):
        import hashlib

        hasher = hashlib.sha256()
        with open(runtime["checkpoint"], "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != runtime["checkpoint_sha256"]:
            raise ValueError("SAM2 checkpoint hash differs from labeling runtime")
    return build_sam2_video_predictor(
        runtime["model_config"], runtime["checkpoint"], device="cuda"
    )


def run(job, predictor=None):
    directory = Path(job["directory"])
    seed = read_json(directory / "seed.json")
    progress = directory / "progress.json"
    cap = cv2.VideoCapture(job["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video")
    with tempfile.TemporaryDirectory(prefix="sam2-frames-", dir=directory) as temporary:
        frames = np.lib.format.open_memmap(
            directory / "frames.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(job["frames"], seed["height"], seed["width"], 3),
        )
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if count >= len(frames) or frame.shape != frames.shape[1:]:
                    raise ValueError("Video metadata/resolution mismatch")
                frames[count] = frame
                cv2.imwrite(str(Path(temporary) / f"{count:06d}.jpg"), frame)
                count += 1
        finally:
            cap.release()
        if count != job["frames"]:
            raise ValueError("Metadata/video frame count mismatch")
        if any(f.shape[:2] != (seed["height"], seed["width"]) for f in frames):
            raise ValueError("Video resolution changed")
        if job.get("correction"):
            from temporal_mask_eraser import erase_component

            with np.load(directory / "object_masks.npz", allow_pickle=False) as data:
                objects = {key: data[key].copy() for key in data.files}
            correction = job["correction"]
            objects = erase_component(
                objects,
                frames,
                correction["frame"],
                correction["point"],
                correction["scope"],
                lambda i: atomic_json(
                    progress,
                    {
                        "status": "running",
                        "completed": i,
                        "total": job["frames"],
                        "stage": "correcting",
                    },
                ),
            )
        else:
            import torch
            shape = (len(frames), seed['height'], seed['width'])
            objects, reused, changed = {}, [], []
            base = Path(job['base_directory']) if job.get('base_directory') else None
            base_seed = read_json(base/'seed.json') if base else None
            old_objects = {o['id']:o for o in base_seed['objects']} if base_seed else {}
            if base:
                with np.load(base/'object_masks.npz', allow_pickle=False) as masks:
                    for obj in seed['objects']:
                        old = old_objects.get(obj['id'])
                        if old and object_signature(seed,obj) == object_signature(base_seed,old):
                            array = masks[str(obj['id'])]
                            if array.shape != shape:
                                raise ValueError('Existing mask dimensions differ')
                            objects[str(obj['id'])] = array
                            reused.append(obj['id'])
            changed = [obj for obj in seed['objects'] if obj['id'] not in reused]
            if changed and predictor is None:
                predictor = load_predictor(job['runtime'])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                state = None
                for obj in changed:
                    if state is None:
                        state = predictor.init_state(video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=True)
                    else:
                        predictor.reset_state(state)
                    track = np.lib.format.open_memmap(directory/f"track-{obj['id']}.npy", mode='w+', dtype=np.uint8, shape=shape)
                    objects[str(obj['id'])] = track
                    prompts = sorted(object_prompts(seed,obj), key=lambda p:p['frame'])
                    for prompt in prompts:
                        predictor.add_new_points_or_box(state, frame_idx=prompt['frame'], obj_id=obj['id'],
                            points=np.asarray(prompt['points'], dtype=np.float32), labels=np.asarray(prompt['labels'], dtype=np.int32))
                    anchor = prompts[0]['frame']
                    seen = set()
                    for reverse in (False,True):
                        for index, ids, logits in predictor.propagate_in_video(state,start_frame_idx=anchor,reverse=reverse):
                            for oid, mask in zip(ids,(logits>0).cpu().numpy()):
                                objects[str(oid)][index] = mask.reshape(shape[1:]).astype(np.uint8)
                            seen.add(index)
                            atomic_json(progress, {'status':'running','completed':len(seen),'total':len(frames),
                                                  'stage':'propagating', 'label':obj['name'], 'reused_objects':reused})
                    if seen != set(range(len(frames))):
                        raise RuntimeError('SAM2 did not produce every frame')
            atomic_json(directory/'incremental.json', {'base_job_id':job.get('base_job_id'), 'reused_objects':reused,
                                                     'generated_objects':[o['id'] for o in changed],
                                                     'deleted_objects':sorted(set(old_objects)-{o['id'] for o in seed['objects']})})
        validation = check_seed_points(seed, objects)
        atomic_json(directory / "seed_validation.json", validation)
        union = np.zeros((len(frames), seed["height"], seed["width"]), dtype=np.uint8)
        for track in objects.values():
            np.maximum(union, track, out=union)
        atomic_npz(directory / "object_masks.npz", **objects)
        atomic_npz(directory / "masks.npz", arr_0=union)
        if job["runtime"].get("render_review_video", True):
            render_review(frames, union, fps, directory / "review.mp4", progress)
        atomic_json(
            directory / "annotations.json",
            {
                "schema_version": 2,
                "frames": len(frames),
                "objects": seed["objects"],
                "target_only": len({obj["name"] for obj in seed["objects"]}) == 1,
            },
        )
        atomic_json(
            progress,
            {
                "status": "awaiting_review",
                "completed": len(frames),
                "total": len(frames),
                "stage": "review ready",
                "warning": validation["warning"],
            },
        )
        del frames
        (directory / "frames.npy").unlink(missing_ok=True)
        for path in directory.glob("track-*.npy"):
            path.unlink()


def render_review(frames, masks, fps, path, progress):
    height, width = frames[0].shape[:2]
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
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            image = frame.copy()
            image[mask > 0] = (image[mask > 0] * 0.5 + np.array([70, 220, 100]) * 0.5).astype(
                np.uint8
            )
            encoder.stdin.write(image.tobytes())
            atomic_json(
                progress,
                {
                    "status": "running",
                    "stage": "encoding review",
                    "completed": i + 1,
                    "total": len(frames),
                },
            )
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError("Review encoder failed")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    try:
        run(job)
    except Exception as exc:
        atomic_json(
            Path(job["directory"]) / "progress.json", {"status": "failed", "error": str(exc)}
        )
        traceback.print_exc()
        sys.exit(1)
