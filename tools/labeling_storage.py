"""LeRobot episode reading and atomic, revisioned labeling storage."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

import cv2
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.fchmod(fd, 0o644)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def object_prompts(seed, obj):
    """Read legacy single-frame seeds and multi-frame refinements identically."""
    return obj.get('prompts', [{'frame':obj.get('frame', seed['frame']),
                                'points':obj.get('points', []), 'labels':obj.get('labels', [])}])


def object_signature(seed, obj):
    return {'id':obj['id'], 'name':obj['name'],
            'prompts':sorted(object_prompts(seed, obj), key=lambda p:p['frame'])}


class Dataset:
    def __init__(self, root, camera="observation.images.rgb.cam_left_head", output=None):
        self.root = Path(root).resolve()
        self.camera = camera
        self.output = Path(output).resolve() if output else self.root / "labels/sam2_semantic"
        self.info = read_json(self.root / "meta/info.json")
        if not self.info:
            raise ValueError(f"Missing meta/info.json: {self.root}")
        self.episodes = {
            int(ep["episode_index"]): ep
            for ep in map(json.loads, (self.root / "meta/episodes.jsonl").read_text().splitlines())
        }
        self.lock = threading.RLock()

    def episode(self, index):
        return self.episodes[int(index)]

    def video(self, index):
        index = int(index)
        self.episode(index)
        return self.root / self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        ).format(
            episode_chunk=index // self.info.get("chunks_size", 1000),
            episode_index=index,
            video_key=self.camera,
        )

    def instruction(self, index, frame):
        ep = self.episode(index)
        if not 0 <= int(frame) < ep["length"]:
            raise ValueError("Frame out of range")
        for segment in ep.get("sub_tasks", []):
            if segment["start"] <= frame < segment["end"]:
                return segment["text"]
        tasks = ep.get("tasks", [])
        if not ep.get("sub_tasks") and len(tasks) == 1 and isinstance(tasks[0], str):
            return tasks[0]
        raise ValueError("Missing unambiguous per-frame instruction")

    def dataset_mask_path(self, index):
        self.episode(index)
        pattern = self.info.get("mask_path")
        if not pattern:
            return None
        return self.root / pattern.format(
            episode_chunk=int(index) // self.info.get("chunks_size", 1000),
            episode_index=int(index), mask_key=self.camera,
        )

    def frame(self, index, frame):
        self.instruction(index, frame)
        cap = cv2.VideoCapture(str(self.video(index)))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError("Video frame unavailable")
            return image
        finally:
            cap.release()

    def directory(self, index):
        self.episode(index)
        camera = self.camera.removeprefix("observation.images.rgb.")
        return self.output / "jobs" / f"episode_{int(index):06d}_{camera}"

    def latest(self, index):
        return read_json(self.directory(index) / "latest.json", {"status": "unlabelled"})

    def job_dir(self, index, job_id):
        if not job_id or Path(job_id).name != job_id or job_id in (".", ".."):
            raise ValueError("Invalid job id")
        return self.directory(index) / job_id

    def export_path(self, index):
        return (
            self.root
            / f"masks/chunk-{int(index) // self.info.get('chunks_size', 1000):03d}/{self.camera}/episode_{int(index):06d}_masks.npz"
        )

    def unapprove(self, index):
        # Prior review jobs stay as revision history; only the active training export is removed.
        for path in (self.export_path(index), self.export_path(index).with_suffix(".json")):
            path.unlink(missing_ok=True)

    def validate_seed(self, seed, allow_empty=False):
        if seed.get("schema_version") != 2 or seed.get("camera") != self.camera:
            raise ValueError("Seed schema/camera mismatch")
        index, frame = int(seed["episode"]), int(seed["frame"])
        image = self.frame(index, frame)
        height, width = image.shape[:2]
        if (seed["width"], seed["height"]) != (width, height):
            raise ValueError("Seed/image shape mismatch")
        if seed.get("instruction") != self.instruction(index, frame):
            raise ValueError("Seed instruction mismatch")
        objects = seed.get("objects", [])
        if (not objects and not allow_empty) or len({o["id"] for o in objects}) != len(objects):
            raise ValueError("Unique target objects required")
        for obj in objects:
            if not isinstance(obj["id"], int) or obj["id"] < 1:
                raise ValueError("Object ids must be positive integers")
            prompts = object_prompts(seed, obj)
            if not obj.get('name') or not prompts or len({p['frame'] for p in prompts}) != len(prompts):
                raise ValueError('Unique prompt frames and label name required')
            if not any(1 in p['labels'] for p in prompts):
                raise ValueError('Each object needs at least one Include point')
            for prompt in prompts:
                if not isinstance(prompt['frame'], int) or not 0 <= prompt['frame'] < self.episode(index)['length']:
                    raise ValueError('Invalid prompt frame')
                points, labels = prompt['points'], prompt['labels']
                if not points or len(points) != len(labels):
                    raise ValueError('Points and labels must have matching nonzero lengths')
                for (x, y), label in zip(points, labels):
                    if label not in (0, 1) or not (0 <= x < width and 0 <= y < height):
                        raise ValueError('Invalid original-coordinate prompt')
        return seed

    def list_state(self):
        episodes = [
            {"episode": index, "length": ep["length"], **self.latest(index)}
            for index, ep in sorted(self.episodes.items())
        ]
        return {
            "dataset": str(self.root),
            "camera": self.camera,
            "episodes": episodes,
            "next_episode": next(
                (e["episode"] for e in episodes if e["status"] != "approved"), None
            ),
        }
