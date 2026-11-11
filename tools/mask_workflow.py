"""Worker lifecycle, stale-approval protection, reviewed mask exports."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2
from labeling_storage import atomic_json, atomic_npz, digest, read_json
import numpy as np


class Workflow:
    def __init__(self, dataset, runtime):
        self.dataset, self.runtime = dataset, runtime
        self.processes = {}

    def current(self, index, expected=None):
        # Polling must not observe the provisional job before worker submission.
        with self.dataset.lock:
            return self._current(index, expected)

    def _current(self, index, expected=None):
        current = self.dataset.latest(index)
        if expected is not None and current.get("job_id") != expected:
            raise ValueError("Stale job: reload this episode")
        job_id = current.get("job_id")
        if job_id:
            path = self.dataset.job_dir(index, job_id)
            current.update(read_json(path / "progress.json", {}))
            if current.get("status") in ("awaiting_review", "approved", "failed"):
                atomic_json(self.dataset.directory(index) / "latest.json", current)
            process = self.processes.get(job_id)
            alive = process is not None and process.poll() is None
            if process is None and current.get("pid"):
                try:
                    os.kill(current["pid"], 0)
                    cmdline = Path(f"/proc/{current['pid']}/cmdline").read_bytes()
                    alive = str(path / "job.json").encode() in cmdline
                except (ProcessLookupError, PermissionError, FileNotFoundError):
                    alive = False
            if current.get("resident_instance") and current.get("status") == "running":
                try:
                    from labeling_sam_client import request
                    state = request(current["resident_socket"], command="status")
                    alive = state["instance"] == current["resident_instance"] and job_id in state["jobs"]
                    if not alive:
                        current.update(read_json(path / "progress.json", {}))
                except (OSError, ValueError, RuntimeError):
                    alive = False
            if current.get("status") == "running" and not alive:
                current.update(
                    status="failed",
                    error="Worker exited or server restarted; retry from saved seed",
                )
                atomic_json(path / "progress.json", current)
                atomic_json(self.dataset.directory(index) / "latest.json", current)
        return current

    def save_seed(self, seed):
        with self.dataset.lock:
            seed = self.dataset.validate_seed(seed, allow_empty=True)
            index = seed["episode"]
            previous = self.current(index)
            if previous.get("status") == "running":
                raise ValueError("Worker is already running")
            self.dataset.unapprove(index)
            atomic_json(self.dataset.directory(index) / "seed.json", seed)
            base = previous.get('job_id') if previous.get('status') in ('awaiting_review','approved') else previous.get('base_job_id')
            state = {'status':'seed_selected', 'seed_hash':digest(seed), 'base_job_id':base}
            atomic_json(self.dataset.directory(index) / 'latest.json', state)
            return state

    def generate(self, index):
        with self.dataset.lock:
            previous = self.current(index)
            if previous.get("status") == "running":
                raise ValueError("Worker is already running")
            seed = read_json(self.dataset.directory(index) / "seed.json")
            if seed is None:
                raise ValueError("Save a seed first")
            self.dataset.validate_seed(seed, allow_empty=bool(previous.get("base_job_id")))
            runtime = self.runtime
            if not Path(runtime["checkpoint"]).is_file():
                raise ValueError("SAM2 checkpoint not installed; run setup-labeler.sh")
            try:
                import sam2  # noqa: F401
            except ImportError:
                if not Path(runtime.get("source", "") + "/sam2/build_sam.py").exists():
                    raise ValueError("SAM2 source/dependencies missing; run setup-labeler.sh")
            self.dataset.unapprove(index)
            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            directory = self.dataset.job_dir(index, job_id)
            directory.mkdir(parents=True)
            job = {
                "job_id": job_id,
                "episode": int(index),
                "seed_hash": digest(seed),
                "status": "running",
                "directory": str(directory),
                "video": str(self.dataset.video(index)),
                "frames": self.dataset.episode(index)["length"],
                "runtime": runtime,
                "base_job_id": previous.get('job_id') if previous.get('status') in ('awaiting_review','approved') else previous.get('base_job_id'),
            }
            if job['base_job_id']:
                job['base_directory'] = str(self.dataset.job_dir(index, job['base_job_id']))
            atomic_json(directory / "seed.json", seed)
            atomic_json(directory / "job.json", job)
            atomic_json(
                directory / "progress.json",
                {"status": "running", "completed": 0, "total": job["frames"]},
            )
            atomic_json(self.dataset.directory(index) / "latest.json", job)
            if runtime.get("resident_socket"):
                from labeling_sam_client import request
                try:
                    state = request(runtime["resident_socket"], command="submit", job=str(directory / "job.json"))
                except Exception as exc:
                    atomic_json(directory / "progress.json", {"status":"failed","error":str(exc)})
                    raise RuntimeError(f"SAM server is not ready: {exc}") from exc
                job.update(pid=state["pid"], resident_instance=state["instance"], resident_socket=runtime["resident_socket"])
                atomic_json(self.dataset.directory(index) / "latest.json", job)
            else:
                with (directory / "worker.log").open("wb") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("propagate_masks.py")),
                            "--job",
                            str(directory / "job.json"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                self.processes[job_id] = process
                job["pid"] = process.pid
                atomic_json(self.dataset.directory(index) / "latest.json", job)
            return job

    def approve(self, index, job_id, seed_hash):
        with self.dataset.lock:
            current = self.current(index, job_id)
            if current["status"] != "awaiting_review" or current["seed_hash"] != seed_hash:
                raise ValueError("Only the current reviewed revision can be approved")
            directory = self.dataset.job_dir(index, job_id)
            if digest(read_json(self.dataset.directory(index) / "seed.json")) != seed_hash:
                raise ValueError("Seed changed since generation")
            needs_video = read_json(directory / "job.json", {}).get("runtime", {}).get("render_review_video", True)
            artifacts = ["masks.npz", "object_masks.npz", "annotations.json"]
            if needs_video:
                artifacts.append("review.mp4")
            for artifact in artifacts:
                if not (directory / artifact).is_file():
                    raise ValueError(f"Missing artifact: {artifact}")
            if needs_video:
                capture = cv2.VideoCapture(str(directory / "review.mp4"))
                try:
                    if (
                        not capture.isOpened()
                        or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        != self.dataset.episode(index)["length"]
                    ):
                        raise ValueError("Review video incomplete")
                finally:
                    capture.release()
            with np.load(directory / "masks.npz", allow_pickle=False) as masks:
                arr = masks["arr_0"]
                seed = read_json(directory / "seed.json")
                if (
                    arr.shape
                    != (self.dataset.episode(index)["length"], seed["height"], seed["width"])
                    or arr.dtype != np.uint8
                    or not np.isin(arr, [0, 1]).all()
                ):
                    raise ValueError("Incomplete/invalid mask sequence")
                path = self.dataset.export_path(index)
                atomic_npz(path, arr_0=arr)
            provenance = {
                "approved": True,
                "job_id": job_id,
                "seed_hash": seed_hash,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "camera": self.dataset.camera,
                "episode": int(index),
                "target_only": len({obj["name"] for obj in seed["objects"]}) == 1,
            }
            atomic_json(path.with_suffix(".json"), provenance)
            current.update(status="approved")
            atomic_json(directory / "progress.json", current)
            atomic_json(self.dataset.directory(index) / "latest.json", current)
            return current

    def erase(self, index, job_id, frame, point, scope):
        with self.dataset.lock:
            current = self.current(index, job_id)
            if current["status"] not in ("approved", "awaiting_review"):
                raise ValueError("No completed masks to correct")
            if scope not in ("frame", "video"):
                raise ValueError("Invalid erase scope")
            if not 0 <= int(frame) < self.dataset.episode(index)["length"]:
                raise ValueError("Frame out of range")
            seed = read_json(self.dataset.job_dir(index, job_id) / "seed.json")
            if (
                len(point) != 2
                or not 0 <= point[0] < seed["width"]
                or not 0 <= point[1] < seed["height"]
            ):
                raise ValueError("Erase point out of bounds")
            source = self.dataset.job_dir(index, job_id)
            new_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            directory = self.dataset.job_dir(index, new_id)
            directory.mkdir()
            import shutil

            for name in ("seed.json", "job.json", "object_masks.npz"):
                shutil.copy2(source / name, directory / name)
            job = read_json(directory / "job.json")
            job.update(
                job_id=new_id,
                directory=str(directory),
                correction={"frame": frame, "point": point, "scope": scope},
                status="running",
            )
            atomic_json(directory / "job.json", job)
            atomic_json(
                directory / "progress.json",
                {"status": "running", "completed": 0, "total": job["frames"]},
            )
            self.dataset.unapprove(index)
            atomic_json(self.dataset.directory(index) / "latest.json", job)
            with (directory / "worker.log").open("wb") as log:
                self.processes[new_id] = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("propagate_masks.py")),
                        "--job",
                        str(directory / "job.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            job["pid"] = self.processes[new_id].pid
            atomic_json(self.dataset.directory(index) / "latest.json", job)
            return job
