"""Local target-only SAM2 video labeling server; GPU work runs in subprocesses."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlparse

import cv2
from labeling_storage import Dataset, read_json
from mask_workflow import Workflow
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset")
    parser.add_argument("--config", default=str(ROOT / "configs/labeling_active.json"))
    parser.add_argument("--runtime", default=str(ROOT / "configs/labeling_runtime.json"))
    args = parser.parse_args()
    active, runtime = read_json(args.config), read_json(args.runtime)

    def local(path):
        return str((ROOT / path).resolve()) if not Path(path).is_absolute() else path

    dataset = Dataset(
        local(args.dataset or active["dataset"]),
        active.get("camera", "observation.images.rgb.cam_left_head"),
        local(active["output"]) if active.get("output") and not args.dataset else None,
    )
    for key in ("checkpoint", "source", "dependencies"):
        if runtime.get(key):
            runtime[key] = local(runtime[key])
    runtime["gpu"] = active.get("gpu", 0)
    runtime["render_review_video"] = active.get("render_review_video", True)
    if active.get("resident_socket"):
        runtime["resident_socket"] = local(active["resident_socket"])
    workflow = Workflow(dataset, runtime)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def json(self, value, status=200):
            self.send_bytes(json.dumps(value).encode(), "application/json", status)

        def send_bytes(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def media(self, path):
            length = path.stat().st_size
            start, end = 0, length - 1
            byte_range = self.headers.get("Range")
            if byte_range:
                import re

                match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
                if not match:
                    raise ValueError("Unsupported byte range")
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
                if start > end:
                    raise ValueError("Invalid byte range")
            self.send_response(206 if byte_range else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = stream.read(min(1024 * 1024, remaining))
                    self.wfile.write(data)
                    remaining -= len(data)

        def valid_host(self):
            return self.headers.get("Host", "").split(":")[0] in ("localhost", "127.0.0.1")

        def do_GET(self):
            if not self.valid_host():
                return self.json({"error": "Localhost access only"}, 403)
            try:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == "/":
                    return self.send_bytes(
                        Path(__file__).with_name("labeling_page.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                if parsed.path == "/api/server":
                    checkpoint_ready = Path(runtime["checkpoint"]).is_file()
                    return self.json(
                        {
                            "token": token,
                            "automatic_available": False,
                            "render_review_video": runtime["render_review_video"],
                            "automatic_reason": "No verified detector configured. Select target points manually.",
                            "sam_checkpoint_ready": checkpoint_ready,
                            "model": runtime["model_name"],
                        }
                    )
                if parsed.path == "/api/state":
                    for index in dataset.episodes:
                        workflow.current(index)
                    return self.json(dataset.list_state())
                index = int(query.get("episode", ["-1"])[0])
                dataset.episode(index)
                if parsed.path == "/api/job":
                    return self.json(workflow.current(index))
                if parsed.path == "/api/seed":
                    return self.json(read_json(dataset.directory(index) / "seed.json", {}))
                if parsed.path == "/api/frame":
                    frame = int(query.get("frame", ["0"])[0])
                    image = dataset.frame(index, frame)
                    return self.json(
                        {
                            "instruction": dataset.instruction(index, frame),
                            "width": image.shape[1],
                            "height": image.shape[0],
                        }
                    )
                if parsed.path == "/frame.png":
                    frame = int(query.get("frame", ["0"])[0])
                    image = dataset.frame(index, frame)
                    current = workflow.current(index)
                    if query.get("overlay", ["0"])[0] == "1":
                        mask_path = dataset.dataset_mask_path(index)
                        if current.get("job_id") and current["status"] in ("approved", "awaiting_review"):
                            mask_path = dataset.job_dir(index, current["job_id"]) / "masks.npz"
                        elif current.get('base_job_id'):
                            mask_path = dataset.job_dir(index, current['base_job_id']) / 'masks.npz'
                        label = query.get("label", [""])[0]
                        object_path = mask_path.with_name("object_masks.npz") if mask_path else None
                        if object_path is not None and object_path.is_file():
                            seed = read_json(object_path.with_name("seed.json"))
                            active_seed = read_json(dataset.directory(index)/'seed.json', seed)
                            active_ids = {obj['id'] for obj in active_seed['objects']}
                            colors = ([70, 220, 100], [255, 160, 64], [70, 180, 255],
                                      [230, 107, 214], [255, 224, 80], [165, 138, 255])
                            with np.load(object_path, allow_pickle=False) as masks:
                                for obj in seed["objects"]:
                                    if obj["id"] not in active_ids or (label and obj["name"] != label):
                                        continue
                                    mask = masks[str(obj["id"])][frame] > 0
                                    if mask.shape != image.shape[:2]:
                                        raise ValueError("Mask/image dimensions differ")
                                    color = colors[sum(map(ord, obj["name"])) % len(colors)]
                                    image[mask] = (image[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                        elif not label and mask_path is not None and mask_path.is_file():
                            with np.load(mask_path, allow_pickle=False) as data:
                                mask = data["arr_0"][frame] > 0
                                if mask.shape != image.shape[:2]:
                                    raise ValueError("Mask/image dimensions differ")
                                image[mask] = (
                                    image[mask] * 0.5 + np.array([70, 220, 100]) * 0.5
                                ).astype(np.uint8)
                    return self.send_bytes(cv2.imencode(".png", image)[1].tobytes(), "image/png")
                if parsed.path in ("/video.mp4", "/review.mp4"):
                    path = (
                        dataset.video(index)
                        if parsed.path == "/video.mp4"
                        else dataset.job_dir(index, query["job"][0]) / "review.mp4"
                    )
                    return self.media(path)
                self.json({"error": "Not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.json({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.valid_host():
                return self.json({"error": "Localhost access only"}, 403)
            try:
                if self.headers.get("X-Label-Token") != token:
                    return self.json({"error": "Invalid session token"}, 403)
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self.json({"error": "Cross-origin request refused"}, 403)
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length < 1024 * 1024:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path == "/api/seed":
                    return self.json(workflow.save_seed(value))
                if path == "/api/complete":
                    return self.json(workflow.generate(int(value["episode"])))
                if path == "/api/review":
                    return self.json(
                        workflow.approve(int(value["episode"]), value["job_id"], value["seed_hash"])
                    )
                if path == "/api/correct":
                    return self.json(
                        workflow.erase(
                            int(value["episode"]),
                            value["job_id"],
                            int(value["frame"]),
                            value["point"],
                            value.get("scope", "frame"),
                        )
                    )
                if path == "/api/auto":
                    raise ValueError("Automatic selection disabled: no verified detector installed")
                self.json({"error": "Not found"}, 404)
            except Exception as exc:
                self.json({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Labeling UI: http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
