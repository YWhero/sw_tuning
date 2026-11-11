"""Runtime preflight for the documented arms-only GR00T recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


def check_dataset(
    root,
    masks=False,
    camera="observation.images.rgb.cam_left_head",
    allow_legacy_masks=False,
    mask_shapes=None,
):
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not episodes:
        raise ValueError("Dataset has no episodes")
    chunks = info.get("chunks_size", 1000)
    legacy_count = 0
    for checked, ep in enumerate(episodes, start=1):
        eid = ep["episode_index"]
        args = dict(episode_chunk=eid // chunks, episode_index=eid, video_key=camera)
        video = root / info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        ).format(**args)
        parquet = root / info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ).format(**args)
        if not video.is_file() or not parquet.is_file():
            raise FileNotFoundError(f"Missing RGB/parquet for episode {eid}: {video}, {parquet}")
        if not ep.get("sub_tasks"):
            raise ValueError(
                f"Episode {eid} has no sub_tasks; run tools/prepare_dataset.py explicitly"
            )
        if masks:
            mask = root / info.get(
                "mask_path",
                "masks/chunk-{episode_chunk:03d}/{mask_key}/episode_{episode_index:06d}_masks.npz",
            ).format(**args, mask_key=camera)
            if not mask.is_file():
                raise FileNotFoundError(f"Episode {eid}: missing mask file: {mask}")
            provenance = mask.with_suffix(".json")
            approval = json.loads(provenance.read_text()) if provenance.exists() else {}
            if (
                ("approved" in approval and approval["approved"] is not True)
                or (approval.get("approved") and "sha256" not in approval)
                or ("sha256" in approval
                    and approval["sha256"] != hashlib.sha256(mask.read_bytes()).hexdigest())
            ):
                raise ValueError(f"Episode {eid}: invalid/stale approval provenance")
            if not approval.get("approved"):
                if allow_legacy_masks:
                    legacy_count += 1
                else:
                    raise ValueError(
                        f"Episode {eid}: mask exists but has no approval record. "
                        "Review the target masks or explicitly set ALLOW_LEGACY_MASKS=1 to use existing masks without claiming approval."
                    )
            with np.load(mask, allow_pickle=False) as data:
                arr = data["arr_0"]
                if (
                    arr.dtype != np.uint8
                    or arr.ndim != 3
                    or len(arr) != ep["length"]
                    or arr.size == 0
                    or arr.max() > 1
                ):
                    raise ValueError(f"Episode {eid}: invalid binary masks")
                if mask_shapes is not None:
                    mask_shapes[eid] = arr.shape[1:]
            if (len(episodes) > 100) and (checked % 100 == 0):
                print(f"Mask validation: {checked}/{len(episodes)} episodes checked", flush=True)
    if legacy_count:
        print(
            f"Explicit legacy-mask mode: {legacy_count} existing mask files have no recorded approval; validation remains enabled."
        )
    return len(episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--masks", action="store_true")
    parser.add_argument(
        "--allow-legacy-masks",
        action="store_true",
        help="Explicitly accept valid pre-existing masks without tool approval metadata",
    )
    args = parser.parse_args()
    root = Path(args.dataset)
    mask_shapes = {}
    count = check_dataset(
        root, masks=args.masks, allow_legacy_masks=args.allow_legacy_masks, mask_shapes=mask_shapes
    )
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    modalities = json.loads((root / "meta/modality.json").read_text())
    arm_slices = {}
    for group in ("state", "action"):
        arm_slices[group] = []
        for key in ("arm_left", "arm_right"):
            item = modalities[group][key]
            start, end = item["start"], item["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError(f"Invalid {group}/{key} channel range")
            if any(start < other.stop and other.start < end for other in arm_slices[group]):
                raise ValueError(f"Overlapping {group} arm channels")
            arm_slices[group].append(slice(start, end))
    for episode in episodes:
        index = episode["episode_index"]
        parquet = root / info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ).format(episode_chunk=index // info.get("chunks_size", 1000), episode_index=index)
        file = pq.ParquetFile(parquet)
        if file.metadata.num_rows != episode["length"]:
            raise ValueError(f"Episode {index}: parquet/metadata length mismatch")
        data = file.read(columns=["observation.state", "action"]).to_pydict()
        for group, key in (("state", "observation.state"), ("action", "action")):
            values = np.stack(data[key])
            for channels in arm_slices[group]:
                if (values.ndim != 2 or values.shape[1] < channels.stop
                        or not np.isfinite(values[:, channels]).all()):
                    raise ValueError(f"Episode {index}: invalid {key} arm channels")
        if args.masks:
            import cv2
            video = root / info.get(
                "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ).format(episode_chunk=index // info.get("chunks_size", 1000), episode_index=index,
                     video_key="observation.images.rgb.cam_left_head")
            cap = cv2.VideoCapture(str(video))
            try:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Episode {index}: video frame unavailable")
                shape = frame.shape[:2]
            finally:
                cap.release()
            if mask_shapes[index] != shape:
                raise ValueError(f"Episode {index}: original video/mask geometry mismatch")
    print(
        f"Runtime preflight passed: {count} episodes; arm channels from metadata; masks={args.masks}, legacy mode={args.allow_legacy_masks}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        sys.exit(
            f"Training inputs not ready: {exc}. Check dataset metadata and reviewed target masks."
        )
