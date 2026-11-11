import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parent))
from check_training_inputs import check_dataset


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    (tmp_path / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2, "sub_tasks": [{"text": "Pick up the can."}]})
    )
    for name in (
        "videos/chunk-000/observation.images.rgb.cam_left_head/episode_000000.mp4",
        "data/chunk-000/episode_000000.parquet",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    mask = (
        tmp_path / "masks/chunk-000/observation.images.rgb.cam_left_head/episode_000000_masks.npz"
    )
    mask.parent.mkdir(parents=True)
    np.savez_compressed(mask, arr_0=np.ones((2, 3, 4), dtype=np.uint8))
    return tmp_path, mask


def test_legacy_requires_explicit_opt_in(dataset):
    root, _ = dataset
    with pytest.raises(ValueError, match="no approval record"):
        check_dataset(root, masks=True)
    assert check_dataset(root, masks=True, allow_legacy_masks=True) == 1


@pytest.mark.parametrize(
    "array",
    [
        np.ones((1, 3, 4), dtype=np.uint8),
        np.ones((2, 3, 4), dtype=np.float32),
        np.full((2, 3, 4), 2, dtype=np.uint8),
    ],
)
def test_legacy_still_validates_masks(dataset, array):
    root, mask = dataset
    np.savez_compressed(mask, arr_0=array)
    with pytest.raises(ValueError, match="invalid binary masks"):
        check_dataset(root, masks=True, allow_legacy_masks=True)


@pytest.mark.parametrize("approval", [{"approved": False}, {"approved": True, "sha256": "stale"}])
def test_opt_in_cannot_bypass_invalid_approval(dataset, approval):
    root, mask = dataset
    mask.with_suffix(".json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="invalid/stale"):
        check_dataset(root, masks=True, allow_legacy_masks=True)


def test_approved_masks_pass_without_legacy_mode(dataset):
    root, mask = dataset
    mask.with_suffix(".json").write_text(
        json.dumps({"approved": True, "sha256": hashlib.sha256(mask.read_bytes()).hexdigest()})
    )
    assert check_dataset(root, masks=True) == 1


def test_missing_mask_is_not_accepted(dataset):
    root, mask = dataset
    mask.unlink()
    with pytest.raises(FileNotFoundError, match="missing mask file"):
        check_dataset(root, masks=True, allow_legacy_masks=True)


@pytest.mark.parametrize("arm_width", [7, 8])
def test_runtime_preflight_uses_metadata_arm_slices(dataset, monkeypatch, arm_width):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from check_training_inputs import main

    root, _ = dataset
    (root / "meta/info.json").write_text('{"chunks_size":1000}')
    layout = {"arm_left": {"start": 0, "end": arm_width},
              "arm_right": {"start": arm_width, "end": arm_width * 2}}
    (root / "meta/modality.json").write_text(json.dumps({"state": layout, "action": layout}))
    parquet = root / "data/chunk-000/episode_000000.parquet"
    values = np.zeros((2, arm_width * 2)).tolist()
    pq.write_table(pa.table({"observation.state": values, "action": values}), parquet)
    monkeypatch.setattr(sys, "argv", ["check_training_inputs.py", str(root)])
    main()
    # The selected channels still must exist and be finite.
    values[0][-1] = float("nan")
    pq.write_table(pa.table({"observation.state": values, "action": values}), parquet)
    with pytest.raises(ValueError, match="invalid.*arm channels"):
        main()
    values[0][-1] = 0.0
    pq.write_table(pa.table({"observation.state": values, "action": values}), parquet)
    layout["arm_right"]["end"] += 1
    (root / "meta/modality.json").write_text(json.dumps({"state": layout, "action": layout}))
    with pytest.raises(ValueError, match="invalid.*arm channels"):
        main()


def test_preflight_checks_metadata_mask_path_not_unused_default(dataset):
    root, mask = dataset
    actual = root / "target_masks/chunk-000/observation.images.rgb.cam_left_head/episode_000000_masks.npz"
    actual.parent.mkdir(parents=True)
    mask.rename(actual)
    (root / "meta/info.json").write_text(json.dumps({
        "mask_path": "target_masks/chunk-{episode_chunk:03d}/{mask_key}/episode_{episode_index:06d}_masks.npz"
    }))
    assert check_dataset(root, masks=True, allow_legacy_masks=True) == 1
    actual.with_suffix(".json").write_text('{"approved":false}')
    with pytest.raises(ValueError, match="invalid/stale"):
        check_dataset(root, masks=True, allow_legacy_masks=True)


def test_legacy_source_metadata_is_not_mistaken_for_an_approval(dataset):
    root, mask = dataset
    mask.with_suffix('.json').write_text(json.dumps({
        'objects': [{'id': 1, 'name': 'target'}],
        'source_metadata': {'approved': True},
        'note': 'target-only extraction; no new human review claimed',
    }))
    with pytest.raises(ValueError, match='no approval record'):
        check_dataset(root, masks=True)
    assert check_dataset(root, masks=True, allow_legacy_masks=True) == 1
    # A stale hash is rejected even when the provenance has no approval flag.
    mask.with_suffix('.json').write_text('{"sha256":"stale"}')
    with pytest.raises(ValueError, match='invalid/stale'):
        check_dataset(root, masks=True, allow_legacy_masks=True)


def test_prepare_requires_dataset_specific_modality_without_mutating_metadata(dataset):
    from prepare_dataset import prepare
    root, _ = dataset
    before = (root / "meta/episodes.jsonl").read_bytes()
    with pytest.raises(ValueError, match="channel metadata"):
        prepare(root)
    assert (root / "meta/episodes.jsonl").read_bytes() == before
    assert not (root / "meta/modality.json").exists()


def test_preflight_decodes_video_and_rejects_misaligned_mask(dataset, monkeypatch):
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq
    from check_training_inputs import main
    root, mask = dataset
    layout = {"arm_left": {"start": 0, "end": 7}, "arm_right": {"start": 7, "end": 14}}
    (root / "meta/modality.json").write_text(json.dumps({"state": layout, "action": layout}))
    pq.write_table(pa.table({"observation.state": np.zeros((2, 14)).tolist(),
                            "action": np.zeros((2, 14)).tolist()}),
                   root / "data/chunk-000/episode_000000.parquet")
    video = root / "videos/chunk-000/observation.images.rgb.cam_left_head/episode_000000.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (16, 16))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(sys, "argv", ["check_training_inputs.py", str(root), "--masks", "--allow-legacy-masks"])
    np.savez_compressed(mask, arr_0=np.ones((2, 16, 16), dtype=np.uint8))
    main()
    np.savez_compressed(mask, arr_0=np.ones((2, 8, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="geometry mismatch"):
        main()
