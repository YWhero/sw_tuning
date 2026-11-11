"""Extract a complete visual overlay and provenance from a full GR00T checkpoint.
The full checkpoint is retained. Inference needs the same full base plus this overlay.
Usage: extract_vision.py CHECKPOINT [OUTPUT.safetensors]
"""

import json
from pathlib import Path
import sys

from safetensors.torch import load_file, save_file


PREFIX = "backbone.model.model.visual."


def extract(ckpt: Path, out: Path, prefix: str = PREFIX) -> None:
    index_path = ckpt / "model.safetensors.index.json"
    state = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = {s for k, s in index["weight_map"].items() if k.startswith(prefix)}
        for shard in sorted(shards):
            for k, v in load_file(ckpt / shard).items():
                if k.startswith(prefix):
                    state[k] = v
    else:
        for k, v in load_file(ckpt / "model.safetensors").items():
            if k.startswith(prefix):
                state[k] = v
    assert state, f"{ckpt} 에 {prefix}* 키가 없습니다"
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out))
    import hashlib

    manifest = {
        "checkpoint": str(ckpt),
        "prefix": prefix,
        "tensors": len(state),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    config = ckpt / "config.json"
    if config.exists():
        manifest["model_config"] = json.loads(config.read_text())
    processor = ckpt / "processor_config.json"
    if processor.exists():
        manifest["processor"] = json.loads(processor.read_text())
    run = ckpt.parent / "run_manifest.json"
    if not run.exists():
        run = ckpt.parent.parent / "run_manifest.json"
    if run.exists():
        manifest["run"] = json.loads(run.read_text())
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    size_gb = out.stat().st_size / 1e9
    print(f"추출 완료: {out} — 텐서 {len(state)}개, {size_gb:.2f}GB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ckpt = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt / "vision_only.safetensors"
    extract(ckpt, out)
