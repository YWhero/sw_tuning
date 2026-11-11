"""Export the complete GR00T VLM backbone (vision + language), excluding action."""
import sys
from pathlib import Path
from safetensors import safe_open
from extract_vision import extract as extract_prefix


def extract(ckpt, out):
    import json
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        keys = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(ckpt / "model.safetensors", framework="pt") as weights:
            keys = weights.keys()
    for prefix in ("backbone.model.model.visual.", "backbone.model.model.language_model."):
        if not any(k.startswith(prefix) for k in keys):
            raise ValueError("Full VLM export requires visual AND language weights: " + prefix)
    extract_prefix(ckpt, out, prefix="backbone.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: extract_vlm.py CHECKPOINT OUTPUT.safetensors")
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
