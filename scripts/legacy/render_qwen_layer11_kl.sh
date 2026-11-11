#!/usr/bin/env bash
# Historical renderer: layer 11 image-conditional QK attention, not feature cosine.
# Pass --base, --checkpoint (optional), --dataset, --episodes, --out.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ./scripts/activate-conda.sh
export PYTHONPATH="$PWD/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
exec python tools/language_attention_videos.py "$@" \
  --map-type attention --attention-layer 11 --batch-size 1
