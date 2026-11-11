#!/usr/bin/env bash
# Unified three-camera segmentation labeler; pass a dataset path or --stop.
# ── 여기서 설정하세요 ──
DATASET="${DATASET:-data/Refilling_Dispenser_lerobot_v21}"
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
    [[ "$DATASET" = /* ]] || DATASET="$ROOT/$DATASET"
    set -- "$DATASET"
fi
exec bash "$ROOT/scripts/start_labeling.sh" "$@"
