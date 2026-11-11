#!/usr/bin/env bash
# Compatibility entry point; the root launcher accepts an explicit dataset.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/start_labeling.sh" "${1:-$ROOT/data/Refilling_Dispenser_lerobot_v21}"
