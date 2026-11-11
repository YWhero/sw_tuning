#!/usr/bin/env bash
# 원본 GR00T에 튜닝된 VLM 전체를 얹고 동결한 뒤 액션헤드만 학습합니다.
export DATASET="${DATASET:-data/Refilling_Dispenser_merged_932_934_lerobot_v21}"
export EXP_NAME="${EXP_NAME:-refilling_vlm_action_$(date +%Y%m%d_%H%M%S)}"
export VLM_WEIGHTS="${VLM_WEIGHTS:-checkpoints/vision/refilling_visual_hybrid_upperbg_nopad_1000_20261111-vlm.safetensors}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
# --vlm-weights / --dataset / --name 명령행 옵션이 위 설정보다 우선합니다.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${TUNING_RUNTIME:-conda}" != docker ]]; then
    source ./scripts/activate-conda.sh
fi
exec python3 tools/pipeline.py action "$@"
