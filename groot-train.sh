#!/usr/bin/env bash
# Baseline: 비전·LLM 동결, 액션헤드 학습
# ── 여기서 설정하세요 ──────────────────────────────────────────────
# DATASET: tuning 기준 상대 경로 또는 절대 경로. 비전 학습은 검증된 마스크 입력 필요.
export DATASET="${DATASET:-data/Refilling_Dispenser_merged_932_934_lerobot_v21}"                   # 기본 데이터셋
export EXP_NAME="${EXP_NAME:-refilling_baseline_$(date +%Y%m%d_%H%M%S)}"                 # 기본: 실행 시각을 포함한 새 실험 이름
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"  # 실제 배치
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"  # 유효 배치 = 실제 배치 × 누적
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export GPU="${GPU:-0}"
# 새 학습은 letterbox=false. 환경변수로 전달한 값은 위 기본값보다 우선합니다.
# 실행: ./groot-train.sh             입력 검사만: ./groot-train.sh --check
# tmux 실행: ./groot-train.sh --background
# --dataset / --name / --base 명령행 옵션은 위 설정보다 우선합니다.
# ── 실행부 ─────────────────────────────────────────────────────────
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${TUNING_RUNTIME:-conda}" != docker ]]; then
    source ./scripts/activate-conda.sh
fi
exec python3 tools/pipeline.py groot "$@"
