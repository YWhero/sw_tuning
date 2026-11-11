#!/usr/bin/env bash
# Hybrid + 상단 배경 attention 억제: 비전타워 전체 학습, LLM·액션 동결
# ── 여기서 설정하세요 ──────────────────────────────────────────────
# DATASET: tuning 기준 상대 경로 또는 절대 경로. 비전 학습은 검증된 마스크 입력 필요.
export DATASET="${DATASET:-/home/ubuntu/mission_reasoning/tuning/logs/refilling_visual_hybrid_upperbg_nopad_1000_20261111/inputs/head}"                   # 기본 데이터셋
export EXP_NAME="${EXP_NAME:-refilling_vision_hybrid_$(date +%Y%m%d_%H%M%S)}"                 # 기본: 실행 시각을 포함한 새 실험 이름
# 새 비전튜닝은 항상 원본 사전학습 GR00T에서 시작합니다.
export BASE_MODEL_PATH="nvidia/GR00T-N1.7-3B"
# 기존 사용 승인된 마스크 snapshot 사용 (정식 검토 완료를 뜻하지 않음).
export ALLOW_LEGACY_MASKS="${ALLOW_LEGACY_MASKS:-1}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"  # 실제 배치
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"  # 유효 배치 = 실제 배치 × 누적
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
# 상단 배경 억제 계수: target 겹침 패치 제외. 0이면 기존 hybrid만 사용.
export UPPER_BACKGROUND_ATTENTION_WEIGHT="${UPPER_BACKGROUND_ATTENTION_WEIGHT:-0}"
export GPU="${GPU:-0}"
# 새 학습은 letterbox=false. 환경변수로 전달한 값은 위 기본값보다 우선합니다.
# 실행: ./vision-hybrid.sh             입력 검사만: ./vision-hybrid.sh --check
# tmux 실행: ./vision-hybrid.sh --background
# --dataset / --name 옵션은 위 설정보다 우선합니다. --base는 원본 모델만 허용합니다.
# 완료 후 checkpoints/vision/<실험명>-vlm.safetensors에 비전+LLM 전체 저장.
# ── 실행부 ─────────────────────────────────────────────────────────
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${TUNING_RUNTIME:-conda}" != docker ]]; then
    source ./scripts/activate-conda.sh
fi
exec python3 tools/pipeline.py hybrid "$@"
