#!/usr/bin/env bash
# Cosine feature 히트맵 MP4
# ── 여기서 설정하세요 ──────────────────────────────────────────────
DATASET="data/Refilling_Dispenser_lerobot_v21"                     # 데이터셋 경로 (tuning 기준 상대 경로 또는 절대 경로)
EPISODES=(0 1)               # 원하는 에피소드 번호. 하나만: (0)
CHECKPOINT="checkpoints/vision_runs/refilling_visual_llm15_kl_nopad_1000_20261110/checkpoint-1000"                  # 기본: 완료된 Refilling KL 1000스텝
VLM_WEIGHTS="checkpoints/vision/refilling_visual_llm15_kl_nopad_1000_20261110-vlm.safetensors"              # CHECKPOINT를 비우면 이 VLM 전체 가중치 사용
OUT="results/language_attention/refilling_feature_cosine_$(date +%Y%m%d_%H%M%S)"  # 새 출력 폴더
BASE_MODEL="nvidia/GR00T-N1.7-3B" # 가중치 경로 둘 다 비우면 원본 모델 시각화
ATTENTION_LAYER=15             # 0부터 시작: 15 = LLM 16번째 층
BATCH_SIZE=4
GPU="${GPU:-0}"
# 실행: ./cosine-map.sh   /   도움말: ./cosine-map.sh --help
# 명령행 --dataset / --episodes / --checkpoint / --out 등은 위 값보다 우선합니다.
# 전처리는 체크포인트 설정을 따릅니다. 새 학습의 기본은 letterbox=false입니다.
# ── 실행부 ─────────────────────────────────────────────────────────
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./scripts/activate-conda.sh
export CUDA_VISIBLE_DEVICES="$GPU"
args=(--base "$BASE_MODEL" --attention-layer "$ATTENTION_LAYER"
      --batch-size "$BATCH_SIZE" --episodes "${EPISODES[@]}" --out "$OUT")
[[ -z "$DATASET" ]] || args+=(--dataset "$DATASET")
# CLI로 가중치를 지정하면 파일의 가중치 기본값은 생략합니다.
cli_weights=false
for arg in "$@"; do
    case "$arg" in --checkpoint|--checkpoint=*|--vision-weights|--vision-weights=*|--vlm-weights|--vlm-weights=*) cli_weights=true ;; esac
done
if ! "$cli_weights"; then
    if [[ -n "$CHECKPOINT" ]]; then
        args+=(--checkpoint "$CHECKPOINT")
    elif [[ -n "$VLM_WEIGHTS" ]]; then
        args+=(--vlm-weights "$VLM_WEIGHTS")
    fi
fi
exec python tools/language_attention_videos.py "${args[@]}" "$@" --map-type feature_cosine
