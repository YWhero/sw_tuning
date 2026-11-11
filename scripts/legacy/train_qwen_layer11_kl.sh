#!/usr/bin/env bash
# Qwen layer 11 Q/K-only mask KL. Does not start or stop other training sessions.
# Supply --base-model-path, --dataset-path, --modality-config-path,
# --embodiment-tag, --output-dir, --experiment-name and --max-steps.
# Dataset/config must provide one image + matching mask per sample.
# Use delta_indices=[0] for state/action in this KL-only data config so episode tails remain eligible.
# Saved FULL checkpoints contain Q/K changes; extract_vision.py does NOT export these changes.
# For subsequent action training: use that full checkpoint as --base-model-path,
# --no-qwen-layer11-qk-kl-only, --no-vision-kl-only, --no-tune-llm, --no-tune-visual.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ./scripts/activate-conda.sh
export PYTHONPATH="$PWD/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
exec python Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  "$@" \
  --no-language-last2-feature-only --qwen-layer11-qk-kl-only --attn-supervision-layers 11 \
  --attn-supervision-loss kl --attn-supervision-weight 1.0 \
  --no-vision-kl-only --no-tune-llm --no-tune-visual \
  --no-tune-projector --no-tune-diffusion-model --no-tune-vlln \
  --no-ttrtc-enabled
