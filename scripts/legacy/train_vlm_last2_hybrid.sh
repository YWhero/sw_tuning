#!/usr/bin/env bash
# Train only visual last-two and language last-two blocks with section7 hybrid loss.
# Supply base, dataset, modality config, output/experiment, batch and step options.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ./scripts/activate-conda.sh
export PYTHONPATH="$PWD/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
exec python Isaac-GR00T/gr00t/experiment/launch_finetune.py "$@" \
  --vlm-last2-feature-only --attn-supervision-layers -1 \
  --attn-supervision-loss feature_hybrid --attn-supervision-weight 1.0 \
  --no-qwen-layer11-qk-kl-only --no-language-last2-feature-only --no-vision-kl-only \
  --no-tune-llm --no-tune-visual --no-tune-projector --no-tune-diffusion-model --no-tune-vlln --no-ttrtc-enabled
