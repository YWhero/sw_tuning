#!/usr/bin/env bash
# Phase 1: whole visual tower, target-mask KL at LLM index 15 (16th layer); LLM/action frozen.
# Default: 1000 steps, microbatch 8 x accumulation 8, learning rate 1e-5.
# Usage: DATASET=<dataset> EXP_NAME=<name> ./vision-tune.sh --check
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python3 tools/pipeline.py vision "$@"
