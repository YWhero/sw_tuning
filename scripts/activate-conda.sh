#!/usr/bin/env bash
# Usage: source /home/ubuntu/mission_reasoning/tuning/scripts/activate-conda.sh
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate tuning
_tuning_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TUNING_RUNTIME=conda
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$_tuning_root/cache/huggingface"
export PYTHONPATH="$_tuning_root/tools:$_tuning_root/Isaac-GR00T:$_tuning_root/cache/segmentation/sam2-source"
export CYCLO_GROOT_MODEL_DIR=/home/ubuntu/mission_reasoning/cyclo_intelligence/docker/workspace/model/groot
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
cd "$_tuning_root"
unset _tuning_root
