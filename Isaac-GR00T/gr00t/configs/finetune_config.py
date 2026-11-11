# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Finetune config used for single node post-training.
from dataclasses import dataclass
import warnings


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to one dataset root, or an os.pathsep-separated list of dataset roots."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment.
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`.
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_vlln: bool = True

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    # --- Training-time real-time chunking (action modules only) ---
    ttrtc_enabled: bool = False
    ttrtc_max_delay_steps: int = 10
    ttrtc_zero_delay_probability: float = 0.1
    ttrtc_policy_hz: float = 15.0  # Effective action rate, saved with checkpoint.

    # --- Vision Attention Supervision (mask-guided) ---
    attn_supervision_layers: list[int] | None = None
    """Retained LLM decoder layer indices; use -1 for the final layer.
    None disables supervision. Requires datasets with mask files and a "mask" modality config."""

    vlm_last2_feature_only: bool = False
    language_last2_feature_only: bool = False
    """Train only the last two retained language blocks with feature hybrid loss."""

    qwen_layer11_qk_kl_only: bool = False
    """Freeze all except Qwen layer index 11 Q/K; skip action and downstream computation."""

    vision_kl_only: bool = False
    """Train visual parameters with KL alone; never evaluate the action head."""

    attn_supervision_weight: float = 0.5
    """Weight of the auxiliary attention supervision loss added to the action loss."""

    upper_background_attention_weight: float = 0.0
    """Hybrid-only upper-background attention penalty; zero preserves existing loss."""

    attn_supervision_loss: str = "kl"
    """KL attention loss, or feature_hybrid for visual-only / selected-block feature training."""

    attn_supervision_margin: float = 0.9
    """Target attention mass inside the mask for the "mass" loss."""

    # --- Vision Weights Overlay ---
    vlm_weights_path: str | None = None
    """Complete backbone overlay; freeze the VLM and train the action head."""

    vision_weights_path: str | None = None
    """Path to a vision-tower-only safetensors file (tools/extract_vision.py output).
    Loaded over the base model's vision tower after model creation — lets Phase 3 start
    from base + tuned vision without materializing a full grafted checkpoint on disk."""

    # --- Memory / Optimizer ---
    optim: str = "adamw_torch"
    """HuggingFace TrainingArguments optimizer. Use "adamw_bnb_8bit" (requires bitsandbytes)
    to shrink optimizer state ~4x on VRAM-limited GPUs (e.g. 32 GB consumer cards)."""

    gradient_checkpointing: bool = False
    """If True, recompute activations during backward to trade compute for VRAM."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """

    use_percentiles: bool = True
    """
    If True, use q01/q99 percentile statistics for state/action min-max normalization.
    If False, use full min/max statistics.
    """

    letter_box_transform: bool = False

    shortest_image_edge: int | None = None
    """
    Resize images so the shortest edge has this size before fractional cropping.
    If set, crop_fraction must also be set and legacy image_crop_size/image_target_size
    preprocessing is disabled.
    """

    crop_fraction: float | None = None
    """
    Fraction of the resized image retained by the random/center crop.
    If set, shortest_image_edge must also be set and legacy image_crop_size/image_target_size
    preprocessing is disabled.
    """

    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total batch summed across all GPUs in one forward/backward, BEFORE
    gradient accumulation."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Forward passes per optimizer step. Multiplies ``global_batch_size`` to
    produce the post-accumulation per-optimizer-step batch."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    ds_weights_alpha: float | None = None
    """Power-law exponent for dataset soup weighting. When set, each dataset's
    sampling weight is len(dataset)^alpha and per-dataset mix_ratio values are ignored."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    resume_from_checkpoint: bool = False
    """If True, resume from the latest ``checkpoint-*`` in ``output_dir``. Default
    False so a rerun against an existing ``output_dir`` starts fresh instead of
    silently merging with a previous experiment. Incompatible with
    ``save_only_model=True`` (enforced by ``experiment.run``)."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    def __post_init__(self) -> None:
        if self.ttrtc_enabled:
            if self.tune_llm or self.tune_visual or self.attn_supervision_layers:
                raise ValueError("TTRTC freezes the entire VLM and disables attention supervision")
            if not (self.tune_projector and self.tune_diffusion_model and self.tune_vlln):
                raise ValueError("TTRTC trains all existing action modules")
            if self.ttrtc_max_delay_steps < 0:
                raise ValueError("ttrtc_max_delay_steps must be nonnegative")
            if not 0 < self.ttrtc_zero_delay_probability <= 1:
                raise ValueError("ttrtc_zero_delay_probability must be in (0, 1]")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.gradient_accumulation_steps > 1:
            accumulated_batch_size = self.global_batch_size * self.gradient_accumulation_steps
            warnings.warn(
                f"global_batch_size={self.global_batch_size} is pre-accumulation; "
                f"accumulated_batch_size={accumulated_batch_size} "
                f"(× gradient_accumulation_steps={self.gradient_accumulation_steps}).",
                stacklevel=2,
            )
