"""Shared launcher for vision tuning and GR00T action-head training."""

import argparse
import json
import math
import os
import shlex
import signal
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("TUNING_RUNTIME") == "conda"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from model_paths import shared_model_root, to_container_path, to_host_path


def container_path(value):
    return to_container_path(value, ROOT)


def host_path(value):
    path = to_host_path(value, ROOT)
    return path if path.is_absolute() else ROOT / path


def runtime_path(value):
    return str(host_path(value)) if NATIVE else container_path(value)


def env(name, default):
    return os.environ.get(name, str(default))


# Clear absent values too: tmux's server may retain a previous run's settings.
BACKGROUND_ENV = (
    "VLM_WEIGHTS", "TUNING_RUNTIME", "GPU", "ACTION_HORIZON", "OUTPUT_DIR", "MAX_STEPS",
    "GLOBAL_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "STATE_DROPOUT_PROB",
    "SAVE_STEPS", "SAVE_TOTAL_LIMIT", "LEARNING_RATE", "OPTIM",
    "DATALOADER_NUM_WORKERS", "SHORTEST_IMAGE_EDGE", "CROP_FRACTION",
    "UPPER_BACKGROUND_ATTENTION_WEIGHT", "RESUME", "ALLOW_LEGACY_MASKS", "CYCLO_GROOT_MODEL_DIR",
    "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_HOME", "HF_HOME",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE",
    "PYTORCH_ALLOC_CONF", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM", "NO_ALBUMENTATIONS_UPDATE",
)


def background_command(arguments):
    command = ["env"]
    for key in BACKGROUND_ENV:
        command += ["-u", key]
    command += [f"{key}={os.environ[key]}" for key in BACKGROUND_ENV if key in os.environ]
    return command + [sys.executable, str(Path(__file__).resolve()), *arguments]


def ensure_no_trainer():
    # Historical sessions use launch_verified.py; check both host and container.
    pattern = r"([l]aunch_finetune|[l]aunch_verified)\.py"
    commands = [["pgrep", "-f", pattern]]
    if not NATIVE:
        commands.append(["docker", "compose", "exec", "-T", "gr00t", "pgrep", "-f", pattern])
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            raise RuntimeError("A trainer is already running; stop it before starting another GPU pipeline")
        if result.returncode != 1:
            raise RuntimeError("Could not check for running trainers")


def check_dataset_policy(dataset):
    root = host_path(dataset).resolve()
    if root.name in {"802_test", "802_test_kl_target", "802_test_kl_union", "802_test_tactile"}:
        raise ValueError("New Task 802 runs must use data/802_mcap_lerobot_v21; historical data is read-only")
    if root.name == "802_mcap_lerobot_v21":
        raise ValueError(
            "Canonical Task 802 data needs a verified left_arm/right_arm + task configuration "
            "and object_1 target-mask loader. The current arms/sub_task recipe is incompatible; "
            "do not use arr_0 (target OR right hand) as a target mask."
        )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["vision", "hybrid", "action", "groot"])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs and print exact command without training",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--foreground", action="store_true", help="Run attached (default)")
    mode.add_argument("--background", action="store_true", help="Run attached command in tmux")
    parser.add_argument("--dataset", default=os.environ.get("DATASET"), required=not os.environ.get("DATASET"))
    parser.add_argument("--name", default=env("EXP_NAME", ""))
    parser.add_argument("--base", default=env("BASE_MODEL_PATH", "nvidia/GR00T-N1.7-3B"))
    parser.add_argument("--vlm-weights", "--vision-weights", dest="vision_weights", default=env("VLM_WEIGHTS", env("VISION_WEIGHTS", "")),
                        help="Full VLM safetensors exported by vision-kl.sh or vision-hybrid.sh; old option name is an alias")
    args = parser.parse_args()
    phase = args.phase
    if phase in ("vision", "hybrid"):
        if args.base != "nvidia/GR00T-N1.7-3B":
            raise ValueError("New vision tuning must start from original nvidia/GR00T-N1.7-3B, not a tuned checkpoint")
        if args.vision_weights:
            raise ValueError("Vision tuning must not overlay previously tuned VLM/vision weights")
    # Compose and host-side validation must resolve the same shared mount.
    os.environ["CYCLO_GROOT_MODEL_DIR"] = str(shared_model_root(ROOT))
    dataset_arg = args.dataset
    if not Path(dataset_arg).is_absolute() and not dataset_arg.startswith("data/"):
        dataset_arg = "data/" + dataset_arg
    dataset = runtime_path(dataset_arg)
    check_dataset_policy(dataset)
    name = args.name or {
        "vision": "language-kl-vision", "hybrid": "language-hybrid-vision", "action": "groot-language-kl", "groot": "groot-base"
    }[phase]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Experiment name must be a single folder name")
    overlay = args.vision_weights
    if phase == "groot" and overlay:
        raise ValueError("groot training does not use VISION_WEIGHTS; unset it or use scripts/action-train.sh")
    base = (
        runtime_path(args.base)
        if Path(args.base).is_absolute() or args.base.startswith("checkpoints/")
        else args.base
    )
    prefix = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        f"CUDA_VISIBLE_DEVICES={env('GPU', 0)}",
        "-e",
        "GROOT_GROUNDING_ONLY=0",
        "-e",
        f"ACTION_HORIZON={env('ACTION_HORIZON', 16)}",
        "gr00t",
        "python",
    ]
    if NATIVE:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={env('GPU', 0)}",
                  "GROOT_GROUNDING_ONLY=0", f"ACTION_HORIZON={env('ACTION_HORIZON', 16)}",
                  sys.executable]
    vision = phase in ("vision", "hybrid")
    default_output = f"checkpoints/{'vision_runs' if vision else 'groot'}"
    output = runtime_path(env("OUTPUT_DIR", default_output))
    maximum = int(env("MAX_STEPS", 1000 if vision else 100000))
    batch = int(env("GLOBAL_BATCH_SIZE", 8 if vision else 64))
    accum = int(env("GRADIENT_ACCUMULATION_STEPS", 8 if vision else 1))
    # One launcher flag controls both processor and action-head dropout.
    state_dropout_arg = env("STATE_DROPOUT_PROB", 0 if vision else 0.2)
    state_dropout_prob = float(state_dropout_arg)
    if not math.isfinite(state_dropout_prob) or not 0 <= state_dropout_prob <= 1:
        raise ValueError("STATE_DROPOUT_PROB must be finite and in [0,1]")
    if maximum < 1 or batch < 1 or accum < 1:
        raise ValueError("Steps and batch sizes must be positive")
    command = prefix + [
        str(ROOT / "Isaac-GR00T/gr00t/experiment/launch_finetune.py") if NATIVE else "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        base,
        "--dataset-path",
        dataset,
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        runtime_path("configs/groot_n17_target_fullframes_config.py" if vision
                     else "configs/groot_n17_arms_config.py"),
        "--num-gpus",
        "1",
        "--output-dir",
        output,
        "--experiment-name",
        name,
        "--max-steps",
        str(maximum),
        "--save-steps",
        str(min(maximum, int(env("SAVE_STEPS", 2000 if vision else 10000)))),
        "--save-total-limit",
        env("SAVE_TOTAL_LIMIT", 2),
        "--global-batch-size",
        str(batch),
        "--gradient-accumulation-steps",
        str(accum),
        "--learning-rate",
        env("LEARNING_RATE", "1e-5" if vision else "1e-4"),
        "--warmup-ratio",
        "0.00625" if vision else "0.05",
        "--weight-decay",
        "1e-5",
        "--optim",
        env("OPTIM", "adamw_torch"),
        "--state-dropout-prob",
        state_dropout_arg,
        "--dataloader-num-workers",
        env("DATALOADER_NUM_WORKERS", 0 if vision else 8),
        "--shortest-image-edge",
        env("SHORTEST_IMAGE_EDGE", 256),
        "--crop-fraction",
        env("CROP_FRACTION", 0.9),
        "--no-letter-box-transform",
        "--no-tune-llm",
    ]
    command += (
        [
            "--tune-visual",
            "--no-tune-projector",
            "--no-tune-diffusion-model",
            "--no-tune-vlln",
            "--attn-supervision-layers",
            "15",
            "--attn-supervision-weight",
            "1.0",
            "--vision-kl-only" if phase == "vision" else "--no-vision-kl-only",
            "--no-ttrtc-enabled",
            "--shard-size",
            "128",
            "--num-shards-per-epoch",
            "100000",
            "--attn-supervision-loss",
            "kl" if phase == "vision" else "feature_hybrid",
        ]
        if vision
        else ["--no-tune-visual", "--tune-projector", "--tune-diffusion-model", "--tune-vlln"]
    )
    upper_weight = 0.0
    if phase == "hybrid":
        upper_arg = env("UPPER_BACKGROUND_ATTENTION_WEIGHT", "0.1")
        upper_weight = float(upper_arg)
        if not math.isfinite(upper_weight) or upper_weight < 0:
            raise ValueError("UPPER_BACKGROUND_ATTENTION_WEIGHT must be finite and nonnegative")
        command += ["--upper-background-attention-weight", upper_arg]
    if phase in ("action", "groot"):
        command += ["--no-ttrtc-enabled", "--no-vision-kl-only"]
    if phase == "action" and not overlay:
        raise ValueError("Action training requires VLM_WEIGHTS from the vision phase")
    if overlay:
        if not host_path(overlay).is_file():
            raise FileNotFoundError(overlay)
        command += ["--vlm-weights-path", runtime_path(overlay)]
    if vision:
        export_host = ROOT / "checkpoints/vision" / f"{name}-vlm.safetensors"
        if export_host.exists() or export_host.with_suffix(".json").exists():
            raise FileExistsError(f"Existing vision export: {export_host}; choose a new name")
    out_host = host_path(output)
    training_host = out_host / name  # Official launcher appends experiment_name.
    run_host = training_host
    # The shared model root can contain other models. Guard this run only.

    writable_parent = run_host
    while not writable_parent.exists():
        writable_parent = writable_parent.parent
    if not os.access(writable_parent, os.W_OK | os.X_OK):
        raise PermissionError(
            f"Checkpoint directory is not writable: {writable_parent}. "
            "Correct its ownership for the host user before running dataset validation."
        )
    resume = env("RESUME", 0) == "1"
    if run_host.exists() and any(run_host.iterdir()) and not resume:
        raise FileExistsError(f"Existing experiment: {run_host}; choose a new name or RESUME=1")
    if resume:
        checkpoints = sorted(
            training_host.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if (
            not checkpoints
            or not all(
                (checkpoints[-1] / f).exists()
                for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
            )
            or not list(checkpoints[-1].glob("rng_state*.pth"))
        ):
            raise ValueError("Latest checkpoint is not resumable")
        previous = json.loads((run_host / "run_manifest.json").read_text())
        if previous["command"] != command:
            raise ValueError("Resume requires identical recorded training configuration")
        state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
        if state.get("train_batch_size") not in (None, batch):
            raise ValueError("Checkpoint train_batch_size differs from requested batch; Trainer would restore it")
        command += ["--resume-from-checkpoint"]
    print(
        f"Micro/global batch={batch}, GPU=1, accumulation={accum}, effective optimizer batch={batch * accum}"
    )
    print(shlex.join(command))
    print(f"Checkpoint directory: {training_host}")
    print(f"Model path: {runtime_path(str(training_host))}/checkpoint-<step>")
    if vision:
        print(f"VLM weights: {export_host}")
    if not NATIVE:
        subprocess.run(["docker", "compose", "up", "-d", "gr00t"], cwd=ROOT, check=True)
    if not args.check:
        ensure_no_trainer()
    check_command = [
        "docker",
        "compose",
        "exec",
        "-T",
        "gr00t",
        "python",
        "/workspace/tools/check_training_inputs.py",
        dataset,
    ]
    if NATIVE:
        check_command = [sys.executable, str(ROOT / "tools/check_training_inputs.py"), dataset]
    if vision:
        check_command.append("--masks")
        if env("ALLOW_LEGACY_MASKS", 0) == "1":
            check_command.append("--allow-legacy-masks")
    subprocess.run(check_command, cwd=ROOT, check=True)
    if args.check:
        return
    if args.background:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                name,
                "-c", str(ROOT),
                *background_command([
                phase,
                "--foreground",
                "--dataset",
                args.dataset,
                "--name",
                name,
                "--base",
                args.base,
                "--vision-weights",
                overlay,
                ]),
            ],
            check=True,
        )
        print(f"tmux attach -t {name}")
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    import fcntl

    with (ROOT / "logs/.pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_no_trainer()
        if not resume and run_host.exists() and any(run_host.iterdir()):
            raise FileExistsError(f"Existing experiment: {run_host}; choose a new name or RESUME=1")
        run_host.mkdir(parents=True, exist_ok=True)
        if not resume:
            manifest_text = json.dumps(
                    {
                        "command": command,
                        "checkpoint_directory": runtime_path(str(training_host)),
                        "host_checkpoint_directory": str(training_host),
                        "dataset": dataset,
                        "base": base,
                        "vision_overlay": overlay,
                        "vlm_overlay": overlay,
                        "state_dropout_prob": state_dropout_prob,
                        "upper_background_attention_weight": upper_weight,
                        "allow_legacy_masks": env("ALLOW_LEGACY_MASKS", 0) == "1",
                        "created": time.time(),
                        "method": (
                            "whole visual tower KL; LLM layer index 15 (16th layer) final prompt-query image-conditional attention; all frames eligible"
                            if phase == "vision" else
                            "whole visual tower feature_hybrid + anchor + configured upper-background attention penalty; frozen LLM/action; final VLM features"
                            if phase == "hybrid" else
                            "action flow loss; frozen vision and LLM; no visual overlay"
                            if phase == "groot" else
                            "action flow loss; frozen vision and LLM; tuned full VLM overlay"
                        ),
                        },
                    indent=2,
                )
            with (run_host / "run_manifest.json").open("x") as manifest:
                manifest.write(manifest_text)
        with (ROOT / "logs" / f"{name}.log").open("a") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                start_new_session=True,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"Pipeline failed; see logs/{name}.log")
            except KeyboardInterrupt:
                if NATIVE:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait()
                else:
                    subprocess.run([
                        "docker",
                        "compose",
                        "exec",
                        "-T",
                        "gr00t",
                        "pkill",
                        "-INT",
                        "-f",
                        "[l]aunch_finetune.py",
                    ], cwd=ROOT)
                    process.terminate()
                raise
        if vision:
            checkpoints = sorted(
                training_host.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
            )
            if not checkpoints:
                raise RuntimeError("Training completed without a full checkpoint")
            subprocess.run(
                prefix
                + [
                    runtime_path("tools/extract_vlm.py"),
                    runtime_path(str(checkpoints[-1])),
                    runtime_path(str(export_host)),
                ],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        main()
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        BlockingIOError,
        subprocess.CalledProcessError,
    ) as exc:
        sys.exit(str(exc))
