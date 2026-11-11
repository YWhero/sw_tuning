"""Command/path regression tests: all subprocesses are mocked; no training."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "training_pipeline", Path(__file__).resolve().parents[1] / "pipeline.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class TrainingCLITest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "tuning"
        self.root.mkdir()
        self.runs = self.root / "checkpoints/groot"
        self.shared = Path(temporary.name) / "shared-models"
        checkpoint = self.root / "checkpoints/base"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(json.dumps({"action_horizon": 40}))
        dataset = self.root / "data/test/meta"
        dataset.mkdir(parents=True)
        (dataset / "info.json").write_text(json.dumps({"fps": 15}))
        self.environment = {"CYCLO_GROOT_MODEL_DIR": str(self.shared)}

    def invoke(self, phase="groot", *, check=True, name="example", environment=None,
               base=None, on_launch=None, on_active_check=None,
               native=False, background=False):
        base = base or ("nvidia/GR00T-N1.7-3B" if phase in ("vision", "hybrid") else "checkpoints/base")
        arguments = ["pipeline.py", phase, "--dataset", "test", "--base", base, "--name", name]
        if check:
            arguments.append("--check")
        if background:
            arguments.append("--background")

        def fake_launch(command, **kwargs):
            if on_launch is not None:
                on_launch(command)
            return SimpleNamespace(stdout=[], wait=lambda: 0)

        def fake_run(command, **kwargs):
            if "pgrep" in command and on_active_check is not None:
                on_active_check()
            return SimpleNamespace(returncode=1)

        with (
            mock.patch.object(pipeline, "ROOT", self.root),
            mock.patch.object(pipeline, "NATIVE", native),
            mock.patch.object(pipeline.subprocess, "run", side_effect=fake_run) as run,
            mock.patch.object(pipeline.subprocess, "Popen", side_effect=fake_launch) as launch,
            mock.patch.dict(os.environ, {**self.environment, **(environment or {})}, clear=True),
            mock.patch.object(sys, "argv", arguments),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            pipeline.main()
        if check:
            launch.assert_not_called()
        return output.getvalue(), run, launch

    @staticmethod
    def resumable_checkpoint(root, step):
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / filename).touch()
        (checkpoint / "trainer_state.json").write_text("{}")
        return checkpoint

    def test_vision_requires_original_base_without_overlay(self):
        for phase in ("vision", "hybrid"):
            with self.assertRaisesRegex(ValueError, "original"):
                self.invoke(phase, base="checkpoints/previous-tuned-model")
            for key in ("VLM_WEIGHTS", "VISION_WEIGHTS"):
                with self.assertRaisesRegex(ValueError, "must not overlay"):
                    self.invoke(phase, environment={key: "/tmp/old.safetensors"})
            output, _, _ = self.invoke(phase)
            self.assertIn("--base-model-path nvidia/GR00T-N1.7-3B", output)

    def test_vision_uses_kl_only_all_frames_and_separate_run_storage(self):
        output, _, _ = self.invoke("vision")
        for flag in (
            "--vision-kl-only", "--attn-supervision-weight 1.0",
            "--attn-supervision-layers 15", "--attn-supervision-loss kl",
            "--max-steps 1000", "--learning-rate 1e-5",
            "--warmup-ratio 0.00625", "--no-letter-box-transform",
            "--modality-config-path /workspace/configs/groot_n17_target_fullframes_config.py",
            "--shard-size 128", "--num-shards-per-epoch 100000",
            "--global-batch-size 8", "--gradient-accumulation-steps 8",
            "--no-tune-llm", "--tune-visual", "--no-tune-projector",
            "--no-tune-diffusion-model", "--no-tune-vlln",
            "--output-dir /workspace/checkpoints/vision_runs",
        ):
            self.assertIn(flag, output)
        self.assertIn("/workspace/checkpoints/vision_runs/example/checkpoint-<step>", output)
        self.assertNotIn("--qwen-layer11-qk-kl-only", output)
        self.assertNotIn("--select-layer", output)

    def test_hybrid_trains_only_whole_visual_and_checks_masks(self):
        output, run, _ = self.invoke("hybrid")
        for flag in ("--attn-supervision-loss feature_hybrid", "--no-vision-kl-only",
                     "--upper-background-attention-weight 0.1",
                     "--tune-visual", "--no-tune-llm", "--no-tune-projector",
                     "--no-tune-diffusion-model", "--no-tune-vlln", "--no-ttrtc-enabled",
                     "--attn-supervision-layers 15", "--no-letter-box-transform",
                     "--max-steps 1000", "--global-batch-size 8",
                     "--gradient-accumulation-steps 8"):
            self.assertIn(flag, output)
        self.assertNotIn(" --vision-kl-only ", output)
        self.assertNotIn("--vlm-last2-feature-only", output)
        self.assertIn("checkpoints/vision_runs", output)
        self.assertIn("example-vlm.safetensors", output)
        self.assertTrue(any("--masks" in call.args[0] for call in run.call_args_list))

    def test_upper_weight_override_validation_and_nonhybrid_isolation(self):
        for value in ("0", "0.25"):
            output, _, _ = self.invoke("hybrid", environment={"UPPER_BACKGROUND_ATTENTION_WEIGHT": value})
            self.assertIn("--upper-background-attention-weight " + value, output)
        for value in ("-1", "nan", "inf", "bad"):
            with self.assertRaises(ValueError):
                self.invoke("hybrid", environment={"UPPER_BACKGROUND_ATTENTION_WEIGHT": value})
        for phase in ("vision", "groot"):
            output, _, _ = self.invoke(phase, environment={"UPPER_BACKGROUND_ATTENTION_WEIGHT": "0.7"})
            self.assertNotIn("--upper-background-attention-weight", output)

    def test_upper_weight_survives_background_environment(self):
        with mock.patch.dict(os.environ, {"UPPER_BACKGROUND_ATTENTION_WEIGHT": "0.25"}, clear=True):
            command = pipeline.background_command(["hybrid"])
        self.assertIn("UPPER_BACKGROUND_ATTENTION_WEIGHT=0.25", command)
        with mock.patch.dict(os.environ, {}, clear=True):
            command = pipeline.background_command(["hybrid"])
        index = command.index("UPPER_BACKGROUND_ATTENTION_WEIGHT")
        self.assertEqual(command[index-1], "-u")

    def test_hybrid_exports_visual_weights_and_records_method(self):
        folder = self.root / "checkpoints/vision_runs/example"
        _, run, _ = self.invoke("hybrid", check=False,
            on_launch=lambda command: self.resumable_checkpoint(folder, 1000))
        manifest = json.loads((folder / "run_manifest.json").read_text())
        self.assertIn("feature_hybrid", manifest["method"])
        self.assertEqual(manifest["upper_background_attention_weight"], 0.1)
        self.assertTrue(any(any("extract_vlm.py" in str(arg) for arg in call.args[0])
                            for call in run.call_args_list))

    def test_action_trains_whole_head_with_overlay_and_frozen_backbone(self):
        overlay = self.root / "checkpoints/vision.safetensors"
        overlay.touch()
        _, _, launch = self.invoke("action", check=False,
                                  environment={"VISION_WEIGHTS": str(overlay)})
        command = launch.call_args.args[0]
        for flag in ("--no-tune-llm", "--no-tune-visual", "--tune-projector",
                     "--tune-diffusion-model", "--tune-vlln", "--no-vision-kl-only",
                     "--no-ttrtc-enabled", "--no-letter-box-transform"):
            self.assertIn(flag, command)
        self.assertNotIn("--attn-supervision-layers", command)
        self.assertEqual(command[command.index("--vlm-weights-path") + 1],
                         "/workspace/checkpoints/vision.safetensors")

    def test_groot_and_action_differ_only_by_visual_overlay(self):
        overlay = self.root / "checkpoints/vision.safetensors"
        overlay.touch()
        for index, overrides in enumerate(({}, {
            "MAX_STEPS": "1200", "GLOBAL_BATCH_SIZE": "8",
            "GRADIENT_ACCUMULATION_STEPS": "8", "LEARNING_RATE": "2e-5",
            "OPTIM": "adamw_bnb_8bit", "STATE_DROPOUT_PROB": "0.1",
            "ACTION_HORIZON": "24", "SAVE_STEPS": "200",
        })):
            commands = []
            for phase in ("groot", "action"):
                name = f"{phase}-{index}"
                environment = dict(overrides)
                if phase == "action":
                    environment["VISION_WEIGHTS"] = str(overlay)
                _, run, launch = self.invoke(
                    phase, check=False, name=name, environment=environment,
                    base="nvidia/GR00T-N1.7-3B")
                command = list(launch.call_args.args[0])
                self.assertTrue(any("check_training_inputs.py" in str(c.args) for c in run.call_args_list))
                command[command.index("--experiment-name") + 1] = "comparison"
                if phase == "action":
                    position = command.index("--vlm-weights-path")
                    del command[position:position + 2]
                else:
                    self.assertNotIn("--vlm-weights-path", command)
                    manifest = json.loads((self.root / f"checkpoints/groot/{name}/run_manifest.json").read_text())
                    self.assertEqual(manifest["vision_overlay"], "")
                    self.assertNotIn("tuned visual overlay", manifest["method"])
                self.assertIn("--no-letter-box-transform", command)
                self.assertNotIn("--letter-box-transform", command)
                commands.append(command)
            self.assertEqual(commands[0], commands[1])

    def test_groot_rejects_accidental_visual_overlay(self):
        with self.assertRaisesRegex(ValueError, "groot.*VISION_WEIGHTS"):
            self.invoke("groot", environment={"VISION_WEIGHTS": "accidental.safetensors"})

    def test_conda_launch_uses_current_python_without_docker(self):
        _, run, launch = self.invoke("groot", check=False, native=True)
        command = launch.call_args.args[0]
        self.assertIn(sys.executable, command)
        self.assertIn(str(self.root / "Isaac-GR00T/gr00t/experiment/launch_finetune.py"), command)
        self.assertNotIn("docker", command)
        self.assertTrue(all("docker" not in call.args[0] for call in run.call_args_list))
        self.assertTrue(any("check_training_inputs.py" in str(call.args) for call in run.call_args_list))
        self.assertFalse(any("/workspace" in arg for arg in command))
        self.assertEqual(command[command.index("--global-batch-size") + 1], "64")

    def test_background_clears_stale_server_settings_and_preserves_values(self):
        _, run, launch = self.invoke(check=False, background=True, native=True,
            environment={"MAX_STEPS": "222", "GLOBAL_BATCH_SIZE": "64",
                         "PYTHONPATH": "/path with spaces", "TUNING_RUNTIME": "conda"})
        launch.assert_not_called()
        command = next(call.args[0] for call in run.call_args_list if call.args[0][0] == "tmux")
        self.assertIn("MAX_STEPS=222", command)
        self.assertIn("GLOBAL_BATCH_SIZE=64", command)
        self.assertIn("PYTHONPATH=/path with spaces", command)
        self.assertIn("-c", command)
        self.assertIn("LEARNING_RATE", command)  # env -u prevents stale inheritance.
        self.assertNotIn("LEARNING_RATE=", command)

    def test_resume_rejects_trainer_restoring_different_batch(self):
        self.invoke(check=False)
        checkpoint = self.resumable_checkpoint(self.runs / "example", 10)
        (checkpoint / "trainer_state.json").write_text('{"train_batch_size": 4}')
        with self.assertRaisesRegex(ValueError, "train_batch_size"):
            self.invoke(environment={"RESUME": "1"})

    def test_running_historical_trainer_is_rejected(self):
        def running_wrapper(command, **kwargs):
            self.assertIsNotNone(re.search(command[-1],
                "/python /tuning/logs/experiment/launch_verified.py --max-steps 8000"))
            return SimpleNamespace(returncode=0)
        with mock.patch.object(pipeline, "NATIVE", True), \
             mock.patch.object(pipeline.subprocess, "run", side_effect=running_wrapper):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                pipeline.ensure_no_trainer()

    def test_task802_policy_checks_resolved_dataset_not_alias(self):
        canonical = self.root / "data/802_mcap_lerobot_v21"
        canonical.mkdir()
        alias = self.root / "data/alias"
        alias.symlink_to(canonical)
        with self.assertRaisesRegex(ValueError, "object_1"):
            pipeline.check_dataset_policy(str(alias))
        with self.assertRaisesRegex(ValueError, "canonical|802_mcap_lerobot_v21"):
            pipeline.check_dataset_policy(str(self.root / "data/802_test_kl_target"))

    def test_native_export_uses_host_paths(self):
        run_dir = self.root / "checkpoints/vision_runs/native-vision"
        _, run, _ = self.invoke("vision", check=False, native=True, name="native-vision",
            on_launch=lambda command: self.resumable_checkpoint(run_dir, 2))
        export = next(call.args[0] for call in run.call_args_list
                      if str(self.root / "tools/extract_vlm.py") in call.args[0])
        self.assertEqual(export[-2:], [str(run_dir / "checkpoint-2"),
            str(self.root / "checkpoints/vision/native-vision-vlm.safetensors")])



    def test_state_dropout_defaults_for_training_phases(self):
        overlay = self.root / "checkpoints/vision.safetensors"
        overlay.touch()
        for phase, expected in (("vision", "0"), ("action", "0.2"), ("groot", "0.2")):
            with self.subTest(phase=phase):
                environment = {"VISION_WEIGHTS": str(overlay)} if phase == "action" else {}
                output, _, _ = self.invoke(phase, environment=environment)
                self.assertIn(f"--state-dropout-prob {expected} ", output)

    def test_state_dropout_override_is_recorded_and_resume_requires_same_value(self):
        environment = {"STATE_DROPOUT_PROB": "0.15"}
        output, _, launch = self.invoke(check=False, environment=environment)
        self.assertIn("--state-dropout-prob 0.15 ", output)
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--state-dropout-prob") + 1], "0.15")
        manifest = json.loads((self.runs / "example/run_manifest.json").read_text())
        self.assertEqual(manifest["state_dropout_prob"], 0.15)
        self.resumable_checkpoint(self.runs / "example", 10)
        self.invoke(environment={**environment, "RESUME": "1"})
        with self.assertRaisesRegex(ValueError, "identical recorded"):
            self.invoke(environment={"RESUME": "1"})


    def test_state_dropout_rejects_nonfinite_and_out_of_range_values(self):
        for value in ("nan", "inf", "-inf", "-0.1", "1.1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "STATE_DROPOUT_PROB"):
                    self.invoke(environment={"STATE_DROPOUT_PROB": value})
        self.assertFalse(self.shared.exists())


    def test_sibling_experiments_do_not_block_or_receive_root_manifest(self):
        sibling = self.runs / "existing"
        self.resumable_checkpoint(sibling, 9000)
        sentinel = self.runs / "run_manifest.json"
        sentinel.write_text('{"sentinel": true}')
        self.invoke(check=False, name="first")
        self.invoke(check=False, name="second")
        for name in ("first", "second"):
            manifest = json.loads((self.runs / name / "run_manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_directory"], f"/workspace/checkpoints/groot/{name}")
            self.assertEqual(manifest["host_checkpoint_directory"], str(self.runs / name))
            self.assertEqual(manifest["command"][manifest["command"].index("--output-dir") + 1], "/workspace/checkpoints/groot")
        self.assertEqual(sentinel.read_text(), '{"sentinel": true}')
        with self.assertRaisesRegex(FileExistsError, "Existing experiment"):
            self.invoke(name="first")

    def test_locked_guard_rechecks_experiment_created_after_preflight(self):
        sentinel = self.runs / "example/foreign-output.txt"

        def competing_writer():
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("preserve this output")

        with self.assertRaises(FileExistsError):
            self.invoke(check=False, on_active_check=competing_writer)
        self.assertEqual(sentinel.read_text(), "preserve this output")
        self.assertFalse((sentinel.parent / "run_manifest.json").exists())

    def test_manifest_creation_cannot_overwrite_a_racing_writer(self):
        manifest = self.runs / "example/run_manifest.json"
        original_open = Path.open

        def racing_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path == manifest and mode in {"w", "x"}:
                with original_open(path, "w") as stream:
                    stream.write('{"competing_writer":true}')
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", racing_open):
            with self.assertRaises(FileExistsError):
                self.invoke(check=False)
        self.assertEqual(manifest.read_text(), '{"competing_writer":true}')

    def test_resume_reads_only_selected_experiment_and_numeric_latest(self):
        self.invoke(check=False)
        experiment = self.runs / "example"
        manifest_before = (experiment / "run_manifest.json").read_text()
        (experiment / "checkpoint-9").mkdir()  # Incomplete older checkpoint is irrelevant.
        latest = self.resumable_checkpoint(experiment, 10)
        self.resumable_checkpoint(self.runs / "other", 9999)
        output, _, _ = self.invoke(environment={"RESUME": "1"})
        self.assertIn("--resume-from-checkpoint", output)
        self.assertEqual((experiment / "run_manifest.json").read_text(), manifest_before)
        (latest / "rng_state.pth").unlink()
        with self.assertRaisesRegex(ValueError, "not resumable"):
            self.invoke(environment={"RESUME": "1"})

    def test_sibling_checkpoint_cannot_supply_resume_and_config_must_match(self):
        self.invoke(check=False)
        self.resumable_checkpoint(self.runs / "other", 9999)
        with self.assertRaisesRegex(ValueError, "not resumable"):
            self.invoke(environment={"RESUME": "1"})
        self.resumable_checkpoint(self.runs / "example", 10)
        with self.assertRaisesRegex(ValueError, "identical recorded"):
            self.invoke(environment={"RESUME": "1", "MAX_STEPS": "1234"})

    def test_explicit_output_remains_base_directory(self):
        self.invoke(check=False, environment={"OUTPUT_DIR": "checkpoints/custom"})
        manifest_path = self.root / "checkpoints/custom/example/run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["checkpoint_directory"], "/workspace/checkpoints/custom/example")
        command = manifest["command"]
        self.assertEqual(command[command.index("--output-dir") + 1], "/workspace/checkpoints/custom")
        self.assertFalse((self.root / "checkpoints/custom/run_manifest.json").exists())

    def test_shared_host_checkpoint_is_accepted_as_base(self):
        checkpoint = self.shared / "trained/checkpoint-50"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text('{"action_horizon":40}')
        output, _, _ = self.invoke(base=str(checkpoint))
        self.assertIn("--base-model-path /workspace/model/groot/trained/checkpoint-50", output)

    def test_action_uses_single_experiment_directory(self):
        overlay = self.root / "checkpoints/vision.safetensors"
        overlay.touch()
        self.invoke("action", check=False, name="legacy-action",
                    environment={"VISION_WEIGHTS": str(overlay)}, base="nvidia/GR00T-N1.7-3B")
        output_base = self.root / "checkpoints/groot/legacy-action"
        manifest = json.loads((output_base / "run_manifest.json").read_text())
        command = manifest["command"]
        self.assertEqual(command[command.index("--output-dir") + 1], "/workspace/checkpoints/groot")
        self.assertEqual(command[command.index("--base-model-path") + 1], "nvidia/GR00T-N1.7-3B")
        self.assertEqual(manifest["checkpoint_directory"], "/workspace/checkpoints/groot/legacy-action")
        self.assertFalse(self.shared.exists())

    def test_vision_export_scans_run_directory_and_keeps_destination(self):
        output_base = self.root / "checkpoints/vision_runs/legacy-vision"

        def produce_checkpoint(command):
            self.resumable_checkpoint(output_base, 2)

        _, run, _ = self.invoke("vision", check=False, name="legacy-vision", on_launch=produce_checkpoint)
        self.assertTrue((output_base / "run_manifest.json").is_file())
        export = next(call.args[0] for call in run.call_args_list if "/workspace/tools/extract_vlm.py" in call.args[0])
        self.assertEqual(export[-2:], [
            "/workspace/checkpoints/vision_runs/legacy-vision/checkpoint-2",
            "/workspace/checkpoints/vision/legacy-vision-vlm.safetensors",
        ])
        self.assertFalse(self.shared.exists())

    def test_vision_and_action_resume_keep_run_manifest(self):
        overlay = self.root / "checkpoints/resume-vision.safetensors"
        overlay.touch()
        for phase, folder in (("vision", "vision_runs"), ("action", "groot")):
            with self.subTest(phase=phase):
                name = f"resume-{phase}"
                output_base = self.root / f"checkpoints/{folder}/{name}"
                environment = {"VISION_WEIGHTS": str(overlay)} if phase == "action" else {}

                def produce_checkpoint(command):
                    self.resumable_checkpoint(output_base, 10)

                self.invoke(phase, check=False, name=name, environment=environment,
                            on_launch=produce_checkpoint)
                manifest_path = output_base / "run_manifest.json"
                manifest_before = manifest_path.read_text()
                self.assertFalse((output_base / name / "run_manifest.json").exists())
                output, _, _ = self.invoke(phase, name=name,
                    environment={**environment, "RESUME": "1"})
                self.assertIn("--resume-from-checkpoint", output)
                self.assertEqual(manifest_path.read_text(), manifest_before)


if __name__ == "__main__":
    unittest.main()
