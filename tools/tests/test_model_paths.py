"""Shared model mount conversion tests; no containers or real model files."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "shared_model_paths", Path(__file__).resolve().parents[1] / "model_paths.py"
)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "tuning"
        self.root.mkdir()
        self.shared = Path(temporary.name) / "shared"
        self.shared.mkdir()
        environment = mock.patch.dict(os.environ, {"CYCLO_GROOT_MODEL_DIR": str(self.shared)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_shared_root_and_checkpoint_round_trip(self):
        for relative in ("", "experiment/checkpoint-1000", "model with spaces/checkpoint-2"):
            host = self.shared / relative
            container = str(Path("/workspace/model/groot") / relative)
            self.assertEqual(paths.to_container_path(str(host), self.root), container)
            self.assertEqual(paths.to_host_path(container, self.root), host)

    def test_default_and_relative_shared_root_resolution(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.shared_model_root(self.root),
                             self.root.parent / "cyclo_intelligence/docker/workspace/model/groot")
        with mock.patch.dict(os.environ, {"CYCLO_GROOT_MODEL_DIR": "relative-models"}, clear=True):
            self.assertEqual(paths.shared_model_root(self.root), self.root / "relative-models")

    def test_tuning_mapping_and_sibling_prefix_stay_distinct(self):
        self.assertEqual(paths.to_container_path("checkpoints/base", self.root), "/workspace/checkpoints/base")
        self.assertEqual(paths.to_host_path("/workspace/checkpoints/base", self.root), self.root / "checkpoints/base")
        self.assertEqual(paths.to_host_path("/workspace/model/groot-other/run", self.root),
                         self.root / "model/groot-other/run")
        with self.assertRaises(ValueError):
            paths.to_container_path(str(self.shared.with_name("shared-other") / "run"), self.root)

    def test_parent_traversal_and_symlink_escape_are_rejected(self):
        for container in ("/workspace/model/groot/../outside", "/workspace/../outside"):
            with self.subTest(container=container):
                with self.assertRaises(ValueError):
                    paths.to_host_path(container, self.root)
                with self.assertRaises(ValueError):
                    paths.to_container_path(container, self.root)
        outside = self.root.parent / "outside"
        outside.mkdir()
        (self.shared / "escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            paths.to_host_path("/workspace/model/groot/escape/checkpoint-1", self.root)
        with self.assertRaises(ValueError):
            paths.to_container_path(str(self.shared / "escape/checkpoint-1"), self.root)


if __name__ == "__main__":
    unittest.main()
