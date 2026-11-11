"""Shared Cyclo checkpoint mount and training-workspace path conversion."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

CYCLO_MODEL_PATH = "/workspace/model/groot"


def shared_model_root(project_root: Path) -> Path:
    value = os.environ.get("CYCLO_GROOT_MODEL_DIR")
    path = Path(value).expanduser() if value else project_root.parent / "cyclo_intelligence/docker/workspace/model/groot"
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _relative_container_path(value: str, prefix: str):
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError(f"Container path may not traverse parent directories: {value}")
    try:
        return path.relative_to(prefix)
    except ValueError:
        return None


def to_container_path(value: str, project_root: Path) -> str:
    value = str(value)
    if not value:
        return value
    if value.startswith("/workspace/") or value == "/workspace":
        # Validate shared-mount escapes as well as lexical traversal.
        to_host_path(value, project_root)
        return str(PurePosixPath(value))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    for host_root, container_root in (
        (shared_model_root(project_root), CYCLO_MODEL_PATH),
        (project_root.resolve(), "/workspace"),
    ):
        try:
            return str(PurePosixPath(container_root) / path.relative_to(host_root))
        except ValueError:
            pass
    raise ValueError(f"Path must be within the tuning workspace or shared Cyclo model mount: {path}")


def to_host_path(value: str, project_root: Path) -> Path:
    value = str(value)
    if value.startswith("/workspace/") or value == "/workspace":
        for prefix, root in (
            (CYCLO_MODEL_PATH, shared_model_root(project_root)),
            ("/workspace", project_root.resolve()),
        ):
            relative = _relative_container_path(value, prefix)
            if relative is None:
                continue
            path = root / str(relative)
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Path escapes its mounted root: {value}")
            return path
    return Path(value).expanduser()
