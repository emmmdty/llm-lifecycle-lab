"""Stage 4 pretrain: run records (command / config / environment / hardware / revision / seed)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import importlib.metadata

_VERSION_KEYS = ("python", "torch", "tokenizers", "transformers", "numpy")


def git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def gather_environment(
    version_getter: Callable[[str], str] | None = None,
) -> dict[str, str | None]:
    version_getter = version_getter or importlib.metadata.version
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for key in _VERSION_KEYS[1:]:
        try:
            versions[key] = version_getter(key)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _default_cuda() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            driver = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        driver = None
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        "total_memory_mib": props.total_memory // (1024 * 1024),
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
    }


def gather_hardware(cuda_provider: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    cuda_provider = cuda_provider or _default_cuda
    return {
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda": cuda_provider(),
    }


def build_run_record(
    *,
    run_id: str,
    command: str,
    config: dict[str, Any],
    revision: dict[str, Any],
    seed: int,
    environment: dict[str, Any],
    hardware: dict[str, Any],
    git: str | None,
    resume_from: str | None = None,
    notes: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "git_commit": git,
        "resume_from": resume_from,
        "config": config,
        "revision": revision,
        "seed": seed,
        "environment": environment,
        "hardware": hardware,
        "notes": list(notes),
    }
