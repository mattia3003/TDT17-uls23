"""Checkpoints that resume on either machine, and the config check that guards a resume."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

# Keys that may differ between sessions of one run, so a run can move between IDUN and the home PC.
MACHINE_KEYS = ("machine", "paths")


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> None:
    """Write via a temporary file, so a job killed mid-write never leaves a broken checkpoint."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(state), tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load to the CPU (tensors are moved to the GPU by ``load_state_dict``), plain tensors and numbers only."""
    return torch.load(path, map_location="cpu", weights_only=True)


def config_differences(saved: Mapping[str, Any], current: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys whose values differ, ignoring the machine-specific top-level keys."""
    diffs = []
    for key in sorted(set(saved) | set(current)):
        if not prefix and key in MACHINE_KEYS:
            continue
        a, b = saved.get(key), current.get(key)
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            diffs += config_differences(a, b, f"{prefix}{key}.")
        elif a != b:
            diffs.append(f"{prefix}{key}")
    return diffs
