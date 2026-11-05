"""Cheap helpers for locating files in the dataset tree without walking all of it."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = (".nii.gz", ".nii", ".mha", ".nrrd")


def is_image_file(name: str) -> bool:
    """True if ``name`` has a medical-image suffix we can load (``*.nii.gz.zip`` is not one)."""
    return name.lower().endswith(IMAGE_SUFFIXES)


def dataset_dirs(data_root: str | Path, data_cfg: Mapping[str, Any], dataset: str) -> tuple[Path, Path]:
    """Return ``(image_dir, label_dir)`` for ``dataset`` from the ``data`` config section."""
    root = Path(data_root)
    return (
        root / data_cfg["image_dir"].format(dataset=dataset),
        root / data_cfg["label_dir"].format(dataset=dataset),
    )


def first_pair(image_dir: str | Path, label_dir: str | Path) -> tuple[Path, Path] | None:
    """Return the first image in ``image_dir`` (sorted) that has a same-named file in ``label_dir``."""
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    for name in sorted(os.listdir(image_dir)):
        if is_image_file(name) and (label_dir / name).is_file():
            return image_dir / name, label_dir / name
    return None


def describe_tree(root: str | Path, max_depth: int = 2, max_dirs: int = 12, max_files: int = 3) -> list[str]:
    """Summarise the top of a directory tree: per-folder counts plus a few example names."""
    lines: list[str] = []

    def visit(path: Path, depth: int) -> None:
        indent = "  " * depth
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError as err:
            lines.append(f"{indent}{path.name}/  <unreadable: {err}>")
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        lines.append(f"{indent}{path.name}/  ({len(dirs)} dirs, {len(files)} files)")
        for entry in files[:max_files]:
            lines.append(f"{indent}  {entry.name}")
        if len(files) > max_files:
            lines.append(f"{indent}  ... {len(files) - max_files} more files")
        if depth < max_depth:
            for entry in dirs[:max_dirs]:
                visit(Path(entry.path), depth + 1)
            if len(dirs) > max_dirs:
                lines.append(f"{indent}  ... {len(dirs) - max_dirs} more dirs")

    visit(Path(root), 0)
    return lines
