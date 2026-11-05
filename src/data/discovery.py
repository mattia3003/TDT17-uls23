"""Cheap helpers for finding files in a dataset tree without walking all of it."""

from __future__ import annotations

import os
from pathlib import Path

IMAGE_SUFFIXES = (".nii.gz", ".nii", ".mha", ".nrrd")


def is_image_file(name: str) -> bool:
    """True if ``name`` has a medical-image suffix we can load."""
    return name.lower().endswith(IMAGE_SUFFIXES)


def find_first_pair(
    root: str | Path,
    image_dir: str = "images",
    label_dir: str = "labels",
    max_depth: int = 4,
) -> tuple[Path, Path] | None:
    """Return the first ``(image, label)`` pair under ``root``, or ``None``.

    Looks top-down for a directory named ``image_dir`` whose sibling ``label_dir``
    holds a file with the same name. Stops at the first match and never descends
    more than ``max_depth`` levels, so it stays light on large trees.
    """
    root = Path(root)
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames.sort()
        if len(current.parts) - base_depth >= max_depth:
            dirnames.clear()
        if current.name != image_dir:
            continue
        label_root = current.parent / label_dir
        for name in sorted(filenames):
            if is_image_file(name) and (label_root / name).is_file():
                return current / name, label_root / name
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
