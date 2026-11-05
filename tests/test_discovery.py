"""Tests for dataset-tree discovery helpers, on a tiny synthetic tree."""

from pathlib import Path

from src.data.discovery import describe_tree, find_first_pair


def make_tree(root: Path) -> None:
    for rel in [
        "novel/DS_B/images/b_001.nii.gz",
        "novel/DS_B/labels/b_001.nii.gz",
        "novel/DS_A/images/a_001.nii.gz",
        "novel/DS_A/images/a_002.nii.gz",
        "novel/DS_A/labels/a_002.nii.gz",
        "novel/DS_A/readme.txt",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_find_first_pair_skips_unpaired_images(tmp_path: Path) -> None:
    make_tree(tmp_path)
    pair = find_first_pair(tmp_path)
    assert pair is not None
    image, label = pair
    assert image == tmp_path / "novel/DS_A/images/a_002.nii.gz"
    assert label == tmp_path / "novel/DS_A/labels/a_002.nii.gz"


def test_find_first_pair_respects_max_depth(tmp_path: Path) -> None:
    make_tree(tmp_path)
    assert find_first_pair(tmp_path, max_depth=1) is None


def test_describe_tree_counts(tmp_path: Path) -> None:
    make_tree(tmp_path)
    lines = describe_tree(tmp_path, max_depth=2)
    assert lines[0].endswith("(1 dirs, 0 files)")
    assert any("DS_A/  (2 dirs, 1 files)" in line for line in lines)
