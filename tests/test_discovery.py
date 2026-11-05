"""Tests for dataset-tree discovery helpers, on a tiny synthetic tree mirroring the IDUN layout."""

from pathlib import Path

from src.data.discovery import dataset_dirs, describe_tree, first_pair, is_image_file

DATA_CFG = {"image_dir": "data/{dataset}/images", "label_dir": "annotations/{dataset}/labels"}


def make_tree(root: Path) -> None:
    for rel in [
        "data/DS_A/images/a_001.nii.gz",
        "data/DS_A/images/a_002.nii.gz",
        "annotations/DS_A/labels/a_002.nii.gz",
        "annotations/DS_A/labels/a_002.nii.gz.zip",
        "annotations/README.md",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_is_image_file_rejects_zip_copies() -> None:
    assert is_image_file("x_lesion_01.nii.gz")
    assert not is_image_file("x_lesion_01.nii.gz.zip")


def test_dataset_dirs_fill_template(tmp_path: Path) -> None:
    image_dir, label_dir = dataset_dirs(tmp_path, DATA_CFG, "DS_A")
    assert image_dir == tmp_path / "data/DS_A/images"
    assert label_dir == tmp_path / "annotations/DS_A/labels"


def test_first_pair_skips_unlabelled_images(tmp_path: Path) -> None:
    make_tree(tmp_path)
    pair = first_pair(*dataset_dirs(tmp_path, DATA_CFG, "DS_A"))
    assert pair is not None
    image, label = pair
    assert image == tmp_path / "data/DS_A/images/a_002.nii.gz"
    assert label == tmp_path / "annotations/DS_A/labels/a_002.nii.gz"


def test_first_pair_returns_none_without_labels(tmp_path: Path) -> None:
    make_tree(tmp_path)
    (tmp_path / "annotations/DS_A/labels/a_002.nii.gz").unlink()
    assert first_pair(*dataset_dirs(tmp_path, DATA_CFG, "DS_A")) is None


def test_describe_tree_counts(tmp_path: Path) -> None:
    make_tree(tmp_path)
    lines = describe_tree(tmp_path, max_depth=2)
    assert lines[0].endswith("(2 dirs, 0 files)")
    assert any("annotations/  (1 dirs, 1 files)" in line for line in lines)
