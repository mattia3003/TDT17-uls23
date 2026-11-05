"""Tests for manifest building and the EDA report, on tiny synthetic NIfTI files."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.data.eda import crop_fit, dataset_summary, problem_cases, usable_mask, write_report
from src.data.manifest import build_manifest, list_cases, parse_case_id, voi_stats
from src.utils.config import load_yaml

REPO = Path(__file__).resolve().parents[1]
SHAPE = (32, 32, 16)
SPACING = (0.5, 0.5, 2.0)
PAD_VALUE = -2049.0


def write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.diag([*SPACING, 1.0])), path)


def make_voi(lesion_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Image with 4 padded z-slices and a cube lesion containing the VOI centre (16, 16, 8)."""
    image = np.full(SHAPE, 40.0, dtype=np.float32)
    image[:, :, :4] = PAD_VALUE
    label = np.zeros(SHAPE, dtype=np.uint8)
    lo = 16 - lesion_size // 2
    label[lo:lo + lesion_size, lo:lo + lesion_size, 7:9] = 1
    image[label > 0] = 100.0
    return image, label


def make_dataset(root: Path, dataset: str, cases: dict[str, int], unlabelled: tuple[str, ...] = ()) -> None:
    for case, size in cases.items():
        image, label = make_voi(size)
        write_nifti(root / f"data/{dataset}/images/{case}.nii.gz", image)
        if case not in unlabelled:
            write_nifti(root / f"annotations/{dataset}/labels/{case}.nii.gz", label)
            (root / f"annotations/{dataset}/labels/{case}.nii.gz.zip").touch()


DATA_CFG = {
    "image_dir": "data/{dataset}/images",
    "label_dir": "annotations/{dataset}/labels",
    "datasets": {
        "DS_A": {"id_pattern": r"^p(?P<patient>\d+)_lesion_(?P<lesion>\d+)$", "category_dir": "data/{dataset}/categories"},
        "DS_B": {"id_pattern": r"^b_(?P<patient>\d+)_lesion_(?P<lesion>\d+)$"},
    },
}


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    make_dataset(tmp_path, "DS_A", {"p01_lesion_01": 2, "p01_lesion_02": 4, "p02_lesion_01": 6}, unlabelled=("p02_lesion_01",))
    make_dataset(tmp_path, "DS_B", {"b_7_lesion_01": 4, "b_8_lesion_01": 8})
    image, _ = make_voi()
    write_nifti(tmp_path / "data/DS_A/categories/Lung/p01_lesion_02.nii.gz", image)
    return tmp_path


@pytest.mark.parametrize("case, dataset, patient", [
    ("000010_02_02_100_lesion_01", "ULS23_DeepLesion3D", "000010"),
    ("bone_00002_lesion_03", "ULS23_Radboudumc_Bone", "00002"),
    ("diag_pancreas_0004_lesion_01", "ULS23_Radboudumc_Pancreas", "0004"),
])
def test_repo_id_patterns(case: str, dataset: str, patient: str) -> None:
    cfg = load_yaml(REPO / "configs/data/uls23.yaml")["data"]
    assert parse_case_id(case, cfg["datasets"][dataset]["id_pattern"]) == (patient, True)


def test_parse_case_id_falls_back_to_case_name() -> None:
    assert parse_case_id("weird_name", r"^p(?P<patient>\d+)$") == ("weird_name", False)
    assert parse_case_id("weird_name", None) == ("weird_name", False)


def test_voi_stats_on_synthetic_voi(tmp_path: Path) -> None:
    image, label = make_voi(lesion_size=4)
    write_nifti(tmp_path / "img.nii.gz", image)
    write_nifti(tmp_path / "lab.nii.gz", label)
    stats = voi_stats(tmp_path / "img.nii.gz", tmp_path / "lab.nii.gz")
    assert stats["pad_fraction"] == pytest.approx(4 / 16)
    assert stats["image_min"] == PAD_VALUE
    assert stats["hu_p50"] == pytest.approx(40.0)  # padding excluded from the percentiles
    assert stats["lesion_voxels"] == 32
    assert stats["lesion_mm3"] == pytest.approx(32 * 0.5 * 0.5 * 2.0)
    assert (stats["extent_vox_x"], stats["extent_vox_y"], stats["extent_vox_z"]) == (4, 4, 2)
    assert stats["n_components"] == 1
    assert stats["center_in_lesion"] and not stats["touches_border"]
    assert stats["label_geometry_match"]
    assert stats["lesion_hu_median"] == pytest.approx(100.0)
    assert (stats["spacing_x"], stats["spacing_z"]) == (0.5, 2.0)


def test_list_cases_pairs_ids_and_categories(synthetic_root: Path) -> None:
    cases = list_cases(synthetic_root, DATA_CFG, "DS_A").set_index("case")
    assert list(cases.index) == ["p01_lesion_01", "p01_lesion_02", "p02_lesion_01"]
    assert cases.loc["p01_lesion_01", "patient_id"] == "DS_A:01"
    assert cases.loc["p01_lesion_02", "category"] == "Lung"
    assert cases.loc["p02_lesion_01", "label"] == ""
    assert cases.loc["p01_lesion_01", "image"] == "data/DS_A/images/p01_lesion_01.nii.gz"


def test_build_manifest_and_report(synthetic_root: Path, tmp_path: Path) -> None:
    manifest = build_manifest(synthetic_root, DATA_CFG, num_workers=1)
    assert len(manifest) == 5
    assert usable_mask(manifest).sum() == 4

    summary = dataset_summary(manifest, ["DS_A", "DS_B"]).set_index("dataset")
    assert summary.loc["DS_A", "n_missing_label"] == 1
    assert summary.loc["DS_A", "n_patients"] == 1
    assert summary.loc["ALL", "n_usable"] == 4
    assert problem_cases(manifest)["reason"].tolist() == ["missing label"]

    fit = crop_fit(manifest, [[4, 4, 2], [32, 32, 16]], ["DS_A", "DS_B"]).set_index(["dataset", "crop"])
    assert fit.loc[("DS_B", "4x4x2"), "frac_fits_extent"] == 0.5  # 4-cube fits, 8-cube does not
    assert fit.loc[("ALL", "32x32x16"), "frac_fits_centered"] == 1.0

    # Round-trip through CSV, as --reuse-manifest does, then write every table and figure.
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    reread = pd.read_csv(tmp_path / "manifest.csv")
    eda_cfg = {"output_dir": str(tmp_path / "eda"), "crop_sizes": [[16, 16, 8]], "gallery_quantiles": [0.5]}
    write_report(reread, synthetic_root, eda_cfg, ["DS_A", "DS_B"])
    for name in ["summary_per_dataset.csv", "crop_fit.csv", "problem_cases.csv", "categories.csv",
                 "lesion_size.png", "spacing.png", "lesion_hu.png", "padding.png", "crop_fit.png", "gallery.png"]:
        assert (tmp_path / "eda" / name).is_file(), name
