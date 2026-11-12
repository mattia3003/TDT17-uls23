"""Tests for the split generator (course train/test plus patient-grouped validation) and the seeding helper."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.splits import check_no_patient_leak, grouped_sample, make_splits, read_course_listing
from src.utils.seed import seed_everything


def synthetic_manifest() -> pd.DataFrame:
    """Two training datasets (A: 20 patients x 3 lesions, B: 10 x 1) and a held-out dataset C."""
    rows = [(ds, f"{ds}_{p}_{n}", f"{ds}:{p}") for ds, n_patients, n_lesions in [("A", 20, 3), ("B", 10, 1), ("C", 4, 2)]
            for p in range(n_patients) for n in range(n_lesions)]
    manifest = pd.DataFrame(rows, columns=["dataset", "case", "patient_id"])
    manifest["image"] = "img.nii.gz"
    manifest["label"] = "lab.nii.gz"
    manifest["label_geometry_match"] = True
    manifest["lesion_voxels"] = 10
    manifest.loc[0, "label"] = ""  # one unlabelled case, which must be left out
    return manifest


def course_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """Course-style split per lesion: every 5th usable lesion is a test case, so patients can be on both sides."""
    cases = manifest.iloc[1:][["dataset", "case"]].reset_index(drop=True)
    cases["course_split"] = np.where(cases.index % 5 == 0, "test", "train")
    cases["image"] = cases["dataset"] + "/" + cases["course_split"] + "/images/" + cases["case"] + ".nii.gz"
    cases["label"] = cases["image"].str.replace("/images/", "/labels/")
    return cases


def build(seed: int = 0, val_fraction: float = 0.25) -> pd.DataFrame:
    manifest = synthetic_manifest()
    return make_splits(manifest, course_split(manifest), ["A", "B"], ["C"], val_fraction, seed)


def test_splits_follow_the_course_split_and_hold_out_c() -> None:
    splits, course = build(), course_split(synthetic_manifest())
    assert len(splits) == len(synthetic_manifest()) - 1
    merged = splits.merge(course, on=["dataset", "case"], suffixes=("", "_course"))
    trained = merged[merged["dataset"] != "C"]
    assert (trained["split"] == "test").equals(trained["course_split"] == "test")  # test is exactly course test
    assert (merged["image"] == merged["image_course"]).all()  # paths from the course folders
    assert set(splits.loc[splits["dataset"] == "C", "split"]) == {"heldout"}
    assert set(splits.loc[splits["dataset"] != "C", "split"]) == {"train", "val", "test"}


def test_patient_in_both_marks_patients_on_both_course_sides() -> None:
    splits = build()
    sides = splits.groupby("patient_id")["course_split"].nunique()
    assert splits["patient_in_both"].equals(splits["patient_id"].map(sides) > 1)
    assert splits["patient_in_both"].any() and not splits["patient_in_both"].all()


def test_validation_is_patient_grouped_and_near_the_fraction() -> None:
    splits = build()
    check_no_patient_leak(splits[splits["split"].isin(["train", "val"])])
    for dataset in ["A", "B"]:
        part = splits[(splits["dataset"] == dataset) & splits["split"].isin(["train", "val"])]
        frac = (part["split"] == "val").mean()
        assert 0.15 <= frac <= 0.25 + 1 / len(part), (dataset, frac)  # the target is rounded to whole lesions


def test_same_seed_gives_same_splits_and_other_seed_differs() -> None:
    pd.testing.assert_frame_equal(build(seed=1), build(seed=1))
    assert not build(seed=1)["split"].equals(build(seed=2)["split"])


def test_grouped_sample_ignores_row_order() -> None:
    groups = pd.Series([f"p{i // 2}" for i in range(40)])
    picked = groups[grouped_sample(groups, 0.2, np.random.default_rng(5))]
    shuffled = groups.sample(frac=1.0, random_state=3)
    picked_shuffled = shuffled[grouped_sample(shuffled, 0.2, np.random.default_rng(5))]
    assert set(picked) == set(picked_shuffled) and len(picked) == 8


def test_leak_check_and_unmatched_cases_raise() -> None:
    leaky = pd.DataFrame({"patient_id": ["p1", "p1"], "split": ["train", "val"]})
    with pytest.raises(ValueError, match="more than one split"):
        check_no_patient_leak(leaky)
    manifest = synthetic_manifest()
    course = course_split(manifest)
    with pytest.raises(ValueError, match="only in the manifest"):
        make_splits(manifest, course.iloc[1:], ["A", "B"], ["C"], 0.2, 0)
    with pytest.raises(ValueError, match="only in the manifest or only in the course"):
        make_splits(manifest.iloc[2:], course, ["A", "B"], ["C"], 0.2, 0)


def test_read_course_listing(tmp_path: Path) -> None:
    lines = ["DS/test/images/a.nii.gz", "DS/test/labels/a.nii.gz", "DS/train/images/b.nii.gz", "DS/train/labels/b.nii.gz"]
    (tmp_path / "files.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    course = read_course_listing(tmp_path / "files.txt")
    assert course.to_dict("records") == [
        {"dataset": "DS", "case": "a", "course_split": "test", "image": lines[0], "label": lines[1]},
        {"dataset": "DS", "case": "b", "course_split": "train", "image": lines[2], "label": lines[3]}]
    (tmp_path / "bad.txt").write_text("\n".join([lines[0], "DS/train/labels/a.nii.gz", lines[2]]), encoding="utf-8")
    with pytest.raises(ValueError, match="2 cases lack"):
        read_course_listing(tmp_path / "bad.txt")  # a: label in another split; b: no label


def test_seed_everything_repeats_random_draws() -> None:
    gen = seed_everything(7)
    first = (np.random.rand(), torch.rand(1).item(), torch.randint(0, 100, (3,), generator=gen).tolist())
    gen = seed_everything(7)
    second = (np.random.rand(), torch.rand(1).item(), torch.randint(0, 100, (3,), generator=gen).tolist())
    assert first == second
