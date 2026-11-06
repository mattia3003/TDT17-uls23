"""Tests for the patient-grouped split generator and the seeding helper."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.splits import check_no_patient_leak, grouped_sample, load_official_split, make_splits
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


def official_split(manifest: pd.DataFrame) -> dict[str, list[str]]:
    """Official-style split: the last two patients of each dataset are test cases."""
    test_patients = {f"{ds}:{p}" for ds, n in [("A", 20), ("B", 10), ("C", 4)] for p in (n - 2, n - 1)}
    is_test = manifest["patient_id"].isin(test_patients)
    return {"train": manifest.loc[~is_test, "case"].tolist(), "test": manifest.loc[is_test, "case"].tolist()}


def build(seed: int = 0) -> pd.DataFrame:
    manifest = synthetic_manifest()
    return make_splits(manifest, official_split(manifest), ["A", "B"], ["C"], val_fraction=0.25, seed=seed)


def test_splits_assign_every_usable_case() -> None:
    splits = build()
    assert len(splits) == len(synthetic_manifest()) - 1
    counts = splits.groupby(["dataset", "split"]).size()
    assert counts[("A", "test")] == 6 and counts[("B", "test")] == 2  # taken from the official split
    assert counts[("C", "heldout")] == 8
    assert set(splits.loc[splits["dataset"] != "C", "split"]) == {"train", "val", "test"}


def test_validation_is_patient_grouped_and_near_the_fraction() -> None:
    splits = build()
    check_no_patient_leak(splits)
    for dataset in ["A", "B"]:
        part = splits[(splits["dataset"] == dataset) & splits["split"].isin(["train", "val"])]
        frac = (part["split"] == "val").mean()
        assert 0.15 <= frac <= 0.25, (dataset, frac)


def test_same_seed_gives_same_splits_and_other_seed_differs() -> None:
    pd.testing.assert_frame_equal(build(seed=1), build(seed=1))
    assert not build(seed=1)["split"].equals(build(seed=2)["split"])


def test_grouped_sample_ignores_row_order() -> None:
    groups = pd.Series([f"p{i // 2}" for i in range(40)])
    picked = groups[grouped_sample(groups, 0.2, np.random.default_rng(5))]
    shuffled = groups.sample(frac=1.0, random_state=3)
    picked_shuffled = shuffled[grouped_sample(shuffled, 0.2, np.random.default_rng(5))]
    assert set(picked) == set(picked_shuffled) and len(picked) == 8


def test_leak_check_and_missing_official_case_raise() -> None:
    leaky = pd.DataFrame({"patient_id": ["p1", "p1"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="more than one split"):
        check_no_patient_leak(leaky)
    manifest = synthetic_manifest()
    official = official_split(manifest)
    official["train"].remove("A_0_1")
    with pytest.raises(ValueError, match="not in the official split"):
        make_splits(manifest, official, ["A"], [], 0.2, 0)


def test_official_split_checksum(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["a"], "test": ["b"]}), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_official_split(str(path), digest) == {"train": ["a"], "test": ["b"]}
    with pytest.raises(ValueError, match="SHA-256"):
        load_official_split(str(path), "0" * 64)


def test_seed_everything_repeats_random_draws() -> None:
    gen = seed_everything(7)
    first = (np.random.rand(), torch.rand(1).item(), torch.randint(0, 100, (3,), generator=gen).tolist())
    gen = seed_everything(7)
    second = (np.random.rand(), torch.rand(1).item(), torch.randint(0, 100, (3,), generator=gen).tolist())
    assert first == second
