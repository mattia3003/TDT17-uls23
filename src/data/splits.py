"""Train/val/test splits, with train and test taken from the course's folder split (D41).

Split names: ``train`` and ``val`` (course training cases of the training datasets),
``test`` (their course test cases) and ``heldout`` (every case of a held-out dataset, E2).
The course split is per lesion, not per patient, so ``patient_in_both`` marks patients with
lesions in both course train and course test; validation is still drawn by patient.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import usable_mask

SPLIT_COLUMNS = ["dataset", "case", "patient_id", "split", "course_split", "patient_in_both", "image", "label"]


def read_course_listing(path: str | Path) -> pd.DataFrame:
    """One row per case from a listing of ``<dataset>/<train|test>/<images|labels>/<case>.nii.gz`` paths.

    Returns ``dataset, case, course_split, image, label`` (paths relative to the data root).
    Raises if a case lacks its image or label, or has them in different splits.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split():
        dataset, split, kind, name = line.split("/")
        rows.append({"dataset": dataset, "course_split": split, "kind": kind,
                     "case": name.removesuffix(".nii.gz"), "path": line})
    files = pd.DataFrame(rows)
    paths = files.pivot(index=["dataset", "case"], columns="kind", values="path")
    splits = files.pivot(index=["dataset", "case"], columns="kind", values="course_split")
    bad = paths["images"].isna() | paths["labels"].isna() | (splits["images"] != splits["labels"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} cases lack an image or label, or have them in different splits, "
                         f"e.g. {bad[bad].index[0]}")
    return pd.DataFrame({"course_split": splits["images"], "image": paths["images"],
                         "label": paths["labels"]}).reset_index()


def grouped_sample(groups: pd.Series, fraction: float, rng: np.random.Generator) -> pd.Series:
    """Mark whole groups, in random order, while the marked rows stay within ``fraction`` of all rows.

    Groups are sorted before shuffling, so the result depends only on the seed, not on row order.
    """
    sizes = groups.value_counts().sort_index()
    target = round(fraction * len(groups))
    chosen: list[str] = []
    total = 0
    for i in rng.permutation(len(sizes)):
        if total + sizes.iloc[i] <= target:
            chosen.append(str(sizes.index[i]))
            total += int(sizes.iloc[i])
    return groups.isin(chosen)


def make_splits(manifest: pd.DataFrame, course: pd.DataFrame, train_datasets: Sequence[str],
                heldout_datasets: Sequence[str], val_fraction: float, seed: int) -> pd.DataFrame:
    """Assign every usable case of the configured datasets to a split.

    ``course`` is ``read_course_listing`` output. The validation cases are drawn per training
    dataset, grouped by patient, from the course training cases. Raises if the usable cases of
    the manifest and the course listing differ.
    """
    datasets = [*train_datasets, *heldout_datasets]
    usable = manifest.loc[usable_mask(manifest), ["dataset", "case", "patient_id"]]
    usable = usable[usable["dataset"].isin(datasets)]
    cases = usable.merge(course[course["dataset"].isin(datasets)], on=["dataset", "case"],
                         how="outer", validate="one_to_one", indicator=True)
    unmatched = cases[cases["_merge"] != "both"]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} cases are only in the manifest or only in the course listing, "
                         f"e.g. {unmatched['case'].iloc[0]} ({unmatched['_merge'].iloc[0]})")
    sides = cases.groupby("patient_id")["course_split"].nunique()
    cases["patient_in_both"] = cases["patient_id"].map(sides) > 1

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for dataset in train_datasets:
        part = cases[cases["dataset"] == dataset].copy()
        part["split"] = part["course_split"]
        train = part["split"] == "train"
        val = grouped_sample(part.loc[train, "patient_id"], val_fraction, rng)
        part.loc[val[val].index, "split"] = "val"
        parts.append(part)
    for dataset in heldout_datasets:
        parts.append(cases[cases["dataset"] == dataset].assign(split="heldout"))
    splits = pd.concat(parts, ignore_index=True)[SPLIT_COLUMNS]
    check_no_patient_leak(splits[splits["split"].isin(["train", "val"])])
    return splits.sort_values(["dataset", "case"], ignore_index=True)


def check_no_patient_leak(splits: pd.DataFrame) -> None:
    """Raise if any patient has cases in more than one split."""
    n_splits = splits.groupby("patient_id")["split"].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise ValueError(f"{len(leaked)} patients are in more than one split, e.g. {leaked.index[0]}")


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Lesion and patient counts per dataset and split, and how many lesions have a patient in both course splits."""
    return (splits.groupby(["dataset", "split"])
            .agg(lesions=("case", "size"), patients=("patient_id", "nunique"),
                 patient_in_both=("patient_in_both", "sum"))
            .reset_index())
