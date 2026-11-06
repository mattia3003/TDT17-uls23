"""Patient-grouped train/val/test splits, with the test cases taken from the official ULS23 split (D15).

Split names: ``train`` and ``val`` (official training cases of the training datasets),
``test`` (their official test cases) and ``heldout`` (every case of a held-out dataset, E2).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import usable_mask

SPLIT_COLUMNS = ["dataset", "case", "patient_id", "split"]


def load_official_split(source: str, sha256: str) -> dict[str, list[str]]:
    """Read the official split JSON (``{"train": [...], "test": [...]}``) from a URL or local file.

    Raises if its SHA-256 differs from ``sha256``, so a changed upstream file is never used silently.
    """
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            raw = response.read()
    else:
        raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ValueError(f"SHA-256 of {source} is {digest}, expected {sha256}")
    return json.loads(raw)


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


def make_splits(manifest: pd.DataFrame, official: Mapping[str, Sequence[str]], train_datasets: Sequence[str],
                heldout_datasets: Sequence[str], val_fraction: float, seed: int) -> pd.DataFrame:
    """Assign every usable case of the configured datasets to a split.

    The validation cases are drawn per training dataset, grouped by patient, from the official
    training cases. Raises if a training-dataset case is missing from the official split.
    """
    cases = manifest.loc[usable_mask(manifest), ["dataset", "case", "patient_id"]]
    side = {case: name for name, names in official.items() for case in names}
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for dataset in train_datasets:
        part = cases[cases["dataset"] == dataset].copy()
        part["split"] = part["case"].map(side)
        missing = part.loc[part["split"].isna(), "case"]
        if len(missing):
            raise ValueError(f"{len(missing)} {dataset} cases are not in the official split, e.g. {missing.iloc[0]}")
        train = part["split"] == "train"
        val = grouped_sample(part.loc[train, "patient_id"], val_fraction, rng)
        part.loc[val[val].index, "split"] = "val"
        parts.append(part)
    for dataset in heldout_datasets:
        parts.append(cases[cases["dataset"] == dataset].assign(split="heldout"))
    splits = pd.concat(parts, ignore_index=True)[SPLIT_COLUMNS]
    check_no_patient_leak(splits)
    return splits.sort_values(["dataset", "case"], ignore_index=True)


def check_no_patient_leak(splits: pd.DataFrame) -> None:
    """Raise if any patient has cases in more than one split."""
    n_splits = splits.groupby("patient_id")["split"].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise ValueError(f"{len(leaked)} patients are in more than one split, e.g. {leaked.index[0]}")


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Lesion and patient counts per dataset and split."""
    return (splits.groupby(["dataset", "split"])
            .agg(lesions=("case", "size"), patients=("patient_id", "nunique"))
            .reset_index())
