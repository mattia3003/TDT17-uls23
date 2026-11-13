"""Breakdown and paired-comparison tables from per-lesion CSVs (E3, D44, D45). Reads only CSVs, no images.

A lesion counts as missed when its centre-component Dice (D30) is below ``MISSED_BELOW``. Every table
is grouped by a factor (lesion size, lesion-shell contrast, slice spacing, DeepLesion3D category, Bone
intensity group) within each dataset; the factor ``all`` also pools the datasets.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MISSED_BELOW = 0.05
CHANGE_ABOVE = 0.05      # a paired change in centre-component Dice up to this size counts as no change
MIN_CHANGED_FOR_TEST = 10  # fewer changed lesions than this: no Wilcoxon p-value (too few to be reliable)
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0

# factor: (column, bin edges, labels). Bins include their upper edge. Contrast uses the absolute value,
# since faint lesions can be slightly brighter or slightly darker than their shell.
BINS = {
    "size": ("lesion_eq_diam_mm", [0, 10, 15, 20, 30, np.inf], ["0-10 mm", "10-15 mm", "15-20 mm", "20-30 mm", ">30 mm"]),
    "contrast": ("abs_contrast_hu", [-np.inf, 20, 50, 100, np.inf], ["0-20 HU", "20-50 HU", "50-100 HU", ">100 HU"]),
    "slice_spacing": ("spacing_z", [0, 1.5, 3.5, np.inf], ["0-1.5 mm", "1.5-3.5 mm", ">3.5 mm"]),
}
CATEGORICAL = ["category", "intensity_group"]
FACTORS = ["all", *BINS, *CATEGORICAL]


def prepare(lesions: pd.DataFrame) -> pd.DataFrame:
    """Short dataset names and one column per binned factor."""
    d = lesions.copy()
    d["dataset"] = d["dataset"].str.replace(r"^ULS23_(Radboudumc_)?", "", regex=True)
    d["abs_contrast_hu"] = d["lesion_shell_contrast_hu"].abs()
    for factor, (column, edges, labels) in BINS.items():
        d[factor] = pd.cut(d[column], edges, labels=labels)
    return d


def groups(d: pd.DataFrame, factor: str) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """``(dataset, level, rows)`` for one factor, per dataset. Lesions without a value are left out."""
    if factor == "all":
        yield "all", "all", d
        for dataset, part in d.groupby("dataset"):
            yield str(dataset), "all", part
        return
    for key, part in d.groupby(["dataset", factor], observed=True):
        dataset, level = tuple(key)  # type: ignore[arg-type]
        yield str(dataset), str(level), part


def lesion_stats(rows: pd.DataFrame) -> dict[str, Any]:
    return {"lesions": len(rows), "dice_mean": rows["dice"].mean(), "dice_pp_mean": rows["dice_pp"].mean(),
            "dice_pp_median": rows["dice_pp"].median(), "missed": (rows["dice_pp"] < MISSED_BELOW).mean(),
            "precision_mean": rows["precision"].mean(), "precision_pp_mean": rows["precision_pp"].mean(),
            "volume_ratio_median": (rows["pred_voxels"] / rows["true_voxels"]).median()}


def breakdown(lesions: pd.DataFrame) -> pd.DataFrame:
    """One row per (factor, dataset, level) with the Dice, missed fraction, precision and volume ratio."""
    d = prepare(lesions)
    return pd.DataFrame([{"factor": factor, "dataset": dataset, "level": level, **lesion_stats(rows)}
                         for factor in FACTORS for dataset, level, rows in groups(d, factor)])


def pair(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    """One row per lesion with both models' metrics (suffixed with ``names``) and the change ``delta`` in
    centre-component Dice from the first model to the second. Both CSVs must hold exactly the same lesions."""
    if set(a["case"]) != set(b["case"]) or len(a) != len(b):
        raise ValueError(f"the two CSVs hold different lesions ({len(a)} and {len(b)} rows)")
    metrics = ["dice", "dice_pp", "precision", "precision_pp", "pred_voxels"]
    merged = a.merge(b[["case", *metrics]], on="case", suffixes=(f"_{names[0]}", f"_{names[1]}"),
                     validate="one_to_one")
    merged["delta"] = merged[f"dice_pp_{names[1]}"] - merged[f"dice_pp_{names[0]}"]
    return merged


def wilcoxon_p(delta: pd.Series) -> float:
    """Two-sided Wilcoxon signed-rank p-value; lesions with no change are dropped (scipy's default)."""
    if (delta != 0).sum() < MIN_CHANGED_FOR_TEST:
        return float("nan")
    result: Any = wilcoxon(delta)  # scipy's result type is not annotated
    return float(result.pvalue)


def bootstrap_ci(delta: pd.Series) -> tuple[float, float]:
    """95 % percentile interval of the mean change, resampling lesions with a fixed seed."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = delta.to_numpy()
    means = values[rng.integers(0, len(values), (BOOTSTRAP_RESAMPLES, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_stats(rows: pd.DataFrame, names: tuple[str, str]) -> dict[str, Any]:
    first, second = (rows[f"dice_pp_{n}"] for n in names)
    missed_first, missed_second = first < MISSED_BELOW, second < MISSED_BELOW
    delta = rows["delta"]
    low, high = bootstrap_ci(delta)
    return {"lesions": len(rows), f"dice_pp_mean_{names[0]}": first.mean(), f"dice_pp_mean_{names[1]}": second.mean(),
            "delta_mean": delta.mean(), "delta_ci_low": low, "delta_ci_high": high, "delta_median": delta.median(),
            "better": int((delta > CHANGE_ABOVE).sum()), "worse": int((delta < -CHANGE_ABOVE).sum()),
            f"missed_{names[0]}": missed_first.mean(), f"missed_{names[1]}": missed_second.mean(),
            "rescued": int((missed_first & ~missed_second).sum()), "lost": int((~missed_first & missed_second).sum()),
            "wilcoxon_p": wilcoxon_p(delta)}


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group comparison of two models on the same lesions, and the per-lesion table sorted by change."""
    d = prepare(pair(a, b, names))
    table = pd.DataFrame([{"factor": factor, "dataset": dataset, "level": level, **paired_stats(rows, names)}
                          for factor in FACTORS for dataset, level, rows in groups(d, factor)])
    columns = ["case", "dataset", "category", "lesion_eq_diam_mm", "lesion_shell_contrast_hu",
               *[f"{m}_{n}" for m in ["dice", "dice_pp"] for n in names], "delta"]
    return table, d[[c for c in columns if c in d]].sort_values("delta", ignore_index=True)
