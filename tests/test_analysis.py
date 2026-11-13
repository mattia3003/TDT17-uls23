"""Tests for the breakdown and paired-comparison tables on small synthetic per-lesion tables."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.analysis import BOOTSTRAP_RESAMPLES, breakdown, paired_comparison, wilcoxon_p


def lesions(dice_pp: list[float]) -> pd.DataFrame:
    """Six lesions: four DeepLesion3D (with categories), two Bone (no category)."""
    return pd.DataFrame({
        "case": [f"c{i}" for i in range(6)],
        "dataset": ["ULS23_DeepLesion3D"] * 4 + ["ULS23_Radboudumc_Bone"] * 2,
        "dice": [d / 2 for d in dice_pp], "dice_pp": dice_pp,
        "precision": [0.5] * 6, "precision_pp": [1.0] * 6,
        "pred_voxels": [200, 200, 200, 200, 100, 100], "true_voxels": [100] * 6,
        "lesion_eq_diam_mm": [5, 12, 12, 40, 8, 25],
        "lesion_shell_contrast_hu": [10, -30, 60, 300, 500, 20],   # -30 counts as 30 (absolute value)
        "spacing_z": [1.0, 5.0, 2.5, 2.5, 1.0, 1.0],
        "category": ["Liver", "Liver", "Kidney", "Lung", np.nan, np.nan],
        "intensity_group": ["stored in HU"] * 4 + ["shifted", "stored in HU"],
    })


def row(table: pd.DataFrame, factor: str, dataset: str, level: str) -> pd.Series:
    match = table[(table["factor"] == factor) & (table["dataset"] == dataset) & (table["level"] == level)]
    assert len(match) == 1
    return match.iloc[0]


def test_breakdown_groups_bins_and_missed() -> None:
    table = breakdown(lesions([0.0, 0.8, 0.04, 0.6, 1.0, 0.0]))
    assert row(table, "all", "all", "all")["lesions"] == 6
    assert row(table, "all", "DeepLesion3D", "all")["missed"] == pytest.approx(0.5)  # 0.0 and 0.04
    assert row(table, "all", "Bone", "all")["dice_pp_mean"] == pytest.approx(0.5)
    assert row(table, "all", "all", "all")["volume_ratio_median"] == pytest.approx(2.0)
    assert row(table, "contrast", "DeepLesion3D", "20-50 HU")["lesions"] == 1       # the -30 HU lesion
    assert row(table, "size", "DeepLesion3D", "10-15 mm")["dice_pp_mean"] == pytest.approx(0.42)
    assert row(table, "category", "DeepLesion3D", "Liver")["lesions"] == 2
    assert not ((table["factor"] == "category") & (table["dataset"] == "Bone")).any()  # no category, left out
    assert row(table, "intensity_group", "Bone", "shifted")["lesions"] == 1


def test_paired_comparison_counts_and_statistics() -> None:
    a = lesions([0.0, 0.8, 0.0, 0.6, 0.9, 0.5])
    b = lesions([0.7, 0.8, 0.0, 0.4, 0.93, 0.0])   # c0 rescued, c3 worse, c4 same (+0.03), c5 lost
    table, per_lesion = paired_comparison(a, b.sample(frac=1, random_state=0), ("E1", "E1b"))  # row order must not matter
    overall = row(table, "all", "all", "all")
    delta = np.array([0.7, 0.0, 0.0, -0.2, 0.03, -0.5])
    assert overall["delta_mean"] == pytest.approx(delta.mean())
    assert overall["better"] == 1 and overall["worse"] == 2 and overall["rescued"] == 1 and overall["lost"] == 1
    assert overall["missed_E1"] == pytest.approx(2 / 6) and overall["missed_E1b"] == pytest.approx(2 / 6)
    assert overall["delta_ci_low"] <= overall["delta_mean"] <= overall["delta_ci_high"] and BOOTSTRAP_RESAMPLES > 1000
    assert np.isnan(overall["wilcoxon_p"])                                            # 4 changed lesions: too few
    assert list(per_lesion["case"][[0, 5]]) == ["c5", "c0"] and "dice_pp_E1b" in per_lesion
    assert wilcoxon_p(pd.Series([0.1] * 12 + [0.0] * 5)) < 0.001
    assert np.isnan(wilcoxon_p(pd.Series([0.1] * 9)))
    with pytest.raises(ValueError, match="different lesions"):
        paired_comparison(a, b.iloc[:5], ("E1", "E1b"))
