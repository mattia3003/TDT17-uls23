"""Qualitative figures from an evaluation folder (D42, D46): the worst, median and best lesion of each
dataset, the worst lesions overall, and for the failure review (E3) random samples of missed and of
partly found lesions per dataset.

Each lesion gets two panels on the axial slice where its label is largest: a zoom around the
lesion, and the whole VOI slice, which also shows false positives away from it. The label is
outlined in amber and the raw prediction in blue. Intensities are offset-corrected (D17) and shown
in the fixed display windows of the EDA gallery (D23).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.data.datasets import case_table, data_dicts
from src.data.eda import CONTOUR, INK_2, PALETTE
from src.data.intensity import intensity_shift
from src.data.manifest import load_nifti, spatial_array
from src.evaluation.analysis import MISSED_BELOW

PREDICTION = PALETTE[0]
ALL_SPLITS = ["train", "val", "test", "heldout"]
PARTIAL_BELOW = 0.5  # "partly found": centre-component Dice from MISSED_BELOW up to this


def pick_examples(lesions: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    """Worst, median and best lesion of each dataset by ``metric``; ties are broken by case name."""
    picks = []
    for _, part in lesions.groupby("dataset", sort=False):
        ordered = part.sort_values([metric, "case"]).reset_index(drop=True)
        for label, i in [("worst", 0), ("median", (len(ordered) - 1) // 2), ("best", len(ordered) - 1)]:
            picks.append({"pick": label, **ordered.iloc[i].to_dict()})
    return pd.DataFrame(picks)


def worst_lesions(lesions: pd.DataFrame, n: int, metric: str = "dice") -> pd.DataFrame:
    """The ``n`` lesions with the lowest ``metric`` over all datasets. Ties are broken by case name, so
    when many lesions score 0 this is not a fair sample of them; ``sample_band`` is (D46)."""
    worst = lesions.sort_values([metric, "case"]).head(n).reset_index(drop=True)
    return worst.assign(pick=[f"worst {i + 1}" for i in range(len(worst))])


def sample_band(lesions: pd.DataFrame, low: float, high: float, n: int, label: str, seed: int = 0) -> pd.DataFrame:
    """Up to ``n`` lesions per dataset drawn at random (fixed ``seed``) from those with
    ``low <= dice_pp < high``, sorted by dataset and case."""
    band = lesions[(lesions["dice_pp"] >= low) & (lesions["dice_pp"] < high)]
    picks = [part.sample(n=min(n, len(part)), random_state=seed) for _, part in band.groupby("dataset")]
    if not picks:
        return band.assign(pick="")
    sample = pd.concat(picks).sort_values(["dataset", "case"]).reset_index(drop=True)
    return sample.assign(pick=[f"{label} {i + 1}" for i in range(len(sample))])


def load_case(item: Mapping[str, str], prediction: Path,
              intensity_cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image in HU (offset-corrected), label and predicted mask, as 3-D arrays."""
    image = spatial_array(load_nifti(item["image"])[0], dtype=np.float32)
    image += intensity_shift(float(image.min()), intensity_cfg)
    truth = spatial_array(load_nifti(item["label"])[0]) > 0
    pred = spatial_array(load_nifti(prediction)[0]) > 0
    return image, truth, pred


def _draw(ax: Any, image: np.ndarray, truth: np.ndarray, pred: np.ndarray, window: Sequence[float],
          title: str) -> None:
    ax.imshow(np.rot90(image), cmap="gray", vmin=window[0], vmax=window[1])
    for mask, color in ((truth, CONTOUR), (pred, PREDICTION)):
        if mask.any():
            ax.contour(np.rot90(mask), levels=[0.5], colors=color, linewidths=1.0)
    ax.set_title(title, fontsize=7, fontweight="normal", color=INK_2)
    ax.set_axis_off()


def lesion_panels(ax_zoom: Any, ax_full: Any, image: np.ndarray, truth: np.ndarray, pred: np.ndarray,
                  row: pd.Series, windows: Mapping[str, Any]) -> None:
    """Zoomed and whole-VOI panels for one lesion, with its metrics in the titles."""
    z = int(truth.sum(axis=(0, 1)).argmax())
    is_bone = truth.any() and float(np.percentile(image[truth], 95)) > float(windows["bone_above_hu"])
    window = windows["bone" if is_bone else "soft_tissue"]
    xs, ys = np.nonzero(truth[:, :, z]) if truth.any() else ([image.shape[0] // 2], [image.shape[1] // 2])
    lo, hi = np.array([min(xs), min(ys)]), np.array([max(xs), max(ys)]) + 1
    half = max(32, int((hi - lo).max()))
    x0, y0 = np.clip((lo + hi) // 2 - half, 0, None)
    x1, y1 = np.minimum((lo + hi) // 2 + half, image.shape[:2])
    zoom = (slice(x0, x1), slice(y0, y1), z)
    dataset = str(row["dataset"]).removeprefix("ULS23_").removeprefix("Radboudumc_")
    _draw(ax_zoom, image[zoom], truth[zoom], pred[zoom], window,
          f"{dataset}, {row['pick']}: {row['case']}\n"
          f"Dice {row['dice']:.2f} (centre {row['dice_pp']:.2f}), {row['true_voxels']:.0f} voxels")
    _draw(ax_full, image[:, :, z], truth[:, :, z], pred[:, :, z], window,
          f"whole VOI, slice {z}\nwindow {window[0]:g} to {window[1]:g} HU")


def plot_lesions(picks: pd.DataFrame, items: Mapping[str, Mapping[str, str]], prediction_dir: Path,
                 intensity_cfg: Mapping[str, Any], windows: Mapping[str, Any], path: Path, per_row: int,
                 title: str) -> None:
    """Two panels per lesion, ``per_row`` lesions per figure row, under a title saying how they were chosen."""
    n_rows = max(1, -(-len(picks) // per_row))
    fig, axes = plt.subplots(n_rows, 2 * per_row, figsize=(5.2 * per_row, 3.0 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for i, (_, row) in enumerate(picks.iterrows()):
        r, c = divmod(i, per_row)
        case = str(row["case"])
        image, truth, pred = load_case(items[case], prediction_dir / f"{case}.nii.gz", intensity_cfg)
        lesion_panels(axes[r, 2 * c], axes[r, 2 * c + 1], image, truth, pred, row, windows)
    fig.legend(handles=[Line2D([], [], color=CONTOUR, label="label"),
                        Line2D([], [], color=PREDICTION, label="prediction (raw)")],
               loc="lower center", ncol=2)
    fig.suptitle(title, fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_examples(cfg: Mapping[str, Any], eval_dir: Path, windows: Mapping[str, Any], worst_n: int = 12,
                  metric: str = "dice", sample_n: int = 10, seed: int = 0) -> Path:
    """Write ``examples``, ``worst``, ``missed`` and ``partial`` (each .png and .csv) to ``<eval_dir>/figures``
    and return that folder. A sample with no lesions is skipped."""
    lesions = pd.read_csv(eval_dir / "per_lesion.csv")
    table = case_table(cfg["splits"]["manifest_path"], cfg["splits"]["path"])
    items = {item["case"]: item
             for item in data_dicts(table, cfg["paths"]["data_root"], ALL_SPLITS, lesions["case"].tolist())}
    out = eval_dir / "figures"
    out.mkdir(exist_ok=True)
    metric_name = "raw Dice" if metric == "dice" else "centre-component Dice"
    figures = [
        ("examples", pick_examples(lesions, metric), 3, f"Worst, median and best lesion per dataset by {metric_name}"),
        ("worst", worst_lesions(lesions, worst_n, metric), 4,
         f"Lowest {metric_name}; ties are broken by case name, so this is not a random sample of the tied lesions"),
        ("missed", sample_band(lesions, 0.0, MISSED_BELOW, sample_n, "missed", seed), 5,
         f"Missed lesions (centre-component Dice < {MISSED_BELOW:g}): random sample of up to {sample_n} "
         f"per dataset, seed {seed}"),
        ("partial", sample_band(lesions, MISSED_BELOW, PARTIAL_BELOW, sample_n, "partial", seed), 5,
         f"Partly found lesions (centre-component Dice {MISSED_BELOW:g} to {PARTIAL_BELOW:g}): random sample of "
         f"up to {sample_n} per dataset, seed {seed}"),
    ]
    for name, picks, per_row, title in figures:
        if len(picks):
            plot_lesions(picks, items, eval_dir / "predictions", cfg["data"]["intensity"], windows,
                         out / f"{name}.png", per_row, title)
            picks.to_csv(out / f"{name}.csv", index=False)
    return out
