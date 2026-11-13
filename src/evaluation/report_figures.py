"""Summary figures for the report and video (D49), drawn from the training logs and the results/ tables only.

Colours: datasets keep their EDA colours (DeepLesion3D, Bone, Pancreas in config order). The two models
get two other slots of the same validated palette, so a colour never means a dataset in one figure and a
model in another. The E1 pink is below 3:1 contrast against the surface, so it is always named in a legend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.eda import INK_2, MUTED, PALETTE, SURFACE, dataset_colors

MODEL_COLORS = {"E1": PALETTE[4], "E1b": PALETTE[6]}
DATASET_COLORS = dataset_colors(["DeepLesion3D", "Bone", "Pancreas"])
GAP: dict[str, Any] = {"edgecolor": SURFACE, "linewidth": 2}  # 2 px surface gap between adjacent bars


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    (ax.set_ylim if axis == "y" else ax.set_xlim)(0, 1)
    ticks = np.linspace(0, 1, 6)
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ticks, [f"{t:.0%}" for t in ticks])


def training_curves(logs: Mapping[str, pd.DataFrame], path: Path) -> None:
    """Validation Dice per epoch for each model: raw, and centre component (the selection metric, D43)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, column, title in [(axes[0], "val_dice_all", "Raw Dice"),
                              (axes[1], "val_dice_pp_all", "Centre-component Dice (selection metric)")]:
        for name, log in logs.items():
            rows = log.dropna(subset=[column])
            ax.plot(rows["epoch"], rows[column], color=MODEL_COLORS[name], marker="o", markersize=4, label=name)
            last = rows.iloc[-1]
            ax.annotate(f"{name} {last[column]:.3f}", (last["epoch"], last[column]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 290)
    axes[0].set_ylabel("Mean validation Dice (173 lesions)")
    axes[0].set_ylim(0, 0.5)
    axes[0].legend(loc="upper left")
    _save(fig, path)


def missed_by_contrast(paired: Mapping[str, pd.DataFrame], names: Sequence[str], path: Path) -> None:
    """Fraction of lesions missed per contrast bin, both models; one row per split, one column per dataset."""
    datasets = ["DeepLesion3D", "Bone"]
    fig, axes = plt.subplots(len(paired), len(datasets), figsize=(11, 3.4 * len(paired)), squeeze=False, sharey=True)
    for r, (split, table) in enumerate(paired.items()):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            rows = table[(table["factor"] == "contrast") & (table["dataset"] == dataset)]
            x = np.arange(len(rows))
            for i, name in enumerate(names):
                ax.bar(x + (i - 0.5) * 0.4, rows[f"missed_{name}"], width=0.4, color=MODEL_COLORS[name], label=name, **GAP)
            ax.set_xticks(x, [f"{level}\nn={n}" for level, n in zip(rows["level"], rows["lesions"])])
            ax.set_title(f"{dataset}, {split}")
            ax.grid(axis="x", visible=False)
            _percent_axis(ax)
            if c == 0:
                ax.set_ylabel("Lesions missed")
    axes[-1, 0].set_xlabel("Lesion–shell contrast (absolute)")
    axes[-1, 1].set_xlabel("Lesion–shell contrast (absolute)")
    axes[0, 0].legend(loc="upper right")
    _save(fig, path)


def test_breakdown(breakdown: pd.DataFrame, path: Path) -> None:
    """Final model on the test split: missed fraction by contrast per dataset, and by DeepLesion3D category."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    contrast = breakdown[breakdown["factor"] == "contrast"]
    levels = list(dict.fromkeys(contrast["level"]))
    x = np.arange(len(levels))
    for i, dataset in enumerate(["DeepLesion3D", "Bone"]):
        rows = contrast[contrast["dataset"] == dataset].set_index("level").reindex(levels)
        left.bar(x + (i - 0.5) * 0.4, rows["missed"], width=0.4, color=DATASET_COLORS[dataset],
                 label=dataset, **GAP)
        for xi, (missed, n) in enumerate(zip(rows["missed"], rows["lesions"])):
            left.annotate(f"n={n:.0f}", (xi + (i - 0.5) * 0.4, missed), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=7, color=MUTED)
    left.set_xticks(x, levels)
    left.set_xlabel("Lesion–shell contrast (absolute)")
    left.set_ylabel("Lesions missed")
    left.set_title("By contrast")
    left.grid(axis="x", visible=False)
    _percent_axis(left)
    left.legend(loc="upper right")

    category = breakdown[(breakdown["factor"] == "category")].sort_values("missed")
    y = np.arange(len(category))
    right.barh(y, category["missed"], height=0.6, color=DATASET_COLORS["DeepLesion3D"], **GAP)
    right.set_yticks(y, [f"{level} (n={n})" for level, n in zip(category["level"], category["lesions"])])
    for yi, missed in enumerate(category["missed"]):
        right.annotate(f"{missed:.0%}", (missed, yi), xytext=(4, 0), textcoords="offset points", va="center",
                       fontsize=8, color=INK_2)
    right.set_xlabel("Lesions missed")
    right.set_title("DeepLesion3D by category")
    _percent_axis(right, "x")
    right.grid(axis="y", visible=False)
    _save(fig, path)


def heldout_comparison(table: pd.DataFrame, path: Path, reference: str = "DeepLesion3D",
                       heldout: str = "Pancreas") -> None:
    """E2: missed fraction of the held-out lesions next to reference lesions of similar contrast and size."""
    rows = table.iloc[::-1].reset_index(drop=True)  # first group on top
    labels = [f"{g.replace('reference', reference).replace('held out', heldout)} (n={n})"
              .replace("<=", "≤").replace(">=", "≥") for g, n in zip(rows["group"], rows["lesions"])]
    colors = [DATASET_COLORS[heldout if g.startswith("held out") else reference] for g in rows["group"]]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = np.arange(len(rows))
    ax.barh(y, rows["missed"], height=0.6, color=colors, **GAP)
    for yi, missed in enumerate(rows["missed"]):
        ax.annotate(f"{missed:.0%}", (missed, yi), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Lesions missed by the final model (E1b)")
    ax.set_title(f"{heldout} against {reference} lesions of similar contrast and size", loc="right")
    _percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
