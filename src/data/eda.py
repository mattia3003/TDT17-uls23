"""Phase 1 analysis on top of the manifest: summary tables, crop-fit table and figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: Slurm nodes have no display

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.data.manifest import load_nifti

# Validated categorical palette (fixed order, never cycled) and neutral inks; see REFERENCES.md.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CONTOUR = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 10, "lines.linewidth": 1.8,
    "axes.formatter.useoffset": False,
})


def _bool_frac(series: pd.Series) -> float:
    """Fraction of True in a column that may hold bools, 'True'/'False' strings or NaN."""
    values = series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).dropna()
    return float(values.mean()) if len(values) else float("nan")


def _has(series: pd.Series) -> pd.Series:
    """True where a text column is non-empty (NaN counts as empty)."""
    return series.fillna("").astype(str) != ""


def _column(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    """Column ``name``, or a constant series if the column is absent (e.g. no read errors at all)."""
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.Series(pd.to_numeric(_column(frame, name), errors="coerce"), index=frame.index).fillna(0)


def usable_mask(manifest: pd.DataFrame) -> pd.Series:
    """Cases with an image, a non-empty label and no read error."""
    return (_has(manifest["image"]) & _has(manifest["label"]) & ~_has(_column(manifest, "error"))
            & (_numeric(manifest, "lesion_voxels") > 0))


def dataset_colors(datasets: Sequence[str]) -> dict[str, str]:
    """Fixed colour per dataset in config order."""
    if len(datasets) > len(PALETTE):
        raise ValueError(f"At most {len(PALETTE)} datasets can get distinct colours; use small multiples instead.")
    return dict(zip(datasets, PALETTE))


def _summarise(group: pd.DataFrame) -> dict[str, Any]:
    usable = group[usable_mask(group)]
    per_patient = usable.groupby("patient_id").size()
    q = lambda col, p: float(usable[col].quantile(p)) if len(usable) else float("nan")  # noqa: E731
    errors = _column(group, "error")
    return {
        "n_images": int(_has(group["image"]).sum()),
        "n_labels": int(_has(group["label"]).sum()),
        "n_usable": len(usable),
        "n_missing_label": int((_has(group["image"]) & ~_has(group["label"])).sum()),
        "n_missing_image": int((~_has(group["image"]) & _has(group["label"])).sum()),
        "n_errors": int(_has(errors).sum()),
        "n_patients": int(per_patient.size),
        "lesions_per_patient_median": float(per_patient.median()) if len(per_patient) else float("nan"),
        "lesions_per_patient_max": int(per_patient.max()) if len(per_patient) else 0,
        "id_parsed_frac": _bool_frac(group["id_parsed"]),
        "spacing_xy_median": q("spacing_x", 0.5),
        "spacing_z_min": q("spacing_z", 0.0),
        "spacing_z_median": q("spacing_z", 0.5),
        "spacing_z_max": q("spacing_z", 1.0),
        "lesion_mm3_p05": q("lesion_mm3", 0.05),
        "lesion_mm3_median": q("lesion_mm3", 0.5),
        "lesion_mm3_p95": q("lesion_mm3", 0.95),
        "eq_diam_mm_median": q("lesion_eq_diam_mm", 0.5),
        "lesion_hu_median_p05": q("lesion_hu_median", 0.05),
        "lesion_hu_median_median": q("lesion_hu_median", 0.5),
        "lesion_hu_median_p95": q("lesion_hu_median", 0.95),
        "image_min_median": q("image_min", 0.5),
        "pad_fraction_median": q("pad_fraction", 0.5),
        "frac_touches_border": _bool_frac(usable["touches_border"]),
        "frac_center_in_lesion": _bool_frac(usable["center_in_lesion"]),
        "frac_multi_component": float((usable["n_components"] > 1).mean()) if len(usable) else float("nan"),
        "frac_label_nonbinary": float((usable["label_n_values"] > 2).mean()) if len(usable) else float("nan"),
        "frac_geometry_mismatch": 1.0 - _bool_frac(usable["label_geometry_match"]),
        "axcodes": "|".join(sorted(usable["axcodes"].dropna().astype(str).unique())),
        "disk_dtypes": "|".join(sorted(usable["disk_dtype"].dropna().astype(str).unique())),
        "disk_gb": float((_numeric(group, "image_mb") + _numeric(group, "label_mb")).sum() / 1000),
    }


def dataset_summary(manifest: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    """One row per dataset plus an ``ALL`` row."""
    rows = {ds: _summarise(manifest[manifest["dataset"] == ds]) for ds in datasets}
    rows["ALL"] = _summarise(manifest)
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("dataset").reset_index()


def problem_cases(manifest: pd.DataFrame) -> pd.DataFrame:
    """Cases that are not usable, with the reason."""
    errors = _column(manifest, "error", "")
    read_error = errors.fillna("").astype(str).map(lambda e: f"read error: {e}")
    reason = np.select(
        [~_has(manifest["image"]), ~_has(manifest["label"]), _has(errors)],
        ["missing image", "missing label", read_error],
        default="empty label",
    )
    out = manifest.loc[~usable_mask(manifest), ["dataset", "case"]].copy()
    out["reason"] = reason[~usable_mask(manifest).to_numpy()]
    return out.reset_index(drop=True)


def crop_fit(manifest: pd.DataFrame, crop_sizes: Sequence[Sequence[int]], datasets: Sequence[str]) -> pd.DataFrame:
    """Fraction of lesions whose bounding box fits a crop (anywhere, and when centred on the VOI centre)."""
    rows = []
    usable = manifest[usable_mask(manifest)]
    for ds in [*datasets, "ALL"]:
        group = usable if ds == "ALL" else usable[usable["dataset"] == ds]
        extent = group[["extent_vox_x", "extent_vox_y", "extent_vox_z"]].to_numpy(float)
        lo = group[["bbox_lo_x", "bbox_lo_y", "bbox_lo_z"]].to_numpy(float)
        hi = group[["bbox_hi_x", "bbox_hi_y", "bbox_hi_z"]].to_numpy(float)
        center = group[["shape_x", "shape_y", "shape_z"]].to_numpy(float) // 2
        for crop in crop_sizes:
            size = np.asarray(crop, dtype=float)
            start = center - size // 2
            fits_any = (extent <= size).all(axis=1)
            fits_centered = ((lo >= start) & (hi <= start + size - 1)).all(axis=1)
            rows.append({
                "dataset": ds, "crop": "x".join(str(c) for c in crop), "n": len(group),
                "frac_fits_extent": float(fits_any.mean()) if len(group) else float("nan"),
                "frac_fits_centered": float(fits_centered.mean()) if len(group) else float("nan"),
            })
    return pd.DataFrame(rows)


def _step_hist(ax: plt.Axes, data: Mapping[str, np.ndarray], colors: Mapping[str, str],
               bins: Sequence[float]) -> None:
    for ds, values in data.items():
        if len(values):
            ax.hist(values, bins=list(bins), histtype="step", linewidth=1.8, color=colors[ds],
                    weights=np.full(len(values), 1 / len(values)), label=f"{ds} (n={len(values)})")
    ax.set_ylabel("Fraction of lesions")


def _by_dataset(usable: pd.DataFrame, col: str, datasets: Sequence[str]) -> dict[str, np.ndarray]:
    return {ds: usable.loc[usable["dataset"] == ds, col].dropna().to_numpy(float) for ds in datasets}


def plot_lesion_size(usable: pd.DataFrame, datasets: Sequence[str], colors: Mapping[str, str], path: Path) -> None:
    data = _by_dataset(usable, "lesion_eq_diam_mm", datasets)
    values = np.concatenate([v for v in data.values() if len(v)])
    fig, ax = plt.subplots(figsize=(7, 4))
    lo = max(values.min(), 1.0)
    _step_hist(ax, data, colors, np.geomspace(lo, max(values.max(), 2 * lo), 30).tolist())
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    ax.set_xlabel("Lesion size: equivalent-sphere diameter (mm, log scale)")
    ax.set_title("Lesion size per dataset")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _boxplot(ax: plt.Axes, data: Mapping[str, np.ndarray], colors: Mapping[str, str], labels: bool = True) -> None:
    """Horizontal box per dataset (top to bottom in config order), tinted with the dataset colour."""
    boxes = ax.boxplot(list(data.values()), vert=False, widths=0.5, patch_artist=True,
                       medianprops={"color": INK, "linewidth": 1.5},
                       flierprops={"marker": "o", "markersize": 3, "markeredgecolor": MUTED})
    for patch, ds in zip(boxes["boxes"], data):
        patch.set_facecolor(colors[ds] + "55")
        patch.set_edgecolor(colors[ds])
    ax.set_yticks(range(1, len(data) + 1), list(data) if labels else [""] * len(data))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def plot_spacing(usable: pd.DataFrame, datasets: Sequence[str], colors: Mapping[str, str], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.9 + 0.7 * len(datasets)))
    _boxplot(axes[0], _by_dataset(usable, "spacing_x", datasets), colors)
    _boxplot(axes[1], _by_dataset(usable, "spacing_z", datasets), colors, labels=False)
    axes[0].set_xlabel("In-plane spacing (mm)")
    axes[1].set_xlabel("Slice spacing (mm)")
    fig.suptitle("Voxel spacing per dataset", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lesion_hu(usable: pd.DataFrame, datasets: Sequence[str], colors: Mapping[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.7 * len(datasets)))
    _boxplot(ax, _by_dataset(usable, "lesion_hu_median", datasets), colors)
    ax.set_xlabel("Median intensity inside the lesion (HU)")
    ax.set_title("Lesion intensity per dataset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_padding(usable: pd.DataFrame, datasets: Sequence[str], colors: Mapping[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    _step_hist(ax, _by_dataset(usable, "pad_fraction", datasets), colors, [i / 25 for i in range(26)])
    ax.set_xlabel("Fraction of VOI voxels at the minimum value (padding)")
    ax.set_title("Padding per VOI")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_crop_fit(fit: pd.DataFrame, datasets: Sequence[str], colors: Mapping[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for ds in datasets:
        rows = fit[fit["dataset"] == ds]
        ax.plot(rows["crop"], rows["frac_fits_extent"], marker="o", markersize=6, color=colors[ds], label=ds)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Crop size (voxels, x × y × z)")
    ax.set_ylabel("Fraction of lesions that fit")
    ax.set_title("Lesions whose bounding box fits in the crop")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gallery_panel(ax: plt.Axes, row: pd.Series, data_root: Path) -> None:
    """Axial slice with the largest lesion area, zoomed around the lesion, with the mask contour."""
    image = np.asarray(load_nifti(data_root / row["image"])[0].dataobj, dtype=np.float32)
    mask = np.asarray(load_nifti(data_root / row["label"])[0].dataobj) > 0
    z = int(mask.sum(axis=(0, 1)).argmax())
    lo = np.array([row["bbox_lo_x"], row["bbox_lo_y"]], dtype=int)
    hi = np.array([row["bbox_hi_x"], row["bbox_hi_y"]], dtype=int)
    half = max(32, int((hi - lo).max()))
    mid = (lo + hi) // 2
    x0, y0 = np.clip(mid - half, 0, None)
    x1, y1 = np.minimum(mid + half, image.shape[:2])
    img, msk = image[x0:x1, y0:y1, z], mask[x0:x1, y0:y1, z]
    body = img[img > image.min()]
    vmin, vmax = np.percentile(body if body.size else img, [1, 99])
    ax.imshow(np.rot90(img), cmap="gray", vmin=vmin, vmax=vmax)
    if msk.any():
        ax.contour(np.rot90(msk), levels=[0.5], colors=CONTOUR, linewidths=1.2)
    ax.set_title(f"{row['case']}\n{row['lesion_mm3']:.0f} mm³, Ø≈{row['lesion_eq_diam_mm']:.0f} mm",
                 fontsize=8, fontweight="normal", color=INK_2)
    ax.set_axis_off()


def plot_gallery(usable: pd.DataFrame, datasets: Sequence[str], quantiles: Sequence[float],
                 data_root: Path, path: Path) -> pd.DataFrame:
    """Rows = datasets, columns = lesion-volume quantiles. Returns the chosen cases."""
    fig, axes = plt.subplots(len(datasets), len(quantiles), figsize=(3 * len(quantiles), 3.2 * len(datasets)),
                             squeeze=False)
    chosen = []
    for r, ds in enumerate(datasets):
        group = usable[usable["dataset"] == ds]
        for c, qtl in enumerate(quantiles):
            ax = axes[r, c]
            if group.empty:
                ax.set_axis_off()
                continue
            target = float(group["lesion_mm3"].quantile(qtl))
            row = group.iloc[int(np.argmin(np.abs(group["lesion_mm3"].to_numpy(float) - target)))]
            chosen.append({"dataset": ds, "quantile": qtl, "case": row["case"], "lesion_mm3": row["lesion_mm3"]})
            _gallery_panel(ax, row, data_root)
        axes[r, 0].text(-0.08, 0.5, ds, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", color=INK, fontweight="bold")
    for c, qtl in enumerate(quantiles):
        axes[0, c].annotate(f"volume quantile {qtl:.2f}", (0.5, 1.25), xycoords="axes fraction",
                            ha="center", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(chosen)


def write_report(manifest: pd.DataFrame, data_root: str | Path, eda_cfg: Mapping[str, Any],
                 datasets: Sequence[str]) -> pd.DataFrame:
    """Write all tables and figures to ``eda_cfg['output_dir']``; return the dataset summary."""
    out = Path(eda_cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    colors = dataset_colors(datasets)
    usable = manifest[usable_mask(manifest)]

    summary = dataset_summary(manifest, datasets)
    summary.to_csv(out / "summary_per_dataset.csv", index=False)
    problem_cases(manifest).to_csv(out / "problem_cases.csv", index=False)
    fit = crop_fit(manifest, eda_cfg["crop_sizes"], datasets)
    fit.to_csv(out / "crop_fit.csv", index=False)
    if "category" in usable:
        cats = usable[_has(usable["category"])]
        if not cats.empty:
            cats.groupby(["dataset", "category"]).agg(
                n=("case", "size"), lesion_mm3_median=("lesion_mm3", "median"),
                lesion_hu_median=("lesion_hu_median", "median"),
            ).reset_index().to_csv(out / "categories.csv", index=False)

    plot_lesion_size(usable, datasets, colors, out / "lesion_size.png")
    plot_spacing(usable, datasets, colors, out / "spacing.png")
    plot_lesion_hu(usable, datasets, colors, out / "lesion_hu.png")
    plot_padding(usable, datasets, colors, out / "padding.png")
    plot_crop_fit(fit, datasets, colors, out / "crop_fit.png")
    plot_gallery(usable, datasets, eda_cfg["gallery_quantiles"], Path(data_root), out / "gallery.png") \
        .to_csv(out / "gallery_cases.csv", index=False)
    return summary
