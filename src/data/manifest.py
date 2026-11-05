"""Build the per-VOI manifest: identifiers, geometry, lesion and intensity statistics.

One row per case (image and/or label file). Paths are stored relative to
``paths.data_root`` so the manifest is portable between machines.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from src.data.discovery import dataset_dirs, is_image_file, strip_image_suffix

# Below this fraction of voxels at the VOI minimum we treat the VOI as unpadded.
PAD_FRACTION_THRESHOLD = 0.01
# Thickness (voxels) of the shell around the lesion used for the lesion-vs-surroundings contrast.
SHELL_VOXELS = 3


def load_nifti(path: str | Path) -> tuple[nib.Nifti1Image, nib.Nifti1Header]:
    """Load a NIfTI file (lazily; voxel data is read on access) with its header."""
    image = cast(nib.Nifti1Image, nib.load(path))
    return image, cast(nib.Nifti1Header, image.header)


def header_affine(header: nib.Nifti1Header) -> np.ndarray:
    """Voxel-to-world affine stored in the header (sform/qform), as a plain array."""
    return cast(np.ndarray, header.get_best_affine())


def spatial_array(image: nib.Nifti1Image, dtype: Any = None) -> np.ndarray:
    """Voxel data as a 3-D array; trailing size-1 axes (e.g. a stored ``(x, y, z, 1)``) are dropped.

    MONAI's ``EnsureChannelFirst`` hides such an axis, plain nibabel does not.
    """
    data = np.asarray(image.dataobj, dtype=dtype)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {data.shape}")
    return data


def parse_case_id(case: str, pattern: str | None) -> tuple[str, bool]:
    """Return ``(patient, parsed)`` from a case name using the regex group ``patient``.

    Falls back to the case name itself (``parsed=False``) so every case still gets a group.
    """
    if pattern:
        match = re.match(pattern, case)
        if match and match.groupdict().get("patient"):
            return match.group("patient"), True
    return case, False


def read_categories(category_dir: Path) -> dict[str, str]:
    """Map case name -> category from ``category_dir/<category>/**/<case>.nii.gz``."""
    categories: dict[str, str] = {}
    if not category_dir.is_dir():
        return categories
    for category in sorted(p.name for p in category_dir.iterdir() if p.is_dir()):
        for _, _, filenames in os.walk(category_dir / category):
            for name in filenames:
                if is_image_file(name):
                    case = strip_image_suffix(name)
                    categories[case] = f"{categories[case]}|{category}" if case in categories else category
    return categories


def list_cases(data_root: str | Path, data_cfg: Mapping[str, Any], dataset: str) -> pd.DataFrame:
    """List every image and/or label of ``dataset`` with relative paths, patient ID and category."""
    root = Path(data_root)
    ds_cfg = data_cfg["datasets"][dataset] or {}
    image_dir, label_dir = dataset_dirs(root, data_cfg, dataset)
    images = {strip_image_suffix(n): n for n in os.listdir(image_dir) if is_image_file(n)}
    labels = {strip_image_suffix(n): n for n in os.listdir(label_dir) if is_image_file(n)} if label_dir.is_dir() else {}
    categories = {}
    if ds_cfg.get("category_dir"):
        categories = read_categories(root / ds_cfg["category_dir"].format(dataset=dataset))

    rows = []
    for case in sorted(images.keys() | labels.keys()):
        patient, parsed = parse_case_id(case, ds_cfg.get("id_pattern"))
        rows.append({
            "dataset": dataset,
            "case": case,
            # Prefix with the dataset so equal IDs in different datasets never merge.
            "patient_id": f"{dataset}:{patient}",
            "id_parsed": parsed,
            "category": categories.get(case, ""),
            "image": (image_dir / images[case]).relative_to(root).as_posix() if case in images else "",
            "label": (label_dir / labels[case]).relative_to(root).as_posix() if case in labels else "",
        })
    return pd.DataFrame(rows)


def _xyz(prefix: str, values: Iterable[float]) -> dict[str, float]:
    return {f"{prefix}_{axis}": float(v) for axis, v in zip("xyz", values)}


def voi_stats(image_path: Path, label_path: Path | None) -> dict[str, Any]:
    """Geometry, padding, intensity and lesion statistics for one VOI (array axes x, y, z)."""
    image_nii, header = load_nifti(image_path)
    image = spatial_array(image_nii, dtype=np.float32)
    spacing = np.array(header.get_zooms()[:3], dtype=float)
    slope, inter = header.get_slope_inter()
    stats: dict[str, Any] = {
        **_xyz("shape", image.shape),
        "image_ndim_disk": len(image_nii.shape),
        **_xyz("spacing", spacing),
        "axcodes": "".join(nib.aff2axcodes(header_affine(header))),
        "disk_dtype": str(image_nii.get_data_dtype()),
        "scl_slope": slope,
        "scl_inter": inter,
        "image_mb": os.path.getsize(image_path) / 1e6,
    }

    # Padding uses (volume minimum - 1), so padded voxels sit exactly at the VOI minimum.
    vmin = float(image.min())
    at_min = image == vmin
    pad_fraction = float(at_min.mean())
    body = image[~at_min] if pad_fraction > PAD_FRACTION_THRESHOLD else image.ravel()
    p_lo, p_mid, p_hi = np.percentile(body, [0.5, 50, 99.5])
    stats.update(image_min=vmin, image_max=float(image.max()), pad_fraction=pad_fraction,
                 hu_p00_5=p_lo, hu_p50=p_mid, hu_p99_5=p_hi)
    if label_path is None:
        return stats

    label_nii, label_header = load_nifti(label_path)
    label = spatial_array(label_nii)
    same_shape = label.shape == image.shape
    same_affine = bool(np.allclose(header_affine(label_header), header_affine(header), atol=1e-3))
    stats.update(
        label_mb=os.path.getsize(label_path) / 1e6,
        label_ndim_disk=len(label_nii.shape),
        label_shape_match=same_shape,
        label_affine_match=same_affine,
        label_geometry_match=same_shape and same_affine,
        label_n_values=int(np.unique(label).size),
        label_max=float(label.max()),
    )
    mask = label > 0
    n_voxels = int(mask.sum())
    volume_mm3 = n_voxels * float(np.prod(spacing))
    stats.update(lesion_voxels=n_voxels, lesion_mm3=volume_mm3,
                 lesion_eq_diam_mm=(6 * volume_mm3 / math.pi) ** (1 / 3))
    if n_voxels == 0 or mask.shape != image.shape:
        return stats

    _, n_components = cast(tuple[np.ndarray, int], ndimage.label(mask, structure=np.ones((3, 3, 3))))
    coords = np.argwhere(mask)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    extent = hi - lo + 1
    center = np.array(mask.shape) // 2
    lesion_hu = image[mask]
    stats.update(
        n_components=int(n_components),
        **_xyz("bbox_lo", lo),
        **_xyz("bbox_hi", hi),
        **_xyz("extent_vox", extent),
        **_xyz("extent_mm", extent * spacing),
        touches_border=bool((lo == 0).any() or (hi == np.array(mask.shape) - 1).any()),
        center_in_lesion=bool(mask[tuple(center)]),
        centroid_offset_mm=float(np.linalg.norm((coords.mean(axis=0) - center) * spacing)),
        lesion_hu_mean=float(lesion_hu.mean()),
        lesion_hu_median=float(np.median(lesion_hu)),
        lesion_hu_p05=float(np.percentile(lesion_hu, 5)),
        lesion_hu_p95=float(np.percentile(lesion_hu, 95)),
        **label_quality(image, mask, ~at_min, lo, hi),
    )
    return stats


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, ndimage.binary_fill_holes(mask))


def label_quality(image: np.ndarray, mask: np.ndarray, body: np.ndarray,
                  lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
    """Indicators of possibly odd labels, computed in a crop around the lesion bounding box.

    Holes are counted per axial slice (catches holes that are open along z) and in 3-D.
    The contrast is the median intensity in the lesion minus that in a thin shell around it,
    leaving out padding.
    """
    start = np.maximum(lo - SHELL_VOXELS - 1, 0)
    stop = np.minimum(hi + SHELL_VOXELS + 2, mask.shape)
    region = tuple(slice(a, b) for a, b in zip(start, stop))
    sub_mask = mask[region]
    n_voxels = int(sub_mask.sum())
    filled_2d = np.stack([_fill_holes(sub_mask[:, :, k]) for k in range(sub_mask.shape[2])], axis=2)
    dilated = cast(np.ndarray, ndimage.binary_dilation(sub_mask, iterations=SHELL_VOXELS))
    shell = dilated & ~sub_mask & body[region]
    shell_hu = image[region][shell]
    shell_median = float(np.median(shell_hu)) if shell_hu.size else float("nan")
    return {
        "hole_voxels_2d": int(filled_2d.sum()) - n_voxels,
        "hole_voxels_3d": int(_fill_holes(sub_mask).sum()) - n_voxels,
        "shell_hu_median": shell_median,
        "lesion_shell_contrast_hu": float(np.median(image[region][sub_mask])) - shell_median,
    }


def _safe_voi_stats(paths: tuple[Path | None, Path | None]) -> dict[str, Any]:
    """Worker wrapper: never raise, so one unreadable file cannot stop the whole job."""
    image_path, label_path = paths
    if image_path is None:
        return {"error": "no image"}
    try:
        return voi_stats(image_path, label_path)
    except Exception as err:  # noqa: BLE001 - recorded in the manifest instead
        return {"error": f"{type(err).__name__}: {err}"}


def _check_errors(results: list[dict[str, Any]], max_error_fraction: float) -> None:
    """Raise if too many of the cases so far failed, so a systematic bug stops the job early."""
    errors = [r["error"] for r in results if "error" in r and r["error"] != "no image"]
    if len(errors) > max_error_fraction * len(results):
        raise RuntimeError(f"{len(errors)} of the first {len(results)} cases failed "
                           f"(limit {max_error_fraction:.0%}); first error: {errors[0]}")


def build_manifest(data_root: str | Path, data_cfg: Mapping[str, Any], num_workers: int = 1,
                   progress_every: int = 100, check_after: int = 50,
                   max_error_fraction: float = 0.05) -> pd.DataFrame:
    """List all configured datasets and compute per-VOI statistics in parallel.

    After the first ``check_after`` cases the job stops if more than ``max_error_fraction``
    of them could not be read, so a systematic bug fails in minutes instead of hours.
    """
    root = Path(data_root)
    cases = pd.concat([list_cases(root, data_cfg, ds) for ds in data_cfg["datasets"]], ignore_index=True)
    tasks = [(root / img if img else None, root / lab if lab else None)
             for img, lab in zip(cases["image"], cases["label"])]
    print(f"Computing statistics for {len(tasks)} cases with {num_workers} workers")
    results: list[dict[str, Any]] = []
    with Pool(num_workers) as pool:
        for i, stats in enumerate(pool.imap(_safe_voi_stats, tasks, chunksize=4), start=1):
            results.append(stats)
            if i == min(check_after, len(tasks)):
                _check_errors(results, max_error_fraction)
            if i % progress_every == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    n_errors = sum("error" in r for r in results)
    print(f"Read errors: {n_errors} of {len(results)} cases (details in the 'error' column)")
    return pd.concat([cases, pd.DataFrame(results)], axis=1)
