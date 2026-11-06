"""Per-lesion overlap metrics and the centre-component post-processing (D30), on binary NumPy masks.

Written directly from voxel counts rather than with MONAI's metric classes: one lesion per VOI
needs no one-hot or batch handling, and the formulas stay visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy import ndimage


def overlap_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision and recall of one predicted mask against its label.

    Precision is NaN for an empty prediction and recall for an empty label (undefined there);
    Dice and IoU are then 0, or 1 if both masks are empty.
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    n_pred, n_true = int(pred.sum()), int(truth.sum())
    union = n_pred + n_true - tp
    return {
        "dice": 2 * tp / (n_pred + n_true) if n_pred + n_true else 1.0,
        "iou": tp / union if union else 1.0,
        "precision": tp / n_pred if n_pred else float("nan"),
        "recall": tp / n_true if n_true else float("nan"),
        "pred_voxels": float(n_pred),
        "true_voxels": float(n_true),
    }


def keep_centre_component(mask: np.ndarray, spacing: Sequence[float] | None = None) -> np.ndarray:
    """Keep only the connected component (26-connectivity) containing the VOI centre.

    If no component contains the centre, keep the one with the voxel nearest to it (distance in mm
    when ``spacing`` is given). ULS23 guarantees the target lesion contains the centre voxel.
    """
    labels, n = cast(tuple[np.ndarray, int], ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3))))
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    centre = np.array(mask.shape) // 2
    keep = labels[tuple(centre)]
    if keep == 0:
        coords = np.argwhere(labels > 0)
        scale = np.asarray(spacing if spacing is not None else np.ones(mask.ndim), dtype=float)
        nearest = coords[np.argmin(np.linalg.norm((coords - centre) * scale, axis=1))]
        keep = labels[tuple(nearest)]
    return labels == keep
