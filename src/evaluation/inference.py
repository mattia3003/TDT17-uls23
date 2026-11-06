"""Whole-VOI prediction with MONAI sliding-window inference, and per-lesion validation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.inferers.utils import sliding_window_inference
from torch import nn

from src.evaluation.metrics import keep_centre_component, overlap_metrics


def predict_mask(model: nn.Module, image: torch.Tensor, roi_size: Sequence[int],
                 inference_cfg: Mapping[str, Any], amp: bool) -> torch.Tensor:
    """Foreground mask (bool, ``B x spatial``) for whole volumes ``B x 1 x spatial``, by sliding window."""
    with torch.no_grad(), torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=amp):
        logits = cast(torch.Tensor, sliding_window_inference(
            image, roi_size=list(roi_size), sw_batch_size=inference_cfg["sw_batch_size"], predictor=model,
            overlap=inference_cfg["overlap"], mode=inference_cfg["mode"]))
    return logits.argmax(dim=1) > 0


def evaluate_cases(model: nn.Module, batches: Iterable[dict[str, Any]], device: torch.device,
                   roi_size: Sequence[int], inference_cfg: Mapping[str, Any], amp: bool,
                   max_cases: int | None = None) -> pd.DataFrame:
    """Per-lesion metrics, raw and after keeping the centre component (``*_pp``), one row per VOI.

    ``batches`` must yield one VOI at a time (batch size 1), with the metadata of the eval transforms.
    """
    model.eval()
    rows = []
    for i, batch in enumerate(batches):
        if max_cases is not None and i >= max_cases:
            break
        start = time.perf_counter()
        image = cast(MetaTensor, batch["image"])
        pred = predict_mask(model, image.to(device).float(), roi_size, inference_cfg, amp)[0].cpu().numpy()
        truth = batch["label"][0, 0].numpy() > 0
        spacing = affine_to_spacing(image.affine[0], r=truth.ndim).tolist()
        raw = overlap_metrics(pred, truth)
        post = overlap_metrics(keep_centre_component(pred, spacing), truth)
        rows.append({"case": batch["case"][0], "dataset": batch["dataset"][0], **raw,
                     **{f"{k}_pp": v for k, v in post.items() if k != "true_voxels"},
                     "seconds": time.perf_counter() - start})
    return pd.DataFrame(rows)


def mean_by_dataset(per_case: pd.DataFrame, column: str) -> dict[str, float]:
    """Mean of ``column`` overall and per dataset (keys ``all`` and dataset names)."""
    means = {"all": float(np.mean(per_case[column]))}
    means.update({str(k): float(v) for k, v in per_case.groupby("dataset")[column].mean().items()})
    return means
