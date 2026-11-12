"""Config-driven MONAI transform chains for training and evaluation.

Both chains start with the same deterministic preprocessing: load, offset correction (D17),
clip and scale to [0, 1] (D18), with one channel per HU window when ``extra_windows`` is set
(D43). It ends in compact dtypes (image float16, label uint8), so a
``PersistentDataset`` cache stays small; float16 changes intensities by at most about 1 HU.
Training then samples patches (D19), casts to float32 and applies the configured augmentations.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import monai.transforms as mt
import torch

from src.data.intensity import intensity_shift

KEYS = ("image", "label")


class CorrectIntensityOffsetd(mt.MapTransform):
    """Bring VOIs stored with a ``+shift`` offset back to HU, using the data-driven rule of D17."""

    def __init__(self, keys: Sequence[Hashable] | Hashable, intensity_cfg: Mapping[str, Any]) -> None:
        super().__init__(keys)
        self.intensity_cfg = dict(intensity_cfg)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] + intensity_shift(float(d[key].min()), self.intensity_cfg)
        return d


class WindowChannelsd(mt.MapTransform):
    """One channel per HU window ``(low, high)``, each clipped and scaled to [0, 1] (D18, D43).

    With a single window this is exactly ``ScaleIntensityRanged``, so single-channel runs are unchanged.
    """

    def __init__(self, keys: Sequence[Hashable] | Hashable, windows: Sequence[Sequence[float]]) -> None:
        super().__init__(keys)
        self.scalers = [mt.ScaleIntensityRange(a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True)
                        for low, high in windows]

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.cat([torch.as_tensor(scale(d[key])) for scale in self.scalers], dim=0)
        return d


def input_windows(intensity_cfg: Mapping[str, Any]) -> list[Sequence[float]]:
    """The ``clip`` window (D18), then any ``extra_windows`` (D43): one input channel each."""
    return [intensity_cfg["clip"], *(intensity_cfg.get("extra_windows") or [])]


def preprocessing(intensity_cfg: Mapping[str, Any]) -> list[mt.Transform]:
    """Deterministic part, shared by training and evaluation (and cached when caching is on)."""
    return [
        mt.LoadImaged(keys=KEYS),
        mt.EnsureChannelFirstd(keys=KEYS),  # also drops the stored trailing size-1 axis (D11)
        CorrectIntensityOffsetd(keys="image", intensity_cfg=intensity_cfg),
        WindowChannelsd(keys="image", windows=input_windows(intensity_cfg)),
        mt.EnsureTyped(keys="image", dtype=torch.float16),
        mt.EnsureTyped(keys="label", dtype=torch.uint8),
    ]


def to_float32() -> mt.Transform:
    """Cast patches back to float32 for training (metadata is not needed there)."""
    return mt.EnsureTyped(keys=KEYS, dtype=torch.float32, track_meta=False)


def from_specs(specs: Sequence[Mapping[str, Any]]) -> list[mt.Transform]:
    """Build MONAI transforms from ``{name: <monai.transforms class>, <its arguments>...}`` entries."""
    transforms = []
    for spec in specs:
        args = dict(spec)
        transforms.append(getattr(mt, args.pop("name"))(**args))
    return transforms


def train_transforms(intensity_cfg: Mapping[str, Any], transforms_cfg: Mapping[str, Any]) -> mt.Compose:
    """Preprocessing, then ``samples_per_volume`` random patches per VOI, then augmentation."""
    return mt.Compose([
        *preprocessing(intensity_cfg),
        mt.RandSpatialCropSamplesd(keys=KEYS, roi_size=transforms_cfg["patch_size"],
                                   num_samples=transforms_cfg["samples_per_volume"], random_size=False),
        to_float32(),
        *from_specs(transforms_cfg.get("augment") or []),
    ])


def eval_transforms(intensity_cfg: Mapping[str, Any]) -> mt.Compose:
    """Preprocessing only: the whole VOI on its native grid, with metadata (affine) kept.

    It stays in the compact dtypes and ends exactly where the training chain's cached part ends,
    so both chains share one cache file per VOI; the prediction code casts to float32.
    """
    return mt.Compose(preprocessing(intensity_cfg))
