"""Intensity conventions shared by the EDA and the training transforms (D17, D18)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def intensity_shift(volume_min: float, intensity_cfg: Mapping[str, Any]) -> float:
    """Value to add to a VOI to bring it to HU: ``-shift`` for VOIs stored shifted, else 0.

    A CT VOI in HU always contains negative values (air, fat), so a VOI whose minimum is at
    least ``shifted_if_min_at_least`` is taken to be stored with a ``+shift`` offset.
    """
    if volume_min >= float(intensity_cfg["shifted_if_min_at_least"]):
        return -float(intensity_cfg["shift"])
    return 0.0
