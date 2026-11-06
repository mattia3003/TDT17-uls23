"""Case lists from the split file and manifest, and MONAI datasets with optional on-disk caching."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.transforms import Compose

# Bump when the code of transforms.preprocessing changes, so an old cache is never reused.
PREPROCESSING_VERSION = 1


def case_table(manifest_path: str | Path, splits_path: str | Path) -> pd.DataFrame:
    """The split file joined with the manifest: paths plus the metadata used for the E3 breakdowns."""
    splits = pd.read_csv(splits_path)
    manifest = pd.read_csv(manifest_path).drop(columns="patient_id")
    table = splits.merge(manifest, on=["dataset", "case"], how="left", validate="one_to_one")
    missing = table.loc[table["image"].isna(), "case"]
    if len(missing):
        raise ValueError(f"{len(missing)} split cases are not in the manifest, e.g. {missing.iloc[0]}")
    return table


def data_dicts(table: pd.DataFrame, data_root: str | Path, splits: Sequence[str],
               max_per_dataset: int | None = None) -> list[dict[str, str]]:
    """MONAI-style items (absolute image/label paths plus case and dataset names) for the given splits.

    ``max_per_dataset`` keeps the first cases of each dataset (sorted by name), e.g. for E0.
    """
    rows = table[table["split"].isin(splits)].sort_values(["dataset", "case"])
    if max_per_dataset is not None:
        rows = rows.groupby("dataset", sort=False).head(max_per_dataset)
    root = Path(data_root)
    return [{"image": str(root / str(image)), "label": str(root / str(label)), "case": str(case), "dataset": str(ds)}
            for image, label, case, ds in zip(rows["image"], rows["label"], rows["case"], rows["dataset"])]


def cache_dir(work_dir: str | Path, intensity_cfg: Mapping[str, Any]) -> Path:
    """Cache folder named by a hash of the preprocessing settings, so changed settings get a new cache."""
    key = json.dumps({"version": PREPROCESSING_VERSION, "intensity": intensity_cfg}, sort_keys=True)
    return Path(work_dir) / "cache" / hashlib.sha1(key.encode()).hexdigest()[:10]


def build_dataset(items: Sequence[dict[str, str]], transform: Compose, cache: Path | None = None) -> Dataset:
    """Plain ``Dataset``, or a ``PersistentDataset`` that stores the deterministic part under ``cache``."""
    if cache is None:
        return Dataset(list(items), transform)
    return PersistentDataset(list(items), transform, cache_dir=cache)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                 generator: torch.Generator | None = None) -> DataLoader:
    """MONAI ``DataLoader``: its default ``worker_init_fn`` seeds each worker's random transforms."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=generator, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)
