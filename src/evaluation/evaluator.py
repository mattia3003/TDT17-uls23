"""Evaluation of a trained run on whole VOIs: a per-lesion CSV and grouped summaries (E2, E3).

Outputs go to ``<run folder>/eval/<name>/`` (default name ``<splits>_<checkpoint>``):
    per_lesion.csv  metrics, raw and after D30, with the case metadata for the E3 breakdowns
    summary.csv     mean and median Dice per group     predictions/  raw predicted masks, <case>.nii.gz
    power.csv       nvidia-smi samples                 compute.json  elapsed time, GPU energy, checkpoint
An existing ``per_lesion.csv`` is never overwritten, so a split is evaluated once unless its folder is deleted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.datasets import build_dataset, build_loader, case_table, check_files_exist, data_dicts
from src.data.eda import with_intensity_correction
from src.data.transforms import eval_transforms
from src.evaluation.inference import evaluate_cases
from src.models.factory import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.compute import GpuPowerLogger, energy_summary, read_power_log

# Case metadata copied into the per-lesion CSV when present: the E3 breakdowns (dataset, size, slice
# spacing, Bone intensity group, DeepLesion3D category), the clean subset (D41) and the failure review.
METADATA_COLUMNS = ["patient_id", "split", "course_split", "patient_in_both", "category", "intensity_group",
                    "lesion_voxels", "lesion_mm3", "lesion_eq_diam_mm", "spacing_x", "spacing_z",
                    "lesion_shell_contrast_hu", "touches_border"]


def lesion_table(per_case: pd.DataFrame, table: pd.DataFrame, intensity_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Rows of ``evaluate_cases`` joined with each case's metadata, one row per lesion."""
    meta = with_intensity_correction(table, intensity_cfg)
    columns = ["dataset", "case", *[c for c in METADATA_COLUMNS if c in meta]]
    return per_case.merge(meta[columns], on=["dataset", "case"], how="left", validate="one_to_one")


def summarise(lesions: pd.DataFrame) -> pd.DataFrame:
    """Mean and median Dice, raw and after D30, per group.

    Groups: all lesions; each dataset; each dataset's course train and test parts when both are
    present (Pancreas in E2); and each dataset's test lesions whose patient has no lesion in
    course training (the clean subset of D41).
    """
    groups: dict[str, pd.DataFrame] = {"all": lesions}
    for dataset, part in lesions.groupby("dataset", sort=False):
        groups[str(dataset)] = part
        if part["course_split"].nunique() > 1:
            for course_split, sub in part.groupby("course_split"):
                groups[f"{dataset}: course {course_split}"] = sub
        test = part[part["split"] == "test"]
        if len(test):
            groups[f"{dataset}: test, patient not in course train"] = test[~test["patient_in_both"].astype(bool)]
    return pd.DataFrame([{"group": name, "lesions": len(g), "dice_mean": g["dice"].mean(),
                          "dice_median": g["dice"].median(), "dice_pp_mean": g["dice_pp"].mean(),
                          "dice_pp_median": g["dice_pp"].median()} for name, g in groups.items()])


def evaluate_run(cfg: Mapping[str, Any], run_dir: Path, splits: Sequence[str], checkpoint: str = "best",
                 cases: Sequence[str] | None = None, name: str | None = None) -> Path:
    """Predict every VOI of ``splits`` (or only ``cases``) with ``<checkpoint>.pt``; return the output folder."""
    name = name or f"{'+'.join(splits)}_{checkpoint}" + ("_subset" if cases else "")
    out_dir = run_dir / "eval" / name
    if (out_dir / "per_lesion.csv").is_file():
        raise FileExistsError(f"{out_dir} already holds results; delete that folder to evaluate again")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    intensity = cfg["data"]["intensity"]

    table = case_table(cfg["splits"]["manifest_path"], cfg["splits"]["path"])
    items = data_dicts(table, cfg["paths"]["data_root"], splits, cases)
    check_files_exist(items)
    loader = build_loader(build_dataset(items, eval_transforms(intensity)), 1, False, cfg["machine"]["num_workers"])
    state = load_checkpoint(run_dir / f"{checkpoint}.pt")
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(state["model"])
    print(f"Evaluating {run_dir.name}/{checkpoint}.pt (epoch {state['epoch']}) on {device}: "
          f"{len(items)} VOIs from {list(splits)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "power.csv").unlink(missing_ok=True)  # left by an attempt that stopped early
    start = time.perf_counter()
    with GpuPowerLogger(out_dir / "power.csv", cfg["power"]["interval_s"]):
        per_case = evaluate_cases(model, loader, device, cfg["transforms"]["patch_size"], cfg["inference"], amp,
                                  save_dir=out_dir / "predictions")
    elapsed = time.perf_counter() - start

    lesions = lesion_table(per_case, table, intensity)
    summary = summarise(lesions)
    lesions.to_csv(out_dir / "per_lesion.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    compute = {**energy_summary(read_power_log(out_dir / "power.csv"), elapsed), "vois": len(items),
               "mean_predict_s": float(per_case["predict_s"].mean()), "checkpoint": f"{checkpoint}.pt",
               "epoch": state["epoch"]}
    (out_dir / "compute.json").write_text(json.dumps(compute, indent=2))
    print(summary.to_string(index=False, float_format="%.3f"))
    print(f"Outputs in {out_dir}")
    return out_dir
