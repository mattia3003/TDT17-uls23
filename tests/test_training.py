"""Tests for metrics, post-processing, compute logging, a tiny end-to-end training run with resume, and evaluation."""

import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

import src.training.trainer as trainer
from src.data.manifest import load_nifti, spatial_array
from src.evaluation.evaluator import evaluate_run, summarise
from src.evaluation.figures import pick_examples, save_examples, worst_lesions
from src.evaluation.metrics import keep_centre_component, overlap_metrics
from src.training.checkpoint import config_differences, load_checkpoint
from src.utils.compute import energy_summary, read_power_log


def test_overlap_metrics_on_known_masks() -> None:
    truth = np.zeros((10, 10, 10), bool)
    truth[2:6, 2:6, 2:6] = True  # 64 voxels
    pred = np.zeros_like(truth)
    pred[2:6, 2:6, 4:8] = True  # 64 voxels, 32 overlapping
    m = overlap_metrics(pred, truth)
    assert m["dice"] == pytest.approx(0.5) and m["iou"] == pytest.approx(32 / 96)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert overlap_metrics(truth, truth)["dice"] == 1.0
    empty = overlap_metrics(np.zeros_like(truth), truth)
    assert empty["dice"] == 0.0 and math.isnan(empty["precision"]) and empty["recall"] == 0.0


def test_keep_centre_component() -> None:
    mask = np.zeros((20, 20, 20), bool)
    mask[8:12, 8:12, 8:12] = True  # contains the centre (10, 10, 10)
    mask[0:2, 0:2, 0:2] = True     # false positive far away
    kept = keep_centre_component(mask)
    assert kept.sum() == 64 and not kept[0, 0, 0]
    # No component contains the centre: keep the nearest one, measured in mm.
    off = np.zeros((20, 20, 20), bool)
    off[10, 10, 13] = True  # 3 voxels away along z
    off[10, 14, 10] = True  # 4 voxels away along y
    assert keep_centre_component(off)[10, 10, 13]
    assert keep_centre_component(off, spacing=(1.0, 1.0, 5.0))[10, 14, 10]  # 15 mm vs 4 mm
    assert not keep_centre_component(np.zeros((4, 4, 4), bool)).any()


def test_config_differences_ignore_machine_keys() -> None:
    saved = {"seed": 1, "train": {"lr": 0.1}, "paths": {"data_root": "a"}, "machine": {"name": "idun"}}
    current = {"seed": 1, "train": {"lr": 0.2}, "paths": {"data_root": "b"}, "machine": {"name": "home_pc"}}
    assert config_differences(saved, current) == ["train.lr"]


def test_power_log_and_energy(tmp_path: Path) -> None:
    path = tmp_path / "power.csv"
    path.write_text("2026/11/06 10:00:00.000, 0, Tesla V100, 200.0, 90, 5000\n"
                    "2026/11/06 10:00:10.000, 0, Tesla V100, [N/A], 90, 5000\n"
                    "2026/11/06 10:00:20.000, 0, Tesla V100, 300.0, 95, 5000\n")
    summary = energy_summary(read_power_log(path), elapsed_s=3600)
    assert summary["mean_gpu_power_w"] == pytest.approx(250.0)
    assert summary["gpu_energy_kwh"] == pytest.approx(0.25) and summary["gpu"] == "Tesla V100"
    assert energy_summary(read_power_log(tmp_path / "missing.csv"), 60)["gpu_energy_kwh"] is None


SHAPE = (64, 64, 32)


def write_case(root: Path, case: str, seed: int) -> dict[str, Any]:
    """A noisy VOI with a bright cube lesion at the centre, stored as (x, y, z, 1)."""
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 30.0, SHAPE).astype(np.float32)
    label = np.zeros(SHAPE, np.float32)
    label[28:36, 28:36, 13:19] = 1.0
    image[label > 0] += 300.0
    for kind, array in [("images", image), ("labels", label)]:
        (root / kind).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(array[..., None], np.eye(4)), root / kind / f"{case}.nii.gz")
    return {"dataset": "DS", "case": case, "image": f"images/{case}.nii.gz", "label": f"labels/{case}.nii.gz"}


def tiny_config(tmp_path: Path, run_name: str) -> dict[str, Any]:
    """Real config structure, with a 3-level DynUNet and 32x32x16 patches so it runs on a CPU in seconds."""
    data = tmp_path / "data"
    rows = [write_case(data, f"c{i}", i) for i in range(4)]
    pd.DataFrame(rows).assign(patient_id="unused").to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame({"dataset": "DS", "case": [r["case"] for r in rows], "patient_id": [f"DS:{i}" for i in range(4)],
                  "split": ["train", "train", "train", "val"], "course_split": "train", "patient_in_both": False,
                  "image": [r["image"] for r in rows], "label": [r["label"] for r in rows],
                  }).to_csv(tmp_path / "splits.csv", index=False)
    return {
        "seed": 0, "deterministic": True, "run": {"name": run_name},
        "machine": {"name": "test", "num_workers": 0},
        "paths": {"data_root": str(data), "work_dir": str(tmp_path / "work")},
        "data": {"intensity": {"shifted_if_min_at_least": 0, "shift": 1024, "clip": [-1000, 1500]}},
        "splits": {"manifest_path": str(tmp_path / "manifest.csv"), "path": str(tmp_path / "splits.csv")},
        "transforms": {"patch_size": [32, 32, 16], "samples_per_volume": 2,
                       "augment": [{"name": "RandFlipd", "keys": ["image", "label"], "prob": 0.5, "spatial_axis": 0}]},
        "cache": {"enabled": False},
        "train": {"epochs": 3, "batch_size": 2, "amp": True, "grad_clip": 12,
                  "loss": {"name": "DiceCELoss", "to_onehot_y": True, "softmax": True, "include_background": False},
                  "optimizer": {"name": "SGD", "lr": 0.01, "momentum": 0.9},
                  "scheduler": {"name": "PolynomialLR", "power": 0.9}},
        "validation": {"splits": ["val"], "every": 2, "check_at_start": 1},
        "inference": {"sw_batch_size": 2, "overlap": 0.25, "mode": "gaussian"},
        "power": {"interval_s": 1},
        "model": {"name": "dynunet", "spatial_dims": 3, "in_channels": 1, "out_channels": 2,
                  "kernel_size": [3, 3, 3], "strides": [1, 2, 2], "filters": [4, 8, 16], "norm_name": "instance"},
    }


def test_fit_writes_outputs_and_resume_matches_an_uninterrupted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU: deterministic and runs anywhere
    full = trainer.fit(tiny_config(tmp_path, "full"))
    for name in ["config.yaml", "run_info.json", "log.csv", "last.pt", "best.pt", "val_last.csv", "compute.json"]:
        assert (full / name).is_file(), name
    log = pd.read_csv(full / "log.csv")
    assert log["epoch"].tolist() == [1, 2, 3] and log["val_dice_all"].notna().tolist() == [False, True, True]
    with pytest.raises(FileExistsError):
        trainer.fit(tiny_config(tmp_path, "full"))  # never overwrite a run

    # Stop a second run after epoch 1 (as a time limit would), then resume it.
    real_epoch = trainer.train_one_epoch
    calls = {"n": 0}

    def stop_on_second_epoch(*args: Any, **kwargs: Any) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(trainer, "train_one_epoch", stop_on_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(tiny_config(tmp_path, "resumed"))
    monkeypatch.setattr(trainer, "train_one_epoch", real_epoch)
    with pytest.raises(ValueError, match="Config differs"):
        trainer.fit({**tiny_config(tmp_path, "resumed"), "seed": 1}, resume=True)
    resumed = trainer.fit(tiny_config(tmp_path, "resumed"), resume=True)

    assert pd.read_csv(resumed / "log.csv")["epoch"].tolist() == [1, 2, 3]
    a, b = load_checkpoint(full / "last.pt")["model"], load_checkpoint(resumed / "last.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_evaluate_run_matches_validation_and_saves_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = tiny_config(tmp_path, "run")
    run_dir = trainer.fit(cfg)
    out = evaluate_run(cfg, run_dir, ["val"])
    lesions = pd.read_csv(out / "per_lesion.csv")
    assert lesions["case"].tolist() == ["c3"]
    assert {"dice", "dice_pp", "predict_s", "split", "course_split", "intensity_group"} <= set(lesions)
    # Same weights and code path as the validation that chose best.pt, so the same Dice.
    assert lesions["dice"].tolist() == pytest.approx(pd.read_csv(run_dir / "val_best.csv")["dice"].tolist())
    mask, _ = load_nifti(out / "predictions" / "c3.nii.gz")
    assert spatial_array(mask).shape == SHAPE
    assert (out / "summary.csv").is_file() and json.loads((out / "compute.json").read_text())["vois"] == 1
    with pytest.raises(FileExistsError):
        evaluate_run(cfg, run_dir, ["val"])  # results are never overwritten

    windows = {"soft_tissue": [-160, 240], "bone": [-450, 1250], "bone_above_hu": 500}
    figures = save_examples(cfg, out, windows, worst_n=2)
    assert (figures / "examples.png").is_file() and (figures / "worst.png").is_file()
    assert pd.read_csv(figures / "examples.csv")["pick"].tolist() == ["worst", "median", "best"]


def test_two_channel_input_trains_and_evaluates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = tiny_config(tmp_path, "two_channels")
    cfg["data"]["intensity"]["extra_windows"] = [[-160, 240]]
    cfg["model"]["in_channels"] = 2
    cfg["train"]["epochs"] = 1
    run_dir = trainer.fit(cfg)
    assert pd.read_csv(evaluate_run(cfg, run_dir, ["val"]) / "per_lesion.csv")["case"].tolist() == ["c3"]


def test_pick_examples_and_worst_lesions() -> None:
    lesions = pd.DataFrame({"dataset": ["A"] * 4 + ["B"], "case": ["a1", "a2", "a3", "a4", "b1"],
                            "dice": [0.9, 0.1, 0.5, 0.7, 0.3]})
    picks = pick_examples(lesions)
    assert list(zip(picks["pick"], picks["case"])) == [("worst", "a2"), ("median", "a3"), ("best", "a1"),
                                                       ("worst", "b1"), ("median", "b1"), ("best", "b1")]
    assert worst_lesions(lesions, 2)["case"].tolist() == ["a2", "b1"]


def test_summarise_groups() -> None:
    lesions = pd.DataFrame({
        "dataset": ["A", "A", "A", "P", "P"], "split": ["test", "test", "test", "heldout", "heldout"],
        "course_split": ["test", "test", "test", "train", "test"], "patient_in_both": [True, False, False, False, False],
        "dice": [0.2, 0.6, 0.8, 0.5, 0.1], "dice_pp": [0.3, 0.6, 0.8, 0.5, 0.2]})
    summary = summarise(lesions).set_index("group")
    assert summary.loc["all", "lesions"] == 5 and summary.loc["A", "dice_mean"] == pytest.approx(1.6 / 3)
    assert summary.loc["A: test, patient not in course train", "dice_mean"] == pytest.approx(0.7)
    assert summary.loc["P: course test", "lesions"] == 1 and summary.loc["P: course test", "dice_pp_mean"] == 0.2
    assert "A: course test" not in summary.index  # one course split only: no extra row
