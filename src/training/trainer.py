"""Training: mixed precision, checkpoint and resume, CSV log, periodic validation and compute logging.

A run lives in ``paths.work_dir/runs/<run.name>/``:
    config.yaml   resolved config          run_info.json  seed, git commit, one entry per session
    log.csv       one row per epoch        val_last.csv / val_best.csv  per-lesion validation results
    last.pt       resume checkpoint        best.pt        weights with the best mean validation Dice
    power.csv     nvidia-smi samples       compute.json   elapsed time, mean GPU power, GPU energy
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import monai.losses
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from src.data.datasets import build_dataset, build_loader, cache_dir, case_table, data_dicts
from src.data.transforms import eval_transforms, train_transforms
from src.evaluation.inference import evaluate_cases, mean_by_dataset
from src.models.factory import build_model
from src.training.checkpoint import config_differences, load_checkpoint, save_checkpoint
from src.utils.compute import GpuPowerLogger, energy_summary, read_power_log
from src.utils.config import load_yaml, save_config
from src.utils.run_info import session_info
from src.utils.seed import seed_everything


def build_from(module: ModuleType, spec: Mapping[str, Any], *args: Any) -> Any:
    """Instantiate ``module.<spec["name"]>(*args, **other keys of spec)``, e.g. a MONAI loss."""
    kwargs = dict(spec)
    return getattr(module, kwargs.pop("name"))(*args, **kwargs)


def build_scheduler(spec: Mapping[str, Any], optimizer: torch.optim.Optimizer, epochs: int) -> Any:
    """Per-epoch LR scheduler; one with a ``total_iters`` argument (e.g. PolynomialLR) gets ``epochs``."""
    cls = getattr(torch.optim.lr_scheduler, spec["name"])
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    if "total_iters" in inspect.signature(cls).parameters:
        kwargs.setdefault("total_iters", epochs)
    return cls(optimizer, **kwargs)


def train_one_epoch(model: nn.Module, loader: Iterable[dict[str, Any]], loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, amp: bool,
                    grad_clip: float | None, max_steps: int | None = None) -> dict[str, float]:
    """One pass over ``loader`` (or ``max_steps`` steps). Raises on a non-finite loss, so a bad run stops early.

    ``data_s`` is time spent waiting for batches; if it is large compared with ``compute_s``,
    data loading is the bottleneck and caching (``cache.enabled``) may help.
    """
    model.train()
    losses: list[float] = []
    data_s = compute_s = 0.0
    tick = time.perf_counter()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        loaded = time.perf_counter()
        data_s += loaded - tick
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss is {loss.item()} at step {step}")
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())  # waits for the GPU, so compute_s is the real step time
        tick = time.perf_counter()
        compute_s += tick - loaded
    return {"train_loss": float(np.mean(losses)), "steps": len(losses), "data_s": data_s, "compute_s": compute_s}


def prepare_run_dir(cfg: Mapping[str, Any], resume: bool) -> Path:
    """Create or check the run folder. A new run never overwrites one; a resume needs the same config."""
    run_dir = Path(cfg["paths"]["work_dir"]) / "runs" / cfg["run"]["name"]
    if resume:
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError(f"--resume given, but {run_dir / 'last.pt'} does not exist")
        diffs = config_differences(load_yaml(run_dir / "config.yaml"), cfg)
        if diffs:
            raise ValueError(f"Config differs from the saved run in {diffs}; start a new run.name instead")
    elif (run_dir / "last.pt").is_file():
        raise FileExistsError(f"{run_dir} already holds a run; pass --resume, or set another run.name=...")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if resume and info_path.is_file() else {"seed": cfg["seed"], "sessions": []}
    info["sessions"].append(session_info(cfg["machine"]["name"]))
    info_path.write_text(json.dumps(info, indent=2))
    return run_dir


def fit(cfg: Mapping[str, Any], resume: bool = False) -> Path:
    """Train the configured model and return the run folder."""
    run_dir = prepare_run_dir(cfg, resume)
    tcfg, vcfg, intensity = cfg["train"], cfg["validation"], cfg["data"]["intensity"]
    seed, epochs, roi = int(cfg["seed"]), int(tcfg["epochs"]), cfg["transforms"]["patch_size"]
    generator = seed_everything(seed, cfg["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    table = case_table(cfg["splits"]["manifest_path"], cfg["splits"]["path"])
    root, cases, workers = cfg["paths"]["data_root"], tcfg.get("only_cases"), cfg["machine"]["num_workers"]
    cache = cache_dir(cfg["paths"]["work_dir"], intensity) if cfg["cache"]["enabled"] else None
    train_tf = train_transforms(intensity, cfg["transforms"])
    train_items = data_dicts(table, root, ["train"], cases)
    val_items = data_dicts(table, root, vcfg["splits"], cases)
    train_loader = build_loader(build_dataset(train_items, train_tf, cache), tcfg["batch_size"], True, workers, generator)
    val_loader = build_loader(build_dataset(val_items, eval_transforms(intensity), cache), 1, False, workers)
    print(f"Run {run_dir} on {device}: {len(train_items)} training and {len(val_items)} validation VOIs")

    model = build_model(cfg["model"]).to(device)
    loss_fn = build_from(monai.losses, tcfg["loss"])
    optimizer = build_from(torch.optim, tcfg["optimizer"], model.parameters())
    scheduler = build_scheduler(tcfg["scheduler"], optimizer, epochs)
    scaler = GradScaler(device.type, enabled=amp)
    start_epoch, best, elapsed_before = 0, {"val_dice": -1.0, "epoch": 0}, 0.0
    log = pd.DataFrame()
    if resume:
        state = load_checkpoint(run_dir / "last.pt")
        for name, obj in [("model", model), ("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
            obj.load_state_dict(state[name])
        start_epoch, best, elapsed_before = state["epoch"], state["best"], state["elapsed_s"]
        log = pd.read_csv(run_dir / "log.csv")
        log = log[log["epoch"] <= start_epoch]  # drop rows logged after the last checkpoint
        best_text = f"best mean validation Dice {best['val_dice']:.3f}" if best["epoch"] else "no validation yet"
        print(f"Resuming after epoch {start_epoch} ({best_text})")

    session_start = time.perf_counter()
    with GpuPowerLogger(run_dir / "power.csv", cfg["power"]["interval_s"]):
        n_check = int(vcfg.get("check_at_start") or 0)
        if n_check:  # fail fast: the validation code runs before hours of training, not after
            check = evaluate_cases(model, val_loader, device, roi, cfg["inference"], amp, max_cases=n_check)
            print(f"Validation check passed on {len(check)} VOIs, {check['seconds'].mean():.1f} s per VOI")

        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            generator.manual_seed(seed + epoch)  # this epoch's shuffling and worker seeds
            train_tf.set_random_state(seed=seed + epoch)  # used when num_workers is 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            row: dict[str, Any] = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
            row.update(train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, amp,
                                       tcfg.get("grad_clip"), tcfg.get("max_steps_per_epoch")))
            scheduler.step()

            if vcfg["every"] and ((epoch + 1) % vcfg["every"] == 0 or epoch + 1 == epochs):
                val_start = time.perf_counter()
                per_case = evaluate_cases(model, val_loader, device, roi, cfg["inference"], amp)
                per_case.to_csv(run_dir / "val_last.csv", index=False)
                row.update({f"val_dice_{k}": v for k, v in mean_by_dataset(per_case, "dice").items()})
                row.update(val_dice_pp_all=float(per_case["dice_pp"].mean()), val_s=time.perf_counter() - val_start)
                if row["val_dice_all"] > best["val_dice"]:
                    best = {"val_dice": row["val_dice_all"], "epoch": epoch + 1}
                    save_checkpoint(run_dir / "best.pt", {"model": model.state_dict(), **best})
                    per_case.to_csv(run_dir / "val_best.csv", index=False)

            if device.type == "cuda":
                row["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
            row["epoch_s"] = time.perf_counter() - epoch_start
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
            log.to_csv(run_dir / "log.csv", index=False)
            elapsed = elapsed_before + time.perf_counter() - session_start
            save_checkpoint(run_dir / "last.pt", {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "elapsed_s": elapsed})
            summary = energy_summary(read_power_log(run_dir / "power.csv"), elapsed)
            (run_dir / "compute.json").write_text(json.dumps({**summary, "epochs_done": epoch + 1}, indent=2))
            print(format_row(row), flush=True)

    print(f"Done. Best mean validation Dice {best['val_dice']:.3f} at epoch {best['epoch']}; outputs in {run_dir}")
    return run_dir


def format_row(row: Mapping[str, Any]) -> str:
    """One log line per epoch for the console / Slurm log."""
    parts = [f"epoch {row['epoch']}", f"loss {row['train_loss']:.4f}", f"lr {row['lr']:.2e}",
             f"{row['epoch_s']:.0f} s (data wait {row['data_s']:.0f} s)"]
    if "val_dice_all" in row:
        parts.append(f"val Dice {row['val_dice_all']:.3f} (centre component {row['val_dice_pp_all']:.3f})")
    if "peak_mem_gb" in row:
        parts.append(f"peak {row['peak_mem_gb']:.1f} GB")
    return " | ".join(parts)
