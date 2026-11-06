"""GPU power sampling with ``nvidia-smi``, for the energy estimate (plan, section 7).

GPU energy = mean sampled power x elapsed time. The same code runs on IDUN and the home PC.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import IO, Any

import pandas as pd

QUERY = ["timestamp", "index", "name", "power.draw", "utilization.gpu", "memory.used"]


class GpuPowerLogger:
    """Context manager that appends ``nvidia-smi`` samples to a CSV every ``interval_s`` seconds.

    Only the GPUs in ``CUDA_VISIBLE_DEVICES`` (set by Slurm) are sampled. Without ``nvidia-smi``
    (e.g. a CPU-only machine) it does nothing.
    """

    def __init__(self, path: str | Path, interval_s: float) -> None:
        self.path = Path(path)
        self.interval_s = interval_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._file: IO[bytes] | None = None

    def __enter__(self) -> GpuPowerLogger:
        if shutil.which("nvidia-smi") is None:
            print("WARNING  nvidia-smi not found; GPU power is not logged.")
            return self
        cmd = ["nvidia-smi", f"--query-gpu={','.join(QUERY)}", "--format=csv,noheader,nounits",
               f"--loop-ms={int(self.interval_s * 1000)}"]
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if devices and devices != "-1":
            cmd.append(f"--id={devices}")
        self._file = open(self.path, "ab")
        self._proc = subprocess.Popen(cmd, stdout=self._file, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *_: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._file is not None:
            self._file.close()


def read_power_log(path: str | Path) -> pd.DataFrame:
    """Samples written by ``GpuPowerLogger``; rows that are not numbers (e.g. "[N/A]") are dropped."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=QUERY)
    log = pd.read_csv(path, names=QUERY, skipinitialspace=True)
    log["power.draw"] = pd.to_numeric(log["power.draw"], errors="coerce")
    return log.dropna(subset=["power.draw"])


def energy_summary(power_log: pd.DataFrame, elapsed_s: float) -> dict[str, Any]:
    """Mean GPU power (summed over the sampled GPUs), elapsed hours and GPU energy in kWh."""
    if power_log.empty:
        return {"elapsed_h": elapsed_s / 3600, "gpu": "", "mean_gpu_power_w": None, "gpu_energy_kwh": None}
    mean_w = float(power_log.groupby("index")["power.draw"].mean().sum())
    return {
        "elapsed_h": elapsed_s / 3600,
        "gpu": ", ".join(sorted(power_log["name"].astype(str).unique())),
        "n_power_samples": len(power_log),
        "mean_gpu_power_w": mean_w,
        "gpu_energy_kwh": mean_w * elapsed_s / 3.6e6,
    }
