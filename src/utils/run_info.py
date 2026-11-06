"""What is saved with each run so it can be traced and reproduced: git commit, versions, machine."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any

import monai
import torch


def git_commit() -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes (``None`` outside git)."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def session_info(machine: str) -> dict[str, Any]:
    """Facts about this training session (a resumed run has one entry per session)."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "start": datetime.now().isoformat(timespec="seconds"),
        "machine": machine,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": gpu,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "monai": monai.__version__,
        **git_commit(),
    }
