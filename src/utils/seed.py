"""Central seeding, so every run is reproducible as far as is practical.

Remaining non-determinism (documented in DECISIONS.md): some CUDA kernels (e.g. the backward
pass of 3-D transposed convolutions and atomic adds in loss reductions) are not bit-exact
between runs, even with cuDNN set to deterministic.
"""

from __future__ import annotations

import torch
from monai.utils import set_determinism


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy, PyTorch (CPU and CUDA) and MONAI; return a seeded ``torch.Generator``.

    ``set_determinism`` seeds ``random``, NumPy and PyTorch. Pass the returned generator to the
    ``DataLoader``: it fixes the shuffling order and the base seed that MONAI's default
    ``worker_init_fn`` uses to seed the random transforms in each worker. With ``deterministic``,
    cuDNN uses deterministic algorithms and no auto-tuning (somewhat slower).
    """
    set_determinism(seed=seed)  # also sets cudnn.deterministic=True, benchmark=False
    if not deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return torch.Generator().manual_seed(seed)
