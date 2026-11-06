"""Train a segmentation model. Uses the GPU, so the student starts it (locally or with sbatch).

Examples:
    python scripts/train.py --config configs/experiments/e0_overfit.yaml --machine configs/machines/home_pc.yaml paths.data_root=data/sample/ULS23
    python scripts/train.py --config configs/experiments/e1_dynunet3d.yaml --machine configs/machines/idun.yaml
    # Continue a stopped run (same config; machine and paths may differ):
    python scripts/train.py --config configs/experiments/e1_dynunet3d.yaml --machine configs/machines/idun.yaml --resume
"""

from __future__ import annotations

import argparse

from src.training.trainer import fit
from src.utils.config import add_config_arguments, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="Continue the run from its last.pt checkpoint.")
    args = parser.parse_args()
    fit(config_from_args(args), resume=args.resume)


if __name__ == "__main__":
    main()
