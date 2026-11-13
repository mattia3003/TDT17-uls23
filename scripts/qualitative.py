"""Qualitative figures from an evaluation folder written by scripts/evaluate.py (no GPU).

Writes <work_dir>/runs/<run>/eval/<eval>/figures/, each figure with a CSV of the chosen cases:
    examples.png  worst, median and best lesion per dataset
    worst.png     the lowest-scoring lesions overall (ties broken by case name)
    missed.png    random sample of missed lesions per dataset (centre-component Dice < 0.05), for E3
    partial.png   random sample of partly found lesions per dataset (centre-component Dice 0.05-0.5)
It reads about 60 VOIs, so it is light enough for the IDUN login node. See src/evaluation/figures.py.

Examples:
    python scripts/qualitative.py --run e1_dynunet3d --eval val_best --machine configs/machines/idun.yaml
    python scripts/qualitative.py --run e0_overfit_v2 --eval train+val+heldout_best_subset --worst 4 --machine configs/machines/home_pc.yaml paths.data_root=data/sample/ULS23 paths.work_dir=work
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.figures import save_examples
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run name: a folder in <work_dir>/runs/.")
    parser.add_argument("--eval", required=True, help="Evaluation folder in <run>/eval/, e.g. val_best.")
    parser.add_argument("--worst", type=int, default=12, help="Number of lesions in worst.png.")
    parser.add_argument("--metric", default="dice", choices=["dice", "dice_pp"], help="Metric that ranks lesions.")
    parser.add_argument("--sample", type=int, default=10, help="Lesions per dataset in missed.png and partial.png.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of those random samples.")
    parser.add_argument("--machine", required=True, help="Machine config, e.g. configs/machines/idun.yaml.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. paths.data_root=data/sample/ULS23.")
    args = parser.parse_args()

    work_dir = load_config([args.machine], args.overrides)["paths"]["work_dir"]
    run_dir = Path(work_dir) / "runs" / args.run
    # configs/eda.yaml supplies the display windows of the EDA gallery (D23).
    cfg = load_config([run_dir / "config.yaml", "configs/eda.yaml", args.machine], args.overrides)
    out = save_examples(cfg, run_dir / "eval" / args.eval, cfg["eda"]["gallery_windows"], args.worst, args.metric,
                        args.sample, args.seed)
    print(f"Figures in {out}")


if __name__ == "__main__":
    main()
