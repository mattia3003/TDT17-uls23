"""Summary figures for the report and video from the training logs and the results/ tables (no GPU, no images).

Writes to --out (default results/figures):
    training_curves.png     validation Dice per epoch, E1 and E1b (raw and centre component)
    missed_by_contrast.png  lesions missed per contrast bin, E1 and E1b, val and training split
    test_breakdown.png      final model on the test split, by contrast and by DeepLesion3D category
    e2_pancreas.png         Pancreas against DeepLesion3D lesions of similar contrast and size

The tables come from scripts/analyse.py; the logs are each run's log.csv, copied to work/ on the PC.

Example:
    python scripts/report_figures.py --logs work/e1_log.csv work/e1b_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import report_figures as rf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", nargs=2, required=True, help="log.csv of E1 and of E1b, in that order.")
    parser.add_argument("--results", default="results", help="Folder holding the analyse.py tables.")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rf.training_curves({"E1": pd.read_csv(args.logs[0]), "E1b": pd.read_csv(args.logs[1])},
                       out / "training_curves.png")
    paired = {split: pd.read_csv(results / f"e1_vs_e1b_{split}" / "paired.csv") for split in ["val", "train"]}
    rf.missed_by_contrast({"validation": paired["val"], "training split": paired["train"]}, ["E1", "E1b"],
                          out / "missed_by_contrast.png")
    rf.test_breakdown(pd.read_csv(results / "e3_test" / "breakdown.csv"), out / "test_breakdown.png")
    rf.heldout_comparison(pd.read_csv(results / "e2_pancreas" / "heldout_comparison.csv"), out / "e2_pancreas.png")
    print(f"Figures in {out}")


if __name__ == "__main__":
    main()
