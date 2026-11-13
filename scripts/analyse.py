"""Breakdown and paired-comparison tables from per-lesion CSVs written by scripts/evaluate.py (no GPU, no images).

Each input is a per_lesion.csv or the evaluation folder holding it. The tables go to --out (small CSVs,
committed under results/) and are printed. See src/evaluation/analysis.py for the groups and statistics.

    breakdown  one model: Dice, missed fraction, precision and volume ratio per dataset and factor
               (size, contrast, slice spacing, category, intensity group)  ->  breakdown.csv
    compare    two models on the same lesions: change in centre-component Dice per group, with lesions
               better/worse, rescued/lost, a bootstrap interval and a Wilcoxon test  ->  paired.csv,
               paired_lesions.csv (every lesion, sorted by change)

Examples (with the CSVs copied to work/ on the PC):
    python scripts/analyse.py compare work/e1_val_per_lesion.csv work/e1b_val_per_lesion.csv --names E1 E1b --out results/e1_vs_e1b_val
    python scripts/analyse.py breakdown work/e1b_test_per_lesion.csv --out results/e3_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.analysis import breakdown, paired_comparison


def read_lesions(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p / "per_lesion.csv" if p.is_dir() else p)


def show(table: pd.DataFrame) -> None:
    for factor, part in table.groupby("factor", sort=False):
        shown = part.drop(columns="factor")
        if "wilcoxon_p" in shown:
            shown["wilcoxon_p"] = shown["wilcoxon_p"].map("{:.2g}".format)
        print(f"\n--- {factor} ---\n{shown.to_string(index=False, float_format='%.3f')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("breakdown", help="Tables for one model.")
    one.add_argument("lesions", help="per_lesion.csv or its evaluation folder.")
    two = commands.add_parser("compare", help="Paired comparison of two models on the same lesions.")
    two.add_argument("first", help="per_lesion.csv (or folder) of the reference model.")
    two.add_argument("second", help="per_lesion.csv (or folder) of the model compared with it.")
    two.add_argument("--names", nargs=2, default=["a", "b"], help="Short model names used in the column names.")
    for sub in (one, two):
        sub.add_argument("--out", required=True, help="Output folder, e.g. results/e3_test.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "breakdown":
        table = breakdown(read_lesions(args.lesions))
        table.to_csv(out / "breakdown.csv", index=False)
    else:
        table, lesions = paired_comparison(read_lesions(args.first), read_lesions(args.second), tuple(args.names))
        table.to_csv(out / "paired.csv", index=False)
        lesions.to_csv(out / "paired_lesions.csv", index=False)
    show(table)
    print(f"\nTables in {out}")


if __name__ == "__main__":
    main()
