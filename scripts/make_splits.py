"""Build the committed split file from the manifest and the official ULS23 split (D15).

Light: reads data/manifest.csv and downloads one small JSON file (checked against its SHA-256).

Example:
    python scripts/make_splits.py --config configs/splits.yaml --machine configs/machines/home_pc.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.splits import load_official_split, make_splits, split_summary
from src.utils.config import add_config_arguments, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    cfg = config_from_args(parser.parse_args())["splits"]

    official = load_official_split(cfg["official"]["source"], cfg["official"]["sha256"])
    splits = make_splits(pd.read_csv(cfg["manifest_path"]), official, cfg["train_datasets"],
                         cfg["heldout_datasets"], cfg["val_fraction"], cfg["seed"])
    path = Path(cfg["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    print(split_summary(splits).to_string(index=False))
    print(f"\nWrote {path} ({len(splits)} cases); no patient is in more than one split.")


if __name__ == "__main__":
    main()
