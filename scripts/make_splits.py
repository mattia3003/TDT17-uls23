"""Build the committed split file from the manifest and the course's train/test folders (D41).

Light: reads data/manifest.csv and the committed listing of the course folders. The listing is
made on IDUN, from the folder that holds the dataset folders (only file names are read):
    cd /cluster/projects/vc/courses/TDT17/2026/2_mic/ULS23/data
    find . -name "*.nii.gz" | sed "s|^\\./||" | sort > ~/course_files.txt

Example:
    python scripts/make_splits.py --config configs/splits.yaml --machine configs/machines/home_pc.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.splits import make_splits, read_course_listing, split_summary
from src.utils.config import add_config_arguments, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    cfg = config_from_args(parser.parse_args())["splits"]

    splits = make_splits(pd.read_csv(cfg["manifest_path"]), read_course_listing(cfg["course_listing"]),
                         cfg["train_datasets"], cfg["heldout_datasets"], cfg["val_fraction"], cfg["seed"])
    path = Path(cfg["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    print(split_summary(splits).to_string(index=False))
    print(f"\nWrote {path} ({len(splits)} cases); no patient is in both train and val.")


if __name__ == "__main__":
    main()
