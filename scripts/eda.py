"""Phase 1 data analysis: build the manifest, then write summary tables and figures.

Reads every configured VOI once (CPU only), so run it as a Slurm job, not on a login node.

Examples:
    python scripts/eda.py --config configs/data/uls23.yaml configs/eda.yaml --machine configs/machines/idun.yaml
    # Re-draw tables/figures from an existing manifest without re-reading the data:
    python scripts/eda.py --config configs/data/uls23.yaml configs/eda.yaml --machine configs/machines/idun.yaml --reuse-manifest
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data.eda import write_report
from src.data.manifest import build_manifest
from src.utils.config import add_config_arguments, config_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--reuse-manifest", action="store_true", help="Load the existing manifest instead of rebuilding.")
    args = parser.parse_args()
    cfg = config_from_args(args)

    data_root = Path(cfg["paths"]["data_root"])
    datasets = list(cfg["data"]["datasets"])
    manifest_path = Path(cfg["eda"]["manifest_path"])
    start = time.perf_counter()

    if args.reuse_manifest:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_manifest(data_root, cfg["data"], num_workers=cfg["machine"]["num_workers"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(f"Wrote {manifest_path} ({len(manifest)} rows)")

    summary = write_report(manifest, data_root, cfg["eda"], datasets)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.set_index("dataset").T.to_string())
    print(f"\nWrote tables and figures to {cfg['eda']['output_dir']} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
