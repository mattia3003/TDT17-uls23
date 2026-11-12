"""Evaluate a trained run on whole VOIs. Uses the GPU, so the student starts it (locally or with sbatch).

Uses the run's saved config.yaml, with the machine config and any overrides merged on top. Writes
<work_dir>/runs/<run>/eval/<splits>_<checkpoint>/ (see src/evaluation/evaluator.py). The test split
is evaluated only once the model is final (Phase 4), and never to choose anything (D15, D41).

Examples:
    # Gate 2: validation predictions to inspect
    python scripts/evaluate.py --run e1_dynunet3d --splits val --machine configs/machines/idun.yaml
    # Phase 4: the test split, and E2 on all of Pancreas
    python scripts/evaluate.py --run e1_dynunet3d --splits test --machine configs/machines/idun.yaml
    python scripts/evaluate.py --run e1_dynunet3d --splits heldout --machine configs/machines/idun.yaml
    # Real-file check on the PC sample, with the E0 weights (no test cases)
    python scripts/evaluate.py --run e0_overfit_v2 --splits train val heldout --cases 000010_02_02_100_lesion_01 bone_00002_lesion_01 diag_pancreas_0004_lesion_01 diag_pancreas_2066_lesion_01 --machine configs/machines/home_pc.yaml paths.data_root=data/sample/ULS23 paths.work_dir=work machine.num_workers=0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.evaluator import evaluate_run
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run name: a folder in <work_dir>/runs/.")
    parser.add_argument("--splits", nargs="+", required=True, choices=["train", "val", "test", "heldout"],
                        help="Splits from the split file to predict.")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--cases", nargs="+", help="Only these case names, e.g. the real-file sample.")
    parser.add_argument("--name", help="Output folder name; default <splits>_<checkpoint>.")
    parser.add_argument("--machine", required=True, help="Machine config, e.g. configs/machines/idun.yaml.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. paths.data_root=data/sample/ULS23.")
    args = parser.parse_args()

    work_dir = load_config([args.machine], args.overrides)["paths"]["work_dir"]
    run_dir = Path(work_dir) / "runs" / args.run
    cfg = load_config([run_dir / "config.yaml", args.machine], args.overrides)
    evaluate_run(cfg, run_dir, args.splits, args.checkpoint, args.cases, args.name)


if __name__ == "__main__":
    main()
