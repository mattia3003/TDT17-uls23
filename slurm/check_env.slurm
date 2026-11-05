#!/bin/bash
# Phase 0 GPU check: loads one ULS23 VOI with MONAI and runs one DynUNet step.
# Submit from the repo root:  sbatch slurm/check_env.slurm  (account from $SBATCH_ACCOUNT, or add --account=share-ie-idi)
#SBATCH --job-name=uls23-check-env
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/idun_env.sh

nvidia-smi
python scripts/check_env.py --machine configs/machines/idun.yaml
