#!/bin/bash
# Evaluation of a trained run on one GPU. All arguments go to scripts/evaluate.py.
# Submit from the repo root:
#   sbatch slurm/evaluate.slurm --run e1_dynunet3d --splits val
#SBATCH --job-name=uls23-eval
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/idun_env.sh

nvidia-smi
python -u scripts/evaluate.py --machine configs/machines/idun.yaml "$@"
