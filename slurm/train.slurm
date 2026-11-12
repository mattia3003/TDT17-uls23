#!/bin/bash
# Training on one GPU. First argument: experiment config; anything after it goes to train.py.
# Submit from the repo root:
#   sbatch --time=00:30:00 slurm/train.slurm configs/experiments/timing.yaml
#   sbatch slurm/train.slurm configs/experiments/e1_dynunet3d.yaml
#   sbatch slurm/train.slurm configs/experiments/e1_dynunet3d.yaml --resume   # continue after a time limit
# Memory: E1 peaked at 32.0 GB MaxRSS (summed over 8 data workers), so 64G leaves a margin.
#SBATCH --job-name=uls23-train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/idun_env.sh

nvidia-smi
# -u: unbuffered output, so the log file shows each epoch as it finishes (tail -f works)
python -u scripts/train.py --config "$1" --machine configs/machines/idun.yaml "${@:2}"
