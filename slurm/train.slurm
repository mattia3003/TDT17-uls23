#!/bin/bash
# Training on one GPU. First argument: experiment config; anything after it goes to train.py.
# Submit from the repo root:
#   sbatch --time=00:30:00 slurm/train.slurm configs/experiments/timing.yaml
#   sbatch slurm/train.slurm configs/experiments/e1_dynunet3d.yaml
#   sbatch slurm/train.slurm configs/experiments/e1_dynunet3d.yaml --resume   # continue after a time limit
#SBATCH --job-name=uls23-train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/idun_env.sh

nvidia-smi
python scripts/train.py --config "$1" --machine configs/machines/idun.yaml "${@:2}"
