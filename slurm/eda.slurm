#!/bin/bash
# Phase 1 data analysis (CPU only): manifest + tables + figures for all configured datasets.
# Submit from the repo root:  sbatch slurm/eda.slurm
#SBATCH --job-name=uls23-eda
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/idun_env.sh

python scripts/eda.py --config configs/data/uls23.yaml configs/eda.yaml --machine configs/machines/idun.yaml
