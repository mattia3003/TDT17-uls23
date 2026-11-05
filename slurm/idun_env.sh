#!/bin/bash
# Shared environment for all IDUN jobs. Sourced by every slurm/*.slurm script.
# Set these two once (or export them in ~/.bashrc). List available modules with: module avail Python
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/uls23}"

module purge
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} python=$(which python)"
