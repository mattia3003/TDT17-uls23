# TDT17 mini-project: 3D lesion segmentation on ULS23 (MONAI)

3D lesion segmentation on a subset of the ULS23 CT dataset with MONAI `DynUNet`. See `PROJECT_PLAN.md` for scope, `DECISIONS.md` for design choices and `REFERENCES.md` for sources.

## Layout

```
configs/     YAML configs; configs/machines/ holds per-machine paths
src/         data, models, training, evaluation, utils (installed as package `src`)
scripts/     thin CLI entry points
slurm/       IDUN job scripts (wrappers around the same scripts/ commands)
splits/      committed split files
tests/       unit tests on synthetic data
results/     small CSVs and final figures
```

Data, caches, checkpoints and predictions live outside git, under `paths.work_dir` from the machine config.

## Data

On IDUN, `paths.data_root` holds `data/<dataset>/images/` and `annotations/<dataset>/labels/`, paired by filename (see `configs/data/uls23.yaml`). Only the 3 novel ULS23 datasets are present. For the home PC, copy the same two subtrees for the selected datasets and keep the folder structure.

## Setup: home PC (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # ~3 GB download (CUDA PyTorch)
pip install -e .
pytest
python scripts/check_env.py --machine configs/machines/home_pc.yaml --no-data
```

## Setup: IDUN

IDUN is a Slurm cluster. You log in to a *login node* to edit files, install packages and submit jobs. GPU work runs on *compute nodes* through `sbatch`. Don't run heavy work on the login nodes.

1. **Log in** (NTNU network or VPN): `ssh <ntnu-username>@idun-login1.hpc.ntnu.no`
2. **Find your Slurm account and partitions:**
   ```bash
   sacctmgr show assoc where user=$USER format=account%30,partition%20
   sinfo -s
   ```
   This project uses `share-ie-idi`. To avoid typing it on every submit, add `export SBATCH_ACCOUNT=share-ie-idi` to `~/.bashrc` (Slurm reads it automatically). GPU jobs use the `GPUQ` partition.
3. **Storage:** code and the venv go in your home directory (`/cluster/home/$USER`). Large outputs go in `/cluster/work/$USER/tdt17-uls23` (`paths.work_dir` in `configs/machines/idun.yaml`). Neither home nor work is backed up, so GitHub is the backup for code. Check quota with `lfs quota -u $USER /cluster`. The dataset under `/cluster/projects/vc/courses/...` is read-only.
4. **Get the code:** `git clone <your-github-repo-url> ~/tdt17-uls23` once the GitHub repo exists. Otherwise copy the folder with `scp -r`.
5. **Python environment** (on the login node):
   ```bash
   module avail Python                  # pick a 3.11 version, then set PYTHON_MODULE in slurm/idun_env.sh
   module load Python/3.11.3-GCCcore-12.3.0
   python -m venv ~/venvs/uls23
   source ~/venvs/uls23/bin/activate
   cd ~/tdt17-uls23
   pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
   ```
6. **GPU check job** (from the repo root):
   ```bash
   sbatch slurm/check_env.slurm         # or: sbatch --account=share-ie-idi slurm/check_env.slurm
   squeue -u $USER                      # wait until it finishes
   cat logs/uls23-check-env-<jobid>.out
   ```
