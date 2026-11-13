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

On IDUN, `paths.data_root` holds the course's train/test split (D41): `<dataset>/<train|test>/<images|labels>/<case>.nii.gz`. Only the 3 novel ULS23 datasets are present. Training and evaluation take each case's file paths and split from `splits/uls23_splits.csv`. For the home PC, copy the same folders for the selected datasets and keep the folder structure.

The manifest and EDA (Phase 1) were built from the course's earlier layout, `data/<dataset>/images/` and `annotations/<dataset>/labels/` (see `configs/data/uls23.yaml`). The course removed that layout on 12 Nov 2026, so the EDA can't be rerun on IDUN as it is. Its results are committed, and the files were byte-identical (D41). The real-file sample in `data/sample/ULS23` holds both layouts.

`splits/course_files.txt` lists the course folders. To rebuild the split file after the course data changes, see `scripts/make_splits.py`.

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
4. **Get the code:** `git clone https://github.com/mattia3003/TDT17-uls23.git ~/tdt17-uls23`.
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

## Working routine: PC and IDUN

The PC repo is the master copy. All edits and commits happen there. The IDUN clone only runs code, and its outputs are copied back to the PC and committed from there.

**Before every `git pull` on IDUN**, discard IDUN's copies of generated outputs. Otherwise git refuses to pull files that were committed from the PC:

```bash
cd ~/tdt17-uls23
git checkout -- data results   # reset tracked outputs to the committed version
git clean -n data results      # dry run: list untracked outputs that would be deleted
git clean -f data results      # delete them (files covered by .gitignore are kept)
git pull
```

**Before any long IDUN job,** run the same command locally on the real-file sample, which is seconds of work. The sample is 2 VOIs per dataset in `data/sample/ULS23`, gitignored, with the same folder layout as the course data. For example, for the EDA:

```powershell
python scripts/eda.py --config configs/data/uls23.yaml configs/eda.yaml --machine configs/machines/home_pc.yaml paths.data_root=data/sample/ULS23 eda.manifest_path=data/sample/manifest.csv eda.output_dir=data/sample/eda
```

**Copying results to the PC** (PowerShell, from the repo root). Give the full destination file name, because a trailing `\` before a closing quote breaks the path:

```powershell
scp mattiald@idun-login1.hpc.ntnu.no:~/tdt17-uls23/data/manifest.csv data/manifest.csv
scp -r mattiald@idun-login1.hpc.ntnu.no:~/tdt17-uls23/results/eda results
```

**Training** uses one experiment config (`configs/experiments/`) and one machine config. A run writes to `paths.work_dir/runs/<run.name>/`, which holds `log.csv`, the checkpoints and `compute.json`. A run that hits a time limit is continued with `--resume`, which works on either machine. Put `--resume` before any `key=value` overrides.

```powershell
# Home PC: E0 on the real-file sample, outputs in work/ (gitignored)
python scripts/train.py --config configs/experiments/e0_overfit.yaml --machine configs/machines/home_pc.yaml paths.data_root=data/sample/ULS23 paths.work_dir=work machine.num_workers=0
```

```bash
# IDUN
sbatch --time=00:30:00 slurm/train.slurm configs/experiments/timing.yaml
sbatch slurm/train.slurm configs/experiments/e1_dynunet3d.yaml
sbatch slurm/train.slurm configs/experiments/e1_dynunet3d.yaml --resume
```

Before training starts, `train.py` checks that every training and validation file exists. If any are missing, it stops within seconds and names one of the missing files.

**Evaluation** predicts whole VOIs with a run's `best.pt` (or `--checkpoint last`). It uses the run's saved config, with the machine config merged on top. The outputs go to `<work_dir>/runs/<run>/eval/<splits>_<checkpoint>/`:
- `per_lesion.csv`: metrics plus the metadata for the E3 breakdowns;
- `summary.csv`: results per group;
- `predictions/`: the predicted masks;
- `compute.json`: inference time and energy.

An existing result is never overwritten, so delete its folder to evaluate again (D42). Evaluate the test split only once the model is final.

```powershell
# Home PC: real-file check on the sample with the E0 weights (no test cases)
python scripts/evaluate.py --run e0_overfit_v2 --splits train val heldout --cases 000010_02_02_100_lesion_01 bone_00002_lesion_01 diag_pancreas_0004_lesion_01 diag_pancreas_2066_lesion_01 --machine configs/machines/home_pc.yaml paths.data_root=data/sample/ULS23 paths.work_dir=work machine.num_workers=0
```

```bash
# IDUN (the Slurm script adds --machine configs/machines/idun.yaml)
sbatch slurm/evaluate.slurm --run e1_dynunet3d --splits val       # Gate 2: inspect validation predictions
sbatch slurm/evaluate.slurm --run e1_dynunet3d --splits test      # Phase 4, once
sbatch slurm/evaluate.slurm --run e1_dynunet3d --splits heldout   # E2: all of Pancreas
```

**Figures** of the worst, median and best lesion per dataset, and of the 12 worst lesions overall, are drawn from an evaluation folder. This needs no GPU and is fine on the login node:

```bash
python scripts/qualitative.py --run e1_dynunet3d --eval val_best --machine configs/machines/idun.yaml
```

**Analysis tables** come from the per-lesion CSVs alone, so they run anywhere in seconds (D45). `breakdown` describes one model per dataset and factor (size, contrast, slice spacing, category, intensity group). `compare` pairs two models on the same lesions. The small CSVs go to `results/` and are committed. With the evaluation CSVs copied to `work/` on the PC:

```powershell
python scripts/analyse.py compare work/e1_val_per_lesion.csv work/e1b_val_per_lesion.csv --names E1 E1b --out results/e1_vs_e1b_val
python scripts/analyse.py compare work/e1_train_per_lesion.csv work/e1b_train_per_lesion.csv --names E1 E1b --out results/e1_vs_e1b_train
python scripts/analyse.py breakdown work/e1b_test_per_lesion.csv --out results/e3_test
python scripts/analyse.py breakdown work/e1b_heldout_per_lesion.csv --out results/e2_pancreas
```

`--splits` and `--cases` read every word after them until the next `--option`. Any `key=value` overrides must therefore come after another option, such as `--machine` or `--checkpoint best`. Otherwise they're read as split names and rejected.

**Light work on the login node is fine:** things that take seconds and read only a few files, such as `scripts/eda.py --reuse-manifest`. Use the same environment as the jobs: `source slurm/idun_env.sh`. Anything that reads the whole dataset, or uses a GPU, goes through `sbatch`.

**Logging:** after every significant job, run `sacct -j <jobid> --format=JobID,Elapsed,State,AllocCPUS,MaxRSS` and add a row to `COMPUTE_LOG.md`, including for failed jobs.
