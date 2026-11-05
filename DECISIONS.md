# Decisions

Short log of non-obvious choices, so each one can be explained in the video. Newest last.

## Phase 0: Setup

**D1. Python 3.11 + venv.** Already installed on the home PC and a standard module on HPC clusters. All pinned packages ship wheels for it. Python 3.14 (also installed) is too new for some scientific wheels.

**D2. PyTorch 2.5.1 (CUDA 12.4 wheel) + MONAI 1.4.0, all versions pinned in `requirements.txt`.** A well-tested, mutually compatible pair. MONAI ≥ 1.2.0 is required by the ULS23 repo because of loading problems in older versions. The CUDA 12.4 wheel bundles its own CUDA runtime, so the same `requirements.txt` works on Windows and on IDUN, given a recent NVIDIA driver. *Risk:* GPUs newer than Hopper (e.g. Blackwell, sm_100) need a newer PyTorch build. `scripts/check_env.py` warns if this happens. NumPy is held at 1.26 to avoid NumPy 2 incompatibilities in the stack.

**D3. Config system: plain YAML files merged in order, plus dotted `key=value` overrides.** This is a small custom loader (`src/utils/config.py`), with no Hydra/OmegaConf. It has few dependencies, is easy to explain, and is enough for this project. Environment variables such as `${USER}` are expanded at load time.

**D4. One machine config per machine (`configs/machines/*.yaml`), merged last.** Experiment configs never contain paths, so the same experiment runs unchanged on IDUN and the home PC.

**D5. Code installed as an editable package (`pip install -e .`).** This lets `scripts/` and `tests/` import `src.*` without `sys.path` hacks.

**D6. Slurm scripts are thin wrappers.** Each one sources `slurm/idun_env.sh` (module + venv) and runs the same Python command used locally. The Slurm account (`share-ie-idi`) is not hardcoded. It comes from `SBATCH_ACCOUNT` in `~/.bashrc` or from `sbatch --account=...`.

**D7. The dataset layout is set in config (`configs/data/uls23.yaml`), with no assumptions in the code.** On IDUN, images and labels are in separate trees, `data/<ds>/images` and `annotations/<ds>/labels`, and pair by filename. Adding a dataset with the same layout only means adding its name to the list.

## Phase 1: Data analysis

**D8. The EDA job runs on CPUs only (`CPUQ`, 8 cores) and reads each VOI once with nibabel.** Statistics need no GPU, so using one would waste a scarce resource and energy. nibabel gives the raw header fields (spacing, data type, scaling, orientation). These are needed to explain intensity oddities such as the Bone minimum of −8477. MONAI's `LoadImage` uses nibabel for NIfTI files anyway.

**D9. Padding is detected as "voxels equal to the VOI minimum".** ULS23 pads with (volume minimum − 1), so padded voxels are exactly at the VOI minimum. If under 1 % of voxels are at the minimum, the VOI is treated as unpadded. Intensity percentiles leave padded voxels out.

**D10. Grouping IDs come from filename regexes in the data config.** The ID is the DeepLesion patient for DeepLesion3D, and `bone_<id>` / `diag_pancreas_<id>` for Radboud. Each ID is prefixed with the dataset name, so equal numbers in different datasets never merge. *Limitation:* we can't tell whether a Radboud ID is a patient or a single study, so the same patient could appear under two IDs.
