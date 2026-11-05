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

**D11. Volumes are read as 3-D by dropping trailing size-1 axes, and labels must be on the image's voxel grid.** The first EDA run (job 25934982) failed in 1547 of 1618 cases with "structure and input must have equal rank": nibabel returned more than 3 dimensions, while MONAI's `EnsureChannelFirst` had quietly hidden the extra axis in the Phase 0 check. The number of dimensions on disk is now recorded (`image_ndim_disk`, `label_ndim_disk`). Labels whose shape or affine differs from the image are listed in `problem_cases.csv` and left out of the statistics instead of being silently skipped.

**D12. Before any long IDUN job: run the same command locally on a few real VOIs, and let the job fail fast.** Synthetic test files only check our own assumptions about the data, and that is how D11 slipped through. A sample of 2 VOIs per dataset (copied to the gitignored `data/sample/`) runs the real pipeline in seconds. The EDA job also stops if more than 5 % of its first 50 cases cannot be read (`eda.check_after`, `eda.max_error_fraction`).

**D13. Label quality is measured, never corrected.** A visual check of the gallery showed masks that looked partly off, such as a square hole in `diag_pancreas_2066_lesion_01`. Without radiological training we can't judge individual labels, and editing ground truth would make results incomparable with ULS23. So the manifest records three indicators:
- holes per axial slice and in 3-D
- the lesion-vs-surroundings contrast (median inside the lesion minus a 3-voxel shell around it, padding left out)
- flatness (smallest over largest physical extent, computed in the report)

They are summarised per dataset and listed per case in `label_quality.csv`. Label noise is reported as a limitation. Cases are only left out if they are clearly broken, and then they are documented.
