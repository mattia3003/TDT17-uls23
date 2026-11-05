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

## Gate 1: Data decisions (confirmed by the student, 2026-11-05)

The evidence comes from EDA job 25936150: `data/manifest.csv` and `results/eda/`.

**D14. Train on ULS23_DeepLesion3D + ULS23_Radboudumc_Bone (1440 lesions). Hold out ULS23_Radboudumc_Pancreas (120 lesions) entirely, for E2.**
- *Why:* only these 3 datasets are on IDUN (section 3 of the plan). Training on the two largest keeps the most data: 743 DeepLesion3D lesions from many body regions plus 697 bone lesions. Pancreas is a separate organ and lesion type, so it makes a meaningful generalization test with no extra training. Its tumours are large (median Ø 27 mm, against 11 and 15 mm) and have very low contrast to their surroundings (median 4 HU, against 67 and 135).
- *Alternative considered:* holding out Bone. That is the more extreme shift, but it would cut training data to 863 lesions, and E2 would then mostly test sclerotic bone appearance.
- *Limitation:* DeepLesion3D includes abdominal lesions, so E2 tests a new dataset, annotation protocol and lesion type, not entirely unseen anatomy.

**D15. Splits:**
- *Test:* the official ULS23 test cases of the two training datasets (74 DeepLesion3D + 85 Bone).
- *Validation:* about 15 % of the official training cases, grouped by patient, with its own fixed seed.
- *E2 test:* all 120 Pancreas lesions, never used for any selection.
- *Why:* the official split (`baseline_model/train_test_split.json`, ULS23 repo commit `f45a172`, SHA-256 `4d781d5c…`) covers all 1560 labelled lesions. Not one of our 819 patient groups is on both sides. Reusing it makes our test results comparable with the authors' own evaluation on these datasets, and it avoids a test split of our own making. The file is downloaded and checked against its checksum rather than copied into git, because the repo shows no license. The derived split file we commit lists only case IDs.
- *Limitation:* for the Radboud datasets we can't tell whether an ID is a patient or a single study (D10). The official split agrees with our grouping, though.

**D16. Keep the native voxel grid (no resampling).**
- *Why:* the ULS23 nnU-Net baseline also used no resampling. It is the simplest choice. Predictions are made directly in the original VOI space, so metrics need no resampling back and lose nothing to interpolation.
- *Limitation:* slice spacing ranges from 0.45 to 5 mm (DeepLesion3D: 343 VOIs at 1 mm, 366 at 5 mm). A patch of 64 slices therefore covers 32–320 mm in z, and the network sees lesions at varying depth scales. A breakdown by slice spacing in E3 would show whether this matters (suggested plan change).

**D17. Correct the Bone intensity offset: VOIs with no negative values are shifted by +1024, so subtract 1024.**
- *Evidence:* 555 of the 744 Bone VOIs (115 of 151 patients) have a minimum of exactly 0, with air near +23 instead of −1000. The other 189 Bone VOIs, and all DeepLesion3D and Pancreas VOIs, have minimums below −500. No VOI falls in between, and every patient is entirely on one scale. After subtracting 1024, the shifted group matches the HU group:

  | Statistic | HU group | Shifted group − 1024 |
  |---|---|---|
  | 0.5th percentile (air) | −1010 | −1003 |
  | median | −47 | −45 |
  | 99.5th percentile | 866 | 819 |

  This also explains the README's Bone lesion median of "1276 HU".
- *Why a data-driven rule:* "No negative value in a CT VOI" can't happen in HU, because every 256×256 region contains air or fat. So the rule is safe and isn't tied to one dataset name. It never fires on the other datasets. Without the correction, the same tissue would get two different intensities in training.

**D18. Clip intensities to [−1000, 1500] HU, then scale linearly to [0, 1].**
- *Why:* the 0.5th and 99.5th percentiles of lesion intensities in the training split are −905 and 1447 HU (after D17). The window is rounded outward so that air and lung context are kept, and so that dense sclerotic bone lesions are not cut off. A linear scaling loses no contrast, which matters for the low-contrast pancreas lesions. Clipping removes padding and metal extremes (minimums of −2048, −3024, −8477 and −10405).
- *Consequence:* padding becomes −1000, the same as air. That is acceptable because padding sits outside the body, but it means DeepLesion3D's empty slabs look like air. They make up a median of 70 % of each DeepLesion3D VOI, because DeepLesion only provides slices near each lesion.

**D19. Train on 128×128×64-voxel patches, and predict on the whole 256×256×128 VOI with a sliding window.**
- *Why:* 98 % of lesion bounding boxes fit in the patch (DeepLesion3D 99.8 %, Bone about 96.5 %, Pancreas 96.7 %). Sliding-window prediction still covers the few larger lesions, such as the 8 Bone lesions that touch the VOI edge (30–55 mm). The patch fits on both GPUs (2.3 GB at batch size 2 in the Phase 0 check) and is the plan's starting point.
- *Centre prior:* every lesion contains the VOI centre, by construction. Random patch positions in training weaken the model's use of absolute position, but the prior stays and is reported as a limitation.

**D20. No lesions are excluded for label quality. The 58 images without a label are left out.**
- *Why:* nothing is clearly broken. There are no read errors, no geometry mismatches, no labels with more than two values, and no masks in several pieces. The indicators (D13) show small holes in some slice for 43 % of Pancreas masks (6 % overall), and 1.7 % of masks lie on a single slice. These are reported in E3 and in the limitations.

**D21. Figures kept for the video:** `lesion_size.png`, `spacing.png`, a before/after figure of the Bone intensity offset (D17), the gallery redrawn with a fixed display window and corrected intensities, and optionally `crop_fit.png`.

**D22. Plan changes accepted after Gate 1.**
- E3 gets three extra breakdowns: slice spacing, Bone intensity group and DeepLesion3D category. They test D16 and D17 and need no extra training.
- The real-file check (D12) is now part of the working agreement.
- Deterministic preprocessing is cached with MONAI `PersistentDataset`, if E0 shows that loading is a bottleneck.
- Per-dataset baseline numbers from the ULS23 paper are used as a reference point. The paper (Table 2) reports Bone (Dice 0.674, same split) and a pooled Pancreas number, but nothing for DeepLesion3D. See section 3 of the plan.
- The video gets a "finding → decision" map.
- *Why:* each change either tests a Gate 1 decision or makes a finding easier to present, at almost no extra compute.

**D23. DeepLesion3D categories are matched by key slice, and figures use corrected intensities with fixed display windows.**
- *Categories:* the files in `categories/<type>/` are key-slice PNGs named `<patient>_<study>_<series>_<slice>.png`, one per VOI (750 in total), without our `_lesion_<nn>` suffix. A key pattern in the data config strips the suffix before matching. Categories come from a folder listing, so `--reuse-manifest` refreshes them without reading any volume.
- *Intensity settings:* D17 and D18 live in `configs/data/uls23.yaml` under `data.intensity`, and one function (`src/data/intensity.py`) applies the rule. The EDA uses it now, and the Phase 2 transforms will use the same code.
- *Gallery windows:* the gallery uses fixed windows instead of per-slice percentiles, so brightness means the same thing in every panel. Soft tissue is −160 to 240 HU; Bone is −450 to 1250 HU, so that sclerotic lesions aren't saturated.
