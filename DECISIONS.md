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
- *How the model is tested (added in Phase 2):*
  - *No access to the challenge test set:* it is kept hidden by design (725 lesions from 268 patients, Radboudumc and JBZ). Algorithms are submitted as containers and run on grand-challenge.org: *"the test set to remain fully hidden, making it a reliable long-term benchmark"* (paper, §3.1.3 and §5). We therefore test on held-out cases from the public training data, as the authors did during development: *"All three models were first evaluated on held-out training data consisting of 10% of each fully annotated training dataset, split on a patient level"* (§4.1). Their Table 2 reports per-lesion-type results on this held-out data.
  - *Which cases:* the `test` cases of the official split file are about 10 % of each dataset (74 of 743 DeepLesion3D, 85 of 697 Bone, 11 of 120 Pancreas) and patient-grouped. That is consistent with the paper's "random, patient-level, 10% split" (Appendix A.3), although the paper doesn't name the file.
- *Why a public, already known split is acceptable:*
  - A split is valid if the model never learns from or is tuned on its test cases, not if the case names are secret. Ours are used once, at the end (working agreement). All selection uses `val`, and Pancreas (E2) is never used for selection.
  - We fixed the split before training anything, and we didn't choose it ourselves. So it can't have been picked, even unknowingly, to make our model look good.
  - It is the same held-out population the authors report on. That makes the Bone comparison with Table 2 meaningful, which a split of our own wouldn't allow.
  - It is patient-grouped, which we checked independently (D10): none of our 819 groups is on both sides.
- *Caveats:*
  - Our test result is a development estimate on 159 lesions from the public datasets, not a challenge score. It shouldn't be compared directly with the challenge leaderboard or the 0.703 baseline (section 3 of the plan).
  - The authors retrained on *all* training data before the challenge submission (§4.1), so the released baseline weights have seen our test cases. We train from scratch and never use those weights. They must not be used as a comparison model on this split.

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
- *Category counts:* 200 Abdominal, 150 Other, 100 each of Bone, Lung and Mediastinal, and 50 each of Kidney and Liver. The round numbers suggest the ULS23 authors sampled a fixed number of VOIs per lesion type.
- *Gallery windows:* the gallery uses two fixed windows instead of per-slice percentiles, so brightness means the same thing in every panel: soft tissue (−160 to 240 HU) and bone (−450 to 1250 HU). The window is chosen per lesion, not per dataset: a lesion gets the bone window when the 95th percentile of its intensities is above 500 HU, i.e. when part of it is bone-dense. That covers 78 % of DeepLesion3D's Bone category and 65 % of Radboud Bone, and almost none of the other lesions. A per-dataset rule showed DeepLesion3D bone lesions on 1 mm slices as black-and-white noise.
- *Observation for E3:* DeepLesion3D lesions on thin slices (≤ 1.5 mm) have about twice the intensity spread of those on 5 mm slices (median p95−p05 of 211 vs 106 HU). Thin slices are noisier, which supports the slice-spacing breakdown.

## Phase 2: Pipeline

**D24. Seeding goes through one function, `src/utils/seed.py`.** MONAI's `set_determinism` seeds Python, NumPy and PyTorch (CPU and CUDA) and sets cuDNN to deterministic. The function returns a seeded `torch.Generator` for the `DataLoader`. That fixes the shuffling order and the base seed that MONAI's default `worker_init_fn` uses to seed each worker's random transforms. The training seed (`seed` in `configs/base.yaml`) is separate from the split seed (`splits.seed`).
- *Remaining non-determinism:* a few CUDA kernels, such as the backward pass of transposed 3-D convolutions and atomic adds, are not bit-exact between runs. Two runs with the same seed can therefore differ slightly. `deterministic: false` swaps reproducibility for cuDNN auto-tuning (faster).

**D25. The split file is built by `scripts/make_splits.py` and committed as `splits/uls23_splits.csv`.**
- *Inputs:* the committed manifest, plus the official split, which is downloaded and checked against its SHA-256 on every run.
- *Validation:* for each training dataset, whole patients are taken in a seeded random order, as long as the total stays within 15 % of that dataset's official training cases. Patients are sorted before shuffling, so the result depends only on the seed (2311). Doing this per dataset keeps both datasets at about 15 % (DeepLesion3D 100 of 669, Bone 92 of 612).
- *Checks:* the script fails if a training-dataset case is missing from the official split, or if any patient ends up in two splits.
- *Result:*

  | Dataset | train | val | test | heldout |
  |---|---|---|---|---|
  | DeepLesion3D | 569 (420 patients) | 100 (80) | 74 (49) | – |
  | Bone | 520 (119) | 92 (18) | 85 (14) | – |
  | Pancreas | – | – | – | 120 (119) |

- *Limitation:* the 85 Bone test lesions come from only 14 patients. Lesions from the same patient are correlated, so the Bone test mean is less certain than N = 85 suggests. Report it with this caveat, and possibly also per patient.

**D26. Preprocessing is one deterministic chain shared by training and evaluation (`src/data/transforms.py`).**
- *Chain:* load → channel first (this also drops the stored size-1 axis, D11) → offset correction (D17, same `intensity_shift` rule as the EDA) → clip to [−1000, 1500] and scale to [0, 1] (D18).
- *Compact storage:* the chain ends with the image as float16 and the label as uint8, then casts back to float32 after patch sampling. This keeps an optional `PersistentDataset` cache at about 24 MB per VOI instead of 64 MB. float16 moves intensities by at most about 1 HU. The same chain runs with caching on or off, so results don't depend on it.
- *Cache folder:* the cache folder is named by a hash of the intensity settings and a code version number, so changing the window can never reuse a stale cache. MONAI's own transform hash usually falls back to transform names, which would miss such a change.
- *No reorientation:* all 1618 VOIs are stored as RAS, 256×256×128, so no orientation transform is needed.
- *Real-file check (D12):* on the sample, the shifted Bone VOI `bone_00149_lesion_08` comes out with a lesion median of 239 HU, which is the manifest's raw 1263 − 1024. The VOIs stored in HU are unchanged. Loading and preprocessing one VOI takes about 0.4 s on the home PC.

**D27. Model: MONAI `DynUNet`, built by a small factory (`src/models/factory.py`, `configs/model/dynunet3d.yaml`).**
- *Architecture:* 6 levels with 3×3×3 kernels, nnU-Net default widths (32 to 320 filters), instance norm and 2 output channels (softmax). That is 30.8 M parameters. For a 128×128×64 patch the encoder halves x/y five times and z four times (last stride [2, 2, 1]), giving a 4×4×4 bottleneck.
- *Factory:* the model is chosen by `model.name`, and the other keys are passed on as arguments. `spatial_dims: 2` gives a 2-D DynUNet, which is enough for the optional E4. Adding a model means adding one builder function.
- *Not used:* deep supervision is off, to keep the loss simple.
- *Kernel shape:* isotropic 3×3×3 kernels are used even though slice spacing varies (D16). nnU-Net would use in-plane-only kernels for 5 mm data, but our data mixes 0.45–5 mm, so one fixed choice can't suit all of it. The slice-spacing breakdown in E3 shows whether this matters.

## Phase 2 choices (confirmed by the student, 2026-11-06)

**D28. Training patches are taken at uniformly random positions, 2 per loaded VOI (`RandSpatialCropSamplesd`).**
- *Finding:* a 128×128×64 patch inside a 256×256×128 VOI covers the VOI centre for all but about 1.5 % of positions (patch starts at 0–128 in x/y and 0–64 in z). Every lesion contains the centre (Phase 1), so almost every random patch contains the lesion. The sample check confirmed this. A "lesion-centred vs random patches" ratio therefore makes little difference here.
- *Why random:* the lesion lands at a different place in each patch, which weakens the centre prior compared with lesion-centred patches. Taking 2 patches per loaded VOI halves the loading cost per patch, since decompressing a VOI is the slow step.
- *Side effect:* training almost never shows the model a lesion-free patch. Sliding-window prediction, however, also covers windows without the centre, where the model may produce false positives. D30 deals with this.

**D29. Light augmentation:**
- flips along all three axes (p = 0.5 each)
- in-plane 90° rotations (p = 0.5)
- intensity scaling of ±10 % (p = 0.3)
- intensity shift of ±0.02, i.e. ±50 HU (p = 0.3)

*Why:* these are cheap and need no interpolation, so the native grid is kept (D16). x and y spacing are equal, so 90° rotations don't distort the voxels. Arbitrary-angle rotation and scaling, as in nnU-Net, would resample the patch and are left out to keep the setup simple. The augmentations are listed in `configs/base.yaml` and can be changed there.

**D30. Centre-component post-processing (plan change): after prediction, keep only the connected piece of the mask that contains the VOI centre (or, if none does, the one nearest to it). Dice is reported both with and without it.**
- *Why:* the ULS23 task guarantees one target lesion at the VOI centre, also at test time, so using this is part of the task and not leakage. It removes false positives in windows far from the lesion (D28) and in nearby unannotated lesions (their masks were removed, §3 of the plan), at no training cost.
- *For the video:* the difference between the two Dice values shows how much the result depends on the centre prior.

**D31. The home PC gets no copy of the training data for now.** This is decided after E0, from the PC vs IDUN timing comparison. `configs/machines/home_pc.yaml` keeps its placeholder paths until then.
