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

**D15. Splits:** *(Train/test part superseded by D41 on 2026-11-12: the course now provides its own split. The reasoning below is kept as the record of the original choice.)*
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

**D25. The split file is built by `scripts/make_splits.py` and committed as `splits/uls23_splits.csv`.** *(Inputs and result superseded by D41. The validation method is unchanged.)*
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

## Phase 2: Training code

**D32. Experiment configs name their shared defaults with `include:` (extends D3).**
- *How it works:* `configs/experiments/e1_dynunet3d.yaml` includes the data, split, base and model configs. Includes are merged first, and the file's own values win. E0 and the timing run include E1 and change only what differs.
- *Why:* one file then describes one experiment, so a command or Slurm job needs only `--config configs/experiments/<name>.yaml`, and the experiment's differences from E1 are visible in a few lines. The loader stays small, with plain YAML and no interpolation.

**D33. Each run gets its own folder under `paths.work_dir/runs/<run.name>/`, and checkpoints resume exactly, on either machine.**
- *What the folder holds:* the resolved config, `run_info.json` (seed, plus git commit, dirty flag, versions, GPU, host and Slurm job ID for every session), `log.csv` with one row per epoch, per-lesion validation CSVs, `last.pt` (saved every epoch, written via a temporary file) and `best.pt`.
- *Overwrite guard:* a new run refuses to start in a folder that already has a checkpoint.
- *Resume guard:* `--resume` refuses if the config differs from the saved one, except `machine` and `paths`, so a run can move between IDUN and the PC.
- *Exact resume:* each epoch's shuffling and augmentation are seeded with `seed + epoch`, and the data workers are restarted every epoch so the new seed takes effect. A resumed run therefore sees the same data as an uninterrupted one. `tests/test_training.py` checks that a run stopped after epoch 1 and resumed ends with bit-identical weights (on the CPU; GPU runs can still differ slightly, D24).
- *Fail-fast:* a non-finite loss stops the run at once, and 2 validation VOIs are predicted before training starts, so a bug in the validation code shows up in the first minutes, not after hours.

**D34. Compute is logged inside the training code, the same way on both machines.**
- *Sampling:* `nvidia-smi` samples power, utilisation and memory of the job's GPU every 10 s into `power.csv`.
- *Summary:* `compute.json` is updated every epoch with the elapsed time (summed over sessions), the mean GPU power and GPU energy = mean power × elapsed time.
- *Bottleneck check:* `log.csv` separates time waiting for data (`data_s`) from GPU step time (`compute_s`), which shows directly whether caching is needed (D22).
- *Cross-check:* on IDUN, `sacct` checks the elapsed time.

**D35. E0 and the timing run.**
- *E0:* trains on the 4 training VOIs in `data/sample` (2 DeepLesion3D, 2 Bone, including the shifted Bone VOI), with no augmentation, and validates on the same VOIs. It runs on the home PC without copying the dataset, and passes when the validation Dice approaches 1.
- *Timing run:* the E1 setup on the full data on IDUN, for 60 steps plus 10 validation VOIs. Seconds per step × 545 steps per epoch, plus the validation time, gives the cost of E1 before the epoch count is chosen.

**D36. Training settings (confirmed by the student, 2026-11-06). The epoch count follows from the timing run.**
- *Batch:* 2 VOIs per step × 2 patches each = 4 patches of 128×128×64, the same on IDUN and the home PC. Runs are then comparable and can move between machines with `--resume` (D33). E0's peak memory confirms that this fits the 11 GB card.
- *Optimiser and loss:* nnU-Net's defaults, which DynUNet is modelled on:
  - SGD with Nesterov momentum 0.99, learning rate 0.01 and weight decay 3·10⁻⁵
  - polynomial learning-rate decay (power 0.9) over the epochs
  - Dice + cross-entropy loss, with the background channel left out of the Dice part. Background is almost all voxels, and including it would make the Dice term easy to satisfy.
  - gradient-norm clipping at 12
  - float16 mixed precision

  *Why:* these settings are well tested for this architecture on CT segmentation, so no tuning is needed. That fits the project's minimum-effort scope and the "no hyperparameter search" rule.
- *Model selection:* every 10 epochs (and after the last), the model predicts all 192 validation VOIs with a sliding window. `best.pt` keeps the weights with the highest mean raw Dice. The centre-component Dice (D30) is logged next to it, but not used for selection, so the choice of checkpoint doesn't depend on the post-processing. The test split is never used (D15).

## Phase 2: E0 results

**D37. First E0 (500 steps on the home PC): the pipeline learns, but the two tiny lesions failed. One caching bug was found and fixed.**
- *Result:* mean val Dice 0.459. Per lesion:

  | Lesion | Size (voxels) | Dice | Dice after D30 |
  |---|---|---|---|
  | `bone_00002` | 8 356 | 0.92 | 0.93 |
  | `bone_00149` (shifted VOI, D17) | 80 792 | 0.89 | 0.91 |
  | `000156` (DeepLesion3D) | 169 | 0.04 | 0.23 |
  | `000010` (DeepLesion3D, 659 HU) | 117 | 0.00 | 0.00 |

  The loss stayed near 1.0 for the first 70 epochs, while the model predicted almost nothing, and then fell to 0.53.
- *Diagnosis* (inference with `best.pt` on the two small VOIs):
  - The model gives `000010` a lesion probability of about 0.001, so it hasn't learned it yet.
  - `000156` is found (recall 1.0), but as a blob 8× too large.
  - The false positives sit near the VOI edges and corners: 100–170 voxels from the lesion, with the largest (6403 voxels) in a corner. Random patches rarely cover those regions, since a corner voxel is in roughly 1 % of patches (D28), so the model has barely seen them. The D30 step removes such false positives.
- *Interpretation:* not a pipeline bug. The large lesions, the offset correction, the loss and the sliding window all work. 500 steps, with the learning rate decaying to 0, are too few to memorise lesions of 117–169 voxels inside patches of about 1 M voxels.
- *For E1 and the video:* watch the gap between raw and centre-component Dice on validation. A large gap means edge false positives. Small lesions are the expected weak point (pitfall 6), so the size breakdown in E3 matters.
- *Caching bug (found while preparing the rerun with caching on, and fixed before any run used the cache):* the train and eval chains wrote the same cache file per VOI, but the eval chain cast to float32 before caching. Whichever chain wrote first then decided what the other one received. The eval chain now ends exactly where the training chain's cached part ends (float16 image, uint8 label, metadata kept), and the prediction code casts to float32. `tests/test_pipeline.py` covers both write orders.
- *Timing on the home PC:* 0.43 s of GPU time per step of 4 patches, and 4.7 GB peak memory, which confirms the batch size of D36. Validating a whole VOI takes about 0.85 s. With `num_workers` 0, the GPU waited for data about 0.8 s per step.

**D38. E0 rerun with 4× the steps (2000): passed. The small lesions were slow to learn, not unlearnable, so the E1 setup stays as it is.**
- *Why a rerun, and not a change to the setup:* only 4 VOIs doesn't mean the task is easy. The two DeepLesion3D lesions fill about 0.01 % of a 128×128×64 patch, so almost every voxel the loss sees is background. Early on, cross-entropy is minimised by predicting background everywhere. The softmax then saturates, and its gradient (scaled by p(1−p)) becomes tiny for those few lesion voxels. SGD with momentum 0.99 also starts slowly, and in the first run the learning rate had decayed to 0 after 500 steps. The rerun was set up as a test with criteria fixed in advance:
  - *slow:* both small lesions reach Dice above about 0.8, so the setup is fine for E1;
  - *stuck:* `000010` stays near 0, so the loss or optimiser must change before E1 (for example Adam, or deep supervision).
- *Run:* `e0_overfit_v2`, home PC, 1000 epochs = 2000 steps, no augmentation, validation every 50 epochs, caching on. It was interrupted after epoch 13 (to silence a repeated warning) and resumed from `last.pt`, which also exercised the resume path on a real run. `configs/experiments/e0_overfit.yaml` now holds these settings, so the README's E0 command reproduces this run.
- *Result:* mean val Dice 0.900 (0.902 after D30), at the last epoch.

  | Lesion | Size (voxels) | Dice (first E0) | Dice (rerun) | Dice after D30 (rerun) |
  |---|---|---|---|---|
  | `bone_00002` | 8 356 | 0.92 | 0.94 | 0.94 |
  | `bone_00149` (shifted VOI, D17) | 80 792 | 0.89 | 0.94 | 0.94 |
  | `000156` (DeepLesion3D) | 169 | 0.04 | 0.89 | 0.89 |
  | `000010` (DeepLesion3D, 659 HU) | 117 | 0.00 | 0.83 | 0.83 |

- *Learning curve (mean DeepLesion3D Dice):* about 0 until epoch 450 (900 steps), then 0.37 at epoch 500, 0.44 at 700, and 0.85 at 800, where the training loss also dropped from 0.34 to 0.14. The bone lesions were above 0.85 from epoch 250. So the small lesions were learned late and suddenly, which fits the saturation explanation above.
- *Raw vs. D30 Dice:* almost equal now (0.900 vs 0.902), so the edge false positives of the first run are gone after enough training.
- *What it means for E1:* an E1 epoch is about 545 steps (1 089 training VOIs, batch 2), so E1 gets far more steps than this. It also sees many small lesions, not 2. Expect validation Dice on small lesions to rise later than on large ones, and don't stop E1 early because small lesions look poor in the first epochs. The epoch count still comes from the IDUN timing run.
- *Timing on the home PC (with caching):* about 0.8 s per epoch of 2 steps, data wait about 0, validation of 4 VOIs about 3.5 s, peak memory 4.7 GB.

## Phase 2: Timing

**D39. IDUN timing run: the job got a P100, which is about 5× slower than the home PC's RTX 2080 Ti. The E1 GPU type and epoch count are still open.**
- *Run:* job 25947781, `configs/experiments/timing.yaml`: 60 training steps on the full training set, after a 10-VOI validation check, with no cache and 8 data workers. It completed in 3.5 minutes.
- *Result:*

  | | Home PC (RTX 2080 Ti, E0) | IDUN (P100-PCIE-16GB) |
  |---|---|---|
  | GPU time per step (4 patches) | 0.43 s | 2.14 s |
  | Data wait per step | about 0 (cache) | 0.03 s (no cache, 8 workers) |
  | Validation per VOI | 0.85 s | 4.5 s |
  | Peak GPU memory | 4.7 GB | 4.7 GB |
  | Mean GPU power | 209 W | 130 W |

- *Interpretation:* the GPU is the bottleneck, not the data. With 8 workers, the cache isn't needed on IDUN. The P100 has no tensor cores, so mixed precision gains little there, while the 2080 Ti does have them.
- *What it would mean for E1 on a P100:* an epoch of about 545 steps takes about 20 minutes, and validating the 192 VOIs takes about 14 minutes (every 10 epochs). A 24-hour job then fits about 65–70 epochs, or about 37 000 steps. E0 showed that small lesions are learned late (D38), so this is on the low side.
- *GPUQ mixes GPU types:* the Phase 0 check got a V100, this run got a P100. The epoch count should therefore be based on the GPU type E1 actually runs on. The options are: request a faster type with a Slurm constraint and repeat the timing run on it; train E1 on the home PC (about 4 minutes per epoch, but it needs a data copy, see D31); or accept the P100 and resume across several 24-hour jobs.
- *A100 chosen and timed:* `sinfo` showed 5 GPU types on GPUQ (P100, V100, A100, H100, H200). A100 is the largest pool (about 100 GPUs), so queue waits should be short, and it has tensor cores. Using one type for all training also keeps time and energy comparable between runs. A repeat of the timing run with `--constraint=a100` (job 25948190) got an A100-SXM4-80GB:

  | | P100 | A100 | Home PC |
  |---|---|---|---|
  | GPU time per step | 2.14 s | 0.175 s | 0.43 s |
  | Data wait per step | 0.03 s | 0.065 s | about 0 (cache) |
  | Whole epoch of 60 steps | 131 s | 15 s | – |
  | Validation per VOI | 4.5 s | 1.3 s | 0.85 s |

  On the A100, the data wait is about a quarter of the epoch time: the GPU is now fast enough that loading and decompressing the VOIs starts to matter. Part of it may be worker start-up, which 60 steps can't separate from steady-state loading.
- *E1 cost on an A100:* an epoch of about 545 steps takes about 2.3 minutes, and validating the 192 VOIs about 4 minutes. With validation every 10 epochs, that's about 2.7 minutes per epoch on average.

**D40. E1 settings (confirmed by the student, 2026-11-12): 250 epochs, caching on, and an A100 or newer GPU.**
- *250 epochs:* about 136 000 steps, about 11–12 hours on an A100, so one job.
  - *Why:* that's more than half of nnU-Net's default of 250 000 steps. Small lesions are learned late (D38), so a much shorter run risks undertraining them.
  - *Alternatives:* 150 epochs (about 7 h) would be cheaper. 450 epochs (about 20 h, matching nnU-Net) would roughly double the energy. 250 also leaves time before the deadline for the optional E4 and E5.
- *Caching on:* on an A100, the data wait was about 25 % of the epoch (D39).
  - *Cost:* the cache stores each preprocessed VOI once, about 25 MB each, so about 32 GB in `/cluster/work`. The quota is 1 TB, of which 6.3 GB is used.
  - *Results:* unchanged, since the cached data is exactly what the uncached chain produces (D37, tested).
  - The first epoch is slower, because it builds the cache.
  - `timing.yaml` keeps caching off, so it still measures loading from the raw files.
- *GPU:* `slurm/train.slurm` requests `--constraint="a100|h100|h200"`.
  - *Why:* the student accepts any of the three. Allowing all three shortens the queue wait compared with A100 only. A Slurm constraint is an exact feature match, so `a100` alone would exclude the newer types.
  - *Memory (added after E1):* `sacct` showed a MaxRSS of 32.0 GB for E1, exactly the 32 GB requested, and Slurm kills a job that exceeds its request. The figure is a sum over the 8 data workers, so shared memory is partly counted twice. `slurm/train.slurm` now requests 64 GB, which leaves a margin and hardly affects queue time.
    - *Corrected after E1b:* E1b's MaxRSS was 61.3 GB against its 64 GB request, so MaxRSS again filled whatever was requested. Neither job was killed, although both reached their limits. So most of this figure is file cache from reading the cache files (about 32 GB for E1, about 55 GB for E1b), which the system frees as needed, not memory the job truly needs. The Phase 1 EDA showed the same (MaxRSS ≈ data read, COMPUTE_LOG). The 64 GB request is kept as a harmless margin, but the original risk was smaller than it looked.
  - *Trade-off:* runs on different types aren't directly comparable in time and energy. Each run's `compute.json` records the GPU model, and the report states it. If E4 is run, its cost comparison with E1 should use the same GPU type: pin it with `--constraint=<type>` on the command line, which overrides the script.

## Phase 2: Course split

**D41. Use the course's own train/test split (confirmed by the student, 2026-11-12). It replaces the official ULS23 split of D15 and D25. Pancreas stays held out (D14).**
- *What happened:* the first E1 job (26089848) failed after 2 minutes with "file not found". On 11–12 Nov, the course staff had reorganised the dataset folder:
  - The old `data/` + `annotations/` layout moved to `2_mic/ULS23_tmp/`.
  - `2_mic/ULS23/` now holds `<dataset>/<train|test>/<images|labels>/`, i.e. a train/test split made by the course.
  - The rename also explains why the pre-flight check on 6 Nov had passed.
- *Same data:* the new folders hold exactly our 1560 labelled cases. All 3120 files have the same byte size as in the old layout. The only cases missing are the 58 unlabelled images, which D20 had already excluded. So the manifest, EDA and preprocessing decisions remain valid.
- *Why switch:* the course provides this split, so we assume students are expected to use it. Results on a shared test set are what the course can compare.
- *How the course split differs from the official one:* it is a random 20 % of each dataset, taken **per lesion, not per patient**. Only about 23 % of the old test cases are in the new test set, which is what chance alone gives.

  | Dataset | Course train | Course test | Course test lesions whose patient also has lesions in course train |
  |---|---|---|---|
  | DeepLesion3D | 594 | 149 | 61 (41 %) |
  | Bone | 557 | 140 | **132 (94 %)** |
  | Pancreas | 96 | 24 | 1 |

  For Bone the "patient" is the `bone_<id>` group (D10), which is at least the same CT study. So almost every Bone test lesion has another lesion from the same scan in the training data.
- *Our splits (`splits/uls23_splits.csv`):*
  - *test:* exactly the course test cases of DeepLesion3D and Bone.
  - *val:* about 15 % of the course training cases, grouped by patient (seed 2311, as in D25), so no patient is in both train and val.
  - *heldout:* all 120 Pancreas cases (E2). The split file also records each Pancreas case's course split, so the 24 Pancreas course-test lesions can be reported separately.

  | Dataset | train | val | test | heldout |
  |---|---|---|---|---|
  | DeepLesion3D | 505 (392 patients) | 89 (74) | 149 (133) | – |
  | Bone | 473 (121) | 84 (22) | 140 (82) | – |
  | Pancreas | – | – | – | 120 (119) |

  New columns: `course_split` (the course's train/test for every case); `patient_in_both` (the patient has lesions in both course splits); `image` and `label` (paths in the new layout). The course folder listing is committed as `splits/course_files.txt` (file names only), so the split file can be rebuilt without access to IDUN.
- *Options considered:*
  - **A (chosen):** use the course split as given. Report the patient overlap as a limitation, and in E3 also report the test results on the "clean" subset: test lesions with `patient_in_both` false (88 DeepLesion3D, 8 Bone).
  - **B:** keep the course test set, but remove from training every lesion whose patient is in course test. That would remove 88 DeepLesion3D lesions and 412 of 557 Bone lesions. It gives a clean test set, but a much weaker Bone model and a setup that differs from the course's.
  - Keeping the official split was rejected, to stay with the course's shared test set.
  - Pancreas: P1 (chosen) keeps E2 as a held-out dataset. P2, training on all three datasets, would have dropped E2.
- *Consequences and limitations for the report:*
  - **Optimistic test scores, mostly for Bone.** A model can do well on a test lesion partly because it has seen the same patient's anatomy, scanner and lesion appearance in training. The clean-subset comparison in E3 shows how large the effect is. The Bone clean subset has only 8 lesions, so it's indicative at most.
  - **Table 2 of the paper is no longer the same split.** The Bone comparison with Table 2 (0.674) is now only a rough reference, not a like-for-like one.
  - **Validation doesn't share this overlap.** Val is grouped by patient, while test isn't. Validation Dice may therefore come out *lower* than test Dice, and this doesn't mean the model is better on test.
  - **D18 used the old training split.** The intensity window was rounded from lesion percentiles of the old training split, which includes some of today's course test lesions. The window is a generic, rounded CT range (−1000 to 1500 HU), and no model was trained or selected on it, so the effect is negligible. It is still stated for transparency.
  - **E0 (D37, D38) used the old split.** Two of its four lesions (`000156`, `bone_00149`) are now course test cases. E0 was only a pipeline check on discarded weights, so this has no effect on results. `e0_overfit.yaml` no longer runs as-is and is kept as the record of D38.
- *Fail-fast added:* `train.py` now checks that every training and validation file exists, on the node that trains, before any GPU work. If not, it stops within seconds with "N of M image/label files are missing". The failed job above only found out when a data worker crashed.
- *Second move, same split (12 Nov, 15:36–15:58):* the first evaluation jobs (26092329, 26092333) stopped after 1.5 minutes, because the fail-fast check found none of the files.
  - *What changed:* the course had moved the split folders one level down, to `2_mic/ULS23/data/<dataset>/<train|test>/`, and removed `ULS23_tmp`. The course's `make_splits.ipynb` shows how its split was made: a random 80/20 per dataset (seed 42), copied from `data/<dataset>/images|labels`.
  - *Check:* a fresh listing of the new folders is identical, line for line, to the committed `splits/course_files.txt`. So every case is in the same dataset, split and folder as before.
  - *Fix:* only `paths.data_root` in `configs/machines/idun.yaml` changed, to `.../2_mic/ULS23/data`. The split file and code are unchanged.
  - *E1 is unaffected:* it started at about 15:05 and read the original files only in its first epoch, before the move. After that it trained from its own cache.
  - *Side effect:* the old layout that the manifest and EDA were built from is gone, so the EDA can't be rerun on IDUN as it is. Its results are committed.
- *Effect on E1 cost:* 978 training VOIs instead of 1089, so about 489 steps per epoch. 250 epochs (D40) then take about 10 h on an A100. Validation covers 173 VOIs instead of 192.

## Phase 2: Evaluation script

**D42. `scripts/evaluate.py` (build step 4): a per-lesion CSV, grouped summaries and saved masks. Results are never overwritten (confirmed by the student, 2026-11-12).**
- *What it does:* it loads a run's own `config.yaml` (with the machine config merged on top) and its `best.pt` (or `last.pt`). It then predicts every VOI of the chosen splits with the same sliding-window code as validation, and writes `<run>/eval/<splits>_<checkpoint>/`:
  - `per_lesion.csv`: raw and D30 Dice, IoU, precision, recall and prediction time per lesion, plus the case metadata for E3 (category, Bone intensity group, lesion size, slice spacing, lesion–shell contrast, `split`, `course_split`, `patient_in_both`).
  - `summary.csv`: mean and median Dice for all lesions, each dataset, Pancreas course train/test separately (E2), and the clean test subset of D41.
  - `predictions/<case>.nii.gz`: the raw predicted masks on the VOI's own grid and affine, for the qualitative figures and the failure review.
  - `compute.json` and `power.csv`: inference time and GPU energy, for the energy estimate.
- *Choices:*
  - **Results are never overwritten.** If `per_lesion.csv` exists, the script stops, and evaluating again means deleting the folder by hand. This turns "evaluate the test split once" (D15, D41) into a rule the code enforces.
  - **No cache.** Evaluation is a single pass, so a cache would not save time, and it would add about 10 GB of test and Pancreas VOIs to `/cluster/work`.
  - **Pancreas uses exactly the same settings** as the training datasets. No Pancreas-specific choice is made, so E2 stays zero-shot.
- *Checks:* a unit test trains a tiny model and checks that `evaluate.py` gives exactly the Dice that training's validation recorded for `best.pt`. On the real-file sample with the E0 weights, it reproduced D38's per-lesion Dice (`000010` 0.829, `bone_00002` 0.939). The two Pancreas lesions scored 0, as expected for a model that memorised 4 non-Pancreas VOIs.
- *Qualitative figures (`scripts/qualitative.py`):* the script reads an evaluation folder and writes `figures/examples.png` and `figures/worst.png`, each with a CSV of the chosen cases. It needs no GPU and only a few dozen VOIs, so it runs on the login node.
  - `examples.png` shows the worst, median and best lesion of each dataset by raw Dice.
  - `worst.png` shows the 12 worst lesions overall, for the failure review in E3. (Ties are broken by case name, so it is not a fair sample when many lesions score 0; see D46.)
  - Each lesion gets two panels on its largest label slice: a zoom around the lesion, and the whole VOI slice, so false positives away from the lesion are visible. The label is outlined in amber and the raw prediction in blue. The display windows and offset correction are the same as in the EDA gallery (D23).
  - On the sample, the outlines matched the anatomy, which confirms that the saved masks keep the VOI's orientation.

**D45. `scripts/analyse.py`: the breakdown and paired-comparison tables come from the per-lesion CSVs, in the repo, written to `results/` (confirmed by the student, 2026-11-13).**
- *Why:* the D43 and D44 numbers were first computed with throwaway scripts. Every table in the report should be reproducible from the repo. The script reads only `per_lesion.csv` files, so it needs no GPU, no images and no IDUN, and the outputs are small enough to commit.
- *Two commands:*
  - **`breakdown`** (one model) writes `breakdown.csv`. For each dataset and factor it gives the lesion count, the mean raw Dice, the mean and median centre-component Dice, the missed fraction, raw and D30 precision, and the median ratio of predicted to true volume.
  - **`compare`** (two models on the same lesions) writes:
    - `paired.csv`, with per group: both means, the mean change with a 95 % bootstrap interval, the median change, the lesions better or worse by more than 0.05, the missed fraction of each model, rescued and lost lesions, and a Wilcoxon p-value;
    - `paired_lesions.csv`, with every lesion sorted by change, for the figures and the review.
- *Fixed choices* (constants in `src/evaluation/analysis.py`, the same as in D43 and D44):
  - **Missed:** a lesion is missed when its centre-component Dice is below 0.05.
  - **Bins** include their upper edge:
    - size by equivalent diameter: 10, 15, 20 and 30 mm;
    - contrast by the absolute lesion–shell difference: 20, 50 and 100 HU (the sign is dropped, since faint lesions can be slightly brighter or darker than their shell);
    - slice spacing: 1.5 and 3.5 mm.
  - **Wilcoxon p-values** only for groups where at least 10 lesions changed; smaller groups get no p-value, because it would not be reliable.
  - **Bootstrap:** 5000 resamples of lesions with a fixed seed, so the intervals are the same on every run.
- *Check:* unit tests on synthetic tables check the bins, the counts and that row order doesn't matter. The script reproduces D44 exactly: +0.049 (95 % interval +0.014 to +0.083), 43 better, 22 worse, 21 rescued, 7 lost, p = 0.002.
- *Outputs so far:* `results/e1_vs_e1b_val`, `results/e1_vs_e1b_train`, `results/e3_test` and `results/e2_pancreas`.

**D46. The E3 failure review uses random samples of missed and of partly found lesions per dataset, not `worst.png` (confirmed by the student, 2026-11-13).**
- *Problem:* 92 of the 289 test lesions have raw Dice exactly 0 (60 DeepLesion3D, 32 Bone). `worst.png` breaks ties by case name, so its "12 worst" were the first 12 zero-Dice lesions alphabetically. They were all DeepLesion3D from low patient numbers, and Bone did not appear at all. The same applied to the E1 validation figure that D43 described, so that sentence has been corrected.
- *First look at those 12 anyway* (E1b, test). All are complete misses:
  - 6 have low contrast (< 50 HU);
  - 3 have good contrast (87–362 HU), but the model segmented a vessel or nodule a few millimetres away. Nothing tells the model that the lesion is at the VOI centre;
  - 2 are the same chest-wall lesion in two studies of one patient;
  - 1 is small (6.8 mm).
  - None shows a boundary error or an obvious annotation issue.
- *Fix (`scripts/qualitative.py`, `src/evaluation/figures.py`):* two new figures, each with a CSV of its lesions.
  - **`missed.png`:** up to 10 lesions per dataset, drawn at random with a fixed seed (0) from the missed lesions (centre-component Dice < 0.05). It shows what a typical miss looks like in each dataset.
  - **`partial.png`:** the same, drawn from lesions with centre-component Dice from 0.05 to 0.5. This is where boundary errors and partly found lesions show up, which `worst.png` cannot show.
  - Every figure now has a title saying how its lesions were chosen. `worst.png` is kept, and its title says that ties are broken by case name.
  - `--sample` and `--seed` change the size and the seed of the samples.
- *Checks:* a unit test checks that the sample is per dataset, stays within its Dice band, isn't simply the first lesions by name, and is the same on every run.
- *Figure layout:* row spacing was increased after the first test figures, because each row's titles overlapped the images above them.

## Phase 4: Test, E2 and E3

**D47. Results of the final model E1b: test split (once), E2 on Pancreas and the E3 failure review (review checked by the student, 2026-11-13).**
- *Test split* (`results/e3_test/breakdown.csv`, job 26098860):

  | | Lesions | Raw Dice | Centre-component Dice, mean (median) | Missed |
  |---|---|---|---|---|
  | All | 289 | 0.258 | 0.418 (0.424) | 36 % |
  | DeepLesion3D | 149 | 0.279 | 0.412 (0.557) | 41 % |
  | Bone | 140 | 0.236 | 0.424 (0.400) | 31 % |
  | DeepLesion3D, patient not in course train | 88 | 0.261 | 0.385 | |

  - **Test against val:** test is a little below val (0.418 against 0.454). That is expected, since the model was chosen on val, and each split has only about 150 lesions per dataset.
  - **Bone:** 94 % of the Bone test lesions share a patient with training (D41), yet Bone scored lower on test than on the patient-grouped val. So the overlap did not visibly inflate the result. The clean Bone subset has only 8 lesions, too few to report on its own.
- *Breakdowns (test):*
  - **Contrast is the strongest factor.** DeepLesion3D lesions under 50 HU are missed about 61 % of the time (51 lesions), against 26 % above 100 HU.
  - **By category:** Liver (73 % missed), Kidney (71 %) and Abdominal (61 %) are the weak categories. Bone-type lesions in DeepLesion3D are missed only 12 % of the time.
  - **Small lesions:** more than half of lesions under 10 mm are missed in both datasets.
  - **Slice spacing has no effect,** as on val (D43).
  - **Volume:** the predicted volume is a median 4.3× the lesion's in Bone (12× under 10 mm). DeepLesion3D lesions over 30 mm get only 0.08× (5 lesions).
- *E3 failure review* (`results/e3_test/failure_review.csv`, D46 samples): one primary type per lesion, read from the figures by Claude and checked by the student.

  | Missed (20) | Lesions | | Partly found (20) | Lesions |
  |---|---|---|---|---|
  | Low contrast (faint soft tissue, or lytic in bone) | 10 | | Under-segmented: right lesion, only part of it | 12 |
  | Model segments a nearby structure | 4 | | Boundary error on a small lesion | 5 |
  | Small (5–6 mm) | 2 | | Prediction mostly on a nearby structure | 3 |
  | Unusual appearance or location | 2 | | | |
  | Found only a tiny part | 2 | | | |

  - **Low contrast explains half the misses, in both datasets.** Lytic or faint bone lesions (−98 to +43 HU against their shell) fail the same way as faint soft-tissue lesions.
  - **Under-segmentation explains most partial finds.** The model finds the most visible part, for example the dense part of a mixed bone lesion, and stops there.
  - **Nearby structures (7 of 40):** the model segments a vessel, rib or node beside the lesion. Nothing tells the model that the lesion is at the VOI centre, so a centre prior is a concrete idea for future work.
  - **Repeated studies fail alike.** The same lesion in two studies of one patient fails the same way: the chest-wall lesion is missed twice, and the 000658 node is under-segmented to about 0.3 twice.
  - **No clear annotation errors** in the 40.
  - **Limitation:** each lesion was judged on one slice by one reader.
- *E2: Pancreas, zero-shot* (`results/e2_pancreas/breakdown.csv`, job 26098861):
  - **Result:** mean Dice 0.001 over 120 tumours, with 119 missed.
  - **The model does predict something:** a median of about 2 800 voxels per VOI. But only 6 predictions touch the tumour at all, and the predicted volume is a median 0.31× the tumour's.
  - **Not a bug:**
    - the labels lie on the visible darker mass in the figures;
    - the same pipeline scores 0.42 on the test split;
    - Pancreas is stored in HU, so no intensity correction is involved.
  - **Size and contrast do not explain it alone.** Comparison with DeepLesion3D lesions evaluated by the same model (train, val and test):

    | Lesions | n | Missed |
    |---|---|---|
    | DeepLesion3D, contrast ≤ 20 HU | 98 | 56 % |
    | DeepLesion3D, contrast ≤ 20 HU and ≥ 15 mm | 25 | 44 % (39 % on training lesions) |
    | DeepLesion3D Abdominal, contrast ≤ 20 HU | 31 | 74 % |
    | Pancreas, contrast median 4 HU, median 27 mm | 120 | 99 % |
    | Pancreas tumours with contrast > 40 HU | 7 | 100 % |

    So Pancreas fails far more often than DeepLesion3D lesions with similar contrast and size. Pancreatic tumours are faint and ill-defined inside an organ the training data barely covers, so their whole appearance is new to the model. Two weaknesses seen in E3 are at their extreme here: low contrast and under-segmentation of large lesions.
  - **Caveat:** the lesion–shell contrast compares a tumour with a shell that can include fat and vessels. It is not the contrast against the pancreas itself.
- *A mask-based location analysis* (where exactly the Pancreas predictions land) *was not done.* The figures already show small blobs on kidney, bowel and vessels away from the tumour, and the comparison above answers the main question. The time goes to the report instead.

**D48. No E4 (2D) and no E5 (more seeds). The energy estimate uses the Green Algorithms model with stated assumptions (confirmed by the student, 2026-11-13).**
- *E4 and E5 are not run.* The project focuses on E1 → E1b, the test result, E2 and E3. Together these argue for the procedure and show where the data and the model reach their limits.
  - **E5** would cost about 5 h of A100 time per seed, plus queueing. The paired test already shows that the E1 → E1b gain is significant (p = 0.002, D44). The single seed per model goes in the limitations.
  - **E4** is not run either. A 2D model is mentioned in the discussion as a possibly cheaper alternative that was not tested.
- *Energy method* (details and sources in COMPUTE_LOG.md, "Totals and assumptions"):
  - **GPU energy** is measured from `nvidia-smi` samples.
  - **CPU, memory and cooling** are estimated per job with the Green Algorithms model (Lannelongue et al. 2021): 8 cores at 7.5 W each (3.5 W in the low case), 0.3725 W per GB of requested memory, and PUE 1.67 (1.2 in the low case). The home PC uses PUE 1.
  - **Why this model:** it is published and widely used, and every number in it can be checked. IDUN's own PUE and node CPUs were not available, so they are given as ranges, not single values.
- *Result:* 2.75 kWh measured on GPUs, and about 5.9 kWh in total (range 4.3–5.9). That is about 31 km of EV driving at 19.1 kWh/100 km (EV Database), with a range of 22–31 km. The two training runs are 94 % of it.
- *Is the cost justified?* Points for the discussion:
  - **Small in absolute terms:** the whole project used about the energy of one short car trip. E1b, the experiment that tested the low-contrast hypothesis, cost about 2.6 kWh (14 km) for +0.048 centre-component Dice on val, and it confirmed the hypothesis.
  - **Fewer epochs would have been almost as good.**
    - E1's centre-component Dice was nearly flat from epoch 140 (D43).
    - E1b reached 0.439 by epoch 160, against 0.454 at epoch 250.
    - Stopping at about 150 epochs would have saved roughly 40 % of the training energy (about 2 kWh) for about 0.015 centre-component Dice.
    - Raw Dice was still rising, so the choice depends on the metric.
  - **The cache made training cheaper.** The data wait was about 1 s per epoch, so the GPU was busy almost all the time, and the runs were shorter than they would have been without it.
  - **GPU type matters as much as the method.** E1b ran on an A100-PCIE-40GB. It used about 18 % less energy per run than E1 on the SXM4-80GB, but took about 16 % longer, part of which is the extra input channel.
  - **Not tried:** a 2D model, smaller patches, or a subset of the data. Each could cut the cost, and each would probably lose some accuracy (untested).
  - **Inference is cheap:** 0.3–0.45 s per VOI, about 0.02 Wh of GPU energy. So most of the energy goes into training, not into using the model. The real-file check on the sample drew `missed.png` correctly. The sample has no partly found lesions, so `partial.png` was skipped, as intended.

## Phase 3: E1 and Gate 2

**D43. E1 underfits and misses low-contrast lesions. Gate 2: one targeted retrain, E1b, with a second input channel in a soft-tissue window (confirmed by the student, 2026-11-12).**
- *E1 run:* job 26090388, 250 epochs in 4.3 h on an A100, 1.41 kWh (COMPUTE_LOG).
  - Best val Dice 0.244 raw and 0.406 after D30, both at the last epoch.
  - The epoch-100 check passed: Dice was 0.16–0.19 and still rising.
  - The centre-component Dice was roughly flat (0.36–0.42) from epoch 140, so most of the late gain came from fewer false positives. More epochs would not have fixed the core problem.
- *Underfitting test:* `evaluate.py` on the training split as well as on val.

  | | Training split (978) | Validation (173) |
  |---|---|---|
  | Raw Dice, mean | 0.340 | 0.244 |
  | Centre-component Dice, mean (median) | 0.529 (0.692) | 0.406 (0.443) |
  | Lesions missed (centre-component Dice < 0.05) | 25 % | 40 % |

  The model reaches only 0.53 even on lesions it trained on, while E0 showed that the network can memorise lesions (0.90). So the main problem is underfitting, with a smaller generalisation gap on top. The val evaluation reproduced the training log's 0.244 exactly.
- *Where it fails* (from `per_lesion.csv` and the figures):
  - **Contrast is the strongest factor.** For DeepLesion3D validation lesions:

    | Lesion–shell contrast | Missed (val) | Missed (train) | Centre-component Dice (val) |
    |---|---|---|---|
    | < 50 HU | 73 % | 58 % | 0.14 |
    | 50–100 HU | 44 % | 27 % | 0.42 |
    | > 100 HU | 26 % | 8 % | 0.54 |

  - **By category:** Kidney lesions are missed 91 % of the time and Liver lesions 67 % (val), against 9 % for bone-type lesions. The 12 worst validation lesions are all faint DeepLesion3D soft-tissue lesions, with no prediction at the centre. *Corrected 2026-11-13 (D46):* those 12 all had Dice 0 and were the first such lesions by case name, not a fair sample. The contrast and category numbers above come from all lesions and still stand.
  - **Slice spacing has no effect.** Lesions on 1 mm and 5 mm slices score the same (0.38 against 0.37 on val, 0.50 against 0.48 on train). This rules out anisotropy, which had been a suspect.
  - **Size matters less than contrast.** Lesions under 10 mm do somewhat worse.
  - **False positives:** the predicted volume is about 2.3× the true lesion volume (median). Precision rises from 0.21 to 0.52 after D30, so the false positives sit mostly away from the target.
  - **D30 can hurt:** for `bone_00149_lesion_07`, the model found part of the lesion but not the part with the centre, and D30 kept another component. Raw Dice was 0.19, and 0.00 after D30. This goes into the failure review.
  - **Context:** the DeepLesion3D lesions were chosen as the ones a standard nnU-Net handled worst (course README). The ULS23 baseline also trained on about 38 000 lesions, not 978, so part of the gap is expected.
- *Why the input:* in the wide −1000 to 1500 HU window (D18), a 30 HU lesion–tissue difference is about 1 % of the input range. Instance normalisation largely cancels a linear rescaling, but not clipping. A soft-tissue window clips away air and bone, so faint differences fill a large part of the range. Using only a soft-tissue window would saturate the dense bone lesions, which E1 handles well. Two channels keep both.
- *Options presented:*
  - **A:** accept E1 and explain its limits. It costs nothing, and the analysis above is useful for the video.
  - **B (chosen):** a second input channel. It tests the observed failure directly, and E1 against E1b makes a clean comparison for the video and the reflection.
  - **C:** deep supervision. It is a general nnU-Net improvement, but not aimed at the observed failure.
  - Anisotropy-aware strides were dropped after the spacing breakdown. More epochs were dropped because the centre-component Dice had plateaued.
- *E1b setup (`configs/experiments/e1b_dynunet3d_2ch.yaml`):*
  - The input gets two channels: the D18 window and a soft-tissue window of −160 to 240 HU (the EDA display window, D23). Each channel is clipped and scaled to [0, 1].
  - The model's `in_channels` is 2. Everything else is identical to E1, which a unit test checks: the configs differ only in `extra_windows`, `in_channels` and `run.name`.
  - *Code:* `WindowChannelsd` in `src/data/transforms.py` makes one channel per window. With one window it is exactly the old `ScaleIntensityRanged` (also tested), so E1's preprocessing and cache are unchanged. The cache key includes the windows, so E1b gets its own cache of about 55 GB.
  - *Side effect:* the intensity augmentations act on both channels in the same units, so ±0.02 means ±50 HU in the wide channel but ±8 HU in the soft-tissue channel. This is kept, since the augmentation is mild.
- *How E1 and E1b are compared:* on validation only, with the same breakdowns (contrast, category, size). The test split is evaluated once, after one model has been chosen.
  - *Selection metric (confirmed by the student, 2026-11-12):* the model with the higher mean centre-component Dice on validation goes to test. Raw Dice is reported alongside it.
  - The reason is that D30 is part of the method, and raw Dice is dominated by false positives that D30 removes.
  - This does not change D36: within each run, `best.pt` is still chosen by raw Dice.
- *Real-file check before IDUN:* E1b trained one epoch on the two sample training lesions on the home PC. The validation check, the epoch and the checkpoint all ran: "2 training and 2 validation VOIs", 1.2 s per VOI, peak GPU memory 4.6 GB. So the 2-channel input runs through the whole pipeline, and the extra channel doesn't change memory use noticeably.

**D44. E1b beats E1 on validation, mainly on low-contrast DeepLesion3D lesions. E1b is the final model and goes to the test split, with no further training first (confirmed by the student, 2026-11-13).**
- *E1b run:* job 26092482, 250 epochs in 5.0 h, 1.16 kWh. It ran on an A100-PCIE-40GB, not the SXM4-80GB that E1 used, so time and energy are not directly comparable (COMPUTE_LOG). Best val Dice at the last epoch, as in E1. E1b's centre-component Dice was higher than E1's at most validation epochs from epoch 80, so the gain is not one lucky epoch.
- *Result, the same lesions for both models:*

  | Centre-component Dice, mean (median) | E1 | E1b |
  |---|---|---|
  | Validation, all (173) | 0.406 (0.443) | **0.454 (0.570)** |
  | Validation, DeepLesion3D (89) | 0.367 (0.249) | **0.425 (0.493)** |
  | Validation, Bone (84) | 0.447 (0.485) | **0.486 (0.583)** |
  | Training split, all (978) | 0.529 (0.692) | 0.554 (0.709) |
  | Validation lesions missed (< 0.05) | 40 % | 32 % |

  Raw val Dice rose from 0.244 to 0.261 (DeepLesion3D 0.248 to 0.281, Bone unchanged at 0.240).
- *Paired comparison on validation:*
  - **How many lesions changed.** 43 lesions improved by more than 0.05, 22 got worse by more than 0.05, and 108 stayed within 0.05.
  - **Missed lesions.** 21 lesions that E1 missed were found by E1b, and 7 went the other way.
  - **The median change is 0,** so the gain comes from rescued lesions, not from a general shift.
  - **Significance.** A Wilcoxon signed-rank test gives p = 0.002. A bootstrap over lesions gives a 95 % interval of +0.014 to +0.083 for the mean gain.
  - **Per dataset.** DeepLesion3D p = 0.01, Bone p = 0.08.
- *The gain follows contrast, as the D43 hypothesis predicted:*

  | DeepLesion3D, by lesion–shell contrast | Val: missed, E1 → E1b | Val: Dice change | Train: Dice change |
  |---|---|---|---|
  | < 20 HU | 71 % → 57 % | +0.13 | +0.09 |
  | 20–50 HU | 75 % → 55 % | +0.12 | +0.07 |
  | 50–100 HU | 44 % → 38 % | 0.00 | +0.01 |
  | > 100 HU | 26 % → 22 % | +0.08 | +0.01 |

  The training split (505 DeepLesion3D lesions) confirms the pattern with far more lesions. The val gain above 100 HU (23 lesions) is not seen on train, so it is probably noise.
- *By category:*
  - **Liver** improved most: on the training split, Dice rose from 0.28 to 0.53 and the missed rate fell from 59 % to 19 %. Val has only 6 Liver lesions; they went from 0.26 to 0.38.
  - **Kidney** is still the hardest: 82 % missed on val (91 % for E1), and slightly worse on train. Its failures may be about telling lesion from kidney tissue, not about the window.
  - **Bone** gains less (+0.04 val, +0.02 train), as expected, since bone lesions mostly have high contrast already.
- *Caveats:*
  - **One seed per model.** The 22 lesions that got worse, some from about 0.7 to 0, show that two runs can differ a lot per lesion. The overall gain is significant, but the size of the effect is uncertain.
  - **Still underfitting.** Training-split Dice is only 0.55, and the best epoch was the last one in both runs.
  - **More false positives.** E1b's predicted volume is a median 3.1× the lesion volume on val (E1: 2.35×). Raw Dice stays low because of these blobs elsewhere in the VOI. The median examples score raw 0.17 and 0.13 but 0.85 and 0.88 after D30. D30 removes most of the blobs, which is part of why the centre-component Dice is the selection metric (D43).
- *Choice:* under the selection metric confirmed in D43, E1b wins (0.454 against 0.406).
- *No further training before the test.*
  - **Option:** a longer E1b run, since both runs were still improving at epoch 250. It would cost about 10 h of A100 time plus queueing.
  - **Why not:** with two weeks to the deadline, the E1 → E1b comparison and its analysis matter more for the report and video than a few hundredths of Dice.
  - **Report:** the remaining underfitting and the false positives go in as findings and future work.
  - **Consequence:** the test split is now used once, for E1b, so no later model can be compared on it.
- *Optional experiments:* E4 and E5 were not run before the test. E5 (more seeds of E1b) could still be run afterwards, since it only measures run-to-run spread and selects nothing.
- *Compute for the comparison:* val and training-split evaluation of E1b took 0.034 h and 0.157 h of script time, 0.028 kWh in total (COMPUTE_LOG).

