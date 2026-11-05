# TDT17 Mini-Project: 3D Lesion Segmentation on ULS23 (MONAI)

**Hand-off plan for Claude Code.** Read this whole file before doing anything, then start at Phase 0.

## 1. Project summary

| | |
|---|---|
| Course | TDT17 Visual Intelligence, Task 4 (45% of final grade) |
| Option | 2: Medical Image Computing, 3D lesion segmentation |
| Group size | Individual |
| Deadline | **Fri 27 Nov 2026, 23:59** |
| Main deliverable | Video presentation, **max 12 minutes** |
| Also required | Code, configs and results retained and available on request |
| Framework | PyTorch + MONAI (version >= 1.2.0, see section 3) |
| Compute | NTNU IDUN cluster, with a home PC as fallback (see below) |
| Data | `/cluster/projects/vc/courses/TDT17/2026/2_mic/ULS23/` |

### Project goal: meet the requirements with minimum effort

The student wants to complete this project properly and pass the course, while saving most of their time for other work. **Keep the experimental scope small, but the code well built.** Don't add experiments unless the student asks. The code should still be well structured, reproducible and reusable (see "Code standards" in section 2).

### What the assignment requires of an individual student

1. Develop and evaluate at least one lesion segmentation model on a selected, motivated subset of ULS23.
2. Present dataset analysis, method choices, results, failure cases and key learning points in the video.
3. Track compute use and estimate energy, with a meaningful comparison (e.g. EV driving distance), and discuss whether the cost is justified.
4. Document decisions and acknowledge external resources.

The assignment says the highest achievement also needs one additional aspect, such as generalization to another dataset. This plan covers that with a held-out dataset test, because it needs no extra training.

The project description is **v0**. Splits, subsets and evaluation procedures may change, so the student should check the course channels for updates.

### Compute options

Support both IDUN and the home PC from the start. The student chooses where to run each job, and the same code and configs must work on both.

**Home PC:** RTX 2080 Ti (11 GB VRAM), i9-9900K (8 cores, 16 threads), 32 GB DDR4, and enough SSD space for the selected datasets.

- **Starting 3D settings:** crop around 128 x 128 x 64, batch size 2, mixed precision, and sliding-window inference with a small batch. Adjust the crop based on the EDA and on what fits in memory.
- **Data loading:** start with 6–8 workers. Cache preprocessed data on the SSD instead of in RAM if memory gets tight.
- **Data:** copy only the selected datasets, after Gate 1. The student does this, and the data stays out of git.
- **Out of reach locally:** the full 256 x 256 x 128 VOIs in training, and full-dataset training.
- **Choosing where to run:** give the student a short timing script to compare a few epochs on each machine.

## 2. Working agreement for Claude Code

- **Never initiate training or heavy processing, on IDUN or anywhere else.** The student starts all of these personally. This overrides any step in this plan and has no exceptions, even for short runs.
  - **Claude Code must not start:** model training (including test runs), inference over the dataset, full-dataset analysis, preprocessing or caching, downloads of large files, or anything submitted with `sbatch`, `srun` or `salloc`.
  - **Claude Code does instead:** write the code, config and Slurm script, then give the student the exact command, the expected runtime and resources, and the outputs to look for. It then stops and waits. The student runs it and shares the results.
  - **Claude Code may run:** quick, light checks such as reading files, listing directories, inspecting a single file's header, linting, and unit tests on small synthetic data. If unsure whether something counts as light, ask first.
  - Where a phase below says to run, train or evaluate something, this means preparing it for the student to run, then working from the results the student shares.
- **Code standards.** The code should be well structured and generalizable, without building features the project doesn't use.
  - **Config-driven:** datasets, paths, model, crop size, hyperparameters and seed all come from YAML configs. Nothing is hardcoded in the code.
  - **Modular:** data, models, training and evaluation are separate modules with clear interfaces, and scripts stay as thin entry points.
  - **Easy to extend:** adding a dataset or model should need a new config entry, or a small addition to a model factory, with no changes to the training or evaluation code. Evaluation must work the same for any model.
  - **MONAI first:** prefer MONAI's built-in components over custom code.
  - **Readable:** use type hints, short docstrings and clear names.
- **Seeding and reproducibility.** Every run must be reproducible as far as is practical.
  - Use one central seeding function covering Python, NumPy, PyTorch (including CUDA) and MONAI (`monai.utils.set_determinism`), and seed DataLoader workers too.
  - The seed is set in the config and saved with each run's outputs, along with the resolved config and git commit.
  - Splits use their own fixed seed, so they stay identical when the training seed changes.
  - Document any remaining sources of non-determinism, such as GPU operations.
- **The student must be able to explain every decision in the video.** Briefly explain non-obvious choices.
- **Stop at each gate** (marked 🚦) and wait for the student's decision.
- **Treat section 3 as hypotheses.** Nobody has inspected the data on IDUN yet. If the data disagrees with this plan, report it and update the plan.
- **Treat the course dataset as read-only.** Write derived data to the student's own project or work area.
- **Don't commit data, checkpoints or large outputs to git.**
- **Keep short notes.** Record decisions in `DECISIONS.md` and external code and sources in `REFERENCES.md`. Acknowledging external resources is a project rule.
- **Never tune on the test split.** It is used once, at the end.
- **Check against real files before any long job (D12).** Run the exact same command locally on the small real-data sample (`data/sample/`, 2 VOIs per dataset, gitignored) before handing over an IDUN job. Long jobs fail fast on systematic errors and save intermediate outputs before later steps that could crash.
- **Don't plan around dates.** The student works in periods alongside other projects. The only fixed date is the deadline.

## 3. What is known about ULS23 (verify in Phase 1)

These facts come from the public challenge material. The copy on IDUN may differ.

- **Modality and region:** chest-abdomen-pelvis CT.
- **Size:** 38,693 training lesions in the full public dataset. The official test set is **not public**, so we must create our own test split.
- **Format:** volumes of interest (VOIs) of 256 x 256 x 128 voxels in the **original scan spacing**.
- **Centering:** a voxel randomly chosen from inside the lesion mask sits at the VOI center.
- **Single target:** each VOI has one annotated lesion. Masks of nearby unconnected lesions were removed.
- **Padding:** where needed, VOIs are padded with the array minimum minus 1.
- **MONAI:** the challenge repo warns that MONAI versions below 1.2.0 have had problems loading this data.

**Constituent datasets (per the challenge repo):**

| Group | Datasets | Notes |
|---|---|---|
| Novel annotations | `ULS23_DeepLesion3D`, `ULS23_Radboudumc_Bone`, `ULS23_Radboudumc_Pancreas` | Bone and pancreas are described as challenging lesion types |
| Processed public | `kits21`, `LIDC-IDRI`, `LiTS`, `MDSC` tasks 6/7/10, `NIH-LN` | Check the lesion type and annotation quality of each |
| Weakly annotated | `DeepLesion`, `CCC18` | Out of scope. Confirmed by the IDUN README (`partially_annotated/`, GrabCut masks from 2D measurements) |

**Verified from the IDUN copy's `README.md` (Phase 0):**

- **Folder names:** fully annotated data is in `novel_data/` (`ULS23_DeepLesion3D`: 750, `ULS23_Radboudumc_Bone`: 744, `ULS23_Radboudumc_Pancreas`: 124) and in `processed_data/fully_annotated/` (`kits21`: 333, `LIDC-IDRI`: 2,246, `LiTS`: 888, `MDSC_Task06_Lung`: 76, `MDSC_Task07_Pancreas`: 283, `MDSC_Task10_Colon`: 133, `NIH_LN_ABD`: 558, `NIH_LN_MED`: 379).
- **LNDb:** not included, because of its license.
- **License:** CC BY-NC-SA 4.0.
- **⚠ Only the 3 novel datasets are on IDUN:** `ULS23_DeepLesion3D`, `ULS23_Radboudumc_Bone` and `ULS23_Radboudumc_Pancreas`, about 1,618 lesions in total. There is no `processed_data/`, so kits21, LIDC-IDRI, LiTS, MDSC and NIH-LN are not available. Section 4's train/hold-out design must use these 3 datasets. Decide at Gate 1, and check the course channels in case more data is added.
- **Layout:** images are in `data/<dataset>/images/` and labels in `annotations/<dataset>/labels/`, paired by identical filename. This is set in `configs/data/uls23.yaml`. The label folders also hold `*.nii.gz.zip` copies, which are ignored. `data/ULS23_DeepLesion3D/categories/<type>/` groups DeepLesion3D lesions by type (Abdominal, Bone, Kidney, Liver, Lung, Mediastinal, Other).
- **Counts:** there are 1618 images and 1560 labels. 58 images have no label: 7 DeepLesion3D, 47 Bone and 4 Pancreas. Every label has an image. *(Verified in Phase 1.)*
- **Patient IDs:** DeepLesion3D filenames follow `<patient>_<study>_<series>_<slice>_lesion_<nn>`, and Radboud filenames follow `bone_<id>_lesion_<nn>` and `diag_pancreas_<id>_lesion_<nn>`. That gives 549, 151 and 119 groups. For Bone this means a median of 3 and up to 21 lesions per ID, so grouping is essential. *(Verified in Phase 1; D10.)*
- **Baseline spacing:** the ULS23 nnU-Net baseline used **no resampling**, through a dummy resampling function. This supports keeping the original spacing.
- **Official split:** `baseline_model/train_test_split.json` covers all 1560 labelled lesions (170 test). It is patient-grouped: none of our 819 groups is on both sides. It is used for our test split (D15). *(Verified in Phase 1.)*
- **Intensity offset (verified):** 555 of the 744 Bone VOIs are stored shifted by +1024 (minimum exactly 0, air near +23). The rest of Bone, and all of DeepLesion3D and Pancreas, are in HU. This explains the README's Bone statistics. It is corrected by a data-driven rule (D17).
- **File format (verified):** volumes are stored as `(256, 256, 128, 1)` float32. 13 labels are plain 3-D int16. MONAI hides the extra axis, but plain nibabel does not (D11).
- **Spacing (verified):** in-plane spacing is 0.33–0.98 mm and slice spacing is 0.45–5 mm. DeepLesion3D is split about evenly between 1 mm and 5 mm slices.
- **Padding (verified):** DeepLesion3D VOIs are a median of 70 % padding, as whole empty slabs of slices. The padding value is often −1024, equal to air.
- **Centering (verified):** all lesions contain the VOI centre and are a single connected piece. Only 8 Bone lesions touch the VOI edge.

**Published reference point:** the semi-supervised ULS23 baseline reached Dice 0.703 ± 0.240 on the private challenge test set. This is context only, because our test split is a different population.

**Per-lesion-type reference points (paper, Table 2, 10 % patient-level held-out training data; verify against the paper before quoting):**
- **Bone (N = 86):** nnU-Net 0.674 ± 0.262. This is directly comparable to our Bone test lesions, which come from the same official split. Their models were trained on all of ULS23 (about 38k lesions), so a lower E1 result is expected.
- **Pancreas (N = 38):** 0.556–0.616. This pools Radboudumc Pancreas with MSD Pancreas, and the model was trained on pancreas data. For our zero-shot E2 it is an upper reference only.
- **DeepLesion3D:** no per-dataset result is reported. It falls under "various lesions".

### Pitfalls to check

These affect whether the results are valid, so they stay in scope. Each one is also useful material for the video.

1. **Patient-level leakage.** Several lesions can come from the same patient or scan, so splits must be grouped by patient, or by scan if patient IDs aren't available. Document any limitation.
2. **Center prior.** The lesion is always at the center, and a model will exploit that. Note this as a limitation in the video.
3. **Padding and HU clipping.** Padding values must not distort intensity normalization.
4. **Anisotropic, varying spacing.** Pick a simple spacing strategy and justify it. Report metrics in the original VOI space.
5. **Annotation differences.** Datasets differ in how masks were produced, which can look like model error.
6. **Class imbalance and size range.** Tiny lesions occupy very few voxels, while large lesions may exceed a small crop.
7. **Dataset dominance.** A pooled mean can hide poor results, so always report results per dataset.

## 4. Experiment design

**Training data (confirmed at Gate 1, D14):** `ULS23_DeepLesion3D` + `ULS23_Radboudumc_Bone` (1440 lesions). `ULS23_Radboudumc_Pancreas` (120 lesions) is **held out from training entirely**, for E2.

**Splits (confirmed at Gate 1, D15):**
- *Test:* the official ULS23 test cases of the training datasets (159 lesions).
- *Validation:* about 15 % of the official training cases, grouped by patient, with a fixed seed.
- *E2:* all of Pancreas.

Commit the derived split file.

**Preprocessing (confirmed at Gate 1, D16–D19):**
- native voxel grid, with no resampling
- the Bone +1024 offset corrected
- intensities clipped to [−1000, 1500] HU and scaled to [0, 1]
- 128×128×64 training patches, with sliding-window prediction on the full VOI

| ID | Experiment | Question | Training |
|---|---|---|---|
| E0 | Overfit a few VOIs | Does the pipeline work? | Minutes |
| E1 | **3D model**: MONAI `DynUNet` with Dice + CE loss | How good is a well-configured 3D U-Net? | 1 run |
| E2 | **Generalization**: E1 on the held-out dataset | How does it do on an unseen lesion type? | None |
| E3 | **Breakdown**: results by dataset, lesion size, slice spacing, Bone intensity group and DeepLesion3D category, plus failure review | Where does it fail, and why? | None |
| E4 | **Optional: 2D vs 3D**: 2D U-Net trained on slices, evaluated as reassembled 3D volumes | What is 3D context worth, and at what compute cost? | 1 run |
| E5 | **Optional: multiple seeds**: repeat the final model(s) with different training seeds | How much do results vary between runs? | 2 extra runs per model |

### Optional E4: 2D vs 3D

The student decides at Gate 2 whether to run this. It would strengthen the project by comparing substantially different approaches and giving the sustainability discussion a measured trade-off.

- **Prepare for it from the start:** the model factory, transforms and evaluation code should support 2D as well as 3D, so adding E4 later needs little new code.
- **Keep the comparison fair:** use identical splits and evaluation code and, as far as possible, comparable preprocessing and augmentation. Document what differs.
- **Evaluate in 3D:** stack the 2D predictions back into a volume and compute the same per-lesion metrics as for E1.
- **Compare per lesion:** use a paired test on per-lesion Dice (e.g. Wilcoxon signed-rank).
- **Compare cost:** include training time, inference time and energy for both models.

### Optional E5: multiple seeds

The student decides at Gate 2 whether to run this, and can make the final call once the time per run is known. It repeats the final model(s) with 3 training seeds in total, using the same splits, so only the training randomness changes.

**Why it could aid the project:**

- **Separates real effects from noise.** A single run can be lucky or unlucky. Repeats show how much Dice changes from randomness alone.
- **Makes 2D vs 3D more trustworthy.** If E4 is run, a difference smaller than the variation between seeds shouldn't be presented as a real improvement.
- **Strengthens the methodology.** Reporting mean ± standard deviation across runs supports the "experimental methodology" grading criterion and gives the discussion more substance.
- **Shows which findings are stable.** It reveals whether per-dataset results and the generalization result in E2 hold up across runs.

**Cost:** each extra run adds its full training time and energy, so this is the most expensive way to strengthen the project per insight gained. Include this energy in the sustainability estimate, and discuss whether it was worth it.

**How to report it:** give the mean ± standard deviation of per-run Dice. Keep this separate from the spread between lesions, which measures how much difficulty varies across lesions and not how stable training is.

### Evaluation

- **Metrics:** Dice and IoU per lesion, with mean, median and spread. Add precision and recall if they help explain failures.
- **Breakdown:** per dataset and by lesion-size bin. Three further breakdowns cost no extra training (added after Gate 1):
  - *By slice spacing* (≤ 1.5 mm vs ≥ 3 mm): tests whether keeping the native grid (D16) hurts.
  - *By Bone intensity group* (stored in HU vs shifted by +1024): checks that the offset correction (D17) worked.
  - *By DeepLesion3D category* (Lung, Liver and so on).

  Carry the needed columns from the manifest into the per-lesion CSV.
- **Reference point:** quote the paper's per-lesion-type numbers (section 3) next to E1 for Bone, and as an upper reference for E2 Pancreas, with their caveats.
- **Output:** save a per-lesion CSV, so new analyses don't require re-running inference.
- **Selection:** use the validation split for model and threshold selection.

### Out of scope unless the student asks

Other architectures, nnU-Net, hyperparameter searches, weakly annotated data, and semi-supervised learning.

## 5. Phases

### Phase 0: Setup

- [x] Confirm where the work will run, and ask the student about their IDUN account and storage.
- [x] Create the Python environment and pin versions.
- [x] Create the repo structure from section 6 and initialize git.
- [x] Write a short Slurm test job that loads one VOI with MONAI on a GPU, for the student to submit. (Job 25934176 passed on a V100.)

**Done when:** the student reports that the test job works.

### Phase 1: Data analysis

Claude Code writes one analysis script and its Slurm job, and the student runs it. Results are saved to `results/eda/`.

- [x] Map the dataset: folders, file naming, image/label pairing, size on disk. (15.9 GB in total.)
- [x] Build `data/manifest.csv` with one row per VOI: dataset, paths, patient/scan ID if available, spacing, lesion volume, lesion extents, and padding fraction. (Job 25936150. The manifest also records intensity and label-quality indicators.)
- [x] Plot per-dataset lesion size and spacing distributions.
- [x] Make a small gallery of mask overlays per dataset (small, median and large lesions).
- [x] Measure the fraction of lesions that fit inside candidate crop sizes.
- [ ] Fix the DeepLesion3D category reader (the categories are empty in the manifest), and redraw the figures kept for the video (D21).

**🚦 Gate 1: present and decide**

- Which datasets to train on and hold out, and why
- How to group splits, and any limitations
- Spacing strategy, HU window and crop size

Keep the 3–5 most useful figures for the video.

### Phase 2: Pipeline

- [ ] Write the central seeding function and the config loader.
- [ ] Write the split generator, with its own fixed seed and a check that no patient appears in two splits.
- [ ] Write config-driven MONAI transforms and a model factory that supports 2D and 3D.
- [ ] Cache the deterministic part of preprocessing (load, offset correction, clipping and scaling) with MONAI's `PersistentDataset`, in the work directory. Measure during E0 whether data loading is a bottleneck; if not, caching can stay off.
- [ ] Write a training script with mixed precision, checkpoint and resume, and CSV logging. Each run saves its resolved config, seed and git commit.
- [ ] Write a model-agnostic evaluation script that outputs the per-lesion CSV in original VOI space.
- [ ] Test the metrics on synthetic masks, and test that the same seed gives the same splits.
- [ ] Write a script that saves qualitative examples.
- [ ] Add compute logging inside the training code, so it works on both machines: elapsed time and `nvidia-smi` power samples. Use `sacct` as an extra check on IDUN.
- [ ] Keep Slurm scripts as thin wrappers around the same `train.py` command used locally. Use a separate path config for each machine.
- [ ] Make sure checkpoints can resume on either machine, so a run can move between them.

**Done when:** the student has run **E0** and Dice approaches 1 on the overfit VOIs.

### Phase 3: Training

- [ ] The student trains **E1**.
- [ ] Inspect the curves and some predictions, and fix any bugs.

**🚦 Gate 2: is the model credible?** A retrain is only worthwhile if there is a clear bug or a clear flaw in the setup. Modest performance is acceptable if it's well explained.

The student also decides here whether to run **optional E4 (2D vs 3D)** and **optional E5 (multiple seeds)**, based on the time and interest they have. Report the measured time per run so the cost of each option is clear.

### Phase 3b: Optional experiments

- [ ] **E4:** add the 2D config and any 2D-specific slice sampling. The student trains the 2D model.
- [ ] **E5:** add configs for the extra seeds. The student trains the extra runs, after the final configuration is settled.

### Phase 4: Evaluation

- [ ] Evaluate once on the test split.
- [ ] **E2:** evaluate on the held-out dataset.
- [ ] **E3:** produce the breakdown tables, and review the worst 10–20 cases by failure type (small lesion, low contrast, boundary error, annotation issue, nearby lesion).
- [ ] If E4 was run, evaluate both models on the same test lesions and produce the comparison.
- [ ] If E5 was run, evaluate every seed and report the mean ± standard deviation across runs.
- [ ] Measure inference time per VOI.
- [ ] Complete the energy estimate (section 7).
- [ ] Create final figures, including best, median and worst examples.

### Phase 5: Video and submission

- [ ] Write the script and slides following section 8, and keep to 12 minutes.
- [ ] Add a short README with reproduction steps.
- [ ] The student records the video and submits at the location the course announces.

### Rough pacing (guidance only)

If it helps, aim for data analysis and pipeline by mid-November, training and evaluation by around 20 Nov, and the video in the final week. Leave margin for IDUN queue times near the deadline.

## 6. Repository structure

```
tdt17-uls23/
├── README.md
├── PROJECT_PLAN.md
├── DECISIONS.md
├── REFERENCES.md
├── COMPUTE_LOG.md
├── requirements.txt
├── configs/            # One YAML per experiment, plus shared defaults
├── src/
│   ├── data/           # Manifest, splits, transforms, datasets
│   ├── models/         # Model factory (2D and 3D)
│   ├── training/       # Train loop, losses, checkpointing
│   ├── evaluation/     # Metrics, inference, per-lesion reports
│   └── utils/          # Config loading, seeding, logging, compute tracking
├── scripts/            # Thin entry points: eda, split, train, evaluate
├── slurm/              # Job scripts
├── splits/             # Committed split files
├── tests/              # Metric, split and seeding tests
└── results/            # Small CSVs and final figures
```

Checkpoints, caches and predictions go in the student's work directory on IDUN, outside git.

## 7. Sustainability (required)

Log significant jobs in `COMPUTE_LOG.md`, including failed ones:

| Job ID or run name | Machine | Experiment | GPU type | Elapsed (h) | Mean GPU power (W) | Energy (kWh) |
|---|---|---|---|---|---|---|

- **GPU energy** = mean sampled power x elapsed time.
- **Total estimate:** add a stated assumption for overhead such as CPU, memory and cooling.
- **Comparison:** convert to EV driving distance, using a stated consumption figure with a source (typically around 15–20 kWh per 100 km).
- **Discussion:** Is the cost justified by the performance? Would a smaller subset, smaller crop or fewer epochs have been enough? If E4 was run, use its measured cost and Dice difference to discuss whether 3D is worth it. If not, mention a 2D model as a possibly cheaper alternative that wasn't tested.

## 8. Video outline (max 12 minutes)

The structure is required by the assignment. Timings are suggestions.

| # | Section | Time | Content |
|---|---|---|---|
| 1 | Background / Motivation | 1:00 | Why lesion segmentation matters, and why "universal" is hard |
| 2 | Approach / Strategy | 1:00 | The plan and the scope decision |
| 3 | Data Analysis | 2:30 | Key findings and **how each changed a decision** |
| 4 | Methods / Models | 2:00 | Architecture, preprocessing, loss, splits, and why |
| 5 | Results | 2:00 | Metrics, per-dataset and size breakdowns, held-out dataset, 2D vs 3D if run, examples |
| 6 | Discussion | 2:00 | Strengths, limits, failure types, improvements, **sustainability** |
| 7 | Key Learning Points | 0:30 | What was learned |
| 8 | Contributions / References | 0:30 | Solo work, external code and sources |

**Data analysis → decision map** (for section 3 of the video, from Gate 1):

| Finding | Decision |
|---|---|
| 555 Bone VOIs stored shifted by +1024 (air near +23, not −1000) | Data-driven offset correction (D17), checked by the Bone-group breakdown in E3 |
| Slice spacing 0.45–5 mm; the baseline did not resample | Keep the native grid (D16), checked by the spacing breakdown in E3 |
| 98 % of lesions fit in 128×128×64; a few large Bone lesions do not | 128×128×64 patches plus sliding-window prediction on the full VOI (D19) |
| Up to 21 lesions per Bone patient | Patient-grouped splits; the official split is reused (D15) |
| Lesion intensities −905 to 1447 HU; padding down to −10405 | Clip to [−1000, 1500] and scale to [0, 1] (D18) |
| Pancreas: large, low-contrast (4 HU) lesions from a different organ | Held out for the generalization test E2 (D14) |
| Every lesion contains the VOI centre | Random patch positions; the centre prior is reported as a limitation (D19) |

## 9. Risks and fallbacks

| Risk | Fallback |
|---|---|
| IDUN queues are slow | Shorter jobs with resume, or move the run to the home PC |
| GPU memory is limited | Smaller crop, smaller batch size, or fewer VOIs |
| Data loading is too slow | Cache preprocessed data in the work area with MONAI `PersistentDataset`, or reduce the subset |
| Patient IDs are unavailable | Group by the best available ID, and state the risk in the video |
| Performance is poor | Time-box debugging, then present the failure analysis |
| Time runs short | Skip E5 first, then E4. If still short, skip E2 and use a single dataset for training and testing |
| Course updates the spec (v0) | Adjust the plan |

## 10. References to start from

- de Grauw et al., *The ULS23 Challenge: a Baseline Model and Benchmark Dataset for 3D Universal Lesion Segmentation in CT*: https://arxiv.org/abs/2406.05231
- ULS23 challenge site: https://uls23.grand-challenge.org
- ULS23 repository: https://github.com/MJJdG/ULS23
- Course background material on SharePoint (paper, video and podcast)
- IDUN documentation: https://www.hpc.ntnu.no/idun/

## 11. First prompt for Claude Code

> Read `PROJECT_PLAN.md` fully. Summarize the plan back to me in a few lines, then ask me anything you need to know about my environment (where this is running, my IDUN account, and storage). After that, start Phase 0. Stop at every 🚦 gate. Keep the experiments to what the plan lists, but make the code well structured, config-driven, seeded and generalizable. Never start training or heavy processing yourself, on IDUN or anywhere else: prepare the script and the exact command, and I will run it.
