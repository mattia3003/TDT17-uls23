# Video script: 3D universal lesion segmentation on ULS23

Slide-by-slide content for the 12-minute project video, following the required 8-section structure (PROJECT_PLAN.md, section 8).

**How to use this file**
- **Slide text:** the words to put on each slide, kept short.
- **Visual:** the figure to show. Paths are relative to the repo root.
- **Say:** the spoken script, at about 140 words per minute. The timings add up to about 11:30, which leaves a margin.
- **Sources:** every number comes from `DECISIONS.md` (the D numbers in brackets), `COMPUTE_LOG.md` or the tables in `results/`. The brackets are for checking, not for the slides.

**Colours used in all figures**
- **Datasets:** DeepLesion3D blue, Bone orange, Pancreas green.
- **Models:** E1 pink, E1b purple.
- **Example panels:** label amber, prediction blue.

The thread through the video is the procedure:

> data analysis → decisions → a baseline → a diagnosis of what fails → one targeted retrain → one look at the test split → an honest account of the limits.

---

## 1. Background and motivation (1:00)

### Slide 1: Title
- **Slide text:** "3D universal lesion segmentation in CT: what limits a small model, and how far one targeted fix goes". Name, TDT17, NTNU, November 2026.
- **Visual:** `results/eda/gallery.png` (lesions from the three datasets), faded behind the title.
- **Say (0:15):** "In this project I trained a 3D network to outline lesions in CT scans. It works on any lesion type, from a liver tumour to a bone metastasis. I'll show the procedure, from data analysis to a targeted retrain, and where the data and the model reach their limits."

### Slide 2: Why universal lesion segmentation
- **Slide text:**
  - Radiologists measure lesions to follow cancer treatment, often by hand and in 2D.
  - A 3D outline gives volume and is more reproducible.
  - "Universal" means one model for every organ and lesion type, which is hard: lesions differ in size, contrast and appearance.
  - ULS23 challenge: each case is a 3D volume of interest (VOI) centred on one lesion.
- **Visual:** `results/eda/gallery.png`, which shows small, median and large lesions per dataset.
- **Say (0:45):** "Lesion size is how doctors judge whether a cancer treatment works. Today it's often measured by hand, as a diameter on one slice. A 3D segmentation gives the full volume and is more consistent. A universal model has to handle every organ and every lesion type, and as the gallery shows, they look very different. A small, bright bone lesion and a faint liver lesion are almost different problems. The ULS23 challenge gives one 3D volume per lesion, already centred on it, and the task is to outline it."

## 2. Approach and strategy (1:00)

### Slide 3: Scope and plan
- **Slide text:**
  - **Data on IDUN:** three ULS23 datasets with new annotations: DeepLesion3D (743 lesions), Bone (697), Pancreas (120).
  - **Train** on DeepLesion3D + Bone. **Hold out** Pancreas entirely, as a test of generalization (E2) (D14).
  - **Model:** one 3D U-Net configured like nnU-Net (MONAI DynUNet), no ensembles.
  - **Process:** gates where the evidence decides the next step. Every decision is logged (49 entries in DECISIONS.md).
- **Visual:** a simple flow diagram: EDA → decisions → E0 overfit check → E1 → Gate 2 diagnosis → E1b → test once → E2 + failure review.
- **Say (1:00):** "Three ULS23 datasets are available on IDUN. I trained on DeepLesion3D, with lesions from all over the body, and on a bone metastasis dataset. I held the pancreas tumours out completely, to test how the model handles a lesion type it has never seen. The plan was deliberately small: one 3D U-Net in the nnU-Net style, no ensembles, and clear gates between the phases. First I overfit four lesions to prove the pipeline can learn. Then a full run. Then a diagnosis of what fails, and one retrain aimed at that failure. The test split was used exactly once, at the end. Every decision and its reason is logged in the repository, 49 entries in all."

## 3. Data analysis (2:30)

### Slide 4: Bone intensities were stored on two scales
- **Slide text:**
  - 555 of 744 Bone volumes are stored shifted by +1024: air sits near +23 instead of −1000 HU.
  - Each patient is entirely on one scale, and no volume falls in between.
  - **→ Decision:** detect the shift from each volume's minimum and subtract 1024 (D17). Checked later: on the test split, both Bone groups score alike (centre-component Dice 0.42 and 0.45).
- **Visual:** `results/eda/intensity_offset.png`.
- **Say (0:35):** "The most important finding came first. Three-quarters of the bone volumes were stored with a shift of 1024, so air read as zero instead of minus 1000. Without a fix, the same tissue would have two different values, and the model would have to learn both. Every patient is entirely on one scale, so the shift can be detected reliably from the volume's minimum. I corrected it before training."

### Slide 5: Contrast and size differ between datasets
- **Slide text:**
  - Median lesion diameter: DeepLesion3D 11 mm, Bone 15 mm, Pancreas 27 mm.
  - Median lesion-to-surroundings contrast: 67 HU, 135 HU, and only **4 HU** for Pancreas.
  - Lesion intensities range from −905 to 1447 HU. **→** Clip to [−1000, 1500] HU and scale to [0, 1] (D18).
- **Visual:** `results/eda/lesion_size.png` and `results/eda/lesion_hu.png` side by side.
- **Say (0:40):** "The datasets differ a lot. Pancreas tumours are more than twice as large as DeepLesion3D lesions, and they barely differ from their surroundings: a median of 4 Hounsfield units, against 67 and 135 for the other two. That made Pancreas a hard and honest test of generalization. Lesion intensities span almost 2500 HU, from fat-like lesions to dense bone, so I clipped to minus 1000 to 1500 and scaled to 0 to 1."

### Slide 6: Geometry decisions
- **Slide text:**
  - Slice spacing varies from 0.45 to 5 mm. **→** Keep the native grid, no resampling (D16). Checked later: spacing has no effect on accuracy.
  - 98 % of lesions fit in a 128×128×64 patch. **→** Train on patches, predict the whole 256×256×128 volume with a sliding window (D19).
  - Every lesion contains the VOI centre. **→** Keep only the predicted piece at the centre, and report Dice with and without it (D30).
- **Visual:** `results/eda/crop_fit.png`, with `results/eda/spacing.png` as an inset.
- **Say (0:45):** "Slice spacing varies from under half a millimetre to 5 millimetres. The challenge baseline didn't resample, so I kept the native grid, and the results later showed that spacing made no difference. Almost every lesion fits inside a 128 by 128 by 64 patch, which is what the network trains on. And every lesion contains the centre of its volume. That gave a simple post-processing step: keep only the predicted piece that touches the centre. I report the score both with and without it."

### Slide 7: Splits
- **Slide text:**
  - Course split, with validation grouped by patient: train 978 lesions, validation 173, test 289, held out 120 (Pancreas) (D41).
  - Up to 21 lesions per Bone patient. The course test split is per lesion, so 94 % of Bone test lesions share a patient with training. This is reported as a limitation.
  - All selection on validation. Test used once.
- **Visual:** a small table of the split sizes.
- **Say (0:30):** "I used the course's train and test split and took a patient-grouped validation set out of training. One caveat: a bone patient can have up to 21 lesions, and the course test split is per lesion, so most bone test lesions come from patients the model has seen. I report that as a limitation. Everything was selected on validation, and the test split was evaluated once."

## 4. Methods and models (2:00)

### Slide 8: Model and training
- **Slide text:**
  - **Model:** MONAI DynUNet, 3D, 6 levels, 32 to 320 filters, instance norm (nnU-Net defaults).
  - **Loss:** Dice + cross-entropy. **Optimizer:** SGD with Nesterov momentum 0.99, learning rate 0.01 with polynomial decay.
  - **Training:** 128×128×64 patches, 4 per step, flips, 90° rotations, mild intensity changes, mixed precision.
  - **Run:** 250 epochs on one A100, about 4.3 h. Preprocessed volumes are cached, so the GPU never waits for data.
- **Visual:** a U-Net sketch with the patch size and the two input channels marked (channel 2 is added in E1b).
- **Say (0:50):** "The model is a 3D U-Net with nnU-Net's default design, built with MONAI's DynUNet. It trains on 128 by 128 by 64 patches with a Dice plus cross-entropy loss, SGD with high momentum, and nnU-Net's learning-rate schedule. The augmentations are flips, rotations and small intensity changes. One run is 250 epochs, a little over four hours on an A100 on IDUN. Caching the preprocessed volumes kept the GPU busy almost all the time."

### Slide 9: Checks before spending GPU hours
- **Slide text:**
  - **E0, overfit four lesions:** Dice 0.90, so the pipeline can learn (D38).
  - **Timing run:** a P100 is 12× slower than an A100, so A100s were requested (D39).
  - **Fail-fast file checks:** when the course data was reorganised twice, the jobs stopped within two minutes instead of failing hours in (D41).
  - **Evaluation reproduces training's validation exactly:** 0.244 = 0.244.
- **Visual:** a checklist with ticks.
- **Say (0:40):** "Before the real run, I checked the pipeline cheaply. Overfitting four lesions reached a Dice of 0.9, so the code can learn. A short timing run showed a P100 would be twelve times slower than an A100. A check that stops a job within two minutes if data files are missing paid off twice, when the course data was reorganised. And the evaluation script reproduces the training run's validation score exactly."

### Slide 10: Metrics
- **Slide text:**
  - Dice per lesion, on the whole volume.
  - **Raw Dice:** every predicted voxel counts.
  - **Centre-component Dice:** only the predicted piece at the VOI centre. This is the selection metric between models, chosen before the comparison (D43).
  - **Missed:** centre-component Dice below 0.05.
- **Visual:** `results/figures/examples_test.png`. Point at the median examples: raw Dice 0.17 and 0.14, centre component 0.32 and 0.54.
- **Say (0:30):** "I report Dice per lesion in two ways. Raw Dice counts every predicted voxel, including small false blobs far from the lesion. Centre-component Dice keeps only the piece at the centre. In the median examples here, the lesion itself is found reasonably well, and the raw score is pulled down by blobs elsewhere. I chose the centre-component Dice as the selection metric before comparing any models."

## 5. Results (2:00)

### Slide 11: E1 underfits low-contrast lesions, so E1b targets that
- **Slide text:**
  - **E1:** val Dice 0.244 raw, 0.406 centre component. Only 0.53 even on its own training lesions, so it underfits (D43).
  - DeepLesion3D lesions with less than 50 HU contrast: 73 % missed.
  - **Hypothesis:** in the wide −1000 to 1500 window, a 30 HU difference is about 1 % of the input range.
  - **E1b:** add a second input channel with a soft-tissue window (−160 to 240 HU). Nothing else changes.
- **Visual:** `results/figures/training_curves.png`.
- **Say (0:40):** "The first full run, E1, reached a validation Dice of 0.41 after post-processing. But it scored only 0.53 on its own training lesions, so it underfits. Breaking the misses down, contrast stood out: three in four faint soft-tissue lesions were missed. In the wide intensity window, a 30 HU difference is about one percent of the input. So I made one targeted change: a second input channel with a soft-tissue window. That's E1b. The curves show E1b ahead on the selection metric for most of training."

### Slide 12: The fix works where it was aimed
- **Slide text:**
  - **Validation, centre-component Dice:** 0.406 → **0.454**.
  - **Paired comparison:** 21 missed lesions rescued, 7 lost. Wilcoxon p = 0.002; 95 % interval of the gain +0.014 to +0.083 (D44).
  - **The gain follows contrast:** faint DeepLesion3D lesions 71–75 % → 55–57 % missed; little change above 50 HU. The training split shows the same pattern.
  - **E1b is the final model,** by the rule set in advance.
- **Visual:** `results/figures/missed_by_contrast.png`.
- **Say (0:35):** "On validation, E1b improved the centre-component Dice from 0.41 to 0.45. Pairing the two models lesion by lesion, E1b rescued 21 missed lesions and lost 7, and the gain is statistically significant. More importantly, the gain sits exactly where the hypothesis predicted: the missed rate for faint lesions drops by about 15 to 20 points, and barely changes for high-contrast ones. The much larger training split shows the same pattern. So E1b became the final model."

### Slide 13: Test split, evaluated once
- **Slide text:**

  | Test (289 lesions) | Raw Dice | Centre-component Dice | Missed |
  |---|---|---|---|
  | All | 0.258 | 0.418 | 36 % |
  | DeepLesion3D | 0.279 | 0.412 | 41 % |
  | Bone | 0.236 | 0.424 | 31 % |

  - Close to validation (0.454), as expected after selecting on it.
  - Missed: Liver 73 %, Kidney 71 %, Abdominal 61 %, but bone-type lesions only 12 %. Slice spacing: no effect.
- **Visual:** `results/figures/test_breakdown.png`.
- **Say (0:20):** "On the test split, evaluated once, the final model reaches a centre-component Dice of 0.42, close to validation. Contrast again explains most of the variation. Faint liver, kidney and abdominal lesions are missed most often, while bone-type lesions are rarely missed."

### Slide 14: Pancreas: a near-complete failure, and why
- **Slide text:**
  - **Held-out Pancreas:** Dice 0.001, 119 of 120 tumours missed (D47).
  - **Not a bug:** the labels lie on the visible mass, and the same pipeline scores 0.42 on test.
  - **Not just contrast or size:** DeepLesion3D lesions with similar contrast and size are missed 44 %; Pancreas 99 %. Even its 7 clearer tumours are all missed.
- **Visual:** `results/figures/e2_pancreas.png`, with `results/figures/missed_pancreas.png` as a strip.
- **Say (0:25):** "On the held-out pancreas tumours, the model fails almost completely: it finds 1 of 120. That isn't a bug, and it isn't just low contrast or size. DeepLesion3D lesions with similar contrast and size are found more than half the time. The whole appearance of these tumours is new to the model."

## 6. Discussion (2:00)

### Slide 15: How the model fails
- **Slide text:** review of 40 randomly sampled test lesions (D46, D47).

  | Missed (20) | | Partly found (20) | |
  |---|---|---|---|
  | Low contrast | 10 | Under-segmented | 12 |
  | Nearby structure | 4 | Boundary error | 5 |
  | Small, unusual or tiny part | 6 | Nearby structure | 3 |

  - No clear annotation errors.
- **Visual:** `results/figures/missed_test.png` and `results/figures/partial_test.png`. Crop two or three rows each.
- **Say (0:45):** "I reviewed 40 randomly sampled test lesions. My first worst-cases figure wasn't a fair sample, because it broke ties alphabetically, so I replaced it with random samples. Half the misses are low contrast, in both datasets: faint soft tissue, or lytic bone lesions. When a lesion is found but poorly outlined, the model usually catches the most visible part and stops there. And in seven cases it outlined a vessel or a rib next to the lesion, because nothing tells it the lesion is at the centre."

### Slide 16: Limits and what I would do next
- **Slide text:**
  - **Data:** 978 training lesions, against about 38 000 for the ULS23 baseline. Two datasets. Pancreas shows the limit of that.
  - **Evaluation:** one seed per model. The Bone test lesions share patients with training. One reader for the failure review.
  - **Next:**
    - a centre prior as an input channel (nearby-structure errors);
    - longer training (still improving at epoch 250);
    - more diverse training data (Pancreas).
  - **Reference points from the ULS23 paper** (Table 2): Bone Dice 0.674; Pancreas 0.56–0.62 for models that *were* trained on pancreas data. It is a different split and about 40× more training data, so these are context, not a comparison (PROJECT_PLAN.md, section 3).
- **Visual:** text only.
- **Say (0:45):** "The main limit is data. I trained on under a thousand lesions from two datasets, while the challenge baseline used about 38,000. Pancreas shows what that means for unseen lesion types. Each model was trained once, so the size of the E1b gain is uncertain, even though it's significant. And the bone test lesions share patients with training. Next, I'd tell the model where the centre is with an extra input channel, train longer, since both runs were still improving, and above all train on more varied data."

### Slide 17: Sustainability
- **Slide text:**
  - **Measured GPU energy:** 2.75 kWh. **Estimated total** with CPU, memory and cooling: about 5.9 kWh (4.3–5.9) (D48).
  - **≈ 31 km of EV driving** (19.1 kWh/100 km).
  - The two training runs are 94 %. E1b cost about 2.6 kWh (14 km) for +0.05 Dice, and it confirmed the hypothesis.
  - **Cheaper:** stopping at about 150 epochs saves about 40 % for about 0.015 Dice. GPU type matters too: the PCIe A100 used 18 % less energy per run but took longer. A 2D model was not tested.
- **Visual:** a bar of kWh per group (training, evaluation, development) with the EV equivalent, from `COMPUTE_LOG.md`.
- **Say (0:30):** "All jobs were logged with measured GPU power. With an overhead estimate for CPU, memory and cooling, the whole project used about 6 kilowatt-hours, roughly 30 kilometres in an electric car. Almost all of it was the two training runs. For a project like this, that's cheap. But the curves suggest stopping at 150 epochs would have saved 40 % for almost the same result."

## 7. Key learning points (0:30)

### Slide 18: What I learned
- **Slide text:**
  - Look at the data first: an intensity offset hid in three-quarters of the Bone volumes.
  - Diagnose before you retrain: one targeted change, tested where it was aimed.
  - Decide the selection rule before looking, and use the test split once.
  - Check your own figures: a "worst cases" view can be biased by how ties are broken.
- **Visual:** text only.
- **Say (0:30):** "Four things I'll take with me. Look at the data before modelling: three-quarters of the bone volumes were on a different intensity scale. Diagnose before retraining, and aim the fix at the diagnosis. Fix the selection rule before comparing, and touch the test set once. And check your own analysis tools: my first worst-cases figure was biased by how it broke ties."

## 8. Contributions and references (0:30)

### Slide 19: Contributions and references
- **Slide text:**
  - **Solo project.** Claude Code (Anthropic) was used as a coding assistant. All design decisions were made and reviewed by me and are logged in DECISIONS.md.
  - **Data:** ULS23 (de Grauw et al., Medical Image Analysis 2025), CC BY-NC-SA 4.0. DeepLesion (Yan et al., 2018).
  - **Software:** MONAI (DynUNet), PyTorch, nnU-Net design (Isensee et al., 2021).
  - **Energy:** Green Algorithms (Lannelongue et al., 2021), EV Database.
  - **Compute:** NTNU IDUN.
  - **Code and full decision log:** the repository.
- **Visual:** text only. The full list is in `REFERENCES.md`.
- **Say (0:20):** "This was a solo project. I used Claude Code as a coding assistant, and every decision is logged with its reasoning in the repository. The data is from the ULS23 challenge, the model is built with MONAI and PyTorch following nnU-Net, and the energy estimate follows the Green Algorithms method. Thanks for watching."
