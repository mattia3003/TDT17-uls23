# Compute log

Log every significant job here, including failed ones. GPU energy = mean sampled GPU power × elapsed time.
The training code records elapsed time and `nvidia-smi` power samples. On IDUN, also check with `sacct -j <jobid> --format=JobID,Elapsed,State,AllocTRES`.

| Job ID or run name | Machine | Experiment | GPU type | Elapsed (h) | Mean GPU power (W) | Energy (kWh) | Notes |
|---|---|---|---|---|---|---|---|
| 25934176 | IDUN GPUQ | Phase 0 environment check | V100-PCIE-32GB | < 0.01 | – | ≈ 0 | Passed. A few seconds of GPU use. |
| 25934982 | IDUN CPUQ, 8 CPUs | Phase 1 EDA (first try) | none | 0.057 | – | – (CPU only) | FAILED after scanning all VOIs. 1547 of 1618 VOIs were stored with an extra size-1 axis (D11). Elapsed 00:03:24, MaxRSS 16.7 GB. That is mostly file cache, since about 16 GB of VOIs were read. |
| 25936150 | IDUN CPUQ, 8 CPUs | Phase 1 EDA | none | 0.063 | – | – (CPU only) | Completed, 0 read errors. Elapsed 00:03:45, MaxRSS 16.8 GB (again about the 15.9 GB of data read, so file cache). |
| e0_overfit | Home PC | E0, first try (250 epochs = 500 steps) | RTX 2080 Ti | 0.201 | 135 | 0.027 | Did not pass: val Dice 0.459. Bone 0.92/0.89, the two small DeepLesion3D lesions 0.00/0.04 (D37). 0.43 s GPU time per step of 4 patches, peak memory 4.7 GB. The GPU waited for data most of the time (num_workers 0), hence the low mean power. |
| e0_overfit_v2 | Home PC | E0 rerun (1000 epochs = 2000 steps, caching on) | RTX 2080 Ti | 0.340 | 209 | 0.071 | Passed: val Dice 0.900 (D38). Two sessions (interrupted after epoch 13, resumed); the elapsed time and energy cover both. Data wait about 0 with the cache, hence the higher mean power. |
| 25947781 | IDUN GPUQ, 8 CPUs | Timing run (60 steps + 10-VOI validation check) | P100-PCIE-16GB | 0.058 (training code 0.050) | 130 | 0.006 | Completed. 2.14 s GPU time per step, about 5× the home PC's 0.43 s (D39). Data wait 1.9 s in total for 60 steps (no cache, 8 workers). Validation 4.5 s per VOI. Peak GPU memory 4.7 GB, MaxRSS 4.8 GB. Energy is from the training code's power samples, which cover 0.050 h of the 0.058 h job. |
| 25948190 | IDUN GPUQ, 8 CPUs, `--constraint=a100` | Timing run on A100 (same as above, `run.name=timing_a100`) | A100-SXM4-80GB | 0.044 (training code 0.0085) | 82 | 0.0007 | Completed. 0.175 s GPU time per step, 12× the P100 and 2.5× the home PC (D39). Data wait 3.9 s for 60 steps, now about 25 % of the epoch. Validation 1.3 s per VOI. MaxRSS 5.3 GB. Only 4 power samples, mostly at start-up, so the mean power is not representative of a long run. |
| 26089848 | IDUN GPUQ, 8 CPUs | E1, first try | – | 0.033 | – | ≈ 0 | FAILED after 00:01:57: the course data had been reorganised into train/test folders, so the files were not found (D41). MaxRSS 0.9 GB. |
| 26092329, 26092333 | IDUN GPUQ, 8 CPUs | Evaluation of E1 on val and on the training split (first try) | V100-PCIE-32GB | 0.023 each | – | ≈ 0 | FAILED after 00:01:24: the fail-fast check found no files, because the course data had moved to `ULS23/data/` (D41). MaxRSS 0.6 GB. |
| 26092401 | IDUN GPUQ, 8 CPUs | Evaluation of E1 `best.pt` on val (173 VOIs) | V100-PCIE-32GB | 0.043 (script 0.038) | 125 | 0.005 | Completed. Mean val Dice 0.244 (centre component 0.406), the same as the last validation in training. 0.50 s prediction per VOI. MaxRSS 4.3 GB. |
| 26092402 | IDUN GPUQ, 8 CPUs | Evaluation of E1 `best.pt` on the training split (978 VOIs) | V100-PCIE-32GB | 0.20 (script 0.20) | 151 | 0.030 | Completed. Mean train Dice 0.340 (centre component 0.529): a Gate 2 check for underfitting. 0.47 s prediction per VOI. MaxRSS 11.4 GB. |
| 26090388 (`e1_dynunet3d`) | IDUN GPUQ, 8 CPUs, `a100\|h100\|h200` | E1 (250 epochs, caching on) | A100-SXM4-80GB | 4.33 (training code 4.31) | 326 | 1.41 | Completed. 55 s per epoch (about 0.11 s per step) with 1 s data wait; validation of 173 VOIs about 66 s. Best val Dice 0.244 (centre component 0.406) at the last epoch. Peak GPU memory 4.7 GB. MaxRSS 32.0 GB, right at the 32 GB request (the sum over 8 data workers, partly shared memory). |
| 26092482 (`e1b_dynunet3d_2ch`) | IDUN GPUQ, 8 CPUs, `--constraint=a100` | E1b (as E1, plus a soft-tissue input channel, D43) | A100-PCIE-40GB | 5.01 (training code 4.98) | 233 | 1.16 | Completed. 64 s per epoch with 1 s data wait. Best val Dice 0.261 (centre component 0.454) at the last epoch. Not the same A100 model as E1 (SXM4-80GB): `a100` matches both, so time and energy are not directly comparable. MaxRSS 61.3 GB against a 64 GB request; like E1's 32.0 GB against 32 GB, this is mostly reclaimable file cache from reading the cache files, not a risk of the job being killed. |
| 26097081 | IDUN GPUQ, 8 CPUs | Evaluation of E1b `best.pt` on val (173 VOIs) | V100-PCIE-16GB | 0.055 (script 0.034) | 125 | 0.004 | Completed. Mean val Dice 0.261 (centre component 0.454), the same as the last validation in training. 0.46 s prediction per VOI. |
| 26097082 | IDUN GPUQ, 8 CPUs | Evaluation of E1b `best.pt` on the training split (978 VOIs) | A100-PCIE-40GB | 0.178 (script 0.157) | 154 | 0.024 | Completed. Mean train Dice 0.349 (centre component 0.554). 0.31 s prediction per VOI. |
| 26098860 | IDUN GPUQ, 8 CPUs | Final model E1b on the test split (289 VOIs), run once (D44) | V100-PCIE-16GB | 0.062 (script 0.058) | 146 | 0.008 | Completed. Mean test Dice 0.258 (centre component 0.418). 0.45 s prediction per VOI. |
| 26098861 | IDUN GPUQ, 8 CPUs | E2: E1b on all of Pancreas (120 VOIs) | A100-PCIE-40GB | 0.024 (script 0.020) | 165 | 0.003 | Completed. Mean Dice 0.001: the model finds almost none of the pancreatic tumours. 0.35 s prediction per VOI. |

## Totals and assumptions

Completed 2026-11-13, when all planned jobs were done (D48).

**Method.** The GPU energy is measured (the table above). The rest of each node and the data centre are estimated with the Green Algorithms model (Lannelongue et al. 2021):

energy = elapsed time × (cores × power per core + memory in GB × 0.3725 W/GB) × PUE, plus the measured GPU energy before the PUE is applied.

| Assumption | Value | Source and reasoning |
|---|---|---|
| CPU cores | 8 per job, fully used | The Slurm request. Full use is an upper bound. |
| Power per core | 7.5 W (low case 3.5 W) | Server CPU TDP per core. Intel Xeon Gold 6148: 150 W / 20 cores = 7.5 W. AMD EPYC 7742: 225 W / 64 cores = 3.5 W. The CPU model of each IDUN node was not recorded, so this is a range. |
| Memory power | 0.3725 W per GB requested | Green Algorithms (measured). Uses the requested memory, not MaxRSS: 32 GB, or 64 GB for E1b. |
| PUE (cooling, power distribution) | 1.67 (low case 1.2) | 1.67 is the 2019 global data-centre average used by Green Algorithms (Uptime Institute). IDUN's own PUE was not found. A Nordic centre is probably lower, so 1.67 is conservative. 1.2 is an assumed low case, with no source. |
| Home PC | PUE 1, same CPU and memory model | Green Algorithms recommends PUE 1 for a personal computer. |
| Elapsed time | Slurm elapsed | Covers the whole allocation, including start-up and idle time. |
| EV consumption | 19.1 kWh/100 km | EV Database: average real-world consumption over all listed electric cars, 191 Wh/km (accessed 2026-11-13). |

**Totals** (central case: 7.5 W per core, PUE 1.67):

| Group | Jobs | GPU, measured (kWh) | Estimated total (kWh) | EV distance |
|---|---|---|---|---|
| Training | E1, E1b | 2.57 | 5.51 | 29 km |
| Evaluation | val, training split, test and Pancreas, both models | 0.074 | 0.19 | 1 km |
| Development | Environment check, EDA (CPU), E0 on the home PC, timing runs, failed jobs | 0.105 | 0.19 | 1 km |
| **All** | | **2.75** | **5.9** | **31 km** |

- **Range:** 5.3 kWh (28 km) with 3.5 W per core, and 4.3 kWh (22 km) with PUE 1.2. So the whole project used roughly **4–6 kWh, about 20–30 km of EV driving**.
- **The two training runs are 94 % of it.** E1b alone, the targeted retrain, cost about 2.6 kWh (14 km).

**Not included:**
- development on the home PC (unit tests, sample checks, a few seconds of GPU each);
- light work on the IDUN login node (figures, `git`);
- the energy to build and cool the hardware (embodied energy);
- conversion to CO₂, because the assignment asks for energy.

**Inference cost:** 0.45 s per VOI on a V100 and 0.31–0.35 s on an A100-PCIE, about 0.02 Wh of GPU energy per VOI (146–165 W measured).
