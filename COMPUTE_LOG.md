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

## Totals and assumptions

To be filled in during Phase 4:

- Overhead factor for CPU, memory and cooling, with its source.
- EV consumption figure (kWh/100 km), with its source.
