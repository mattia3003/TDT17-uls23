# Compute log

Log every significant job here, including failed ones. GPU energy = mean sampled GPU power × elapsed time.
The training code records elapsed time and `nvidia-smi` power samples. On IDUN, also check with `sacct -j <jobid> --format=JobID,Elapsed,State,AllocTRES`.

| Job ID or run name | Machine | Experiment | GPU type | Elapsed (h) | Mean GPU power (W) | Energy (kWh) | Notes |
|---|---|---|---|---|---|---|---|
| 25934176 | IDUN GPUQ | Phase 0 environment check | V100-PCIE-32GB | < 0.01 | – | ≈ 0 | Passed. A few seconds of GPU use. |
| 25934982 | IDUN CPUQ, 8 CPUs | Phase 1 EDA (first try) | none | 0.057 | – | – (CPU only) | FAILED after scanning all VOIs. 1547 of 1618 VOIs were stored with an extra size-1 axis (D11). Elapsed 00:03:24, MaxRSS 16.7 GB. That is mostly file cache, since about 16 GB of VOIs were read. |
| 25936150 | IDUN CPUQ, 8 CPUs | Phase 1 EDA | none | 0.063 | – | – (CPU only) | Completed, 0 read errors. Elapsed 00:03:45, MaxRSS 16.8 GB (again about the 15.9 GB of data read, so file cache). |

## Totals and assumptions

To be filled in during Phase 4:

- Overhead factor for CPU, memory and cooling, with its source.
- EV consumption figure (kWh/100 km), with its source.
