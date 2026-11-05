# Compute log

Log every significant job here, including failed ones. GPU energy = mean sampled GPU power × elapsed time.
The training code records elapsed time and `nvidia-smi` power samples. On IDUN, also check with `sacct -j <jobid> --format=JobID,Elapsed,State,AllocTRES`.

| Job ID or run name | Machine | Experiment | GPU type | Elapsed (h) | Mean GPU power (W) | Energy (kWh) | Notes |
|---|---|---|---|---|---|---|---|

## Totals and assumptions

To be filled in during Phase 4:

- Overhead factor for CPU, memory and cooling, with its source.
- EV consumption figure (kWh/100 km), with its source.
