---
id: h4ee4mplr6ecesq61g1skvb
title: Build_supported_datasets_table
desc: ''
updated: 1792305983436
created: 1792305983436
---

## 2026.10.18 - The scan covers KG 4.0's 70 yeast datasets, in parallel

The registry now lists every *S. cerevisiae* dataset the served release `2026.10.17-4295f7ca` carries: the 19 datasets new in KG 4.0 were added (genotype and environment counts from the 2026-10-14 probe cache of `build_bacteria_datasets_table.py`), a ninth section `Genome architecture` holds Puddu 2019, and the two SynLethDB rows point at the stores the loaders build (`syn_leth_db_yeast`, `syn_rescue_db_yeast`; the old subpaths no longer exist, so those two rows had read as not built). Every yeast dev store had been rebuilt for 4.0, so the signal cache held none of them and the whole set was rescanned.

`--workers N` scans the stores the cache misses in a process pool, largest first, and persists each result to `dataset_signals_cache.json` as it lands, so a killed run resumes; the serial pass then hits the cache for all of them. `build_supported_datasets_table.slurm` runs it under slurm (`sbatch --export=ALL,TORCHCELL_SRC=<worktree> ...`).

Measured (job 5777, 24 workers, 128 G reserved): 47 stores, 263.5 GB, 13 min 34 s wall clock, MaxRSS 65.2 GB; the two Costanzo 2016 stores (78 GB each) finished last. Output `experiments/database/results/pre-build/2026-10-18/supported_datasets.json`: 70 rows, all built, all with a signal. The job sat PENDING(Priority) at a 12 h limit behind the 031 GPU packs' reservation and started within seconds of `scontrol update TimeLimit=2:10:00` ([[gilahyper-fifo-hold-release-to-reorder]]).

The figure: [[experiments.database.scripts.plot_supported_datasets_signal]]. The paper views (`render_supported_datasets_table.py`, the SI `datasets_table.tex`) were not re-rendered from this snapshot; that is a manuscript change for its own PR.
