---
id: x6dnlwwdwm7m7yjb5gvrzcq
title: Scaling_grid_table_030
desc: ''
updated: 1791793126154
created: 1791793126154
---

## 2026.10.12 - The tables of the 030 scaling-study document

Script: `experiments/030-solid-growth-multi/scripts/scaling_grid_table_030.py`. Writes the four tables of `notes-tex/030-scaling-study/` from the grid's own files, so no number in the document's tables is typed by hand.

- `t1-sizes.tex`: the N ladder. Each size is instantiated as the trainer instantiates it (CellGraphTransformer at the width, depth and 9 heads of the grid, the per-entry dataset token) and the non-embedding count is read from the model's own `num_parameters`, the count `ComputeAccounting` logs. The 180x8 count is asserted equal to the value W&B holds for the instrumented IGB run 53qqijnu (3,587,350), and it is: 45x2 79,375; 63x3 202,612; 90x4 509,260; 126x6 1,378,444; 180x8 3,587,350. The ladder spans 1.66 decades.
- `t2-subsets.tex`: the D ladder from `subset_s3q_fractions_030_summary.json` and the Q split file: pairs, triples, doubles, records (the 025 unit), gene_interaction entry rows (the 030 unit), rows per record (1.56 at 1/64 to 2.24 at full), the submit script's epoch budget, records seen under it (5.12M / 8.26M / 9.78M / 10.46M) and hours at the ASSUMED 68 records/s pace (21 / 34 / 40 / 43 h), pending the calibration jobs.
- `t3-floors.tex`: the replicate floor per screen in label units, E = (1 - r) sd^2, from `triple_noise_ceiling_030.json`: 2020 same-strain r 0.44, E 2.25e-3 (56% of the score variance); 2018 bounded by Dango 0.59 (E 1.17e-3) and the cross-year 0.89 (E 0.31e-3).
- `t4-jobs.tex`: the 40 jobs of both submit scripts, mirrored from their loops (32 of the 030 grid including the two calibration jobs, 8 of the 025 pair).

The script also writes `results/scaling_grid_table_030.json`, which `notes/assets/drawio/030-scaling-study.gen.py` reads for the grid figure, so the figure and the tables agree.

```bash
PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/scaling_grid_table_030.py
```

The document: [[experiments.030-solid-growth-multi.scripts.scaling_readout_030]] for the readout it describes.
