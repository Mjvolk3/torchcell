---
id: y83hnr2rd0l7lpqs6nn7c57
title: Mle_dist_supcr
desc: ''
updated: 1752609699927
created: 1752609699927
---

## 2026.09.27 - The dist and SupCR terms changed underneath this loss

`MleDistSupCR` builds its `dist` term on `WeightedDistLoss` and its SupCR term on `WeightedSupCRCell` from `multi_dim_nan_tolerant.py`. The fix PR after Phase 6 of [[plan.test-suite-buildout.2026.09.25]] changed both: the soft-sort gradient sign, the pairing of sorted predictions with the theoretical labels, the default dist weight normalization, and the SupCR tie rule (details in [[torchcell.losses.multi_dim_nan_tolerant]]). In this file the `use_buffer=False` branch now applies the scheduled temperature to the unbuffered cell, which it computed and logged but never used. A training run with `lambda_dist > 0` or `lambda_supcr > 0` before and after this change optimizes a different objective; the experiments whose configs actually selected those terms are listed in [[test-campaign.2026.09.25]].

## 2026.10.01 - Schedule validation and per-dimension zero widths; two buffer items left open (#529)

- `TemperatureScheduler` refuses a schedule outside `TEMPERATURE_SCHEDULES` (cosine, exponential) at construction: `Unknown temperature schedule 'linear'; valid schedules: cosine, exponential`. Before, any other name held `init_temp` for the whole run. `MleDistSupCR` with `use_temp_scheduling=True` builds the scheduler in its constructor, so it refuses there too. Every committed `mle_dist_supcr` config uses exponential or cosine.
- A switched-off term (lambda 0) logs `torch.zeros(targets.size(1))` and the buffered terms below `min_samples` return one zero per target or label column, the width an active term logs. Before, all five places logged `torch.zeros(2)`.

Left open, because neither the module docstring nor this note defines the intended formula:

- At buffer weight 1 the dist term computes D([b; B_t]) with B_t the buffer after b was written, so b is counted twice. Candidates: D([b; B_(t-1)]) (read before write) or D(B_t) (the buffer alone). No committed config reaches weight 1 with a buffer: the 42 buffered `mle_dist_supcr` configs in `experiments/006-kuzmin-tmi/conf/` all use adaptive weighting, which tops out at 0.9. Below 1 the random buffer subsample can also draw rows of b.
- The buffered SupCR computes (1 - 0.5 w) * S([z; Z]), scaling the current-batch pairs too. The candidate that matches the comment "Reduce buffer influence" weights only the terms that touch a buffer row by (1 - 0.5 w). Five configs train with the current factor: 006 cabbi_014, cabbi_016, cabbi_017, mmli_015, mmli_018 (lambda_supcr 0.001 to 0.01).

Tests: [[tests.torchcell.losses.test_mle_dist_supcr]].
