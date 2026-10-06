---
id: r1aqq2qsahcrss2g5w4hy6u
title: V22_readout
desc: ''
updated: 1791273294325
created: 1791273294325
---

## 2026.10.06 - Readout of v22, and the v21 pace it exposed

Per run: epoch reached, trailing 20-epoch mean of validation Pearson per feature at a ladder of epochs, the registered window mean (1,000 to 1,200, only for a run that reached 1,199), spread ratio, normalized error, eval-mode train Pearson, seconds per epoch; paired differences against F_ref and, with `--delta-ref`, against v21's S_ref on Delta. Output `results/v22_readout.json`.

First run of it (02:52, everything PARTIAL, v22 at epochs 53 to 120): the `--delta-ref` rows showed all twelve v21 S_ref runs at epoch 69 with `perf/epoch_seconds` 176 to 178 s. At that pace 1,200 epochs take 59 h against the 48 h limit; a pack would time out near epoch 945, short of the registered window. The 22 to 24 h projection recorded on 2026-10-05 read the training progress bar (49 to 59 s), which covers the training batches only.
