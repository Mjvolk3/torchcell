---
id: orf79d684n1tp5o4ohj703a
title: 032 Amino Acid Module
desc: ''
updated: 1790184580070
created: 1790184580070
---

The metabolic module carried to Mulleder's 19 intracellular amino acids: its own experiment as of 2026.09.23, split out of [[experiments.028-gene-essentiality]] (rounds 118 to 129 there) at the user's request. W&B project `torchcell_032_amino_acid_module`; results in `experiments/032-amino-acid-module/results/`; packs in `conf/packs/`; the trainer is `experiments/028-gene-essentiality/scripts/train_essentiality.py` (shared), launched with `ESS_STUDY=032-amino-acid-module ESS_WANDB_PROJECT=torchcell_032_amino_acid_module ESS_WANDB_GROUP=gh_032_aa ESS_JOB_NAME=032-pack`. Scoring: 027's pinned split, epoch selected on validation Pearson (centered five-epoch moving average), mean per-column pinned-test Spearman over the 19 amino acids; every arm paired by seed (2700 to 2705).

## 2026.09.23 - Round map from 028, and state at the split

| round | 028 round | wave | arm | result |
|---|---|---|---|---|
| A1 | 118 | w135 | whole growth-vector affine vs control, width 12, old graph | 0.025 vs 0.034, -0.009 (2/6); ensembles 0.035 vs 0.058 |
| A2 | 121 | w138 | 19 synthesis ratios + growth ratio; growth vector + 42 summaries | -0.003 (2/6); -0.011 (2/6) |
| A3 | 124 | w141 | control on the deletion-only graph (the strain as built) | 0.049 vs 0.034, +0.015 (5/6), no collapsed seed; ensemble 0.062 |
| A4 | 126 | w143 | whole-vector affine on the deletion-only graph | 0.040 vs 0.049, -0.009 (1/6) |
| A5 | 122 | w139 | width-90 control (027's backbone) | running (jobs 2774, 2775) |
| A6 | 129 | w146 | A3's control at 300 epochs | pending (2776) |

State at the split: no form of the flux store beats its control (four forms, two controls). The graph correction is the only gain (+0.015). Comparators (027, same split): concentration readout 0.068, context-only 0.061, pooled 0.048, bilinear ridge 0.045. Records still inherit ARO4/ARO7/BTS1 from the other screens in the shared build; the deletion-only graph removes them from the cell graph but the aggregation is unchanged ([[torchcell.datasets.scerevisiae.mulleder2016]]).

## 2026.09.23 - Training-length audit figure (amino acids)

Produced by `experiments/031-betaxanthin-module/scripts/target_training_length_audit.py` (the module readouts serve both studies; findings in [[experiments.031-betaxanthin-module.scripts.target_training_length_audit]]).

![](./assets/images/032-amino-acid-module/training_length_audit_2026-09-23-12-34-18.svg)

## 2026.09.23 - Jobs resubmitted under 032; the essentiality finalists screened here

- A6 (w146, 300 epochs) resubmitted as job 2781 under this study (W&B project `torchcell_032_amino_acid_module`). A5 (w139_p2/p3, jobs 2774/2775) was already running under 028 and lands there; its result files move here.
- The essentiality finalists screened on the corrected graph, six seeds, paired against A3's control and A4's store affine: A7 (w150, job 2786) the flux vector with the six gene-rule columns beside it, store `ess_solve_encodings_SM_gem_pfbagpr.npz` (built 2026-09-23 with `isozyme_affine_store.py --medium SM --store-tag gem`); A8 (w151, job 2787) the cached solve head, hidden 256, over the SM_gem store.
