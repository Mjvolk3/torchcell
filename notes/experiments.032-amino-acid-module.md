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

## 2026.09.23 - Round A9: the MOMA store on the prototroph

- Same design as 031's B14 (see [[experiments.031-betaxanthin-module]] for the script and the direction-rule finding). Store `ess_solve_encodings_SM_gem_moma.npz` built by `experiments/031-betaxanthin-module/scripts/target_physics_store.py --target amino_acids --variant moma`: 1,161 genes in 29 s, 175 lethal against 161 in the parsimonious store, 63 growth ratios moved by more than 0.01, the log1p vector moved for 567 genes with a median relative distance of 0.0 (the SM parsimonious store is not degenerate, so MOMA and pFBA agree wherever the deletion does not force rerouting). Results `results/target_physics_store_amino_acids_moma.json`.
- The direction rule (sign10, sign30) leaves the prototroph infeasible on SM (84 and 44 fixed directions run against the parsimonious wild-type flux); `results/target_physics_store_amino_acids_sign10.json`, `_sign30.json`; no store written.
- A9 (w155, job 2793): A3's arm with the MOMA store in the affine slot, six seeds, paired against A3 (w141_aadel) and A4 (w143_aadelstore). Pack `conf/packs/w155_p0.pack`.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| A7 | 136 | w150 | flux vector with the six gene-rule columns | pending (2786) |
| A8 | 137 | w151 | cached solve head, hidden 256 | pending (2787) |
| A9 | 138 | w155 | linear MOMA store in the affine slot | pending (2793) |

## 2026.09.23 - CPU baselines over the store representations, and the second amino-acid screen

- `experiments/031-betaxanthin-module/scripts/target_store_baselines.py --target amino_acids` (results `results/store_baselines_amino_acids.json`, 15 min on 8 threads): every representation, every model, sits between 0.006 and 0.058 mean per-column Spearman on the 564 pinned genes (HiGHS parsimonious store kNN 0.045, ProtT5 forest 0.052, the two concatenated under a forest 0.058), against 0.045 for 027's bilinear ridge and 0.049 to 0.062 for the transformer arms. As on betaxanthin, 3,345 of 3,566 training genotypes are outside the model and read the wild-type row.

| representation | columns | ridge | kNN | forest |
|---|---|---|---|---|
| HiGHS parsimonious store (A4's store) | 4131 | +0.006 | +0.045 | +0.020 |
| linear MOMA store (A9's store) | 4131 | +0.018 | +0.014 | +0.032 |
| six gene-rule counts | 6 | +0.025 | +0.026 | +0.029 |
| eight flux statistics of the parsimonious build | 8 | +0.016 | +0.023 | +0.046 |
| ProtT5 | 1024 | +0.035 | +0.025 | +0.052 |
| HiGHS store with ProtT5 | 5155 | +0.033 | +0.050 | +0.058 |

- The second screen: Cooper 2010 does not corroborate Mulleder 2016 at the strain level (matched-pair Spearman -0.006 to +0.059 over 4,051 shared strains, Cooper's own duplicate strains at 0.06); full read in [[experiments.032-amino-acid-module.scripts.amino_acid_dataset_concordance]]. Together with Mulleder's retest at 0.20 Spearman, the amino-acid endpoint's strain-level rank is close to its noise floor, which is where every model above sits.

## 2026.09.23 - Round A11: the driving-force concentration store on the prototroph

- Same construction as 031's B17 (see [[experiments.031-betaxanthin-module]]): `target_driving_force_store.py --target amino_acids --max-abs-dg0 100`, 1,161 genes in 44 s. The wild type's max-min driving force on SM is -1.16 kJ/mol over 249 constrained carried reactions with 59 binding (phosphoglycerate dehydrogenase, glutamate 5-kinase, sulfate adenylyltransferase, triose-phosphate isomerase among them); 1,000 of 1,000 viable deletions negative, 200 move the optimum by more than 0.1 kJ/mol, 405 change the concentration vector. Uncapped the wild type sits at -325. Results `results/target_driving_force_store_amino_acids_mdfcap100.json`, `_mdf.json`; stores `ess_solve_encodings_SM_gem_mdfcap100.npz`, `_mdf.npz`.
- A11 (w158, job 2796): the `mdfcap100` store in the affine slot, six seeds, paired against A3 (w141_aadel) and A4 (w143_aadelstore). Pack `conf/packs/w158_p0.pack`.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| A11 | 141 | w158 | max-min driving-force concentration store (cap 100 kJ/mol) | pending (2796) |

- MEASURED, the ladder rerun with the concentration stores (`results/store_baselines_amino_acids.json`, 22 min): the driving-force concentrations read 0.013 to 0.019 under every model (capped 0.016 ridge, 0.016 kNN, 0.017 forest), inside the same 0.01 to 0.06 band as every other representation on this target.

## 2026.09.23 - Round A5 read (jobs 2774 and 2775, 7 h 36 min and 7 h 48 min)

- The wide control (width 90, two blocks, old graph): 0.034 (sd 0.010) against the width-12 control 0.034 (sd 0.025), +0.000 paired (sd 0.031, 3/6); six-seed ensemble 0.050 against 0.058 (corrected-graph control 0.062). Every seed escapes weakly (selected at epochs 10 to 59, validation Pearson 0.09 to 0.13); validation loss bottoms at epochs 6 to 21 and rises 0.05 to 0.07 by epoch 199. Width removes the collapsed seeds and tightens the spread, and does nothing to the level; the 027 cells of this form read 0.061 to 0.068 over three seeds, the six here 0.016 to 0.043. Written into `sec:target-transfer` as the amino-acid half of round 122; result files moved to `results/w139_aawide_h90l2p1_e200_s27*.json`.

## 2026.09.23 - Round A6 read (job 2781, 5 h 34 min)

- A6 (w146, deletion-only control for 300 epochs): 0.0485 (sd 0.009) vs the 200-epoch control (w141) 0.0492 (sd 0.007); paired -0.0007 (sd 0.003), 1/6; six-seed ensemble 0.060 vs 0.062. Four seeds select the same epoch and score in both runs (148, 88, 75, 37); the two whose peak moved past 200 (287, 217) read 0.036 vs 0.042 and 0.055 vs 0.054. Validation loss bottoms at epochs 20 to 169, up by at most 0.014 by epoch 299. The 200-epoch budget stands. Written into `sec:target-transfer` as round 129.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| A6 | 129 | w146 | deletion-only, 300 epochs | 0.049 (sd 0.009); ens 0.060 |
