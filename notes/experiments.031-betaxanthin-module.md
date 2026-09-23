---
id: 5g7elb30ydkrgcrhvi4r1pc
title: 031 Betaxanthin Module
desc: ''
updated: 1790184572426
created: 1790184572426
---

The metabolic module carried to Cachera's betaxanthin screen: its own experiment as of 2026.09.23, split out of [[experiments.028-gene-essentiality]] (rounds 117 to 132 there) at the user's request so the data can be reviewed per target. W&B project `torchcell_031_betaxanthin_module`; results in `experiments/031-betaxanthin-module/results/`; packs in `conf/packs/`; the trainer is still `experiments/028-gene-essentiality/scripts/train_essentiality.py` (shared code), launched through `freeze_and_submit.sh` with `ESS_STUDY=031-betaxanthin-module ESS_WANDB_PROJECT=torchcell_031_betaxanthin_module ESS_WANDB_GROUP=gh_031_bx ESS_JOB_NAME=031-pack ESS_JOB_OUTPUT=<slurm/output/%x_%j.out>`. Scoring protocol: 027's pinned Merzbacher 639 test genes, epoch selected on validation Pearson (centered five-epoch moving average), pinned-test Spearman and the top-50 enrichment of high producers reported; every arm paired by seed (2700 to 2705) against its control.

## 2026.09.23 - Round map from 028, and state at the split

Betaxanthin rounds, renumbered B1 onward (028 round in parentheses; every result read and published in the 028 chapter, section "Rounds 117 to 125"):

| round | 028 round | wave | arm | result |
|---|---|---|---|---|
| B1 | 117 | w134 | whole-vector store affine vs control, width 12, old graph | 0.121 vs 0.079, +0.043 (5/6); top-50 3.49 vs 4.06 |
| B2 | 119 | w136 | two summary columns; affine decayed and slowed | +0.023 (4/6); -0.005 (3/6) |
| B3 | 120 | w137 | deletion-only graph alone; six rule-count columns alone | -0.020 (2/6); -0.004 (2/6) |
| B4 | 122 | w139 | width-90 control (027's backbone), old graph | 0.109 vs 0.079, +0.030 (4/6); top-50 4.73 |
| B5 | 123 | w140 | store on the deletion-only graph (width 12); store at width 90 (old graph) | 0.121, +0.063 vs graph alone (6/6), top-50 4.93 / 5.59 ensembled; 0.134, +0.025 vs wide control (4/6), ensemble 0.153, top-50 4.09 |
| B6 | 125 | w142 | growth block alone; production block alone | 0.033 (-0.088 vs whole, 0/6); 0.091 (-0.030 vs whole, 1/6) |
| B7 | 127 | w144 | seeds 2706 to 2711 of B5's deletion-only store | twelve-seed ensemble 0.129 / 5.39 vs first six 0.131 / 5.59: saturated |
| B8 | 128 | w145 | store on the deletion-only graph at width 90 | running (jobs 2772, 2773) |
| B9 | 130 | w147 | B5's arm with the affine frozen to epoch 40 | pending (2777) |
| B10 | 131 | w148 | B5's arm with the affine from the backbone's escape (val r 0.15) | pending (2778) |
| B11 | 132 | w149 | width-90 control on the deletion-only graph | pending (2779, 2780) |

Best cell at the split: B5's store on the corrected graph at width 12 (0.121 single-seed, 4.93 top-50; 0.131 / 5.59 ensembled). Best rank: B5's store at width 90 (0.153 ensembled) with a worse call. Comparators (027, same split): precursor readout 0.131, context-only 0.115, cosine kNN 0.145, FCL random forest 0.039.

## 2026.09.23 - Jobs resubmitted under 031; the essentiality finalists screened here

- User direction: the best module from the essentiality study need not be the best here, so the finalists are screened on each target rather than assumed. Screened at width 12 on the corrected graph, six seeds, paired against B3's deletion-only control and B5's store affine: B12 (w152) the flux vector with the six gene-rule columns beside it (028's best single model, 0.934 vs 0.920 for the vector alone), store `ess_solve_encodings_YPD_bx_pfbagpr.npz`; B13 (w153) the cached solve head, a hidden-256 MLP over the store in place of the affine (028 round 57 form). Not portable yet: the graph-convolution-on-GO family (`dcell_style_baseline.py` has no betaxanthin or amino-acid head) and the two-family average that depends on it.
- Pending packs cancelled and resubmitted under this study (W&B project `torchcell_031_betaxanthin_module`, logs in `slurm/output/031-pack_<job>.out`): B9 w147 -> job 2782, B10 w148 -> 2783, B11 w149_p0/p1 -> 2784/2785. B8 (w145, jobs 2772/2773) was already running under 028 and lands there; its result files move here when it finishes.

## 2026.09.23 - Round B14: a physically constrained store, and what the direction rule measured

- User direction (14:30): the primary aim is a physically grounded metabolic module (mass balance, energy, no violation of the second law, kinetic caps, FBA or pFBA solves or polytope samples), so the store affine is screened with other programs in the same slot. New script `experiments/031-betaxanthin-module/scripts/target_physics_store.py` builds a store on the TARGET model (the Cachera strain with the cassette and the BY4741 markers deleted on YPD; the prototroph on SM for 032) over the source store's gene list and reaction order, so a paired cell isolates the program. Variants: `moma` (linear minimization of metabolic adjustment against the source's parsimonious wild type, one HiGHS LP per gene) and `sign10` / `sign30` (two-stage pFBA with reaction directions fixed by the eQuilibrator union-external standard energies at 10 or 30 kJ/mol).
- MEASURED, the direction rule: the wild type is INFEASIBLE under both thresholds on both target models, as it was on the essentiality model (028's `pool125sign` and `pool125sign30` summaries record `wild_type_infeasible: true`; no cell was ever run on them). On the cassette strain the 10 kJ/mol rule fixes 1,754 reactions forward and 1,068 reverse and 97 of those run against the parsimonious wild-type flux; at 30 kJ/mol 1,071 / 905 / 69. On SM: 84 and 44 conflicts. Results `results/target_physics_store_betaxanthin_sign10.json`, `_sign30.json` (032: `_amino_acids_sign10/30.json`); no store is written for an infeasible parent. Reading: standard reaction energies at 1 M cannot be applied as fixed directions; a thermodynamically consistent module needs concentration variables (thermodynamic metabolic flux analysis, a MILP with log-concentration bounds), which is not built.
- MEASURED, the MOMA store: `ess_solve_encodings_YPD_bx_moma.npz`, 1,155 genes in 27 s (8 processes), 107 lethal against 93 in the parsimonious store, 59 genes whose growth ratio moves by more than 0.01, and the log(1+|v|) vector moves for 1,056 genes with a median relative distance of 0.39 to the parsimonious deletion vector (`results/target_physics_store_betaxanthin_moma.json`).
- MEASURED, the source store is degenerate: in `ess_solve_encodings_YPD_bx.npz` (cobra pFBA through GLPK), 1,025 of the 1,031 genes whose deletion leaves growth unchanged have a deletion vector that differs from the wild type's, median relative log1p distance 0.38; in the SM store (HiGHS `parsimonious_reference_flux`) 324 of 918 differ and the median distance is 0.0. Hypothesis (untested which): the GLPK solves return different vertices of the degenerate parsimonious optimum per gene. The MOMA store is anchored to one wild-type vector and carries none of that, so B14 also tests whether the B5 gain survives a store without vertex noise.
- B14 (w154, job 2792): B5's arm with the MOMA store in the affine slot, six seeds, width 12, corrected graph, the growth-ratio CSV unchanged; paired against B3 (w137_bxdel) and B5 (w140_bxdelstore). Pack `conf/packs/w154_p0.pack`; readouts extended (`target_transfer_readout.py` pairs, `target_ensemble_readout.py` arms).

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B12 | 133 | w152 | flux vector with the six gene-rule columns | pending (2789) |
| B13 | 134 | w153 | cached solve head, hidden 256 | pending (2788) |
| B14 | 135 | w154 | linear MOMA store in the affine slot | pending (2792) |
