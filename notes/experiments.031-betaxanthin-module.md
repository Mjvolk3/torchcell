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

### Round B15: the parsimonious store rebuilt on one solver path

- `target_physics_store.py --variant pfba` rebuilds the parsimonious store on the cassette strain through the essentiality stores' HiGHS two-stage solve: 1,155 genes in 78 s, 93 lethal (the same 93), 17 growth ratios moved by more than 0.01 against the GLPK source, the log1p vector moved for 1,062 genes with a median relative distance of 0.015 (`results/target_physics_store_betaxanthin_pfba.json`). Store `ess_solve_encodings_YPD_bx_pfba.npz`.
- Degeneracy restated with both stores measured: among the 1,031 deletions that leave growth unchanged, the GLPK store's vector differs from its wild type's for 1,025 (median relative log1p distance 0.38), the HiGHS store's for 428 (median 0.0, ninetieth percentile 0.39). So the parsimonious optimum has alternative vertices for a large share of no-effect deletions under either solver, and GLPK lands on a different one almost every time. The MOMA store returns the wild-type vector exactly whenever the knocked reactions carry no wild-type flux.
- B15 (w156, job 2794): B5's arm with the HiGHS store in the affine slot, six seeds, paired against B3 (w137_bxdel) and B5 (w140_bxdelstore). With B14 this reads whether the B5 gain rests on the vector or on its vertex noise: if B15 matches B5 the solver path is immaterial; if B14 and B15 both fall below B5, the noise is doing work (a hypothesis until measured).

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B15 | 139 | w156 | HiGHS parsimonious store in the affine slot | pending (2794) |

## 2026.09.23 - CPU baselines over the store representations (user: run CPU-only baselines beside the packs)

- Script `experiments/031-betaxanthin-module/scripts/target_store_baselines.py` (results `results/store_baselines_betaxanthin.json`, 8 min on 8 threads): each deleted gene represented by a store row (log1p of the flux columns, standardized on train), the 027 ladder's ridge and cosine kNN plus a 300-tree random forest (the Flux Cone Learning form on one point), selected on validation per-feature Pearson, scored on the 629 pinned test genes. A gene the store does not cover reads the wild-type row: that is 3,255 of 3,494 training genotypes (the screen is genome-wide, the stores cover the 1,155 GEM genes), so every store baseline learns from about 240 in-model training deletions while the pinned test set is inside the model (628 of 629).

| representation | columns | ridge (rho / top-50) | kNN | forest |
|---|---|---|---|---|
| GLPK parsimonious store, the B5 vector (growth, production, total) | 8355 | -0.005 / 1.80 | +0.108 / 3.19 | +0.027 / 3.19 |
| GLPK growth block alone | 4177 | -0.007 / 1.00 | -0.004 / 0.60 | +0.009 / 2.40 |
| HiGHS parsimonious rebuild (B15's store) | 4177 | +0.098 / 2.00 | **+0.178 / 3.39** | +0.116 / 1.00 |
| linear MOMA (B14's store) | 4177 | +0.134 / 2.20 | +0.091 / 1.00 | +0.072 / 1.00 |
| six gene-rule counts | 6 | +0.059 / 2.00 | +0.035 / 1.40 | +0.056 / 1.40 |
| six flux statistics of the parsimonious build | 6 | +0.107 / 1.80 | +0.146 / 2.40 | +0.102 / 0.80 |
| ProtT5 (027's ladder, reproduced: kNN 0.1446) | 1024 | +0.082 / 3.00 | +0.145 / 4.59 | +0.143 / 4.39 |
| GLPK store and ProtT5 concatenated | 9379 | +0.044 / 2.00 | +0.068 / 2.00 | +0.121 / 4.99 |

- Reading. The GLPK growth block carries nothing on its own (three models within 0.01 of zero); the same program rebuilt with HiGHS carries 0.178 by kNN, the highest single number on this split so far (the best transformer single arm is 0.121, the best six-seed ensemble 0.153, and 027's kNN over ProtT5 0.145). Hypothesis (untested): the vertex noise measured this afternoon (1,025 of 1,031 no-effect deletions differ from the wild type in the GLPK store) is what erases the growth block's signal, and B5's gain came from the production block and the total demand, which the GLPK store alone carries at 0.108 by kNN. The forest is not the best model on any flux representation here, so the Flux Cone Learning form does not rescue the point representation. Rank and call separate again: ProtT5 gives the best top-50 (4.6 to 5.0 with the store concatenated), the clean flux store the best rank.
- Consequence for the module: B15 (job 2794) is the transformer read of the same HiGHS store; if it beats B5 the affine was reading through the noise, and the store should be rebuilt on one solver path for every target from now on.

## 2026.09.23 - Round B17: the metabolite side, a max-min driving-force concentration store

- User question (16:50): the left and right null spaces of Palsson 2015 as a metabolite formulation, and whether steady-state concentrations can be solved with the fluxes. Read from the mirror (`palssonSystemsBiology2015`, sections 9.4.2, 9.4.3, chapter 11): the left null space holds the conservation relationships (pools, combinations of concentrations that are time-invariant), the null space the pathways, and free energy sits in the row space (Delta_r G = S^T mu). Stoichiometry alone does not give steady-state concentrations; the row-space relation does with concentrations as variables: the max-min driving force (Noor 2014) or thermodynamic flux analysis.
- New script `experiments/031-betaxanthin-module/scripts/target_driving_force_store.py`: per deletion, from the HiGHS parsimonious flux pattern, two LPs in log concentration: maximize the smallest driving force -(Delta_r G'0 + RT S^T ln c) over the carried, priced reactions within 1 uM to 10 mM, then the l1-nearest concentrations to the center of the bounds at that optimum. Water and protons at unit activity. Two exclusions, both recorded in the provenance: transport reactions (same species on both sides; MEASURED 505 of 1,041 priced transports carry |Delta_r G'0| above 50 kJ/mol, which no membrane can supply) and proton-translocating reactions (ATP synthase, the respiratory complexes; their driver is the proton-motive force the table lacks); `--max-abs-dg0 100` sets aside the 1,278 priced reactions above 100 kJ/mol (lipid and cofactor species mostly) for the `mdfcap100` variant.
- MEASURED (`results/target_driving_force_store_betaxanthin_mdfcap100.json`, 28 s for 1,155 genes): the wild type's max-min driving force is -1.16 kJ/mol over 172 constrained carried reactions with 32 binding (CDP-diacylglycerol synthases, the malate dehydrogenases, phosphoglycerate dehydrogenase, GAPDH), so the parsimonious wild-type flux is thermodynamically consistent to within about a kilojoule once concentrations are free; 1,062 of 1,062 viable deletions have a negative optimum, the median at the wild type's -1.16, the tenth percentile -5.4, the minimum -41.4; 250 deletions move it by more than 0.1 kJ/mol and 451 change the concentration vector. Uncapped (`_mdf.json`) the wild type sits at -240, set by r_4212 (the lumped pyridoxal-phosphate synthesis, Delta_r G'0 -308, which the parsimonious solution runs BACKWARD, a directionality defect of yeast-GEM 9.0.2 the second law exposes) and by lipid reactions priced at +211 to +268 (PI synthase, PS decarboxylase, dolichyl-phosphate mannosyltransferase), which are pricing artifacts on the written-structure lipids. Stores `ess_solve_encodings_YPD_bx_mdfcap100.npz` and `_mdf.npz` (2,831 concentration columns in mM plus the driving force, the count of constrained reactions and the count binding).
- B17 (w157, job 2795): the `mdfcap100` concentration store in the affine slot, six seeds, paired against B3 (w137_bxdel) and B5 (w140_bxdelstore). Pack `conf/packs/w157_p0.pack`; readouts extended; the CPU ladder (`target_store_baselines.py`) now reads `mdf_conc`, `mdfcap100_conc` and, once built, `sampled_mean`.
- User remark: since the program returns a region, the representation could be probabilistic. Agreed and not built: hit-and-run sampling of the concentration polytope under the driving-force constraints gives a distribution per metabolite per genotype (mean and spread, as the sampled-flux spread store does for fluxes); probabilistic thermodynamic analysis would add the standard-energy covariance, which the current table does not carry (median sigma 4e-15 over 1,755 reactions).

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B17 | 140 | w157 | max-min driving-force concentration store (cap 100 kJ/mol) | pending (2795) |

- MEASURED, the ladder rerun with the concentration stores (`results/store_baselines_betaxanthin.json`, 9 min): the capped driving-force concentrations read 0.151 by ridge (top-50 3.39), 0.082 by kNN and 0.098 by forest on the 629 pinned genes; the uncapped ones 0.082 / 0.089 / 0.063. So the metabolite side carries the rank linearly, second only to the HiGHS flux store under kNN (0.178) and above every transformer single arm (best 0.121), from 2,834 columns of which 451 rows differ from the wild type's. B17 is the transformer read of the same store.

## 2026.09.23 - Round B8 read (jobs 2772 and 2773, 7 h 04 min and 7 h 31 min)

- The store on the corrected graph at width 90: 0.120 (sd 0.024) against the old-graph wide control 0.109 (+0.011, sd 0.067, 3/6) and against B5's width-12 store on the same graph 0.121 (-0.001, sd 0.035, 3/6); top-50 4.43 vs 4.73 and 4.93; six-seed ensemble 0.147 / 4.99 (old-graph wide store 0.153 / 5.19; width-12 corrected store 0.131 / 5.59). Every seed selected off the plateau (epochs 21 to 50, validation Pearson 0.22 to 0.28) where the old-graph wide store had four plateau seeds; validation loss bottoms at epochs 8 to 42 and rises 0.09 to 0.15 by epoch 199. Width and store do not add on the strain as built. Written into `sec:target-transfer` (round 128); result files moved from 028/results to `results/w145_bxwidedelstore_h90l2p1_e200_s27*.json`.

## 2026.09.23 - Round B16: the sampled cone in the affine slot

- The OptGP sampler (`experiments/031-betaxanthin-module/scripts/target_sampled_flux_store.py --target betaxanthin --samples 6 --thinning 10 --growth-fraction 0.99 --processes 12`) finished at 21:20 after 360.8 min nice'd on the node: 1,155 deletions, 93 lethal (zero rows), the median relative distance of the six-sample mean to the parsimonious point 5.6 (`results/target_sampled_flux_store_betaxanthin.json`). Stores `ess_solve_encodings_YPD_bx_sampledmean.npz` (4,177 columns) and `_sampledspread.npz` (8,354: the mean and the spread).
- B16 (w159, job 2797): the sampled mean in the affine slot, six seeds, paired against B3 (w137_bxdel) and B5 (w140_bxdelstore). Pack `conf/packs/w159_p0.pack`; readouts extended. On essentiality the sampled stores read within noise of the parsimonious point (round 87); on betaxanthin the CPU ladder reads the same store first (`sampled_mean` rows, rerun started 21:22).

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B16 | 142 | w159 | sampled-cone mean store (six OptGP samples at 99 percent of optimum) | pending (2797) |

## 2026.09.23 - Rounds B9 and B10 read (jobs 2782 and 2783, 3 h 52 min and 3 h 46 min)

- B9 (w147, affine held at zero until epoch 40): 0.074 (sd 0.046) vs the store arm 0.121 (-0.047, 2/6) and vs the affine-free control 0.058 (+0.016, 4/6); top-50 3.86; six-seed ensemble 0.129 / 5.79. `affine_start_epoch_actual` is 40 in every seed and the selected epochs are 10, 32, 76, 34, 25, 38: five of six select BEFORE the affine starts, so the selected cell is the affine-free trajectory in five seeds. The early validation peak is not overtaken once the affine arrives.
- B10 (w148, affine released from epoch 5 once the trailing validation Pearson reaches 0.15): releases at 5, 6, 25, 31, 6, 40; selected at 2, 57, 35, 32, 78, 38 (four after release). 0.110 (sd 0.050), -0.010 vs the store arm (1/6), +0.052 vs the control (6/6); top-50 4.66; ensemble 0.120 / 6.19 (highest ensemble call of any betaxanthin arm). Seed 2705 released at epoch 40 in B10, so it is the same trajectory as B9's seed 2705 (0.016 at epoch 38 in both): a check that the schedule, not the seed, separates the rounds.
- Read: the store gain does not come from the first epochs. A release rule on the validation curve keeps the store arm's level and call; a fixed late start loses the level because the selected epoch comes first. Written into `sec:target-transfer` as rounds 130 and 131 (599 verifier checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B9 | 130 | w147 | del + store, affine from epoch 40 | 0.074 (sd 0.046); ens 0.129 / 5.79 |
| B10 | 131 | w148 | del + store, affine from escape (val Pearson 0.15) | 0.110 (sd 0.050); ens 0.120 / 6.19 |

## 2026.09.23 - The store ladder made reproducible

- Two runs of `target_store_baselines.py --target betaxanthin` disagreed (HiGHS kNN 0.178 then 0.179 with k 10 then 3; capped concentrations kNN 0.082 then 0.108) while the forest rows were bitwise identical. Cause, measured: 57 percent of the pinned genes read a row identical to the wild type's, so (a) a kNN query on the wild-type row has about 2,000 identical training rows and `argsort` picked an arbitrary k of them, (b) ridge predictions on the no-effect rows agree to 1e-12 and reorder inside the tie, (c) the validation target is 98 percent one value, so the grid's validation Pearson tied and the argmax flipped on 1e-13 noise.
- Fix (same script): every training row tied with the k-th cosine similarity (within 1e-9) is averaged in; predictions are rounded to 1e-9 before ranking; the validation criterion is rounded to 1e-6 with ties going to the later grid point. Two consecutive runs now give identical score lines. New betaxanthin rows: HiGHS store 0.195 by kNN (top-50 3.39), capped driving-force concentrations 0.152 by ridge, MOMA 0.135 by ridge, flux stats 0.145 by kNN, ProtT5 0.144 by kNN; sampled-cone mean 0.007 / 0.064 / 0.102 (ridge / kNN / forest), below the point it is sampled around. Chapter prose and verifier updated; the amino-acid ladder rerun under the same script.

## 2026.09.24 - The study's own notes-tex document

- User (03:00): the target work no longer belongs in the essentiality chapter. The betaxanthin sections (rounds 117 onward, the physically constrained stores on the strain, the store ladder) moved verbatim into `notes-tex/031-betaxanthin-module/` (sections `1-endpoint`, `2-rounds`, `3-physics`, `4-standing`; tables generated by `experiments/031-betaxanthin-module/scripts/target_notes_tex_tables.py`, which serves both target documents; prose numbers checked by `experiments/031-betaxanthin-module/scripts/verify_doc_numbers.py`, 174 checks). The essentiality chapter keeps a one-paragraph pointer under `sec:target-transfer`. The amino-acid half is `notes-tex/032-amino-acid-module/`. First published version 2b802c9f (Zotero collection torchcell/notes-tex/031-betaxanthin-module). Build: `make && make check` in the document directory; publish with `notes-tex/common/zotero_publish.py notes-tex/031-betaxanthin-module`.

## 2026.09.24 - The driving-force polytope sampled: a probabilistic concentration store

- User (2026-09-23 17:30): since the max-min program returns a region, represent it probabilistically. Built: `experiments/031-betaxanthin-module/scripts/target_concentration_polytope_store.py`. Per deletion, stage 1 of the point program gives B*; the region is every log-concentration vector within 1 uM to 10 mM whose carried priced reactions (the point store's set: non-boundary, non-transport, under the 100 kJ/mol cap) all run downhill by at least B* minus a 1 kJ/mol slack; metabolites in no constrained reaction are uniform in the bounds; the rest are drawn by coordinate hit-and-run from the stage-2 point (burn-in 20,000, thinning 2,000, six draws). Random-direction hit-and-run stalled on rows whose start point sits on the box (the wild type's constrained metabolites had zero spread); one axis per step does not. The chain is seeded from a hash of the polytope, not the row, so two deletions with the same region read the same row (seeding by row gave every viable row its own sampling noise: 1,062 of 1,062 differed from the wild type; under the polytope hash 455 do, against the point store's 451).
- MEASURED (`results/target_concentration_polytope_store_betaxanthin_mdfcap100poly.json`, 88 s on eight processes): the wild type's region constrains 234 of 2,803 variable metabolites; over them the median sd of ln c across draws is 2.18 against 2.63 for a free metabolite (the hash-seeded rebuild; the row-seeded first build read 2.25 and 2.64) (uniform over the bounds), so the second law at this cap narrows a typical constrained metabolite only slightly, and the information sits in the few metabolites it pins. Median relative distance of the mean to the point 0.12. Stores `ess_solve_encodings_YPD_bx_mdfcap100polymean.npz` (2,807 columns: mean mM plus mdf, n_active, n_binding, n_constrained_metabolites) and `_mdfcap100polyspread.npz` (plus the sd of ln c per metabolite). The first builds (row-seeded) are in the scratchpad.
- B20 (w164, job 2801): the polytope mean in the affine slot, paired against B3 (w137_bxdel), B5 (w140_bxdelstore) and B17 (w157_bxdelmdf, the point). Ladder rows `mdfcap100_poly_mean` and `mdfcap100_poly_spread` added (`target_store_baselines.py`), rerun started 03:55.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B20 | 147 | w164 | driving-force polytope, mean over draws | pending (2801) |

## 2026.09.24 - Round B11 read (jobs 2784 and 2785, 7 h 18 min and 7 h 08 min)

- B11 (w149, width 90 on the deletion-only graph, no store): 0.092 (sd 0.043) vs the old-graph wide control (B4, w139) 0.109 (-0.017, sd 0.080, 2/6); top-50 4.49 vs 4.73; ensemble 0.122 / 5.19 vs 0.134 / 5.59. Every seed off the plateau (val Pearson 0.22 to 0.29, epochs 19 to 56); validation loss rises 0.08 to 0.17 by epoch 199, the largest of any arm. Paired on the square's second diagonal, B8 (w145, the store on this graph at width 90) reads +0.028 (sd 0.037, 4/6) against B11, with top-50 4.43 vs 4.49: the store adds the same 0.03 at width 90 as at width 12, and no wide cell beats the width-12 store (0.121 / 4.93). Written into the 031 document as round 132 (202 checks; version 3).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B11 | 132 | w149 | width 90, deletion-only graph, no store | 0.092 (sd 0.043); ens 0.122 / 5.19 |

## 2026.09.24 - Round B13 read (job 2788, 3 h 42 min)

- B13 (w153, the cached solve head, hidden 256, over the store instead of the affine): 0.070 (sd 0.049) vs the deletion-only control 0.058 (+0.012, sd 0.074, 3/6) and vs the store affine 0.121 (-0.051, sd 0.061, 1/6); top-50 3.89; ensemble 0.095 / 5.39. Seeds 2700 and 2704 never leave the constant predictor (validation Pearson peaks -0.024 and 0.003; selected at epoch 2; 0.038 and 0.005); the four that escape read 0.043 to 0.136. The head that recovered the affine's loss on the amino acids (A8) loses the affine's gain here and brings the plateau back: a zero-init affine cannot hold the backbone down, a random perceptron over 8,355 columns can. Written into the 031 document as round 134.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B13 | 134 | w153 | del + solve head 256 | 0.070 (sd 0.049); ens 0.095 / 5.39 |

## 2026.09.24 - Round B14 read (job 2792, 3 h 37 min)

- B14 (w154, the linear MOMA store in the affine slot): 0.107 (sd 0.018) vs the deletion-only control 0.058 (+0.049, sd 0.064, 5/6) and vs the parsimonious (GLPK) store 0.121 (-0.014, sd 0.029, 1/6); top-50 4.99 (the highest single-seed call of any arm) vs 3.39 and 4.93; ensemble 0.115 / 5.79. Five seeds off the plateau (epochs 30 to 59); seed 2704 on it at epoch 3 still reads 0.098. A store anchored to one wild-type vector (18 no-effect rows differ vs 1,025) reads the parsimonious store's rank within the seed spread with half the spread and a better call. Written into the 031 document as round 135.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B14 | 135 | w154 | del + linear MOMA store | 0.107 (sd 0.018); ens 0.115 / 5.79 |

## 2026.09.24 - Round B12 read (job 2789, 3 h 45 min)

- B12 (w152, the vector with the six rule-count columns, the essentiality best single store): 0.114 (sd 0.019) vs the deletion-only control 0.058 (+0.056, sd 0.056, 5/6) and vs the plain store 0.121 (-0.007, sd 0.023, 1/6); top-50 3.96 vs 3.39 and 4.93; ensemble 0.137 / 4.79, the highest six-seed rank of any width-12 arm. Five seeds off the plateau (epochs 47 to 91); seed 2704 on it to epoch 183 reads 0.118. The columns keep the rank, cost the single-seed call, add to the ensemble. Written into the 031 document as round 133.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B12 | 133 | w152 | del + vector with rule counts | 0.114 (sd 0.019); ens 0.137 / 4.79 |

## 2026.09.24 - Round B21 queued: the release rule on the MOMA store

- B21 (w166, job 2804): B10's schedule (affine frozen until epoch 5 and until the trailing validation Pearson reaches 0.15) on B14's linear MOMA store, deletion-only graph, six seeds; paired against B14 (w154, affine from epoch 0) and B10 (w148, the GLPK store under the same rule). Round 149.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B21 | 149 | w166 | del + MOMA store, affine on escape | pending (2804) |

## 2026.09.24 - The annotation-only family: GO carries the target

- `experiments/031-betaxanthin-module/scripts/target_go_models.py` (CPU, the ladder's split and selection): DCell's hierarchy (the essentiality shared-input form, a linear head on the root state, auxiliary regressions at 0.3) and a one-hidden-layer perceptron (256) over the propagated GO multi-hot (2,341 columns with train support), the HiGHS store row, ProtT5, and their concatenations; six seeds, 200 epochs, full batch, Adam 1e-3, MSE on standardized targets. MEASURED (`results/go_models_betaxanthin.json`): GO perceptron 0.200 (sd 0.002), top-50 4.79, selected at epoch 21 or 22, three seconds a seed; store row 0.119 / 2.00; ProtT5 0.103 / 4.13; GO + store 0.198 (sd 0.041) / 4.86; GO + ProtT5 0.173 / 4.99 (ensemble 5.39); all three 0.175 / 5.03; DCell hierarchy 0.049 (sd 0.043) / 3.10. GO alone beats every transformer arm (best single 0.121, best ensemble 0.153) and the HiGHS kNN (0.195). Caveat: GO is curated in part from deletion phenotypes (respiratory function among them) and the top hits are respiratory; the split holds out genes, not annotations. A forked two-worker pool deadlocked (torch threads before fork); the runs are in-process, one worker.
- `target_go_affine_store.py`: the GO multi-hot as affine stores over every annotated gene (6,607 genes, 5,501 annotated), `ess_solve_encodings_YPD_bx_go.npz` (2,362 columns) and `_gopfba.npz` (HiGHS flux vector, wild-type row outside the GEM, plus GO; 6,539 columns). B23 (w167, job 2805): GO alone in the affine slot vs B3 and B5. B24 (w168, job 2806): GO appended to the HiGHS store vs B15 (w156) and B23. Written into the 031 document (`3-physics.tex`, table `go_models`; 256 checks; version 5).

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B23 | 150 | w167 | del + GO multi-hot affine | pending (2805) |
| B24 | 151 | w168 | del + HiGHS store with GO appended | pending (2806) |

## 2026.09.24 - Round B15 read (job 2794, 3 h 39 min)

- B15 (w156, the HiGHS parsimonious store in the affine slot): 0.130 (sd 0.034) vs the deletion-only control 0.058 (+0.072, sd 0.067, 5/6) and vs the GLPK store 0.121 (+0.009, sd 0.041, 5/6); top-50 4.46; ensemble 0.148 / 5.79, the highest width-12 ensemble rank (the width-90 store 0.153). Seed 2704 on the plateau (epoch 2) reads 0.160, the highest single cell of any arm: the affine's linear read of the clean vector alone (the GLPK store's plateau cells 0.12 to 0.13). The vertex noise costs about 0.01 on five seeds and 0.02 on the ensemble; one solver path is the right build. Written into the 031 document as round 139.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B15 | 139 | w156 | del + HiGHS parsimonious store | 0.130 (sd 0.034); ens 0.148 / 5.79 |

## 2026.09.24 - Round B25 queued: the release rule on the HiGHS store

- B25 (w169, job 2807): B10's release rule on B15's HiGHS parsimonious store; paired against B15 (w156) and B10 (w148). Round 152.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B25 | 152 | w169 | del + HiGHS store, affine on escape | pending (2807) |
