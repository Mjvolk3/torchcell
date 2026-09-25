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

## 2026.09.24 - Round B17 read (job 2795, 3 h 35 min)

- B17 (w157, the capped driving-force concentration store in the affine slot): 0.118 (sd 0.055) vs the deletion-only control 0.058 (+0.059, sd 0.055, 5/6) and vs the GLPK flux store 0.121 (-0.003, sd 0.037, 2/6); top-50 4.06; ensemble 0.133 / 4.99. Two seeds at 0.167 (epochs 23 and 6), two at 0.058 and 0.040 (epochs 106 and 50). The metabolite side carries the flux vector's rank with twice the spread and a weaker call. Written into the 031 document as round 140 (version 6).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B17 | 140 | w157 | del + driving-force concentration store | 0.118 (sd 0.055); ens 0.133 / 4.99 |

## 2026.09.24 - Round B16 read (job 2797, 3 h 39 min)

- B16 (w159, the sampled-cone mean in the affine slot): 0.088 (sd 0.022) vs the deletion-only control 0.058 (+0.030, sd 0.070, 4/6) and vs the GLPK store 0.121 (-0.033, sd 0.024, 0/6); top-50 2.90 (below the control's 3.39); ensemble 0.083 / 3.00. Every seed off the plateau (epochs 35 to 138). The cone's mean carries less than the optimum it is sampled around on every seed and takes the call below the control; the FCL representation does not travel to the transformer. Written into the 031 document as round 142.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B16 | 142 | w159 | del + sampled-cone mean store | 0.088 (sd 0.022); ens 0.083 / 3.00 |

## 2026.09.24 - Round B18 read (job 2798, 3 h 12 min)

- B18 (w161, the store arm at width 6 with two heads): 0.113 (sd 0.038) vs the width-12 store 0.121 (-0.008, sd 0.055, 4/6); top-50 4.19; ensemble 0.129 / 5.39. Four seeds escape (epochs 39 to 86, 0.110 to 0.137); two never leave the plateau (selected at 185 and 197; 0.136 = the affine alone, 0.039 = the collapse). Half the width: the same level within the spread, and the plateau returns on a third of the seeds. The size axis is read from 6 to 90 and the level does not move on it. Written into the 031 document as round 144.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B18 | 144 | w161 | del + store, width 6 | 0.113 (sd 0.038); ens 0.129 / 5.39 |

## 2026.09.24 - Round B19 read (job 2799, 3 h 34 min)

- B19 (w162, the store arm with transformer dropout 0.3): 0.145 (sd 0.018) vs the store at 0.1 dropout 0.121 (+0.024, sd 0.040, 4/6); top-50 5.03 (best single-seed call); ensemble 0.149 / 5.79 (best width-12 ensemble rank). Five seeds escape at epochs 39 to 81 and read 0.135 to 0.180, the tightest band of any arm; seed 2704 on the plateau reads 0.133. Regularization moves the level where width 6 to 90 did not. Written into the 031 document as round 145 (version 7).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B19 | 145 | w162 | del + store, dropout 0.3 | 0.145 (sd 0.018); ens 0.149 / 5.79 |

## 2026.09.24 - Rounds B26 and B27 queued: the three levers together, and the dropout curve

- B26 (w171, job 2809): the HiGHS store with the affine released on escape and transformer dropout 0.3, the three best single levers on one cell; paired against B25 (w169, the same without dropout) and B19 (w162, the GLPK store at 0.3 without the release). Round 154.
- B27 (w172, job 2810): the store arm at dropout 0.5; paired against B19 (0.3) and B5 (0.1). Round 155.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B26 | 154 | w171 | del + HiGHS store, release rule, dropout 0.3 | pending (2809) |
| B27 | 155 | w172 | del + store, dropout 0.5 | pending (2810) |

## 2026.09.24 - Round B20 read (job 2801, 3 h 39 min)

- B20 (w164, the driving-force polytope mean in the affine slot): 0.125 (sd 0.041) vs the deletion-only control 0.058 (+0.067, sd 0.047, 6/6), vs the GLPK flux store 0.121 (+0.004, sd 0.047, 4/6) and vs the driving-force point 0.118 (+0.008, sd 0.054, 3/6); top-50 3.69; ensemble 0.151 / 5.59, the highest width-12 ensemble rank. Every seed off the plateau (epochs 10 to 76, 0.055 to 0.168). The region's centroid is a third representation at the flux store's level, with more for the ensemble and less for the call; the spread store is unread in the transformer. Written into the 031 document as round 147.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B20 | 147 | w164 | del + driving-force polytope mean | 0.125 (sd 0.041); ens 0.151 / 5.59 |

## 2026.09.24 - Round B21 read (job 2804, 3 h 35 min)

- B21 (w166, the MOMA store with the affine released on escape): 0.106 (sd 0.034) vs the MOMA store 0.107 (-0.001, sd 0.038, 3/6) and vs the GLPK store under the rule 0.110 (-0.004, sd 0.063, 2/6); top-50 4.53; ensemble 0.105 / 5.59. Releases at epochs 5, 5, 22, 30, 9, 5; two seeds select before the release (epochs 2 and 6). The rule adds nothing to the adjustment store and costs its call; its round-131 gain reads as specific to the degenerate store. Written into the 031 document as round 149 (version 8).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B21 | 149 | w166 | del + MOMA store, affine on escape | 0.106 (sd 0.034); ens 0.105 / 5.59 |

## 2026.09.24 - Round B28 queued: the width of the region

- B28 (w174, job 2812): the polytope's mean with the per-metabolite spread of ln c appended (5,666 columns; the ladder read the spread store at 0.138 by kNN); paired against B20 (w164, the mean alone) and B5 (w140). Round 156. The transfer table in the document is now paged at eighteen rows (`target_notes_tex_tables.py`, `PAIRS_PER_TABLE`), after it overran the page.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B28 | 156 | w174 | del + polytope mean and spread | pending (2812) |

## 2026.09.24 - Round B23 read (job 2805, 3 h 39 min)

- B23 (w167, the GO multi-hot in the affine slot over every annotated gene): 0.202 (sd 0.001) vs the deletion-only control 0.058 (+0.144, sd 0.075, 6/6) and vs the GLPK store 0.121 (+0.082, sd 0.025, 6/6); top-50 5.03; ensemble 0.202 / 4.99 (the seeds read the same number). Every seed selects at epochs 8 to 12 on validation Pearson 0.25; validation loss bottoms at epochs 1 to 3 and rises 0.13 to 0.18. The affine's linear read of the binary columns is the whole cell, at the perceptron's 0.200; the backbone adds nothing. The transformer was limited by the annotation it was given. Written into the 031 document as round 150.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B23 | 150 | w167 | del + GO multi-hot affine | 0.202 (sd 0.001); ens 0.202 / 4.99 |

## 2026.09.24 - Rounds B29 and B30 queued: the GO affine released, and at dropout 0.3

- B29 (w175, job 2813): the GO affine under the release rule (frozen until epoch 5 and until validation Pearson 0.15), asking whether a backbone that escapes first adds to the annotation's read; paired against B23 (w167) and B10 (w148). Round 158.
- B30 (w176, job 2814): the GO affine at transformer dropout 0.3; paired against B23 and B19 (w162). Round 159.

| round | 028 | pack | arm | status |
|---|---|---|---|---|
| B29 | 158 | w175 | del + GO affine, released on escape | pending (2813) |
| B30 | 159 | w176 | del + GO affine, dropout 0.3 | pending (2814) |

## 2026.09.24 - Round B24 read (job 2806, 3 h 38 min)

- B24 (w168, the HiGHS deletion vector with GO appended in one affine): 0.211 (sd 0.007) vs the HiGHS store alone 0.130 (+0.081, sd 0.032, 6/6) and vs GO alone 0.202 (+0.009, sd 0.007, 5/6); top-50 4.83 (under GO alone's 5.03); ensemble 0.212 / 4.99, the highest rank of the study. Every seed selects at epochs 8 to 13. The flux vector adds to the annotation on the logit where the perceptron read no gain; what flux adds beside GO is a tenth of what GO adds beside flux. Written into the 031 document as round 151.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B24 | 151 | w168 | del + HiGHS store with GO appended | 0.211 (sd 0.007); ens 0.212 / 4.99 |

## 2026.09.24 - B25 read; standing against the comparators; packs B31 and A17

- B25 (w169, the HiGHS store under the release rule, job 2807): 0.108 (sd 0.021) vs the HiGHS store free from epoch 0 at 0.130 (-0.022, sd 0.042, 2/6) and vs the GLPK store under the rule 0.110 (-0.002, 2/6); top-50 3.99 vs 4.46 and 4.66; ensemble 0.139 / 4.99 vs 0.148 / 5.79. Release at epochs 5, 5, 22, 30, 9, 5 (as round 149); the HiGHS plateau seed (0.160 from the affine alone) reads 0.117 under the rule. The rule's gain was the degenerate GLPK store's; a clean store is read best from epoch 0. Written as round 152; 031 published v9, then v10 with the standing section rewritten through round 153.
- Standing against the comparators on the same split (pinned test Spearman): best arm 0.211 (GO + HiGHS affine, round 151); kNN over the HiGHS store 0.195; best ridge 0.152 (driving-force cap 100); kNN over ProtT5 0.145; FCL RandomForest 0.039. Every comparator is below the best arm; the kNN is the closest.
- Cross-target read (`experiments/032-amino-acid-module/scripts/betaxanthin_amino_acid_correlation.py`, in the 032 note): a deletion's measured Mulleder amino-acid pool reads its betaxanthin level out of fold at 0.226 (permuted 0.00) over 4,477 shared strains. The two endpoints share strain-level signal.
- Submitted B31 = w178 (GO + HiGHS affine at dropout 0.3, job 2816; paired vs w168 and w176) and A17 = w179 (the same on the amino acids, job 2817; vs w177 and w173).
- CPU ladder rerun with GO added (`target_store_baselines.py`, 10 min; every earlier cell reproduced to the printed digit): GO multi-hot ridge 0.076 / kNN 0.179 / forest 0.187 (top-50 5.39, the best call of any read on this split); GO with the HiGHS vector 0.123 / 0.198 / 0.204 (4.79). The forest over the best arm's own input sits 0.008 under round 151's 0.211, the size of that round's seed spread, with a higher call: on this input the transformer's single-seed rank is what a backbone-free model reaches; averaging six seeds adds 0.001. 031 published v12 (392 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B25 | 152 | w169 | del + HiGHS store, affine released on escape | 0.108 (sd 0.021); ens 0.139 / 4.99 |
| B31 | -- | w178 | del + HiGHS store with GO, dropout 0.3 | job 2816, pending |

## 2026.09.24 - B26 read: the three levers on one cell

- B26 (w171, HiGHS store + release rule + dropout 0.3, job 2809, 3 h 40 min): 0.125 (sd 0.036) vs the same without dropout (B25) 0.108 (+0.017, sd 0.018, 5/6) and vs the GLPK store at dropout 0.3 (round 145) 0.145 (-0.020, sd 0.052, 3/6); top-50 4.36 vs 3.99 and 5.03; ensemble 0.148 / 4.39 vs 0.139 / 4.99 and 0.149 / 5.79. Dropout delays the escape (release at 17, 5, 24, 44, 18, 5 vs 5, 5, 22, 30, 9, 5); seeds read 0.060 to 0.155. Dropout gives back about half of what the release took from the HiGHS store (0.130 free, 0.108 under the rule, 0.125 with dropout); the release is the lever that costs on the clean store. Written as round 154; 031 published v13 (409 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B26 | 154 | w171 | del + HiGHS store, release rule, dropout 0.3 | 0.125 (sd 0.036); ens 0.148 / 4.39 |

## 2026.09.24 - B27 read: dropout 0.5, the curve turns

- B27 (w172, store arm at dropout 0.5, job 2810, 3 h 38 min): 0.125 (sd 0.017) vs 0.3 at 0.145 (-0.019, sd 0.019, 2/6) and vs 0.1 at 0.121 (+0.005, sd 0.039, 3/6); top-50 4.16 vs 5.03 and 4.93; ensemble 0.127 / 5.59 vs 0.149 / 5.79 and 0.131 / 5.59. Three seeds never escape the plateau (val Pearson -0.03 to -0.02 through epoch 199; select at 149 to 181; read 0.124 to 0.133 = the affine alone); the three that escape read 0.094 to 0.140. The dropout curve 0.1 -> 0.121, 0.3 -> 0.145, 0.5 -> 0.125: 0.3 is a maximum. Written as round 155; 031 published v14 (426 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B27 | 155 | w172 | del + store, dropout 0.5 | 0.125 (sd 0.017); ens 0.127 / 5.59 |

## 2026.09.25 - B28 read: the polytope's spread beside its mean; packs B32, B33

- B28 (w174, polytope mean + spread of ln c, 5,666 columns, job 2812, 3 h 41 min): 0.116 (sd 0.041) vs the mean alone (B20) 0.125 (-0.010, sd 0.062, 3/6) and vs the GLPK store 0.121 (-0.005, sd 0.048, 4/6); top-50 3.73 vs 3.69 and 4.93; ensemble 0.155 / 4.59 vs 0.151 / 5.59 and 0.131 / 5.59, the highest six-seed rank of any flux-side width-12 arm. Two seeds escape only partly (val 0.09 to 0.10, read 0.061 and 0.089); the rest read 0.097 to 0.164. The region's value is in the ensemble. Written as round 156; 031 published v15 (447 checks).
- Submitted B32 = w180 (GO + HiGHS vector + polytope mean in one affine; store `ess_solve_encodings_YPD_bx_gopfbapoly.npz` from the new `target_store_concat.py`, 9,374 columns; job 2821; vs w168 and w164) and B33 = w181 (seeds 2706 to 2711 of w168, job 2822, for a twelve-seed ensemble of the best arm).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B28 | 156 | w174 | del + polytope mean with the spread | 0.116 (sd 0.041); ens 0.155 / 4.59 |
| B32 | -- | w180 | del + HiGHS store with GO and the polytope mean | job 2821, pending |
| B33 | -- | w181 | w168 seeds 2706 to 2711 | job 2822, pending |

## 2026.09.25 - B29 read: the GO affine under the release rule

- B29 (w175, GO affine released on escape, job 2813, 3 h 40 min): 0.199 (sd 0.007) vs the GO affine from epoch 0 (B23) 0.202 (-0.003, sd 0.007, 4/6) and vs the GLPK store under the rule 0.110 (+0.089, 6/6); top-50 4.89 vs 5.03 and 4.66; ensemble 0.206 / 5.19 vs 0.202 / 4.99 and 0.120 / 6.19. Release at 5, 5, 20, 42, 8, 27; every seed selects 6 to 11 epochs after its release on val 0.24 to 0.28; the two released last read lowest (0.191, 0.189). The backbone escaping first adds nothing to the annotation's single read; the rule buys seed diversity (sd 0.007 vs 0.001), worth 0.003 on the ensemble. Written as round 158; 031 published v16 (468 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B29 | 158 | w175 | del + GO affine, released on escape | 0.199 (sd 0.007); ens 0.206 / 5.19 |

## 2026.09.25 - B30 read: the GO affine at dropout 0.3

- B30 (w176, GO affine at transformer dropout 0.3, job 2814, 3 h 40 min): 0.202 (sd 0.002) vs the GO affine at 0.1 (B23) 0.202 (-0.001, sd 0.001, 0/6) and vs the GLPK store at dropout 0.3 (round 145) 0.145 (+0.057, 6/6); top-50 4.89 vs 5.03 and 5.03; ensemble 0.202 / 4.79. Every seed selects at epochs 8 to 14, reads 0.198 to 0.203. The lever that lifted the store arm by 0.024 moves the annotation arm by less than a thousandth: the annotation arm is the affine alone; no transformer setting has moved it (with B29). Written as round 159; 031 published v17 (485 checks; the first upload attempt failed on the Zotero attachment call and the retry succeeded).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B30 | 159 | w176 | del + GO affine, dropout 0.3 | 0.202 (sd 0.002); ens 0.202 / 4.79 |

## 2026.09.25 - B31 read: GO with the HiGHS vector at dropout 0.3; packs B34, B35

- B31 (w178, GO + HiGHS affine at dropout 0.3, job 2816, 3 h 40 min): 0.212 (sd 0.007) vs the same at 0.1 (B24) 0.211 (+0.000, sd 0.001, 2/6) and vs GO at 0.3 (B30) 0.202 (+0.010, sd 0.007, 5/6); top-50 4.89 vs 4.83 and 4.89; ensemble 0.213 / 4.99. The two levers do not add; the flux gain beside GO reproduces at dropout 0.3 (0.010 vs 0.009). Written as round 161; 031 published v18 (504 checks).
- Since the annotation arms ignore every transformer setting (rounds 158, 159, 161), the next levers act on the affine's input. Two stores built with `target_store_concat.py`: `YPD_bx_gopfbapolyspread` (GO + HiGHS + polytope mean and spread, 12,205 columns) and `YPD_bx_gomoma` (GO + MOMA, 6,539). Submitted B34 = w183 (gopfbapolyspread, job 2824; vs w168 and w180) and B35 = w184 (gomoma, job 2825; vs w167 and w168).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B31 | 161 | w178 | del + HiGHS store with GO, dropout 0.3 | 0.212 (sd 0.007); ens 0.213 / 4.99 |
| B34 | -- | w183 | del + HiGHS store with GO and the polytope mean and spread | job 2824, pending |
| B35 | -- | w184 | del + MOMA store with GO | job 2825, pending |

## 2026.09.25 - B32 read: GO, the HiGHS vector and the polytope mean in one affine

- B32 (w180, gopfbapoly store, 9,374 columns, job 2821, 3 h 39 min): 0.198 (sd 0.011) vs GO + HiGHS (B24) 0.211 (-0.013, sd 0.009, 1/6) and vs the polytope mean alone (B20) 0.125 (+0.073, 6/6); top-50 4.46 vs 4.83 and 3.69; ensemble 0.201 / 4.59 vs 0.212 / 4.99. Seeds select at 4 to 15, read 0.180 to 0.213. The 2,835 concentration columns take 0.013 from the flux-and-annotation affine and the stack reads under GO alone (0.202): the metabolite side does not add to the annotation the way the flux vector does. Hypothesis (untested): dilution of the linear read by a third block the flux block already distinguishes. Written as round 163; 031 published v19 (521 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B32 | 163 | w180 | del + HiGHS store with GO and the polytope mean | 0.198 (sd 0.011); ens 0.201 / 4.59 |

## 2026.09.25 - B33 read: twelve seeds of the best arm

- B33 (w181 = w168 seeds 2706 to 2711, job 2822, 3 h 36 min): the second six read 0.218 (sd 0.009) against the first six's 0.211 (sd 0.007); over twelve 0.214 (sd 0.008, sem 0.002), seeds read 0.202 to 0.232, select at epochs 6 to 13; twelve-seed ensemble 0.216 / 4.99 (six: 0.212 / 4.99); single-seed mean call 4.98. The best arm's level holds: 0.011 above the forest over its own columns (0.204) and 0.019 above the HiGHS-store kNN (0.195). Written as round 164; 031 published v20 (536 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B33 | 164 | w181 | w168 seeds 2706 to 2711 | twelve seeds 0.214 (sem 0.002); ens 0.216 / 4.99 |

## 2026.09.25 - B34 read: the stack with the region's width

- B34 (w183, gopfbapolyspread store, 12,205 columns, job 2824, 3 h 39 min): 0.193 (sd 0.015) vs GO + HiGHS (B24) 0.211 (-0.018, sd 0.016, 2/6) and vs the same stack with the mean alone (B32) 0.198 (-0.005, sd 0.018, 2/6); top-50 4.23 vs 4.83 and 4.46; ensemble 0.201 / 4.59, the same as the mean alone. Seeds select at 9 to 27, read 0.175 to 0.213. The cost grows with the columns: 2,835 concentration columns took 0.013, another 2,831 spread columns take 0.005 more, and the seed spread doubles at each step. The metabolite side does not enter this affine at any width. Written as round 166; 031 published v21 (553 checks).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B34 | 166 | w183 | del + HiGHS store with GO and the polytope mean and spread | 0.193 (sd 0.015); ens 0.201 / 4.59 |

## 2026.09.25 - B35 read: the adjustment store beside the annotation; packs B38, B39

- B35 (w184, GO + MOMA store, 6,539 columns, job 2825, 3 h 39 min): 0.208 (sd 0.003) vs GO alone (B23) 0.202 (+0.006, sd 0.003, 6/6) and vs GO + HiGHS (B24) 0.211 (-0.004, sd 0.006, 2/6); top-50 4.46 vs 5.03 and 4.83; ensemble 0.209 / 4.39. Seeds select at 8 to 16, read 0.205 to 0.212. Both flux programs enter the annotation's read (MOMA +0.006, parsimonious +0.009, difference inside the paired spread) where the metabolite programs do not. Written as round 167; 031 published v22 (569 checks).
- Submitted B38 = w190 (GO + HiGHS + MOMA in one affine, store `YPD_bx_gopfbamoma`, 10,716 columns, job 2830; vs w168 and w184) and B39 = w191 (w167 seeds 2706 to 2711, job 2831, for the twelve-seed annotation-only level).

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B35 | 167 | w184 | del + MOMA store with GO | 0.208 (sd 0.003); ens 0.209 / 4.39 |
| B38 | -- | w190 | del + HiGHS and MOMA stores with GO | job 2830, pending |
| B39 | -- | w191 | w167 seeds 2706 to 2711 | job 2831, pending |
| B36 | -- | w188 | del + HiGHS store with GO, affine weight decay 0.1 | job 2828, pending |
| B37 | -- | w189 | del + HiGHS store with GO, affine weight decay 1.0 | job 2829, pending |

- B36 and B37: since rounds 158 to 164 read the annotation arms as the affine alone, the one setting left that acts on the read is the affine's own regularization: w168 with `--ess-affine-weight-decay 0.1` and `1.0` at the full rate (round 119 read decay with a tenth of the rate on the degenerate store and lost the gain, so the rate is left alone). Paired against w168 and against each other.

## 2026.09.25 - B36 read: the affine's weight decay at 0.1

- B36 (w188, GO + HiGHS affine with the affine's weight decay at 0.1, job 2828, 3 h 40 min): 0.210 (sd 0.007) vs no decay (B24) 0.211 (-0.002, sd 0.008, 4/6); top-50 4.66 vs 4.83; ensemble 0.211 / 4.79 vs 0.212 / 4.99. Seeds select at 8 to 16 on val Pearson 0.24 to 0.27, read 0.199 to 0.221; five seeds select at the same epoch as their control and differ by under 0.004, seed 2703 selects at 16 instead of 9 and reads 0.018 under. A thousandfold decay on the affine's weights leaves the selected epochs and the read where they were. Written as round 169; 031 published v23.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B36 | 169 | w188 | del + HiGHS store with GO, affine weight decay 0.1 | 0.210 (sd 0.007); ens 0.211 / 4.79 |

## 2026.09.25 - B37 read: the affine's weight decay at 1.0

- B37 (w189, the same cell at affine weight decay 1.0, job 2829, 3 h 40 min): 0.204 (sd 0.012) vs no decay (B24) 0.211 (-0.007, sd 0.013, 1/6) and vs decay 0.1 (B36) 0.210 (-0.005, sd 0.013, 2/6); top-50 4.76 vs 4.83 and 4.66; ensemble 0.208 / 4.99. Seeds select at 34 to 114 (vs 8 to 16 at the lighter settings) on val Pearson 0.25 to 0.27, read 0.190 to 0.218. The decay curve none / 0.1 / 1.0 reads 0.211 / 0.210 / 0.204: flat then down, the heavy setting slowing the affine by about fifty epochs and doubling the seed spread. Written as round 170; 031 published v24.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B37 | 170 | w189 | del + HiGHS store with GO, affine weight decay 1.0 | 0.204 (sd 0.012); ens 0.208 / 4.99 |

## 2026.09.25 - B38 read: both flux programs beside the annotation

- B38 (w190, GO + HiGHS + MOMA in one affine, 10,716 columns, job 2830, 3 h 36 min): 0.216 (sd 0.007) vs GO + HiGHS (B24) 0.211 (+0.004, sd 0.003, 5/6) and vs GO + MOMA (B35) 0.208 (+0.008, sd 0.006, 5/6); top-50 4.73 vs 4.83 and 4.46; ensemble 0.217 / 4.59 vs 0.212 / 4.99 and 0.209 / 4.39. Seeds select at 9 to 14 on val Pearson 0.25 to 0.26, read 0.206 to 0.223. The two programs add: the highest single-seed rank of the study at a paired spread of 0.003 against the previous best; the call does not follow. Written as round 171; 031 published v25.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B38 | 171 | w190 | del + HiGHS and MOMA stores with GO | 0.216 (sd 0.007); ens 0.217 / 4.59 |

## 2026.09.25 - B40 read: the control on twelve seeds

- B40 (w192 = w137 seeds 2706 to 2711, job 2832, 3 h 41 min): the control's second six read 0.094 (sd 0.017) vs the first six's 0.058 (sd 0.075); over twelve 0.076 (sd 0.055), seeds -0.089 to 0.126, select at epochs 3 to 69. GO affine vs control paired over twelve: +0.127 (sd 0.054), 12/12, 0.203 vs 0.076; top-50 4.99 vs 4.06. Control ensembles: six 0.082 / 4.79, twelve 0.102 / 5.59; the annotation's twelve-seed ensemble 0.203 stays double. The amino-acid round-168 finding (control ensemble passes the annotation) does not repeat here: the gain is five times what the averaging recovers. Written as round 173; 031 published v30.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B40 | 173 | w192 | w137 seeds 2706 to 2711 | control twelve seeds 0.076 (sd 0.055); GO affine +0.127 (12/12); control ens 0.102 / 5.59 |

## 2026.09.25 - The measured amino-acid pool as an input; DCell's published input; packs B41, B42

- User direction: the cross-endpoint read (pool ridge 0.226, GO + pool 0.237 on its own folds) should be demonstrated as "adding the data helps", in the ladder and in the transformer. New `target_pool_store.py`: the strain's own Mulleder pool (19 mM columns as released, 4,678 strains, 0 missing cells; wild-type row = the loader's reference, the population robust mean) as affine store `YPD_bx_aapool`; `target_store_concat.py` appends it to GO (`YPD_bx_goaapool`, 2,381 columns) and to GO with the HiGHS vector (`YPD_bx_gopfbaaapool`, 6,558). Phenotype to phenotype: the row is the strain's measurement in a different screen, not its sequence, so a gain shows that the data carries signal beyond the gene's identity, not that the module generalizes from genotype.
- Ladder: `target_store_baselines.py` gains `aa_pool`, `go_aapool`, `gopfba_aapool` for betaxanthin (rerun in progress, `slurm/output/store_baselines_betaxanthin_aapool.log`) and a `cooper` target (the 17 Cooper 2010 peaks read from the loader's table by the record's deleted gene on the same pinned split; no reference arm, so the identity check is skipped; `032/slurm/output/store_baselines_cooper.log`).
- Submitted B41 = w194 (GO + pool in the affine, job 2836; vs w167) and B42 = w195 (GO + HiGHS + pool, job 2837; vs w168 and w194).
- DCell: `target_go_models.py` gains `dcell_published_input` (DCell's own per-(gene, term) input; under the split by gene a held-out gene's columns are never trained, so it is the published network's read of an unseen gene and the control for what the shared input buys). Running on CPU: the published input at width 8 (`slurm/output/go_models_betaxanthin_published_w8.log`, merged into `results/go_models_betaxanthin.json`) and both hierarchy arms at width 20 (`go_models_betaxanthin_w20.log`, `results/go_models_betaxanthin_w20.json`), the two widths the essentiality tune grid used. DCell's published sizing, max(20, 0.3 x genes under the term), is not runnable in this implementation: every term is padded to the widest, and the root's 1,650 units would put the edge tensor past 30 GB.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B41 | -- | w194 | del + GO with the measured amino-acid pool | job 2836, pending |
| B42 | -- | w195 | del + HiGHS store with GO and the pool | job 2837, pending |

## 2026.09.25 - B39 read: twelve seeds of the annotation-only arm

- B39 (w191 = w167 seeds 2706 to 2711, job 2831, 3 h 40 min): the second six read 0.204 (sd 0.002) against the first six's 0.202 (sd 0.001); over twelve 0.203 (sd 0.002, sem 0.0005), seeds 0.200 to 0.207, select at epochs 7 to 14; twelve-seed ensemble 0.203 / 4.99. The flux vector's gain beside GO paired over the same twelve seeds (new PAIRS entry): +0.012 (sd 0.009), 11/12, 0.214 vs 0.203 (the first six gave +0.009 on 5/6). Written as round 172; 031 published v27.

| round | 028 | pack | arm | result |
|---|---|---|---|---|
| B39 | 172 | w191 | w167 seeds 2706 to 2711 | twelve seeds 0.203 (sem 0.0005); flux gain beside GO +0.012 (11/12) |

- MEASURED, DCell's published input at width 8 (`slurm/output/go_models_betaxanthin_published_w8.log`, merged into `results/go_models_betaxanthin.json`, six seeds, 8 to 13 min each): 0.028 (sd 0.058), top-50 1.26; ensemble 0.073 / 1.40; seeds select at epochs 0 to 93 and read -0.057 to 0.115. Against the shared input 0.049 (sd 0.043) / 3.10 and the GO perceptron 0.200 / 4.79. A held-out gene's columns are untrained, so the published network reads an unseen gene through random columns; the nonzero read is consistent with the subsystems responding to how many terms a gene lights (hypothesis, untested). Written into the 031 document (section 3; published v28). MEASURED, width 20 (`go_models_betaxanthin_w20.log`, `results/go_models_betaxanthin_w20.json`, 188 min): shared hierarchy 0.069 (sd 0.036) / top-50 3.33, ensemble 0.066 / 3.79, 13 to 24 min a seed; published input 0.041 (sd 0.046) / 1.70, ensemble 0.099 / 2.40, three seeds never reach a validation peak (select at epoch 200). Both a third of the GO perceptron or under. DCell's published sizing (root 1,650 units) is out of reach for the padded implementation (edge tensor past 30 GB). 031 published v29.
- MEASURED, the ladder with the pool (`results/store_baselines_betaxanthin.json`, 13.5 min; all fifteen earlier cells reproduced to the printed digit): the pool alone (19 columns) reads 0.149 by ridge, 0.285 by kNN, 0.280 by forest (top-50 3.99 / 4.19 / 4.79), above every genotype-side read and every transformer arm (0.216); GO + pool 0.102 / 0.174 / 0.264 (forest top-50 5.99); GO + HiGHS + pool 0.156 / 0.195 / 0.247 (forest top-50 6.19, the highest call of any read). Ridge and kNN fall when the 2,362 GO columns join the 19; the forest keeps 0.26 and gains the call (plausible reason, unverified: the distance models weigh every standardized column alike). Written into the 031 document (section 3 and the standing section; 031 published v26).
