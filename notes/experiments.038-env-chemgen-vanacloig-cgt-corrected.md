---
id: 6pl70fzm57ezxkfc3va208w
title: 038 Env Chemgen Vanacloig Cgt Corrected
desc: ''
updated: 1791363906268
created: 1791363906268
---

## 2026.10.07 - Experiment 035 repeated on the corrected store

The same question and design as [[experiments.035-env-chemgen-vanacloig-cgt]], on the
data after the dataset fixes that closed the audit issues: can a model predict the
Vanacloig 2022 chemogenomic profile of a compound it has never seen, from the compound's
structure alone, and does the cell graph transformer beat nested ridge on FCFP4 counts.
Experiment 035 stays the record of the 2026.09.21 store (build 001 of the 033 table); this
experiment reads build 002, the 2026.10.06-4b293d34 store, and because the data differ it
carries its own number rather than a sub-record of 035.

What changed in the data, each traced to its issue:

| change | issue | effect on this experiment |
|---|---|---|
| TMM normalization in place of CPM | #501 | per-compound offsets removed: crystal violet median -2.773 to -0.062, nonylacridine orange +0.44, ethanol +0.21; within-compound rank order across strains unchanged (Spearman 001 vs 002 median 0.9999, minimum 0.971, crystal violet) |
| the nine unreported tokens dropped; DMSO and MBO served as conditions | #501 | 41 served compounds become 34 conditions |
| strain background on the reference's `StrainReferenceGenome`, the genotype is the screened deletion alone | #500 | every strain carries one perturbation; `strain_indices` still deletes the query plus PDR1, PDR3 and SNQ2, since the cell lacks all four |
| Wildenhain, Hillenmeyer HET, Hoepfner input fixes | #504, #505, #506 | not read here (Vanacloig alone), but the pooled table they share changed shape (heterozygous deletions, conditional alleles) |

The compound panel is the 32 published compounds. Two served conditions are left out,
each for a stated reason (`vanacloig_data.EXCLUDED_CONDITIONS`): DMSO is the 1% v/v
vehicle, a control the 035 panel never held; MBO (2-methyl-3-buten-2-ol) is an inhibitor
the paper reports but has no row in any of the twelve 031 embedding tables, so it cannot
be scored inductively until it is embedded. The panel therefore equals the "published"
view of 035, which is what makes the two records comparable: on build 001, nested ridge
scored a median centered Spearman of 0.343 over those 32 compounds (96 compound-evaluations).

Loaded through `vanacloig_data.load_cells`: 111,695 cells, 3,587 strains, 32 compounds
(035: 143,218 cells, 3,598 strains, 41 compounds).

**Code.** The scripts are a copy of 035's at its commit c557479a0 with the experiment
folder renamed, the table pinned to build 002, and the build switch removed; 035 is frozen
as the record of build 001 and development continues here. The W&B project is
`torchcell_038-env-chemgen-vanacloig-cgt-corrected`.

**Protocol**, unchanged from 035 rounds 9 to 11: five compound-cold folds on fold seeds
0, 1 and 2 (96 compound-evaluations), four validation compounds per fold, every choice
nested; the score is the Spearman per held-out compound across strains after each side
subtracts its own mean over the fitted compounds; paired against nested ridge with the
compound-level bootstrap interval.

**Rounds**

1. The ladder (`baseline_ladder.py`, 12 encoders, kernel ridge and kNN, nested): the bar.
2. The round-9 control (CGT + bilinear head, pool fit, 50 epochs, 3 seeds) and the
   environment-encoder head (035 round 10), nine seeds each, and their stacks with ridge.

### Round 1: the ladder on the corrected table (slurm 3374)

`baseline_ladder.py`, fold seeds 0, 1, 2, 96 compound-evaluations (32 compounds x 3 fold
seeds); `results/ladder/ladder_r2_summary.csv`. Centered Spearman per held-out compound
across strains, median and mean over the 96 evaluations:

| model (nested) | median centered Spearman per held-out compound (96 compound-evaluations) | mean centered Spearman (96) | inner leave-one-compound-out mean |
|---|---|---|---|
| kernel ridge, linear kernel on FCFP4 counts (nested ridge, the reference) | **0.359** | 0.323 | 0.331 |
| kernel ridge, RBF on FCFP4 counts | 0.357 | 0.319 | 0.329 |
| kNN, linear kernel on FCFP4 counts | 0.347 | **0.330** | **0.340** |
| kNN, mean of all encoder kernels | 0.334 | 0.314 | 0.306 |
| kNN, RBF on FCFP4 counts | 0.326 | 0.305 | 0.308 |
| kernel ridge, RBF on Uni-Mol | 0.323 | 0.298 | 0.286 |
| kNN, RBF on ChemBERTa-2 MTR | 0.320 | 0.309 | 0.319 |

The bar is nested ridge on FCFP4 counts at median 0.359, against 0.343 on the same 32
compounds of build 001 (035 round 2, "published" view). The ladder's own per-fold pick
split between ridge and kNN on the FCFP4 linear kernel (and once each the all-encoder
combo and ChemBERTa-2 MTR), so, as on build 001, no encoder or kernel separates from the
raw count fingerprint.

### Round 2: the two heads on the corrected store, and both sit below ridge

Slurm 3700 (bilinear control, `conf/factorized/r9_control.yaml`, 4 h 10 min) and 3701
(environment encoder, `r10_envenc.yaml`, 5 h 17 min) on GilaHyper, 2026-10-10, each 15
configs at PARALLEL=2: five compound-cold folds on each of fold seeds 0, 1 and 2, three
seeds per fold, 50 epochs keeping the last, fitted on the non-test pool. The protocol is
035 rounds 9 and 10 unchanged, so the two records compare directly. Centered Spearman per
held-out compound across strains, 96 compound-evaluations (32 published compounds x 3 fold
seeds).

| head | seed 0 | seed 1 | seed 2 | three-seed ensemble (median) | mean |
|---|---|---|---|---|---|
| bilinear control | 0.311 | 0.305 | 0.301 | **0.312** | 0.295 |
| environment encoder | 0.258 | 0.275 | 0.288 | 0.297 | 0.276 |

**Both are below the bar.** Nested ridge on FCFP4 counts reads 0.359 on the same 96
evaluations (round 1, slurm 3374). Paired on compound and fold, the environment encoder
minus the bilinear control is median -0.036, mean -0.018 with a standard error of 0.013,
and 37 of 96 evaluations up: the encoder does not beat the control here, and on this store
neither head reaches ridge.

That is a change of sign against the build-001 record, where the encoder was +0.017 over
the control and the ridge-plus-encoder stack was +0.032 over ridge with a compound-level
interval through zero (035 rounds 10, 11 and 14). Two readings are open and are what the
Delta rounds test. **Hypothesis (untested):** 50 epochs is not saturation for the
environment encoder, whose validation loss was still falling at epoch 50 in 035 round 15;
round 16 trains it to 150. **Second hypothesis (untested):** the corrected store's 32
compounds carry less signal per compound than build 001's 41, so the learning curve in
fitted compounds matters more than the head; round 17 measures it for the encoder beside
the ridge curve already on file. Until those read, the honest statement for the figure is
that on the corrected store the cell graph transformer does not beat nested ridge on
fingerprint counts, and the ensemble of either head is within 0.06 of it.
