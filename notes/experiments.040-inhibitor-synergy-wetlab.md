---
id: ypkhq6j9t1ipiop2xvotn5i
title: 040 Inhibitor Synergy Wetlab
desc: ''
updated: 1791615756027
created: 1791615756027
---

## 2026.10.10 - The 2021 inhibitor runs as validation of what the chemogenomic models can say about mixtures

The environment case study for the paper (R5, "drug exposure"): the in-house Bioscreen runs
of 2021 (strain bAID = BY4742-iAID6, YPD, six sorghum-hydrolysate inhibitors; the private
dataset `InhibitorBioscreenVolk2021Dataset`, 977 wells) were collected to ask whether the
inhibitors interact, and the chemogenomic models of experiments 031 to 038 were built to
predict a compound's deletion profile from its structure. This experiment joins the two.

**What the record says before anything is run.** Compound-cold on Vanacloig 2022, nested
ridge on FCFP4 counts is the bar (median centered Spearman 0.303 on build 001, 0.359 on the
corrected store) and the cell graph transformer, trained to saturation, ties it (035 rounds 8
to 15). Gains came from data (the learning curve rises about 0.05 per doubling of fitted
compounds), not architecture. No trained model predicts the host's own dose response: the
pooled table carries that only as IC30 anchors. Of the six inhibitors, only furfural and
5-HMF are in the corrected Vanacloig store (profile reliability 0.77 and -0.05; levulinic
acid and sodium acetate were among the tokens the paper did not report and were dropped in
build 002, issue #501), sodium acetate is measured in Hoepfner 2014 (125 to 150 mM, YPD, a
different screen), and formic and lactic acid are in no screen. The 2021 data (039, software
generation times): all six singles grew at the ex23 doses (mean fitness 0.788), 14 of 15
pairs (0.465), 5 of 20 triples, none of the 22 combinations of four or more. The served
dataset calls growth from the raw curves (rise of at least 0.3 OD) and disagrees with the
software on 12 of 197 ex23 wells, all three replicates of HMF + LA, HMF + LVA,
FF + AA + HMF and FF + HMF + FA, whose OD rose 0.085 to 0.277: under the served call 12 of
15 pairs and 3 of 20 triples grew (21 combinations, not 25). The served call is primary;
the software call is reported as a sensitivity, because the disputed wells are all HMF
combinations and that is where a synergy call would land.

**Pre-registered claims, each allowed to fail.**

1. Growth under combinations departs from independence. Reference rules from the ex21
   single-agent curves: Bliss independence, Loewe additivity, highest single agent; scored on
   the 63 ex23 combinations (grew within 85 h over 63; fitness over the 25 that grew) and the
   three isobole grids (front against the Loewe line). Hypothesis: synergistic relative to
   Bliss (Bliss gives 0.60 for the four mildest singles together; nothing with four grew).
2. Chemogenomic profile similarity anticipates which pairs deviate. Measured Vanacloig
   profiles where they exist, compound-cold ridge-FCFP4 profiles for formic and lactic acid
   (and, as the fairness check, for the shared compounds from their held-out folds). Tested
   on 15 pairs (permutation) and the three isoboles; HMF x acetic acid is shown but excluded
   (single run, growth islands; 039).
3. Deletion profiles compose under a mixture by a rule measurable on public data: Hillenmeyer
   HET's 26 two-compound environments against their single-compound partners (sum, mean,
   max, Bliss-style product); the base rate of emergent and masked genes.
4. Post-analysis only, not a validated prediction: composed profiles nominate the genes a
   screen under each combination would recover (core, compound-specific, conflict). The
   check is GO enrichment of the nominated sets, and whether it overlaps the terms the
   measured single-compound profiles carry. No in-house screen exists to test them (the
   thesis archive holds only the library-sequencing correspondence for that screen).

**Decisions taken.** Acetic acid is matched to sodium acetate for the measured profile (same
anion at YPD pH; a stated assumption). The learned model in the preliminary round is ridge
FCFP4; the CGT environment encoder enters from 038 round 2 (slurm 3697 control, 3698
encoder, submitted 2026-10-10 02:00 on the two cards named by the user). The wet-lab table
is read from the private dev LMDB with the raw-mirror manifest sha recorded, and is
re-pointed at the knowledge graph once the served build carries the private dataset; the
records are the same. Vanacloig doses come from the rebuilt dev store (Table S1, PR #765).

**Adapting the model to the wet-lab readout (step 3, after claims 1 to 4 are scored).** The
best-performing recipe on record is 035 round 11: the environment-encoder head (compound
token in front of the 6,607 gene tokens, one further layer, one-layer encoder, prior
weight 0, 50 epochs keeping the last, fit on the non-test pool), stacked with nested ridge
(median 0.334, +0.032 over ridge, compound CI through zero). 038 round 2 reproduces that
pair on the corrected store (slurm 3700 control, 3701 encoder, GilaHyper cards 0 and 3,
2026-10-10 02:10; a Delta twin launcher `delta_train_factorized.slurm` under
bfjt-delta-gpu for the follow-on rounds). The adaptation keeps that encoder and changes
three things, each an ablation: (i) a set of compound tokens, one per inhibitor in the
medium, in place of the single token; (ii) a log-molar dose scalar modulating each token
(FiLM), with the Vanacloig and Hoepfner IC30 doses and dose zero as the public anchors;
(iii) a host-growth head read from the mean of the cell-in-medium state and the compound
tokens after they read the genome (the empty-genotype path, since bAID carries no
deletion), fitted on the ex21 single-agent curves and scored on the 63 ex23 combinations
and the isoboles against the model-free Bliss and Loewe references of claim 1. The
gene-level head stays on the Vanacloig task so the compound representation is shared. The
run order is fixed: claims 1 to 4 first on CPU, because if the additive references already
explain the combinations there is nothing for the adapted model to add, and the figure
says so.

**Confounds carried into every caption.** Run lengths 72 / 85 / 96 h, so "no growth" is run
dependent; bAID in aerobic YPD against Vanacloig's haploid deletions in anaerobic SYNH3 at
IC30 and HET's heterozygous diploids in YPD; ex21's uninhibited wells grew slower than its
low-dose wells, so ex21 is normalized by curve fit, not control division.

## 2026.10.10 - Preliminary results, claims 1 to 4 (CPU round)

All numbers from the committed scripts in `experiments/040-inhibitor-synergy-wetlab/scripts/`
and their notes; growth calls as defined above (served = raw-curve, primary; software =
Bioscreen traits, sensitivity).

**Claim 1 holds: the combinations grow less than independence predicts and more than Loewe
additivity predicts** (`mixture_rules.py`). Growth or no growth over all 63 ex23
combinations: Loewe from the ex21 Hill fits reaches accuracy 0.825 and AUROC 0.958 (served)
and 0.825 / 0.952 (software); Bliss from the observed ex23 singles 0.556 / 0.872 and
0.397 / 0.846; Bliss from the ex21 fits 0.333 / 0.777; highest single agent 0.333 / 0.663.
Fitness over the combinations that grew (21 served, 25 software): Bliss from the ex23
singles Spearman 0.573 / 0.865 with observed minus predicted -0.142 / -0.097; Loewe 0.713 /
0.646 with +0.128 / +0.185. Pairs (observed minus Bliss, replicate bootstrap): 12 of 15
synergistic, 2 additive (5-HMF + acetic acid, 5-HMF + formic acid), 1 antagonistic (5-HMF +
furfural, +0.058 [+0.018, +0.103]) under the served call; 13 / 2 / 0 under the software
call; formic + lactic acid grew in no well under either. Isoboles (mean excess over Bliss,
81 interior cells): furfural x acetic acid -0.082 [-0.142, -0.017], formic x acetic acid
-0.446 [-0.536, -0.367] (31 cells where Bliss predicts growth and none grew), 5-HMF x
acetic acid -0.017 [-0.043, +0.007] (single run, flagged); in no grid did a cell grow less
than Loewe predicts. Single agents (`single_agent_curves.py`, ex21 IC30 in mM): furfural
9.8, acetic acid 56.7, 5-HMF 17.9, formic acid 43.8, levulinic acid 75.6, lactic acid 486;
the acids and most isobole axes hit the Hill-slope bound (growth falls to none between
adjacent doses). Vanacloig IC30 against ex21 IC30: furfural 8.0 vs 9.8 mM, 5-HMF 3.3 vs
17.9 mM (anaerobic SYNH3 at 48 h against aerobic YPD at 72 h). Hypothesis (untested): the
steep single-agent curves are what drive Bliss and Loewe this far apart.

**Claim 2 fails as pre-registered** (`similarity_vs_deviation.py`). With measured profiles
in the matrix no similarity measure associates with the pair deviation (15 pairs, every
permutation p above 0.26, signs flip across measures). The all-predicted matrix shows the
opposite sign (raw Spearman -0.686, p 0.006 under the software call; -0.493, p 0.066
served): the small acids have near-identical predicted profiles (fingerprint similarity
through the ridge fit, 0.91 to 0.99) and are the synergistic pairs, a chemistry-class
regularity the fingerprint encodes, not evidence that the chemogenomic profile carries the
interaction.

**Claim 3 holds on public data: deletion profiles compose as the MEAN of the singles**
(`het_mixture_composition.py`, Hillenmeyer HET, 26 two-compound environments, 22 with an
exact or within-twofold single match). The mean is the best fixed rule in 24 of 26 (median
R^2 0.365; sum, which equals Bliss on the log2 scale, 0.091; max 0.050); the free linear fit
gives a 0.529, b 0.593, a + b 1.06. The singles explain the pair far above a control-set
matched null (median Spearman 0.579 vs 0.114; 24 of 26 at p <= 0.05). Emergent (0.355
median) and masked (0.649) gene rates are inside the hit-calling noise: the only two
near-replicate singles disagree on 54 and 64 percent of their hits. On the methotrexate x
5-fluorouracil 3 x 3 grid the mean wins at all nine dose pairs but the weights move with
dose (a 0.13 to 0.76). Carrying the mean rule to the inhibitor profiles (haploid
deletions, anaerobic SYNH3) is a labeled hypothesis.

**Claim 4, post-analysis: the ridge-predicted profiles carry only the generic stress
component** (`inhibitor_profiles.py`, `inhibitor_nominations.py`). Leave-one-compound-out
ridge on the 32 corrected compounds: median centered Spearman 0.387 (IQR 0.241 to 0.482),
furfural 0.341, 5-HMF 0.425 (agreement with a noise profile; its reliability is -0.05).
Predicted acetic acid against the Hoepfner sodium acetate profile: -0.084 (n = 3453), and the
Vanacloig build-001 acetate against Hoepfner's: 0.034, so the two measured acetate profiles
do not agree with each other either. GO enrichment of the top-100 sensitive genes (FDR
0.05): the measured furfural profile carries no significant term (best, ribosome, FDR
0.057); the predicted profiles' terms are those of the gene mean over the 32 compounds (19
of 20 for furfural, 40 of 43 levulinic acid, 48 of 49 acetic acid), and on centered
profiles measured and predicted share no term for any compound. Core nominations (sensitive
in at least four of six singles): 68 genes in the best-available matrix, 105 all-predicted,
led by Golgi transport and chromatin genes (YPT6, RIC1, VPS63, COG5, SNF6). These are
hypotheses for a screen that has not been run.

**What this means for the model step.** The wet-lab data are a clean validation target
(synergy relative to independence, Loewe the better growth predictor, model-free), and the
chemogenomic representation we have adds no measurable signal to it yet: compound-cold ridge
profiles are generic, and similarity does not anticipate deviation once the measured
profiles are in. 038 round 2 (slurm 3700 / 3701) will give the CGT environment encoder's
compound-cold profiles for the same tests; on the 035 record it ties ridge, so the
expectation (hypothesis) is the same null. A dose- and mixture-aware model therefore has a
bar to clear (Loewe AUROC 0.958 on growth; Bliss Spearman 0.57 on fitness) and little
public-data signal to clear it with; the honest preliminary claim is the model-free one.

## 2026.10.10 - Figure plan for the hydrolysate panel (R5) and what fills each cell

Planned as 4 x 3, to be cut to 3 x 3 by merging rows 2 and 3. Status: on file = a committed
script already produces it; running = a job is producing it; to launch = planned round;
to build = code not yet written.

| row | panel | content | source | status |
|---|---|---|---|---|
| 1 concept | a | the task: genotype x environment -> phenotype; a compound enters the cell graph transformer as a token the genes attend to; the strain is read at its deleted genes; the host is read at the mean (empty genotype) | draw.io schematic of the 038 `EnvironmentEncoder` and the 040 host head | to build |
| 1 | b | what is trained on: the pooled chemogenomic table (four screens, 6.04M cells, 5,463 compounds, 5,863 genes; which carry a molar dose; which carry dose series) beside the private Bioscreen data (977 wells, six inhibitors, 63 combinations, three isoboles, one strain) | 033 `store_against_plan.py` tables; 040 `wetlab_table.py` | on file |
| 1 | c | how models are evaluated: compound-cold folds and the per-compound centered Spearman against each compound's reliability ceiling; the wet-lab test = growth/no-growth over 63 and fitness over the grown, against Bliss and Loewe | 038 protocol; 040 `mixture_rules.py` | on file |
| 2 benchmarks | d | the ladder on the corrected store: nested ridge on FCFP4 counts (0.359) vs kernels, kNN and twelve embeddings | 038 round 1 (slurm 3374) | on file |
| 2 | e | CGT vs ridge: bilinear control, environment-encoder head, nine-seed ensembles, the stack; per-compound view | 038 round 2 (GilaHyper 3700 / 3701, ETA 07:00) and round 11 (Delta, to queue) | running |
| 2 | f | learning curve in fitted compounds (ridge on file; encoder to rerun) and the effect of adding Hoepfner / HET sources to the gene-level task | 038 round 17 (Delta), 040 round 1 `sources` arms | to launch |
| 3 environment | g | dose: Vanacloig IC30 vs ex21 IC30 per shared compound (cross-medium potency); the dose-aware encoder's predicted single-agent curves vs ex21 | 040 `single_agent_curves.py` (on file); 040 round 1 `dose` arms | partly |
| 3 | h | mixture rule on public data: HET pair profile = mean of singles; emergent genes inside noise | 040 `het_mixture_composition.py` | on file |
| 3 | i | predicted vs measured profile (furfural, HMF) for ridge and for the CGT encoder; GO overlap against the gene-mean control | 040 `inhibitor_profiles.py` (ridge on file; CGT after 3701) | partly |
| 4 wet lab | j | ex21 single-agent curves with the model's predicted curves (model trained on anchors only, or calibrated on ex21: both shown) | 040 round 1 | to launch |
| 4 | k | the 63 combinations: observed vs Bliss, Loewe and the model, by number of inhibitors; growth AUROC and fitness Spearman in the panel | `mixture_rules.py` (rules on file); model to launch | partly |
| 4 | l | isoboles: observed grid, Loewe front, the model's predicted front; excess over Bliss | `mixture_rules.py` (on file); model to launch | partly |

The claim the bottom row is for: can a model trained on public screens predict the bAID host
under these environment perturbations. The bars are on file (Loewe growth AUROC 0.958,
Bliss-from-singles fitness Spearman 0.573, served call) and the model must beat them to
earn the panel; if it does not, panels j to l show the rules and the model side by side
and the text says so. After this round: joint training over the other datasets.

**Round 1 on Delta (account bfjt-delta-gpu, gpuA40x4, never -preempt; the 031/032 packs
are on bgcg and the 025 jobs on bflt).** 038 rounds 11 (four 15-config halves), 16 and 17
(six jobs, configs on file) as soon as the Delta smoke passes; 040 mixture round 1 (about
ten arms x three seeds, 12 h files at PARALLEL=2) as soon as `train_mixture.py` passes its
CPU smoke.

## 2026.10.10 - Round 1 launched: where each job runs and what it decides

**Delta** (`bfjt-delta-gpu`, the account with no running campaign; `bgcg` carries the
031/032 packs and `bflt` the 025 jobs), partition `gpuA40x4`, 12 h, never `-preempt`. Two
rsynced trees: `/scratch/bbub/mjvolk3/torchcell-038` and `/scratch/bbub/mjvolk3/torchcell-040`.
Three things had to be staged before anything ran, each found by a failing smoke: the
033 cell table and the 031 embedding tables (the trainers read them at GilaHyper absolute
paths, now overridable by `TC038_*` and `TC040_*`), the genome tier
`$DATA_ROOT/torchcell-genomes/sgd_S288C_R64-4-1_20230830` plus the GO release, which the
genome registry resolves and which did not exist on Delta, and the Vanacloig dev store for
the IC30 doses. Smoke 22786687 trained and scored in 22 min, which is what licensed the
round.

| job | round | what it decides |
|---|---|---|
| 22787115, 22787117 | 038 round 11, environment encoder, seed groups 3-5 and 6-8 | the nine-seed ensemble of the head that ties ridge, on the corrected store |
| 22787118, 22787120 | 038 round 11, bilinear control, same seed groups | the paired control for that ensemble |
| 22787121 | 038 round 16 | whether 150 epochs lifts the encoder (its validation loss was still falling at 50) |
| 22787122 | 038 round 17 | the learning curve in fitted compounds for the encoder, beside the ridge curve already on file |

**GilaHyper** keeps the faster lane. 038 round 2 (slurm 3700 bilinear control, 3701
environment encoder) has been running since 02:10 and finishes about 06:00; the first two
040 mixture arms chain onto those two cards as they free. The ten round-1 mixture files
are one arm each, five compound-cold folds of fold seed 0, three seeds per fold; fold
seeds 1 and 2 are round 2, run only on the arms that survive. The arms are the best-guess
joint arm (`sources: vanacloig`, `host_train: anchors+ex21`, `dose: film`, one environment
layer, `lambda_host` 1) and nine variants that change one field each: dose off, host
training on the anchors alone, the two-stage fine-tune with the head alone and with the
environment encoder unfrozen, two environment layers, `lambda_host` 0.1, and Hoepfner and
Hillenmeyer added as gene-level sources.

**Measured while assembling the data, each a correction to the plan.** Of the 32 published
Vanacloig compounds, 28 carry a dose that converts to molar; ethanol, gamma-valerolactone,
isobutanol and methyl methanesulfonate are percent-basis and leave the dose-aware panel.
Acetic acid already has a fingerprint row, so only formic and lactic acid were featurized
from SMILES. Hoepfner serves two compound names against two structures each, so the
compound table is keyed by InChIKey, not by name. Hillenmeyer's three media are pooled
behind one source token and its heterozygotes are read by the deletion operator; both are
stated simplifications.

## 2026.10.10 - Round 1, the first two arms: the model does not predict the wet lab

GilaHyper slurm 3720 (joint arm, `host_train: anchors+ex21`) and 3721 (anchors only, the
zero-shot host), each 2 h 04 at PARALLEL=2: five compound-cold folds of fold seed 0, three
seeds per fold, the recipe of 038 round 10 with a compound token per inhibitor, a log-molar
dose FiLM and the host head. ex23 and the isoboles were never trained on. Values are the
median over the five folds of the three-seed ensemble; the reference is the model-free rule
on the same wells.

| readout (served call) | joint | anchors only | reference |
|---|---|---|---|
| ex23 growth over 63 combinations, AUROC | 0.475 | 0.295 | Loewe **0.958** |
| ex23 worst-scenario Spearman, all 63, no growth as 0 | -0.009 | -0.319 | Loewe **0.772** |
| ex23 fitness over the 21 that grew, Spearman | 0.377 | -0.346 | Bliss **0.573** |
| isobole ex26 / ex27 / ex28, mean observed minus predicted | -0.75 / -0.71 / -0.75 | -0.60 / -0.42 / -0.59 | Bliss -0.08 / -0.45 / -0.02 |
| Vanacloig compound-cold, centered Spearman | 0.261 | 0.266 | ridge **0.359** |
| ex21 over 180 wells, Spearman | 0.945 (train fit) | **0.171** (held out) | |

**The claim fails, and it fails cleanly.** Growth over the 63 combinations is at chance for
the joint arm (0.475, fold range 0.442 to 0.502) and below chance for the anchors arm
(0.295); the worst-scenario ranking is zero and negative. Neither arm approaches Loewe on
either growth readout, and both isobole surfaces sit further from the observed grids than
Bliss does. On the gene-level task both arms read 0.26 against ridge at 0.359, in line with
038 round 2, so the wet-lab failure is not a model that was working elsewhere.

**Two things in the table are worth keeping.** The anchors-only arm, which never sees a
Bioscreen well, reads a Spearman of 0.171 over the 180 ex21 titration wells: weak, but it
is a genuine zero-shot transfer from IC30 anchors in a different medium and strain to a
dose response measured here. And the joint arm's fitness Spearman of 0.377 over the 21
grown combinations is non-trivial for a model that saw no combination, though it is below
Bliss at 0.573 and that arm did train on the single-agent curves.

**Hypothesis (untested), and the queued rounds test it.** The anchors arm is not
uninformative but anti-correlated (-0.346 fitness, -0.319 worst-scenario, both well below
zero with fold ranges that mostly exclude it), which is a systematic inversion rather than
noise. The reading to check is that a host head fitted on single-compound anchors alone
extrapolates the wrong way as compounds are added, so predicted fitness rises with the
number of inhibitors. The dose-off arms separate the FiLM from the token set, the two
fine-tune arms separate calibration from joint training, and a per-combination residual
against the inhibitor count settles the inversion directly; the first two are queued on
Delta (22788113 to 22788118) and the third is an analysis on the predictions already
written.

## 2026.10.10 - Round 1, the dose ablation: the dose FiLM makes the mixture readout worse

The two dose-off arms ran on Delta (bfjt-delta-gpu, jobs 22788113 and 22788114) and
finished between 13:58 and 16:43 CDT. Their per-fold CSVs sit in the Delta tree
(`/scratch/bbub/mjvolk3/torchcell-040/.../results/mixture/r1_mix_van_{ex21,anchors}_off_L1_lh1/`)
and are not yet rsynced here; the numbers below are the same values read from the W&B
run summaries (project `torchcell_040-inhibitor-synergy-wetlab`, runs
`mix_van_ex21_off_L1_lh1_f0` to `_f4` and `mix_van_anchors_off_L1_lh1_f0` to `_f4`).
Median over the five folds of fold seed 0, three-seed ensemble, served call, beside the
two dose-aware arms from GilaHyper 3720 / 3721.

| readout | joint, dose FiLM | joint, dose off | anchors, dose FiLM | anchors, dose off | reference |
|---|---|---|---|---|---|
| ex23 growth over 63, AUROC | 0.475 | 0.586 | 0.295 | 0.413 | Loewe **0.958** |
| ex23 worst-scenario Spearman, all 63 | -0.009 | 0.155 | -0.319 | -0.221 | Loewe **0.772** |
| ex23 fitness over the 21 grown, Spearman | 0.377 | 0.251 | -0.346 | -0.310 | Bliss **0.573** |
| isoboles ex26 / ex27 / ex28, observed minus predicted | -0.75 / -0.71 / -0.75 | -0.39 / -0.39 / -0.46 | -0.60 / -0.42 / -0.59 | -0.60 / -0.42 / -0.59 | Bliss -0.08 / -0.45 / -0.02 |
| Vanacloig compound-cold, centered Spearman | 0.261 | 0.276 | 0.266 | 0.252 | ridge **0.359** |
| ex21 over 180 wells, Spearman | 0.945 (train fit) | 0.433 (train fit, one value per compound) | 0.171 (held out) | 0.066 (held out) | |

**Switching the dose off raises the growth AUROC in both host regimes** (0.475 to 0.586
joint, 0.295 to 0.413 anchors), turns the worst-scenario ranking positive for the joint
arm, and halves the isobole residual (-0.75 to -0.39 on ex26). The dose-off arm cannot
fit ex21 (every well of a compound gets one value, so 0.433 is the between-compound
potency ordering alone), yet it predicts the combinations better than the arm that fits
the curves at 0.945. The isobole residuals are negative for every arm: observed minus
predicted below zero means the model predicts MORE growth in the interior cells than was
seen, and the dose-aware arms predict the most. Hypothesis (untested): the host head
fitted on single-agent curves composes compounds near-additively or less, so adding an
inhibitor lowers predicted fitness by less than it lowers observed fitness, and the
sharper the single-agent fit the larger that gap. Every arm still sits below the
model-free rule on every wet-lab readout; the dose ablation changes which arm is worst,
not the conclusion.

**On reading the zero-shot single-agent numbers.** The anchors arm's per-compound ex21
Spearman swings from -0.93 to +0.91 for the same compound across folds (HMF: -0.72,
-0.93, -0.77, 0.90, 0.91 for folds 0 to 4), and within one compound fitness falls
monotonically with dose, so a model that only knows "more dose, less growth" scores near
1.0 per compound. The per-compound Spearman is therefore not a zero-shot readout. The
readout that carries information is the pooled Spearman over all 180 wells, which needs
the six potencies ordered correctly, and it reads 0.171 (dose FiLM) and 0.066 (dose
off): at most a weak potency ordering, with six compounds.

## 2026.10.10 - Where a learned model can still earn the panel: regimes the rules cannot enter (proposal, awaiting a decision)

The question the user asked after the round-1 table: the data are the paper's
confirmation that Loewe or Bliss is the best predictor when the singles are known, so where
can a model show an advantage. The answer is the information regimes in which the
model-free rules have no prediction at all, or have one they cannot improve with data.
Every regime below needs its own honest baseline, because "the rule cannot run" is not the
same as "the model is good".

| regime | what is known at prediction time | what is predicted | what Loewe / Bliss can do | honest baseline for the model | data (served call) | status |
|---|---|---|---|---|---|---|
| A. zero-shot singles | public screens only; no Bioscreen well | the six ex21 dose-response curves, or their IC30s | nothing; both need the single curves | public IC30 transfer where it exists (Vanacloig furfural 8.0 and HMF 3.3 mM, Hoepfner acetate 125 to 150 mM) and the all-compound mean curve | 180 wells, six potencies | measured: pooled Spearman 0.171 (anchors, dose FiLM), 0.066 (dose off) |
| B. one inhibitor unknown | five ex21 curves and the public anchors; the sixth inhibitor's curve withheld | the sixth curve, and every ex23 combination that contains it | nothing for those combinations unless given a curve | the hybrid: the model's predicted curve for the sixth compound fitted as a Hill and fed to Loewe and Bliss; the ceiling is Loewe with the measured curve | six leave-one-inhibitor-out folds; 32 combinations contain any given inhibitor (15 pairs and 20 triples share it unequally) | to build: a `loco_ex21` host regime that folds over the inhibitor, and the hybrid scorer on the `_host.npy` predictions |
| C. singles and pairs known, higher orders predicted | ex21 and the 15 ex23 pairs | the 20 triples, 15 quadruples, 6 quintuples and the sextuple (48 combinations, 3 grew) | a fixed prediction; neither rule learns from the pairs | the calibrated rule: Loewe or Bliss with one interaction parameter fitted on the 15 pairs (a global combination index) | growth boundary over 48, fitness over 3 | to build: pairs become trainable in this arm only, which changes the pre-registered test set and must be stated |
| D. leave-one-pair-out | ex21 and 14 of the 15 pairs | the fifteenth pair | fixed prediction | the same calibrated rule, refitted per fold | 15 folds; 12 pairs grew, so the fitness ranking over 12 is the test | to build, with C |

The pairs are where the fixed rules are weakest, which is also why they are the regime
with room. On the 15 pairs alone (`mixture_rules.py`, `results/mixture_scores.csv`,
served call, subset 2): Bliss from the ex21 fits ranks the twelve grown pairs at Spearman
0.748 but calls growth at AUROC 0.333; Loewe calls growth at 0.611 and ranks at 0.357.
The two rules also disagree by chemical class: for the weak-acid pairs Loewe is close and
Bliss predicts 0.93 to 1.00 against observed 0.40 to 0.69, while for the aldehyde + acid
pairs Bliss is close and Loewe predicts 0.07 to 0.24 against observed 0.42 to 0.46; and
5-HMF with lactic or levulinic acid grew in no well although both rules predict growth
(Bliss 0.54 to 0.57, Loewe 0.09 but above tau). A rule that picks Loewe within a class
and Bliss across classes was tried as a scratch check (not a committed script, so no
number is recorded here) and does not beat Loewe on growth, because it inherits Bliss's
call on those two 5-HMF pairs. Which rule a pair follows is therefore the quantity a
learned model could predict from compound features, and regime D is its test.

What this costs. Regimes A and B need no new wet-lab data and keep ex23 untouched as a
test set; B needs one trainer regime and one analysis script. Regimes C and D make the 15
pairs training data in those arms, so the pre-registered "ex23 is never trained on" holds
only for the arms of rounds 1 and 2 and the note must say so. Each regime is one arm of
five or six folds at about two hours per card at PARALLEL=2. The decision to take is
whether to open the pairs to training (C and D); A and B can start on the next free card.

## 2026.10.10 - The boundary is the readout: no single rule draws it, and the choice splits by the acid

The user's reframing, agreed and adopted: for several inhibitors the useful prediction is
the shape in dose space inside which the host still grows, the line between alive and
dead at different concentrations. The fixed-dose ex23 set gives one point per
combination, so the boundary lives in the three isobole grids and in the single-agent
curves that fix its intercepts. Two committed scripts now score it:
`mixture_rules.py` (boundary table `results/isobole_boundary.csv`, the Bliss front on the
isobole figure; [[experiments.040-inhibitor-synergy-wetlab.scripts.mixture_rules]]) and
`rule_choice.py` (which rule each pair follows and whether anything predicts the choice;
[[experiments.040-inhibitor-synergy-wetlab.scripts.rule_choice]]).

**Bliss multiplies, Loewe adds.** Bliss multiplies surviving fractions, a sum on the log
scale, two independent hits. Loewe adds doses, each as a fraction of the dose of that
compound alone that gives the same effect, a shared capacity. With the steep ex21 curves
the two fronts sit far apart, so which one a pair follows is a large, visible difference.

**The grids (served call, 81 interior cells each, Jaccard of the predicted and observed
alive regions):** furfural x acetic acid follows Bliss (0.73 against Loewe 0.29; Loewe
kills 10 cells that grew), 5-HMF x acetic acid follows Bliss (0.70 against 0.29; 12 cells),
formic acid x acetic acid follows Loewe (0.88 against 0.44; Bliss keeps 31 dead cells
alive). The growth edge along acetic acid tracks the winning rule within a third to a half
of a dose step and misses by one to three steps under the other. The observed region is
scored raw and after a monotone closure (more inhibitor never rescues growth, so a grown
cell above a dead one is a call error and is dropped); only the 5-HMF grid changes, five
island cells go, and its Bliss overlap rises to 0.77 against Loewe 0.39. Any predicted
surface gets the same closure before it is scored as a boundary.

**The 15 fixed-dose pairs agree with the grids they share and refine the split.** All six
acid + acid pairs follow Loewe. The eight aldehyde + acid pairs split by the acid, not the
class: with acetic or formic acid the aldehyde pairs follow Bliss (HMF+AA, HMF+FA, FF+AA,
FF+FA), with lactic or levulinic acid they follow Loewe (HMF+LA, HMF+LVA, FF+LA, FF+LVA),
and the two 5-HMF pairs among those grew in no well, which Loewe calls and Bliss does not.
This corrects the earlier reading in this note that the split is "aldehyde with acid
follows Bliss": that holds for the two short acids only. Hypothesis (untested): acetic
and formic acid at the ex23 doses act on a different capacity from the aldehydes, while
lactic and levulinic acid at 20 and 6 g/L add to the same load the aldehydes draw on; the
ex21 curves alone do not say which.

**Can the choice be predicted? Not on these 15 pairs** (leave-one-pair-out, prediction =
w Loewe + (1 - w) Bliss): pure Loewe has the lowest error (mean absolute 0.197) and pure
Bliss the best growth call (0.800) and ranking of the grown pairs (0.748); a fitted
global w, the class rule, a fitted class w and six profile-similarity predictors all land
between the two (error 0.196 to 0.219). The no-parameter class hierarchy over all 63
combinations (Loewe within class, Bliss across) reads growth AUROC 0.887 against Loewe's
0.958, with the smallest fitness error over the grown combinations (RMSE 0.148 against
0.208); it does not replace Loewe for the boundary. For the panel: the boundary is
Loewe for the acids and Bliss for the short-acid + aldehyde pairs, both model-free; the
rule choice is the quantity a model would have to supply, and nothing on file supplies
it yet.

**Round 1 closes on GilaHyper; round 2 opens.** The two fine-tune arms (3748 head only,
3750 head + environment encoder, 2 h 04 and 2 h 06) read growth AUROC 0.556 and 0.582,
worst-scenario 0.093 and 0.154, isobole residuals -0.41 to -0.57, Vanacloig 0.271 and
0.256: the same band as the dose-off joint arm, still below every rule. The two-layer
joint arm runs as 3791. Round 2 began with the replication of the dose-off joint arm on
fold seeds 1 and 2 (`conf/mixture/r2_mix_van_ex21_off_L1_lh1.yaml`, ten configs, job
3792, 24 GB requested against a measured 9.7 GB peak). On Delta all six remaining
round-1 arms started about 21:20 CDT (22788115 to 22788120); 22788115, 22788116 and
22788117 duplicate arms GilaHyper has run or is running and can be cancelled. The 038
Delta rounds 11, 16 and 17 finished (W&B project
`torchcell_038-env-chemgen-vanacloig-cgt-corrected`, last run 21:32 CDT); their CSVs
await an rsync from the Delta tree before the 038 note can score them.
