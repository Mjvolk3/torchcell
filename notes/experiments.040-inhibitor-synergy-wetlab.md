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
