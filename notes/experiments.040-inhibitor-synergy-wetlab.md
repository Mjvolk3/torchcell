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
