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
