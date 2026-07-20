---
id: sz4ska0lk5agzfrct0qnyig
title: Construction_validation_doubles
desc: ''
updated: 1784579566973
created: 1784579566973
---

## 2026.07.20 - Doubles for BOTH Triple Reconstruction and Assay Validation

Script: `experiments/010-kuzmin-tmi/scripts/construction_validation_doubles.py`
Data:   `experiments/010-kuzmin-tmi/results/construction_validation_doubles.csv`

### Why the pure set-cover was the wrong objective

The essential wet-lab task is a small set of double mutants that (a) reconstructs many
top-ranked triples AND (b) has **variance in DMF / interaction** to validate the new
echo-plating assay against (vs plate-stamping). The triple-coverage set-cover
([[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]]) does
(a) with 8 doubles but fails (b): those 8 are all **near-neutral** — Costanzo DMF span
0.15, `|ε| ≤ 0.039`, and **zero significant interactions** — so the assay would have almost
nothing to discriminate. An assay validated only on near-1.0 fitness and null interactions
has not been shown to detect signal.

### Two tiers, unioned (13 doubles)

- **coverage (8)** — the greedy set-cover doubles; reconstruct **all 31** within-10 top-k
  triples.
- **validation (5)** — added for dynamic range + real signal: the doubles with a
  **significant Costanzo interaction** (`P<0.05 & |ε|>0.08`, both signs) plus the
  **lowest- and highest-DMF** doubles.

Result: 31/31 triples still reconstructed, DMF range **0.513–1.178** (span 0.665, 4.4× the
coverage-only 0.152), ε from **−0.130 to +0.098**, **3 significant interactions**.

| tier | double | DMF ± SD | ε | p | sig | # triples |
|------|--------|----------|--:|--:|:---:|:---:|
| validation | YGL087C+YJR060W | 0.513 ± 0.172 | −0.075 | 0.27 | | 0 |
| validation | YER079W+YPL081W | 0.863 ± 0.098 | **−0.130** | 0.035 | ✓ | 0 |
| validation | YJR060W+YKL033W-A | 0.871 ± 0.023 | **−0.082** | 0.007 | ✓ | 0 |
| coverage | YJR060W+YPL046C | 0.966 ± 0.018 | +0.003 | 0.45 | | 4 |
| coverage | YDR057W+YLL012W | 1.000 ± 0.069 | −0.036 | 0.30 | | 5 |
| coverage | YLR312C-B+YPL081W | 1.017 ± 0.042 | −0.019 | 0.36 | | 3 |
| coverage | YLL012W+YPL046C | 1.025 ± 0.053 | −0.010 | 0.43 | | 5 |
| coverage | YDR057W+YPL081W | 1.035 ± 0.045 | +0.039 | 0.26 | | 3 |
| coverage | YDR057W+YER079W | 1.096 ± 0.054 | +0.012 | 0.44 | | 5 |
| coverage | YER079W+YLR312C-B | 1.099 ± 0.039 | −0.027 | 0.29 | | 5 |
| coverage | YBR203W+YPL046C | 1.118 ± 0.017 | +0.036 | 0.077 | | 5 |
| validation | YDR057W+YGL087C | 1.137 ± 0.024 | **+0.098** | 0.001 | ✓ | 3 |
| validation | YDR057W+YLR312C-B | 1.178 ± 0.026 | +0.047 | 0.12 | | 1 |

![](assets/images/010-kuzmin-tmi/construction_validation_doubles.svg)

All 44 measured within-10 doubles ranked by DMF, with the 13 selected marked by tier
(coverage = blue `(C)`, validation = red `(V)`, unused = gray; significant ε ringed) — the
tiered companion to `constructed_10_dmf_forest` (which predates the validation tier and marks
only the 8 set-cover doubles):

![](assets/images/010-kuzmin-tmi/construction_validation_doubles_forest.svg)

Notes: DMF/SD/SE and ε/p are the published Costanzo values (SD = sample SD over 4 colonies,
directly comparable to the assay's colony SD — see
[[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]]). The low-DMF end is
dominated by CBF1/YJR060W (its single defect), so a low DMF there is a fitness effect, not
an interaction. Two of the three significant doubles (YER079W+YPL081W, YJR060W+YKL033W-A)
enable no top-k triple — they are pure validation targets; the third (YDR057W+YGL087C,
strongest positive ε) also rebuilds 3 triples. The set is tunable: drop the two triple-less
validation doubles for an 11-double set, or add more near-significant ε to stress the assay.

### There are only 3 significant interactions in the whole panel

Surveyed across all 66 panel-12 doubles and all three sources: exactly **3** clear
Costanzo's bar (`P<0.05 & |ε|>0.08`), and all 3 are within-10 and already in the
validation tier — YDR057W+YGL087C (+0.098), YJR060W+YKL033W-A (−0.082),
YER079W+YPL081W (−0.130). None hide in the dropped-gene doubles; Kuzmin2018/2020 have
too few measured pairs to reach significance. So "construct the significant ones" = these
3, already selected. The panel is near-neutral by design, so strong interactions are scarce.

**High-|ε| but insignificant** doubles (e.g. YGL087C+YJR060W, ε=−0.075, p=0.27, DMF SD
0.172) arise from **large SE(ε)**: ε significance is |ε|/SE(ε), and SE(ε) folds in the SD of
the double-mutant fitness AND both single-mutant fitnesses (n=4 colonies each). A wide
colony SD or a noisy constituent single (CBF1, RPS9A) inflates SE(ε) so even a sizable ε
is indistinguishable from the multiplicative null. Mirror case: YBR203W+YDR057W (ε=+0.062,
below the magnitude bar) is highly significant (p=1.4e-5) because its DMF SD is 0.010. For
assay validation the 3 *significant* doubles are the strong anchors; high-SD/high-|ε| doubles
validate fitness dynamic range, not interaction detection.

**Sampling depth beats SD — the assay's key lever.** The resolving quantity is SE, not SD:
`SE = SD/√n`. SD is a fixed biological property of a strain (colony-to-colony spread), but
**n is ours to choose** — so a "high-SD, insignificant" double is not disqualified, it just
needs more colonies. Because our goal is **sampling specific predictions precisely (depth),
not screening many strains (breadth)**, the right trade is few doubles × many colonies each:
plate a high-SD double like YGL087C+YJR060W (SD 0.172) at large n and its SE shrinks by √n
until the interaction resolves. This is the opposite of a genome-wide screen (breadth, low n
per pair) and is what echo dispensing enables — dense, uniform replication of a handful of
constructs. So the validation tier is chosen for interesting *point estimates* (DMF extremes,
high |ε|); driving each to significance is a sampling decision at the bench, not a property of
the reference. Caveat: deep sampling lowers *our* SE, but the Costanzo reference stays n=4 —
so once our error bar is tighter than theirs, a disagreement may reflect the reference's own
noise as much as the assay's, making our measurement the more trustworthy of the two.

Related: [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]],
[[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]].
