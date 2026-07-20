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

Notes: DMF/SD/SE and ε/p are the published Costanzo values (SD = sample SD over 4 colonies,
directly comparable to the assay's colony SD — see
[[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]]). The low-DMF end is
dominated by CBF1/YJR060W (its single defect), so a low DMF there is a fitness effect, not
an interaction. Two of the three significant doubles (YER079W+YPL081W, YJR060W+YKL033W-A)
enable no top-k triple — they are pure validation targets; the third (YDR057W+YGL087C,
strongest positive ε) also rebuilds 3 triples. The set is tunable: drop the two triple-less
validation doubles for an 11-double set, or add more near-significant ε to stress the assay.

Related: [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]],
[[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]].
