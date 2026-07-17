---
id: qg0fw22zp6wg0giuyqkcqam
title: Validation_panel_smf_reference
desc: ''
updated: 1784302650919
created: 1784302650919
---

## 2026.07.17 - SMF Reference (Costanzo + Kuzmin) for the Droplet Validation Panel

Script: `experiments/010-kuzmin-tmi/scripts/validation_panel_smf_reference.py`
Data:   `experiments/010-kuzmin-tmi/results/validation_panel_smf_costanzo_kuzmin.csv`

Single-mutant fitness (SMF) `fitness ± s.d.` queried from the three SMF datasets
(Costanzo2016, Kuzmin2018, Kuzmin2020) for the **droplet-assay validation panel** —
the public-data comparison column for the wet-lab 2.5 nL / 5.0 nL measurements. This
list includes **YLR313C/SPH1** (authentic gene) and **YPL056C/LCL1**, which the
inference panel-12 never queried (it used YLR312C-B and ELC1 instead). Values come
from the small `Smf*` LMDBs keyed by perturbed systematic gene name (same mechanism as
[[experiments.010-kuzmin-tmi.scripts.investigate_YLR313C_smf_and_interactions]]);
a gene absent from a study is an empty cell (honest missing, never guessed).

| Common | ORF | Costanzo2016 SMF ± SD | Kuzmin2018 SMF ± SD | Kuzmin2020 SMF ± SD |
|--------|-----|-----------------------|---------------------|---------------------|
| SPH1 | YLR313C | 0.984 ± 0.027 | — | — |
| LCL1 | YPL056C | 0.980 ± 0.082 | — | — |
| COS111 | YBR203W | 1.037 ± 0.046 | 1.041 | — |
| — | YKL033W-A | 1.033 ± 0.101 | — | — |
| CBF1 | YJR060W | 0.590 ± 0.114 | 0.752 | — |
| RPS9A | YPL081W | 0.955 ± 0.155 | — | — |
| — | YER079W | 1.039 ± 0.117 | — | — |
| MMS2 | YGL087C | 0.996 ± 0.094 | 0.985 | — |
| YEH1 | YLL012W | 0.993 ± 0.080 | — | 0.969 ± 0.004 |
| ELC1 | YPL046C | 1.043 ± 0.064 | 0.983 | — |
| YOS9 | YDR057W | 1.044 ± 0.053 | — | — |
| — | YLR312C-B | 1.085 ± 0.044 | — | — |

Coverage: Costanzo 12/12, Kuzmin2018 4/12, Kuzmin2020 1/12 — Costanzo2016 is the
genome-wide single-deletion reference; the Kuzmin SMF values are byproducts of their
trigenic arrays (only query/array strains appear, mostly without a released per-strain
SD). Note **SPH1/YLR313C (0.984)** vs the merged small-ORF strain **YLR312C-B (1.085)**
still disagree by ~0.10 — the authentic-gene-vs-merged-feature split from the YLR313C
investigation, now visible in the validation panel. The CSV also carries `*_strain_id`
provenance columns (which Costanzo `_sn` / Kuzmin strain each value came from).

Related: [[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover]].
