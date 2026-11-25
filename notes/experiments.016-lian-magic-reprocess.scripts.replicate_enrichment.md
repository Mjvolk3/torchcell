---
id: jicul1dwxbi4rcz3kot40mu
title: Replicate_enrichment
desc: ''
updated: 1795581737023
created: 1795581737023
---

## 2026.11.24 - What this script is for

Step 7 of `experiments/016-lian-magic-reprocess/scripts/reproduce.sh`. The Lian 2019 MAGIC
pipeline computes `log2(CPM[furfural after] / CPM[untreated before])` per round and per
biological replicate and then, in `enrichment.py`, keeps only the mean and the SD over the
three replicates. This script writes those three numbers out instead, one column per
(round, replicate), as `guide_enrichment_replicates.tsv` (sha256
`46eaf7957ef410395177acf842ed0d36c12fd7c7f20ac5f5b2ee7dfd47d698ab`), the input
`CrisprMagicLian2019Dataset` now reads for one record per guide x round x replicate.

It asserts rather than assumes that it is the same arithmetic: the mean and sample SD over
the emitted columns must reproduce the pinned `guide_enrichment_final.tsv` for every guide
and round at 1e-9 (measured 1.3e-15 over all 91,229 detected guides, 2026-11-24), and the
emitted row order must stay the `guide_map.tsv` order, because the loader joins the CRISPRd
design library onto the enrichment rows positionally. The `is_control` and `corrupted_gene`
flags are carried over from the pinned mean table, not recomputed.

Replicate labels printed in the run log are the authors' own SRA library pairs
(`Ff1a/Ff1b` ... `3Ff3a/3Ff3b`), which is what the loader stores as `replicate_id`.

See [[torchcell.datasets.scerevisiae.lian2019]] (2026.11.24 section) and
[[experiments.016-lian-magic-reprocess.README]].
