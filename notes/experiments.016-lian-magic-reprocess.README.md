---
id: vt6a944n3ekb5tb4ah1xk9c
title: README
desc: ''
updated: 1783967154635
created: 1783967154635
---

## 2026.11.24 - Per-replicate step added to the pipeline (#1177)

`scripts/replicate_enrichment.py` is step 7 of `reproduce.sh`. It writes the nine
per-replicate `r{rnd}_rep{rep}_log2fc` columns the pipeline already computed internally
(`enrichment.py` kept only their mean and SD) as
`guide_enrichment_replicates.tsv`, sha256
`46eaf7957ef410395177acf842ed0d36c12fd7c7f20ac5f5b2ee7dfd47d698ab`, which is now the input
of `CrisprMagicLian2019Dataset` (one record per guide x round x biological replicate).

It refuses to write a table that disagrees with the pinned mean table: the mean and sample
SD over the three emitted columns must reproduce `r{rnd}_log2fc_mean` / `_sd` for every
guide and round at 1e-9. Measured on the 2026-07 counts still in
`$DATA_ROOT/lian2019-magic-reprocess/counts/`: max absolute difference 1.3e-15 in all
three rounds over all 91,229 detected guides, which is also the evidence that those counts
are the ones behind the pinned table.

Replicate labels are the authors' own, not positions: SRA `PRJNA504483`'s run report gives
`LibraryName` = `SampleName` = `Ff1a` ... `3Ff3b`, and `run_manifest.tsv`'s `library`
column matches it for 21/21 runs. The run report is mirrored beside the tables as
`data/sra_runinfo_PRJNA504483.csv`. Details in
[[torchcell.datasets.scerevisiae.lian2019]].
