---
id: 556g3uiuswhlinb392fx63z
title: Generate_040_cql
desc: ''
updated: 1792469998935
created: 1792469998935
---

## 2026.10.19 - The 040 query against KG 4.0

Writes `experiments/040-inhibitor-synergy-wetlab/queries/env_chemgen_mixture_040.cql`, five
`UNION ALL` blocks in the shape of the 033 query: Vanacloig 2022, Hoepfner 2014, Hillenmeyer
2008 HET, Wildenhain 2015, and the private `InhibitorBioscreenVolk2021Dataset`, against release
`2026.10.17-4295f7ca` (KG 4.0, torchcell 1.8.3). Served counts per dataset are pinned in
`SERVED_COUNTS` from the release snapshot, 6,705,334 records in all, and `build_040_store.py`
reads them to report returned against served.

Two shapes of block. The four screens use the 033 gene clause (every perturbed gene in
`$gene_set`, at least one perturbation). The private wells carry `Genotype(perturbations=[])`,
one strain in every well, so that clause would drop all 977; their block has no genotype
clause at all. Every block carries one partition marker for the partitioned raw stage, and the
header must not spell the marker, because the raw stage counts it per block (the first smoke,
slurm 5906, failed on a header that did).

Edit the generator, never the `.cql`.
