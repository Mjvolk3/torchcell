---
id: vhrmvscpbx9umjmbw4o8wef
title: 029 Solid Growth Ko
desc: ''
updated: 1789739318148
created: 1789739318148
---

## 2026.09.18 - Why a new build, and what it changes

Successor of [[experiments.025-solid-growth]] decided from the S3 closure recompute
([[experiments.025-solid-growth.s3-closure]], `notes-tex/025-s3-closure/`). Three changes:

1. Deletions only. The query ([[experiments.029-solid-growth-ko.queries.001_ko_solid_growth]])
   keeps a record only when every perturbation is a deletion (`sga_kanmx_deletion`,
   `sga_natmx_deletion`). TS, DAmP, suppressor and generic alleles leave, so no measured allele
   shares a gene name with a deletion or with the SGD essentiality 0. About a fifth of the
   Kuzmin trigenic rows carry a TS array allele or a non-deletion query allele and leave with
   them; the retained triples keep their 010 split by gene-set transfer.
2. No mean-merge. The deduplication stage is skipped (`deduplicator=None`); the aggregator
   groups every source entry per genotype and each keeps its fitness, SD, score, p-value,
   temperature and screen. Which entry becomes a training label is a label policy applied at
   read time (to be written; the trainer masks a record with two values until then).
3. Both temperatures kept. Costanzo 26 C and 30 C and Kuzmin's 26 C screens are separate
   entries; the policy prefers 30 C for Costanzo and Kuzmin as screened (all Kuzmin selection
   steps ran at 26 C because the diagnostic array carries TS alleles).

Goal stated by the user: include as much deletion data as possible and test whether the
trigenic interaction can be reconstructed from the query's own singles and doubles better
than 025's r 0.230 (digenic 0.445). If it cannot, the join is the problem; if it can, the
approach gains confidence.

Build: `experiments/029-solid-growth-ko/scripts/query.py` under
`scripts/gh_query_build_001.slurm`; root on `/db/experiments/029-solid-growth-ko-001-ko-build`
symlinked from `$DATA_ROOT/data/torchcell/experiments/029-solid-growth-ko/001-ko-build`.
Stage optimizations landed for it in commit 2824b6335 (conversion byte passthrough, batched
raw writes, byte-keyed aggregation).
