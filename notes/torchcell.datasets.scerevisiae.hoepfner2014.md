---
id: dm9g91o4fp9u1qnaoyo00du
title: Hoepfner2014 HIP/HOP Chemogenomic Atlas
desc: ''
updated: 1783988969952
created: 1783988969952
---

## Dataset overview

`torchcell/datasets/scerevisiae/hoepfner2014.py` — `EnvChemgenHoepfner2014Dataset`
(`env_chemgen_hoepfner2014`). The Novartis HIP-HOP chemogenomic atlas (Hoepfner et al. 2014,
Microbiol Res, doi:10.1016/j.micres.2013.11.004): ~1776 compounds profiled at IC30 against the
diploid deletion collections, stored as `env × genotype → (adjusted) MADL sensitivity_score`.

- **29,996,238 records** (L0-L4 PASS), **5,833** measured ORFs.
- **HIP** (heterozygous, YSC1055): one of two copies deleted → `EngineeredCopyNumberPerturbation`
  (copy 1/2, KanMX), diploid, INCLUDES essential genes. 2,956 experiments, 16,939,418 records.
- **HOP** (homozygous, YSC1056): both copies deleted → `KanMxDeletionPerturbation`, diploid,
  non-essential only. 2,923 experiments, 13,056,820 records.
- Source: Dryad doi:10.5061/dryad.v5m8v (`HIP_scores.txt`, `HOP_scores.txt`, `Table_S1.xls`),
  sha256-pinned; downloader solves the Dryad Anubis proof-of-work.

## Data-quality caveats

- **HIP background mutations** — a subset of HIP strains carry undocumented secondary mutations
  (chr XI aneuploidy, WHI2 nonsense, chr V amplification) stored as clean deletions. Authoritative
  list (Table_S5) + risk audit + purge set:
  `[[torchcell.datasets.scerevisiae.hoepfner2014.background-mutations]]`. Experiment:
  `[[experiments.017-hoepfner-background-mutations.analysis]]`.
