---
id: 9pjg636tkqgq1gyiwk55nld
title: Fitness_browser
desc: ''
updated: 1791803958837
created: 1791803958838
---

## 2026.10.12 - The pinned Fitness Browser reader

`torchcell/datasets/fitness_browser.py` is the shared half of the two Fitness Browser loaders ([[torchcell.datasets.ecoli.fitness_browser_ecoli]], [[torchcell.datasets.pputida.fitness_browser_putida]]); the policy and the inventory are in [[torchcell.candidates.aggregation-decomposition]] (2026.10.12).

| piece | what it does |
|---|---|
| `FEBA` (`DepositPin`) | the deposit's identity, verbatim from `DEPOSIT.md` and `headers.txt`: path under the Price 2018 raw mirror, sha256 `250d4a03...`, 10,205,732,864 bytes, URL, `curl` command, retrieved 2026-10-11T04:29:05Z, Last-Modified and ETag |
| `verify_feba`, `Feba(verify=True)` | refuse a file whose size or sha256 differs, then open it `mode=ro&immutable=1` |
| `FbExperiment` | one `Experiment` row, typed; empty strings become None; condition triples kept verbatim (a dose with no condition is kept as released, `set9IT093` of one organism has one) |
| `Feba.values` | an organism's `FitByExp` fit and t as a loci x experiments matrix, refused when the requested experiments do not share one locus set |
| `SourceStudy` | a browser `pubId` as a study: the browser title and URL verbatim (checked against the deposit on every build by `check_studies`), the DOI or PubMed id the records carry, `source_kind` (`primary` or `compendium_only`), and `own_release_dataset` when another registered dataset already loads the study's own release |
| `SampleDecision`, `Fate` | what a loader does with a sample: stored, unattributed release, own release already loaded, author-withdrawn, medium not in the library, condition not representable |
| `derived_se` | `sqrt((f / t)^2 - 0.1^2)`, refusing a zero f or t and a pair below the identity's floor (none on the deposit: smallest `abs(f / t)` 0.1020 Keio, 0.1011 Putida) |
| `compendium_check` | the L4 agreement with a served store: per-sample Pearson r over shared loci, pooled Pearson r over every shared (gene, sample) value, largest difference; a sample absent there needs a stated reason |

Why the deposit is not quoted as a `SourcedValue`: `audit_sourced_value` re-reads its source as text, and the deposit is a 10 GB SQLite file whose rows are not stored as text lines. Every value the loaders store is read from the pinned bytes at build time; nothing about the deposit is hand-transcribed except the study table, which `check_studies` compares with the deposit verbatim.

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from the Fitness Browser deposit (`feba.db`, mirrored under `priceMutantPhenotypesThousands2018`). Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here: the browser recomputes every study's gene fitness under its own statistics version and pipeline, so a value can differ from what the study published at the time; its `Experiment` rows are a database schema (medium, condition columns, `pubId`), and condition detail a paper reports outside those columns is not carried; samples with an empty `pubId` have no attributed study. Where the browser is a study's only release (`source_kind = primary`), it is still a database copy, and the paper's own tables are the source to collect when they exist.
