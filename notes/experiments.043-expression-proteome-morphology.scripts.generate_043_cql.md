---
id: ogpupvqx3sy5s518mwr2jaz
title: Generate_043_cql
desc: ''
updated: 1792393295179
created: 1792393295179
---

## 2026.10.19 - The 15-block query

Writes `expression_proteome_morphology_043.cql` byte-identical to
`experiments/043-expression-proteome-morphology/queries/` (read by
[[experiments.043-expression-proteome-morphology.scripts.build_043_store]]) and
`torchcell/knowledge_graphs/queries/` (the copy the supported-query registry entry
`expression_proteome_morphology_043` points at). Edit the generator, never the `.cql`.

Each block has the property-shaped RETURN of the supported query
`torchcell/knowledge_graphs/queries/expression_proteome_morphology.cql` and keeps the
phenotype `graph_level` filter. The genotype clause is one of three shapes:

| clause | rule | datasets |
|---|---|---|
| `all` | every perturbation names a gene in `$gene_set`, at least one | Kemmeren 2014, Sameith 2015 Sm and Dm, Hu 2007, Jackson 2020, Nadal-Ribelles 2025, Messner 2023, Zelezniak 2018 (node); Ohya 2005 (global) |
| `any` | at least one perturbation names a gene in `$gene_set` | Caudal 2024, Muenzner 2024 proteome and turnover (isolates carry accessory-ORF absences and `sequence_variant` perturbations outside S288C) |
| `none` | no gene clause, unlabeled `(g)` | Boocock 2025, Nguessan 2025 (`SegregantGenotype`), Vaishnav 2022 (promoter library, no gene) |

The phenotype class in each block comment is the one the dataset's recorded closure in
`database/releases/2026.10.17-4295f7ca.closures.json` carries. Two corrections against
the first draft came from that file: Jackson 2020 serves `SingleCellExpressionPhenotype`
and Nadal-Ribelles 2025 serves `PseudobulkExpressionPhenotype`.

`EXPECTED_COUNTS` holds `n_experiments` per dataset from
`database/releases/2026.10.17-4295f7ca.json` (total 173,805).

Validation (run 2026.10.19 on the branch):
`python -m torchcell.knowledge_graphs.supported_queries validate expression_proteome_morphology_043 --release 2026.10.17-4295f7ca`
recorded `dataset_composite 1bbf284e9d159ce8eb8c5fb0355673cf4a2f38dff27742205fa9d35007c15997`,
`since_kg_version 4.0`; `check` reports 5 queries, 0 drifted, 0 failing.
