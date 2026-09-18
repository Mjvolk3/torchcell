---
id: safni1er97kizdb93cnt2vo
title: 001_ko_solid_growth
desc: ''
updated: 1789739325598
created: 1789739325598
---

## 2026.09.18 - The deletion-only query

Generated from `experiments/025-solid-growth/queries/001_all_solid_growth.cql` by adding
`AND p.perturbation_type ENDS WITH 'deletion'` inside every block's `ALL(p IN ...)` predicate
(15 blocks) and requiring two perturbation nodes in the SynthLethDB block. Validated on the
served graph with a 20-gene set on 2026-09-18: Costanzo doubles return deletion x deletion
rows at 30 C; the Kuzmin 2020 triple YAL015C, YOL043C, YHR191C returns its ctf8 deletion row
and not its ctf8-9 TS row; the SynthLethDB self-pairs YPL212C, YHL047C, YKL117W (one
Perturbation node related twice, node count 1) no longer return. Each block is a full scan of
its dataset even for a small gene set (30 to 170 s per block), so the full query is dominated
by streaming the result.

Perturbation types per dataset in the served graph (perturbation nodes, 2026-09-18):
Costanzo 2016 doubles 17.6M kanmx + 14.1M natmx deletions, 7.85M temperature_sensitive_allele,
1.39M damp, 0.41M suppressor_allele; Kuzmin 2018 triples 232k deletion, 27k allele, 14k TS;
Kuzmin 2020 triples 856k deletion, 49k TS; SGD essentiality 1,140 and SynthLethDB 28k, all
kanmx deletions.
