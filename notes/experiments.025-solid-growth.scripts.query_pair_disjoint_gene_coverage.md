---
id: uxr9yhn9yicieryr6r1mubj
title: Query_pair_disjoint_gene_coverage
desc: ''
updated: 1789064751500
created: 1789064751500
---

## 2026.09.10 - How cold is a held-out gene on the query-pair-disjoint split

`experiments/025-solid-growth/scripts/query_pair_disjoint_gene_coverage.py` counts, for
each split arm, how many held-out genes appear in a training triple in any position, from
the recap table (three genes per S0 triple) and the split artifacts. Output:
`experiments/025-solid-growth/results/query_pair_disjoint_gene_coverage.json`.

The question it answers: the 019 expression strand turned the learnable gene table off
because only 4.8 percent of its validation genes are ever perturbed in training, so a free
per-gene row is at initialization at test. Whether the same holds on the 025 Q split
decides how to read a learnable-table arm against a sequence-embedding arm.

| arm | split | records | genes | genes in a training triple | records with all three genes in training | records with a never-trained gene |
|---|---|--:|--:|--:|--:|--:|
| Q | val | 37,705 | 4,021 | 4,013 (99.8 percent) | 81.3 percent | 18.7 percent |
| Q | test | 37,791 | 1,243 | 1,239 (99.7 percent) | 91.4 percent | 8.6 percent |
| R | val | 37,673 | 3,435 | 3,435 (100 percent) | 100 percent | 0 |
| R | test | 37,673 | 3,377 | 3,377 (100 percent) | 100 percent | 0 |

Training has 301,236 records over 4,340 genes on Q (301,386 over 4,352 on R). Only 8 val
genes and 4 test genes never appear in a training triple on Q, but they are query-pair
members, each carried by many triples, so 18.7 percent of val records and 8.6 percent of
test records contain one. So on Q the learnable table is trained for essentially every
held-out gene; what is new at test is the pair, not the genes. A learnable-table arm and
a sequence-embedding arm on Q therefore differ in what the gene vector can contain, not in
whether it was trained, which is the comparison `cgt_s0_q_kl_ctrl_016` against
`cgt_s0_q_kl_emb_017` makes ([[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]]).
