---
id: 2uvvbgqc33pnqr6osjv0hwg
title: Optimized_doubles_setcover
desc: ''
updated: 1784302643804
created: 1784302643804
---

## 2026.07.17 - Minimum Doubles Set-Cover for the Panel-12 Triples

Script: `experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover.py`
Data:   `experiments/010-kuzmin-tmi/results/optimized_doubles_setcover_panel12.csv`
Source triples: `results/inference_3/top_k_constructible_panel12_k200.csv` (52 constructible)

### Why not all C(12,2)=66

In SGA trigenic construction a triple `{A,B,C}` is built by crossing **one** of its
three doubles (`AB`, `AC`, or `BC`) with the remaining single. So the doubles we must
physically build are only a set that **covers** every constructible triple (each triple
shares >=1 double with the set) — a classic minimum **set-cover**, solved greedily
(take the double covering the most still-uncovered triples each round).

Funnel: **66** (all pairs) -> **58** (pairs in >=1 constructible triple) -> **11**
(minimum cover). **83% fewer builds** (55 doubles skipped). Cover verified complete:
all 52 triples contain >=1 chosen double. Genes within each double are ordered
(`gene1 < gene2`, systematic sort).

### Optimized doubles to construct (11)

| rank | gene1 | gene2 | # triples enabled |
|-----:|-------|-------|:-----------------:|
| 1 | YBR203W | YPL046C | 6 |
| 2 | YDR057W | YER079W | 6 |
| 3 | YDR057W | YLL012W | 6 |
| 4 | YER079W | YIL174W | 6 |
| 5 | YER079W | YLR312C-B | 6 |
| 6 | YJR060W | YPL046C | 6 |
| 7 | YLL012W | YPL046C | 6 |
| 8 | YLR104W | YPL046C | 5 |
| 9 | YIL174W | YPL081W | 4 |
| 10 | YLR312C-B | YPL081W | 4 |
| 11 | YDR057W | YPL081W | 3 |

Hub genes: **YPL046C/ELC1** appears in 4 of 11 doubles; **YER079W** in 3 of 11 —
the high-frequency partners the set-cover leans on. Greedy is near-optimal here; an
exact ILP might shave to 10, but 11 is a verified valid cover. The full 66-double
superset lives in `results/inference_3/doubles_table_panel12_k200.csv`; the per-pair
triple-yield weights are in `results/experimental_table_12_genes_k200.csv`
(`gene1_gene2_count` etc.).

Related: [[experiments.010-kuzmin-tmi.scripts.select_12_and_24_genes_top_triples_inference_3]],
[[experiments.010-kuzmin-tmi.scripts.validation_panel_smf_reference]].
