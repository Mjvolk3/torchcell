---
id: 0jksfmomhl0efr4z9wezhnq
title: Split_gene_overlap_audit
desc: ''
updated: 1789720982344
created: 1789720982344
---

## 2026.09.18 - Deleted-gene overlap across the v13, v14 and v16 partitions

Promotes the split audit run before the v16 launch (report 11 under `results/expression_fit_review_2026_09_17/`). The script reads only the cached index JSONs of `fig3_core` and `fig3_proteome` (`processed/*_index.json`, `data_module_cache/index_seed_<k>.json`) and writes `results/split_gene_overlap_audit.json`. Seconds of CPU on GilaHyper.

**Rule.** `CellDataModule` shuffles record indices per index key after `random.seed(split_seed)` and cuts 80/10/10; it never looks at gene identity. One record is one deletion gene set (`GenotypeAggregator`), so an exact-genotype duplicate cannot cross a split (0 on every seed, store and side), but a Sameith double and the singles of its two genes are three records assigned independently.

**Counts.** Held-out genotypes sharing a deleted gene with train (dbl = a double whose parent single is in train; sgl = a single whose gene is deleted in train only inside a double):

| partition | seed | val n | val share (dbl / sgl) | test n | test share (dbl / sgl) |
|---|---|---|---|---|---|
| v13 (fig3_core, expression rows) | 0 | 155 | 21 (15 / 6) | 155 | 13 (3 / 10) |
| | 1 | 151 | 19 (10 / 7) | 150 | 12 (6 / 6) |
| | 2 | 155 | 14 (5 / 7) | 155 | 16 (5 / 10) |
| | 3 | 155 | 18 (8 / 8) | 155 | 16 (8 / 7) |
| v16 (fig3_proteome, expression rows) | 0 | 155 | 18 (14 / 4) | 155 | 15 (14 / 1) |
| | 1 | 155 | 17 (11 / 5) | 155 | 20 (14 / 6) |
| | 2 | 155 | 19 (13 / 6) | 155 | 17 (12 / 4) |
| | 3 | 155 | 20 (14 / 6) | 155 | 22 (16 / 5) |
| v14 (fig3_proteome, proteome rows) | 0 to 3 | 448 | 0 | 447 | 0 |

The full fig3_proteome partition v16 trains on is 3,722 / 478 / 481 on seed 0 (2,478 proteome-only, 141 expression-only, 1,103 both in train); the proteome rows are exactly v14's at every seed because `require_modalities` filters after the draw.

**v13 against v16 held-out genotypes.** Same 1,554 expression genotypes in both stores, different record order (2 of 1,554 sorted positions carry the same genotype), so the held-out overlap at the same seed is at chance: val 9 / 22 / 13 / 10 of 155 and test 12 / 13 / 12 / 9 for seeds 0 to 3 (chance 15.5). v13's partition cannot be reproduced on fig3_proteome by any seed.

**Components of the "shares a deleted gene" relation** over the 4,681 genotypes: 4,545 components; 18 multi-member (sizes 3 x 12, 5 x 2, 7, 9, 11, 81) holding 154 genotypes, the 72 doubles and the 82 singles of their genes (every double gene is also a single; 32 genes sit in 2 to 9 doubles); 70 of the 154 carry both labels and 84 expression only, none proteome only. A gene-set-identity split would move those 18 components as units and change at most 70 of v14's proteome rows.

**Consequences.** Keep the draw (the proteome contrast stays paired with the 16 finished v14 runs); add the expression-only arm `J_expr` on the same partition; read the expression metric on all held-out rows and on the gene-disjoint rows (val 137 / 138 / 136 / 135, test 140 / 135 / 138 / 133); a train-disjoint gene-set split is a separate round if wanted. The shared strains are not replicates: every one is a single-versus-double relation, and the review's seed-0 gap (0.398 shared against 0.178 gene-disjoint, [[experiments.019-simb-multimodal.expression-fit-review]]) is the partition's, carried by every method including a random-embedding kNN ([[experiments.019-simb-multimodal.scripts.graph_retrieval_baseline]]).
