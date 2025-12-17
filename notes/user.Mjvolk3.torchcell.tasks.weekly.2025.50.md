---
id: vkn6qyf19u3m1bluvbza37t
title: '50'
desc: ''
updated: 1765827882782
created: 1765212725596
---
## 2025.12.08

- [x] Clean up investigation on query 006 #M1
- [x] [[Query Issue|experiments.006-kuzmin-tmi.2025.12.08.query-issue]] - First we are checking against all deletion → Need to rebuild db for this. Using as excuse to test entire pipeline to Radiant.
- [x] Rebuild dataset with only deletions on #GH → `009-kuzmin-tmi`
- [x] Rebuild dataset with only deletions on #GH → `010-kuzmin-tmi`

## 2025.12.09

- [x] Launch jobs to check if using only `deletion` in query makes any difference to val Pearson.

## 2025.12.10

- [x] Transfer inference dataset to #IGB and free up GPU for inference. → optimized some.
- [x] Run test of transfer of build from #GH to #Radiant Try to streamline. → using rsync instead. This is fine for now even though we have some plans for globus.
