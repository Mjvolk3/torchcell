---
id: qxq9inv88fvcse4w4kb9k78
title: '51'
desc: ''
updated: 1765839289490
created: 1765827901397
---

## 2025.12.15

- [x] Build inference dataset on #GH → 275 million might be too many.
- [x] #M1 Create `011-kuzmin-tmi` no deletions
- [x] Restart run on #IGB for duplicates → looking good.
- [x] Sync runs on #IGB
- [x] Query size check on `011-kuzmin-tmi` in comparison to `006`, `009`, `010`, and `011` → [[Query Comparison 006 009 010 011|experiments.011-kuzmin-tmi.scripts.query-comparison-006-009-010-011]]
- [x] Inference with current best models on #GH → tried with model from `006` trained on `GH`. This is second best model → Single GPU with batch optimized to 4096 will take 150 hours. On 4 gpu won't get better than 38 hrs. And we need to run on single GPU so we can do for other model ckpts. → Trying to reduce gene set size.

- [ ] Follow up on the dataset outlier comparison by reporting the spearman at snapshot for very best model across the different scenarios.

- [ ] #GH reduce `gene_set` size with tiered selection on intersections. Keep genes in at least $n$ large lists.

- [ ] #Radiant shutdown for Tuesday. Need to check we can go through restart procedure. If not will need to prepare for this.



- [ ] Notes on #GH build

- [ ] Kemmeren, Sameith dataset verify metadata #M1
- [ ] Expression datasets write adapters #M1

- [ ] Start DB build over night #GH
- [ ] Model training on deletions only #IGB

***

- [ ] Email CW
