---
id: 339c3ob8n8fimbon2gv3ihd
title: Subset_s3_query_strict
desc: ''
updated: 1790008352962
created: 1790008352962
---

## 2026.09.21 - S3 for the query-pair-disjoint split

`subset_S3Q_indices.json.gz` = S3 (1,121,645) minus `subset_Q_excluded_doubles.json.gz` (85) = 1,121,560 records, derived from the two committed artifacts with no scan of the build. Asserts that every excluded index is in S3, none is a pinned triple, and every pinned triple of the disjoint split survives. Used by `cgt_s3_q_kl_fit_036` and `cgt_s3_q_kl_embfit_037`.
