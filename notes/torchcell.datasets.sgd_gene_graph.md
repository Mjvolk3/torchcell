---
id: g323mg2405a73bgf5gy03f0
title: Sgd_gene_graph
desc: ''
updated: 1712255620924
created: 1712255620924
---
Works for the local case with chromosome and pathway annotations. Will need to be expanded if want to be more general

## 2026.10.01 - Zero is a value, sorted vocabularies, constant features refused (issue #518)

Previous behavior: `node_data[f] or median` replaced a true 0 with the median; chromosome and pathway indices were `list(growing_set).index(x)`, so one category could get different indices on different genes and string categories changed with `PYTHONHASHSEED`; min-max normalization of a constant feature stored NaN.

Fix: only None takes the median; indices are positions in sorted vocabularies built from the whole graph before the feature loop; under `normalized_chrom_pathways` a feature with min == max raises `ConstantFeatureError` naming every constant feature.

Evidence on the cached `G_gene.pkl` (6,607 genes): 0 genes hold a 0 in any of the seven features, and no feature is constant (`median_value` and `median_abs_dev_value` are None on 1,219 and 1,545 genes, as before). The feature tensors therefore do not change. The categorical indices change, and nothing outside the tests reads `chromosome_index` or `pathways_indices`. The on-disk build under `data/scerevisiae/sgd_gene_graph/processed/` (2024-04-20) is not rebuilt automatically. Tests: `test_unnormalized_build_fills_only_none_with_the_lower_median`, `test_categorical_index_is_a_function_of_the_category`, `test_categorical_indices_do_not_depend_on_pythonhashseed`, `test_single_gene_normalization_refuses_every_constant_feature`, `test_a_constant_feature_is_kept_when_not_normalizing`.
