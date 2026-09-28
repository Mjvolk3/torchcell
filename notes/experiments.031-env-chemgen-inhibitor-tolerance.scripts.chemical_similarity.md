---
id: 64umc54vp8nqh2p0zu1nwht
title: Chemical_similarity
desc: ''
updated: 1790379811194
created: 1790379811194
---

## 2026.09.25 - Nearest chemical neighbors across datasets, and chemistry vs response

For every encoder embedding in `results/embeddings/` (from `embed_compounds.py`):
the nearest Hillenmeyer compound of each Vanacloig compound (Tanimoto for fingerprints,
cosine for dense embeddings) and whether it is an exact InChIKey match; and, over every
(Vanacloig condition, partner condition) pair, the Spearman between chemical similarity
and the cross-dataset response Spearman from `cross_dataset_similarity.py`, plus the
median response similarity of the top decile of pairs by chemistry against the rest.
If chemically similar compounds produce similar per-gene profiles, a model can transfer
through chemistry where no exact match exists. Outputs
`results/chemical_similarity_summary.csv`, `results/nearest_neighbors_<encoder>_<partner>.csv`
and one figure per partner, `chemistry_vs_response_<partner>`.
