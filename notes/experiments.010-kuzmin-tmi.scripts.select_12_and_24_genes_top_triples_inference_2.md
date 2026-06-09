---
id: u7ti4djm6idfqwb7rbvgiap
title: Select_12_and_24_genes_top_triples_inference_2
desc: ''
updated: 1781029075873
created: 1781029075873
---

## 2026.06.09 - Choosing the 12- and 24-Gene Panels That Best Cover the Model's Extreme Triple Predictions

This script turns 479K stricter-filtered inference_2 predictions into a concrete wet-lab plan: it picks the small gene panels (12 and 24 genes) whose pairwise design space yields the most of the model's most extreme (top-k and bottom-k) predicted triple interactions, so that a tractable strain-construction effort captures the predictions worth validating. It exists to answer "which dozen-or-two genes should we actually build strains for" under a combinatorial constraint, while prioritizing genes that already have published Sameith double mutants so the panel doubles as a literature-reproducibility check.

### Specifics worth keeping

- Selection is greedy coverage maximization plus local-swap refinement, with Sameith-overlap genes auto-included first as a priority tier.
- Panels and k sweeps: 12 genes over k in {25, 50, 100, 200} (design ceiling C(12,3)=220), 24 genes over k in {25, 50, 100, 200, 500, 1000, 2000} (ceiling C(24,3)=2024).
- Outputs: `gene_selection_results.csv`, per-(panel,k) `constructible_triples_*.parquet` and `top_k_constructible_*.csv` under `results/inference_2/`, plus coverage/bar/gene-stability/prediction-distribution figures in `ASSET_IMAGES_DIR/010-kuzmin-tmi/`.
- `--plot-only` redraws figures from the saved CSV without reloading predictions or rerunning selection.
- Downstream, the selected k=200 12-gene panel feeds the tables/histogram and queried-data scripts.
