---
id: cgf1ei23twn28mdbl38erz9
title: Visual_score
desc: ''
updated: 1783563850142
created: 1783563850142
---

## 2026.07.08 - Visual-score verifier: gate ordinal colony-color screens beyond the schema bounds

This verifier gates `VisualScorePhenotype` datasets (WS7 Ozaydin carotenoid colony-color screen, WS4), where the phenotype is a bounded ordinal score a human assigns to colony color as a proxy for a target metabolite. The schema's `validate_visual_score` already bounds the score to `[score_scale_min, score_scale_max]` and requires `n_replicates >= 1`, so L0 subsumes those; this module adds the invariants the schema does not encode -- strain uniqueness, a control scored 0, and a named target product.

- L3 `target_product_set` enforces that every record names the metabolite the score PROXIES for -- a visual score is meaningless without its referent, and this is what keeps it interpretable.
- L3 `reference_zero` asserts the control (WT carrying the reporter background) is scored 0 by construction, so scores read as deviations.
- `_deleted_genes` excludes `gene_addition` perturbations (constant engineered background, non-ORF heterologous names) from the L1/L4 deleted-ORF key.
- Exposes `visual_score_gene_set` for the runner's L4 containment against the deletion collection; assembled from [[torchcell.verification.levels]], reported via [[torchcell.verification.report]], driven by [[torchcell.verification.runners]].
