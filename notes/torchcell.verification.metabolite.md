---
id: msup7vtcbf7fixb24b9e01l
title: Metabolite
desc: ''
updated: 1783563798390
created: 1783563798390
---

## 2026.07.08 - Metabolite verifier: one gate spanning centered-score and absolute-quantity assays

This verifier gates `MetabolitePhenotype` datasets that come in two fundamentally different flavors, which is its reason for existing as a distinct module. The WS8 Cachera CRI-SPA betaxanthin screen is population-CENTERED (reference level is identically 0), while WS9 absolute-quantity datasets (Mulleder amino-acid mM, Zelezniak SRM signal) reference each strain against a WT-equivalent baseline. The `reference_centered` flag switches L3 between `reference_zero` and `reference_finite` so one verifier correctly asserts the right baseline semantics for both.

- `reference_finite` accepts a reference whose metabolite keys are a SUBSET of the experiment's -- sparse targeted metabolomics may measure metabolites the WT lacks a baseline for, which is legitimate, not a defect.
- `_deleted_genes` deliberately EXCLUDES `gene_addition` perturbations: the engineered cassette background is constant across strains and its heterologous names are not real ORFs, so it must not pollute the L1/L4 deleted-ORF key.
- SE may be NaN where a metabolite has a single replicate, so NaN SE is dropped before the non-negativity check rather than failing the record.
- Exposes `metabolite_gene_set` for the runner's L4 gene containment; assembled from [[torchcell.verification.levels]], reported via [[torchcell.verification.report]], driven by [[torchcell.verification.runners]].
