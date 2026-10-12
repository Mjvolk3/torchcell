---
id: ngtmyfnjm6mat8cabx81sjn
title: Deletion Vs Perturbation Indices
desc: ''
updated: 1789444670092
created: 1789444670092
---

## 2026.09.14 - A GeneAddition was gating reactions shut

### The defect

Every graph processor built its perturbed-gene set from ALL perturbations regardless of
type:

```python
perturbed_names = {
    p.systematic_gene_name
    for item in data
    for p in item["experiment"].genotype.perturbations
}
```

That is correct for the BACKBONE, which removes a perturbed gene's node either way, and
wrong for the flux layer's availability gate, which reads the same index as "this gene
product is absent" and closes every reaction whose GPR needs it
(`flux_layer.gene_availability`: "Deleted genes are set to zero hard, from the
perturbation").

The Cachera betaxanthin cassette records **ARO4^K229L (YBR249C)** and **ARO7^G141S
(YPR060C)** as GeneAddition perturbations (`cachera2023.py:67-113`). Those are
feedback-resistant alleles the strains OVERPRODUCE through, and the gate was shutting the
reactions they act on.

### Measured on the real 027 data, before the fix

| gene | yeast-GEM reactions | GPR |
|---|---|---|
| ARO7 / YPR060C | `r_0278` chorismate mutase | `YPR060C`, a SINGLE gene |
| ARO4 / YBR249C | `r_0020`, `r_0042` DAHP synthase | |

`r_0278`'s GPR is one gene, so zeroing ARO7 closes chorismate mutase **completely**, with
no isozyme to carry it.

Over a 32-genotype batch of `fig6_pigment_transfer`: **sets equal 0, differ 32.** Every
single betaxanthin genotype carried the defect, and the genes perturbed but not deleted were
`YBR249C` (32/32), `YPR060C` (32/32) and `YPL069C` (BTS1, the carotenoid cassette's
addition, 32/32). So the module was closing the exact two steps the experiment opens, in
100 percent of the strains, for every flux arm run to date.

### The fix, additive

1. **`torchcell/data/graph_processor.py`**: a new `deletion_indices` field (plus
   `deleted_genes`) built from perturbations whose `perturbation_type` ends with
   `"deletion"`. `perturbation_indices` is untouched, so a consumer that has not opted in
   behaves exactly as before. Added in the `Perturbation` processor, which is what 027 runs
   and which builds its names inline, and in all three `_process_gene_info` copies
   (`SubgraphRepresentation`, `IncidenceSubgraphRepresentation`,
   `LazySubgraphRepresentation`) through a `deletion_subset` return key.
2. **`torchcell/models/cell_graph_transformer_metabolism.py`**: `gene_removal_indices(batch)`
   returns `(indices, batch assignment, source)`, preferring `deletion_indices` and falling
   back to `perturbation_indices` for a batch built before the field existed. The model's
   flux-layer call uses it and records the choice as `last_gene_removal_source`.
3. **`train_bx.py`**: `deletion_indices` added to `follow_batch` so
   `deletion_indices_batch` exists; the three readouts that call the layer directly
   (`FluxConcentrationReadout`, `FluxPrecursorConcentrationReadout`, `FluxBiomassReadout`)
   use the helper; the result file records `gene_removal_source`.

**The pooled context still reads every perturbed token.** The context is a representation of
what was CHANGED, and an added allele is a change; the gate is a statement about what is
GONE. Only the gate moved.

### Verification on the real datasets

| build | batch | perturbation_indices | deletion_indices | ARO4 in deletions | ARO7 in deletions | source |
|---|---:|---:|---:|---|---|---|
| `fig6_pigment_transfer` | 64 graphs | 247 | 64 | no | no | `deletion_indices` |
| `fig6_build` | 64 graphs | 171 | 64 | no | no | `deletion_indices` |

64 deletion indices over 64 graphs is one deletion per genotype, which is what both builds
carry. `fig6_build`'s first batch also contains Cachera genotypes (that build holds 4,669
betaxanthin records), which is why ARO4 and ARO7 appear in its perturbation set too; the
deletion set excludes them in both.

A per-genotype assertion that the deletion set is a SUBSET of the perturbation set holds on
every genotype checked.

### Left alone, and it is a separate decision for the user

**The CGT backbone still removes ARO4 and ARO7 from every Cachera cell graph.** That was not
touched. It is a constant across all betaxanthin strains, so it cannot produce a spurious
between-strain difference, but it does mean the backbone never sees those two genes'
tokens in this dataset. Whether the backbone should keep an ADDED gene's node is a modeling
question, not a bug, and it is left open.

### What it does not fix retroactively

Every betaxanthin flux result to date, wave 1 through wave 9, was produced with chorismate
mutase and DAHP synthase closed. The precursor-concentration arms read `thermo_log_c`, which
is a function of the context rather than of the flux, so they are less exposed than the
`flux_scalar` arms, but the constraint terms all saw the gated flux. Any betaxanthin flux
number that is to be compared against a post-fix number has to be re-run.

### The gate at chorismate mutase, measured

`gene_availability` -> `reaction_availability` evaluated on 16 real `fig6_pigment_transfer`
genotypes through a built `flux_projection_anchored` layer (the wave-8 `bxprec` recipe),
reading reaction column 243, `r_0278`:

| index the gate reads | min | max | mean | rows at exactly 0 |
|---|---:|---:|---:|---|
| `perturbation_indices` (old) | 0.0000 | 0.0000 | 0.0000 | **16/16** |
| `deletion_indices` (fixed) | 1.0000 | 1.0000 | 1.0000 | **0/16** |

Chorismate mutase went from fully closed in every genotype to fully open in every genotype.
Measured directly on the gate rather than after an epoch, because the gate is the quantity
in question and an epoch of training would only move the learned sigmoid around a hard zero
that is no longer there.
