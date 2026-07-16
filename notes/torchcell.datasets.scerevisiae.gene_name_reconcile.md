---
id: sed3arw36srp4qkfx9fikdy
title: Gene_name_reconcile
desc: ''
updated: 1784161715361
created: 1784161715361
---

## 2026.07.15 - Shared retain-all reconciliation helper

`reconcile_systematic_names(genome, names, *, label)` + `default_genome()` — the one place
the CalMorph/SCMD loaders (Ohya 2005, Ohnuki 2018/2022) reconcile source systematic ORF
names to the current R64 annotation, wrapping the genome resolver
([[torchcell.sequence.genome.scerevisiae.s288c]]) with the morphology retention policy:

- Resolve each unique name via `genome.resolve_gene_name`.
- Remap to the resolved id when status ∈ {CURRENT, RENAMED, NON_GENE_FEATURE}; keep the
  original name for RETIRED/AMBIGUOUS.
- **Collision-safe**: if a remapped id is already claimed by another record (an SGD merge of
  two distinct 2005 ORFs), keep BOTH originals so the strains stay distinct.
- **Nothing dropped for naming.** Logs status breakdown + remapped/collision/retired lists.

Extracted from Ohya's inline `_reconcile_orf_names` so all three morphology loaders share it
instead of copies. Tests: `tests/torchcell/datasets/scerevisiae/test_gene_name_reconcile.py`.
Batch collision policy is the morphology policy; Cachera keeps its own (common-name,
keep-first) collision handling and calls the per-name resolver directly.
