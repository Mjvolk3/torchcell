---
id: 44a3mr6rwgmgli1ciawj0bp
title: Representation_043
desc: ''
updated: 1792638814585
created: 1792638814585
---

## 2026.10.21 - Resolver of the 043 input side, keys first

Module: `torchcell/data/representation_043.py`; test `tests/torchcell/data/test_representation_043.py` (8 tests). Store binding: [[experiments.043-expression-proteome-morphology.scripts.representation_043]]. Design: the primer's input section (`notes-tex/multimodal/043-datasets-primer/sections/input-design.tex`).

What a `RecordRepresentation` holds, per gene of the shared index (6,607): the key ids of the four composite encoders (FUDT upstream, FUDT downstream, ProtT5, CaLM; id = position in the sorted sha256 vocabulary, -1 = gap), a `GapKind` per slot, the change type under the precedence rule (absent > deletion > loss of function > variant > present > promoter), the allele parent, the operator mask, and the position features (chromosome index 0-16, relative midpoint, chromosome copy number over the basal ploidy, gene copy number). Off-graph tokens (accessory ORFs, the Vaishnav reporter) are `ExtraToken`s with their own keys and gaps.

Decisions made here, each visible in the code:

- **Keys, not vectors.** `AlleleVectorStore` maps vocabulary ids to h5 rows on first read; a requested key with no row raises `KeyError`. Gaps are never filled, not with S288C rows, not with zeros.
- **Route A** (isolate as its own reference context): the isolate's own keys for every gene; genes the record calls absent become `ABSENT_IN_CONTEXT`; the operator mask holds engineered edits only (HO deletion). **Route B**: S288C keys, with the isolate's keys substituted on the edit list (genes whose change type is variant or loss of function); the operator mask holds every mark. A variant gene carries all four of the isolate's keys (its own windows too), since the composite token is the allele's own token.
- **Segregant definite rule over confident runs.** The loaders encode blocks as runs of constant called parent AND constant posterior level (Boocock de novo) or as marker runs, so one parent's haplotype is often several adjacent blocks. A confident run merges consecutive blocks of one parent that all have posterior >= 0.9; a gene is assigned when one run contains it whole. `rule="single_block"` is the literal form, kept to measure the difference (see [[experiments.043-expression-proteome-morphology.scripts.resolve_segregant_alleles_043]]). Unassigned genes are typed gaps with a reason (no block, low posterior, straddle), never the S288C allele: in the YJM145 x YPS163 and CBS2888 x YJM981 crosses S288C is neither parent. In crosses without BY both parents' genes are variant marks.
- **Segregant chromosome copy number is 1** (haploid segregants; the genotype carries no aneuploidy descriptor). This is an assumption, not a measurement. Gene copy number is the assigned parent's pangenome value (1 for BY).
- **Coordinates**: gene spans from the SGD R64-4-1 GFF for every class; the segregant blocks are the loaders' marker positions on the S288C reference. Hypothesis (unverified): the marker coordinates are R64-1-1-based and agree with R64-4-1 to within a few bp for nuclear genes.
