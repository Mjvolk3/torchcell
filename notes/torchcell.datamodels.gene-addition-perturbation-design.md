---
id: u36d39gshgq1ht0wdehwj6z
title: Gene Addition Perturbation Design
desc: ''
updated: 1783309487371
created: 1783309487371
---

## 2026.07.05 - GeneAdditionPerturbation Design (locked decisions)

Motivated by `[[torchcell.datasets.scerevisiae.ozaydin2013]]` +
`[[torchcell.datasets.scerevisiae.cachera2023]]`: the schema had ONLY
loss-of-function perturbations (Deletion/Damp/Allele/Ts/Suppressor), all subclassing
`GenePerturbation` whose `systematic_gene_name` validator requires the native yeast
pattern `Y[A-P][LR]\d{3}[WC]...`. Heterologous cassette genes (crtYB, crtI, CYP76AD1,
DOD) fail it, so the engineered background was invisible in the data. We add a
GAIN-of-function perturbation family.

### Decisions (signed off 2026.07.05)

1. **Shape = per-gene addition perturbation** (chosen over genotype-level background
   or a single nested cassette object). Each added gene is its own `GeneAddition` in
   the SAME `Genotype.perturbations` list as the screened KO, so all genetic
   modifications live in one consumed place. The constant cassette repeats on the
   reference genotype (that is correct -- it is the assay chassis).
2. **Store the RAW form; embeddings are downstream.** `GeneAddition` does NOT embed a
   precomputed vector. It points at raw sequence via a `(plasmid_contig_id,
   locus_tag/coords)` reference into a PLASMID-SEQUENCE STORE that holds the raw
   GenBank. The existing embedding datasets extract the +/-~1kb window and embed --
   exactly as native genes are embedded from genome+GFF. Rationale (user): "want the
   more raw form because we can construct the embedding with the embedding datasets."
3. **Plasmid vs chromosome captured SEPARATELY, each with its own local context.**
   Localization is the demarcation. A plasmid-borne gene is a distinct record from the
   chromosomal copy of the same gene (Ozaydin's extra BTS1 on the 2u plasmid has the
   TDH3p-BTS1-CYC1t context, NOT chromosomal BTS1/YPL069C context). Native
   feedback-resistant alleles integrated ectopically (Cachera ARO4^K229L / ARO7^G141S
   at XII-5) are `GeneAddition`s at that locus carrying the variant -- NOT
   `AllelePerturbation` (which implies an in-place edit at the native locus).
   Copy-number is a later diploid-modeling concern; separate local contexts are wanted
   regardless.

### Why plasmid-derived (the unifying insight)

A plasmid GenBank IS a mini genome+GFF (sequence + feature table). So the plasmid
becomes just another CONTIG in the sequence store, and a `GeneAddition` points at
`(contig=plasmid_id, locus_tag)` exactly as a native perturbation points at
`(chromosome, systematic_name)`. The same "extract CDS +/- window" machinery serves
both. For a plasmid gene the +/-1kb flank is naturally its PROMOTER (TDH3p) +
TERMINATOR (CYC1t) -- the expression-relevant context. Path: raw plasmid GenBank (or
GFF-equivalent) in the library mirror -> COLLAPSE (project CDS features) to per-gene
`GeneAddition` records. Hand-typed gene lists are the fallback only where no sequence
exists.

### Proposed fields (implementation-ready; refine at build)

- `perturbation_type = "gene_addition"`, `description`.
- `added_gene_name: str` -- free string (heterologous names allowed; NO native-only
  validator). For native additions this may be a systematic name.
- `source_organism: str` (e.g. "Xanthophyllomyces dendrorhous", "Beta vulgaris",
  "Saccharomyces cerevisiae").
- `is_heterologous: bool`.
- `localization: Literal["episomal_2micron", "chromosomal_integration", ...]`.
- `construct_name: str | None` -- plasmid/cassette name ("YB/I/BTS1", "Btx-cassette").
  (Named `construct_name`, not `construct`, which shadows a pydantic BaseModel method.)
- `integration_locus: str | None` -- e.g. "XII-5" (chromosomal_integration only).
- Sequence pointer: `plasmid_contig_id: str | None` + `locus_tag: str | None`
  (or coords) into the plasmid-sequence store. `None` until the raw plasmid is
  mirrored (Cachera GenBank download; Ozaydin reconstruct-from-parts).
- Variant: `variant: str | None` -- e.g. "K229L", "G141S" for feedback-resistant
  alleles, so the embedded sequence reflects the expressed (mutant) form.
- CBM linkage (deferred, parallels `target_metabolite_id`): optional reaction / EC /
  metabolite pointer so COBRA/Yeast9 can add the heterologous reactions (crtYB ->
  phytoene synthase + lycopene cyclase; crtI -> phytoene desaturase; CYP76AD1 ->
  tyrosine 3-monooxygenase; DOD -> DOPA-4,5-dioxygenase). `None` now.

Must register in `GenePerturbationType` union + `EXPERIMENT_TYPE_MAP` and pass the
schema-invariant gate (`tests/torchcell/datamodels/test_schema_invariants.py`).

### Blockers before full implementation

- **Plasmid-sequence store** is a NEW subsystem (parallel to the WS10 external genome
  store). The sequence-pointer fields stay `None` until it exists.
- **Raw plasmid artifacts:** Cachera pBTX1/pBTX2 GenBank maps exist (OUP supplementary
  zip, CloudFront-signed -> manual_browser or try the bioRxiv preprint SI); Ozaydin has
  NO assembled-plasmid sequence (Euroscarf P30796 physical-only) -> reconstruct from
  parts. See the plasmid-availability sections in the two dataset notes.
- Interim: `GeneAddition` can be emitted with hand-sourced composition (from Table 2 /
  the Btx-cassette description) and `plasmid_contig_id=None`, then back-filled once the
  store + raw artifacts land. This unblocks the loaders without waiting on the store.
