---
id: adub2w9wfvmui11utqh9fbw
title: Torchcell Perturbation Ontology Consolidation
desc: ''
updated: 1783563871234
created: 1783563871234
---

## 2026.07.08 - Foundational design: perturbation ontology consolidation

`schema.py` is not just a schema -- it is the **torchcell ontology**. After ~12 diverse
datasets (Costanzo, Kuzmin, Sameith, Kemmeren, Ohya, Ozaydin, Cachera, Mülleder, Zelezniak
protein+metabolite, Caudal), the perturbation type hierarchy has accreted to 18 ad-hoc
classes. We now see the principled structure underneath and consolidate it BEFORE the
~75-dataset backlog + BioCypher adapters solidify the ad-hoc shape. This note is the spec.

### The decisions (with user, 2026.07.07-08)

1. **SO-aligned vocabulary.** Gene absence = a `deletion` sequence_alteration (SO:0000159),
   NOT "copy_number = 0"; accessory presence = `insertion` (SO:0000667); dosage change of a
   PRESENT gene = copy_number_gain/loss (SO:0001742/0001743); SNP/indel = SNV (SO:0001483) /
   small indel. PAV is the extreme of CNV in the lit, but we split it out for cleanliness.
2. **Two-layer model = SO's own split.** `sequence_alteration` (the EDIT, ref->alt) is the
   superset; `sequence_variant` (the EFFECT on a feature) is a view. Each perturbation
   carries SO-term annotations (extend the `SORole` pattern already in `sequence/plasmid.py`).
3. **Edits-only records + full reconstruction.** A record stores only DIFFERENCES from the
   S288C reference; a gene not mentioned is present-as-reference (inherited). The full
   genome = reference + edits, reconstructable and COMPLETE -- so NO gene is ever dropped
   (fix the Caudal unmapped-ORF core-loss drop), and records stay compact.
4. **`apply-to-sequence` is a METHOD on a worker-local genome-like accessor** (not a floating
   function). The accessor re-opens the un-picklable store (gffutils SQLite / LMDB env /
   FASTA index) per DataLoader worker; records travel as PLAIN DATA (pointers). Keep the
   biologist-readable `genome[gene]`-style interface. (-> new CLAUDE.md readability principle.)
5. **Full unification (Q5(i)).** Model by STATE, annotate by MECHANISM + PROVENANCE. The
   engineered `KanMxDeletion`/`GeneAddition` become CHILDREN of a presence/absence ABC, so a
   query "all gene absences" is ONE filter (engineered KO + natural loss together), and the
   neutral state word ("absent"/"present") avoids "deletion sounds engineered".

**Identity.** Gene = namespaced CURIE (`SGD:YAL001C`, `pangenome1011:<id>`); allele =
content-addressed (gene + sequence sha256); cross-dataset sameness via xref, never forced
canonicalization. Forward-compatible with BioCypher/Biolink CURIEs.

### Target: the three-axis perturbation ontology

Every genotype difference from the S288C reference is one of three orthogonal AXES, each
crossed with PROVENANCE (engineered | natural) and annotated with an SO MECHANISM term:

```text
GenePerturbation (ABC: systematic_gene_name/CURIE, perturbed_gene_name, provenance)
├── PresenceAbsencePerturbation (ABC: state {present|absent}, mechanism SORole)   # AXIS 1
│   ├── [absent]  DeletionPerturbation            (engineered; SO deletion)
│   │   ├── KanMxDeletionPerturbation / NatMx / Sga* / MeanDeletion   (reparented, unchanged)
│   │   └── NaturalGeneAbsencePerturbation        (natural; Caudal core-loss)   <- NEW
│   └── [present] GeneAdditionPerturbation         (engineered; SO insertion; reparented)
│       └── NaturalGenePresencePerturbation       (natural; Caudal accessory) <- NEW
├── CopyNumberVariantPerturbation (ABC-or-leaf)                                  # AXIS 2
│       copy_number of a PRESENT gene (>=1 or fractional); SO copy_number_gain/loss
├── SequencePerturbation (ABC: mechanism SORole)                                 # AXIS 3
│   ├── SequenceVariantPerturbation               (natural SNP/indel allele; Caudal)
│   └── AllelePerturbation / TsAllele / Damp / Suppressor + Sga*  (engineered; reparented)
```

Reparenting uses **class-level defaults** for the new base fields (`state`, `provenance`,
`mechanism_*`), so EXISTING loaders that construct e.g. `KanMxDeletionPerturbation()` keep
working unchanged -- the migration is mostly reparent + defaults + REBUILD to confirm. Only
the Caudal loader changes (its `CopyNumberVariant`-for-absence -> `NaturalGeneAbsence`;
`CopyNumberVariant`-for-accessory -> `NaturalGenePresence`; reserve CNV for true dosage).

### Provenance + SO mechanism fields (added to the ABCs)

- `provenance: Literal["engineered","natural"]` (required on `GenePerturbation`; class default
  per concrete type).
- `mechanism_so_id: str` + `mechanism_so_name: str` (an SO term, `SO:NNNNNNN`), via a reused
  `SORole`. deletion=SO:0000159, insertion=SO:0000667, SNV=SO:0001483,
  copy_number_gain=SO:0001742, copy_number_loss=SO:0001743.
- state on presence/absence; copy_number/reference_copy_number on CNV; sequence pointer
  (`sequence_source`/`sequence_uri`/`sequence_sha256`) on sequence + presence types.

### Reconstruction as a view -- `apply-to-sequence` (Q4)

The perturbations are pointers (pure data, picklable). A **worker-local genome accessor**
(`PerStrainGenome`-like) exposes `genome[gene]` AND a method `genome.materialize(edits)` /
`apply(reference, edits) -> per-gene sequences`, dereferencing pointers from the off-graph
store (sha256-verified). The un-picklable handle (gffutils/LMDB/FASTA index) is opened lazily
per worker (mirrors the existing `ParsedGenome` disconnection forced by gffutils' SQLite),
never pickled. Full sequence is a VIEW derived from reference+edits (git commits -> working
tree), so we get compactness + provenance + embeddable sequence with no conflict.

### Programmatic ontology-integrity checks (the point of this refactor)

`schema.py` is an ontology; as it grows we must PROVE it keeps good properties. New test
module `tests/torchcell/datamodels/test_ontology_invariants.py` (beyond basic unit tests):

1. **DAGness / single-root hierarchy.** The perturbation inheritance is single-rooted at
   `GenePerturbation` and acyclic (Python guarantees no MRO cycles; we assert single-root +
   that every concrete leaf reaches the root). AND the MODEL-COMPOSITION graph (which model
   references/contains which) is acyclic -- no circular pydantic references.
2. **Liskov substitution.** Every concrete perturbation `isinstance` of all its declared
   bases; a child instance satisfies every parent field-validator (a child never rejects a
   value the parent's contract requires it to accept); substituting a child wherever a
   parent-typed field is expected validates.
3. **Union <-> leaf-set consistency.** `GenePerturbationType` union members == the set of
   CONCRETE (instantiable) perturbation leaves; no concrete leaf missing from the union; no
   abstract base in the union. Same shape already enforced for Experiment/Phenotype.
4. **Discriminator uniqueness.** Every concrete perturbation has a UNIQUE `perturbation_type`
   string (so tagged-union resolution is unambiguous); combined with `ModelStrict`
   forbid-extra, union parsing is provably deterministic (round-trip preserves the concrete
   type).
5. **Round-trip / serialization fidelity.** `model_dump()` -> re-parse through the union ==
   the same concrete type + equal data, for every perturbation and every Experiment/Reference.
6. **SO-annotation coverage.** Every perturbation that claims a mechanism carries a
   well-formed SO id (`^SO:\d{7}$`) that resolves to the expected term name.
7. **Provenance completeness.** Every concrete perturbation declares `provenance in
   {engineered, natural}` -- no unset provenance.
8. **Identity well-formedness.** `systematic_gene_name` is a namespaced CURIE
   (`prefix:local`) or a bare S288C systematic name (legacy-accepted); allele-bearing
   perturbations carry a `sequence_sha256` when a `sequence_uri` is set.

These are GENERIC (they enumerate the union / walk the class tree), so they keep holding as
the ontology grows -- the whole point.

### Migration + rebuild plan (overnight)

1. Write ontology-invariant tests (encode the target properties; some fail until the refactor).
2. Refactor `schema.py`: introduce the ABCs + provenance + SO mechanism; reparent existing
   types with class-default fields; add the 2 natural presence/absence types. Make ALL tests
   (ontology + existing `test_schema*`) pass; mypy --strict clean.
3. Migrate loaders: Caudal (absence/presence + reserve CNV; stop dropping unmapped losses);
   others unchanged (reparent is transparent) -- confirm by rebuild.
4. Rebuild + L0-L4 EVERY dataset (parallel where independent; heavy SGA/TMI may be
   smoke-confirmed if a full rebuild can't finish by morning -- documented).
5. Report: which ontology tests were added, why each protects integrity, rebuild status,
   readiness for BioCypher adapters.

### Toward BioCypher adapters (why this matters now)

Roadmap WS11-14 write BioCypher/Biolink adapters that map these pydantic objects to graph
nodes/edges. A clean, DAG-consistent, CURIE-identified, SO-annotated ontology is exactly what
an adapter consumes -- each perturbation axis -> a node type, `provenance`/mechanism ->
edge/property annotations, CURIE -> node id, xref -> equivalence edges. Consolidating now
means the adapters map a PRINCIPLED ontology, not 18 ad-hoc classes.
See `[[torchcell.datasets.scerevisiae.caudal2024]]`,
`[[torchcell.sequence.plasmid-and-genomic-content-design]]`,
`[[plan.schematization-ingestion-roadmap.2026.06.23]]` (WS11-14).
