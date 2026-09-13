---
id: 1y874r7bup6iwcq7acxurdw
title: Ontology Checks
desc: ''
updated: 1789246469245
created: 1789246469245
---

## 2026.09.12 - Programmatic ontology checks

Source: `torchcell/datamodels/ontology_checks.py`
Tests: `tests/torchcell/datamodels/test_ontology_coherence.py` (81 passed, 4 xfailed)

### Why this module exists

The ontology is not one artifact, it is four, and each already had a guard:

| artifact | existing guard |
|---|---|
| the pydantic trees | `test_ontology_invariants.py`, `test_ontology_all_trees.py` |
| the `*_TYPE_MAP` registries | `test_schema_invariants.py` |
| the adapter conf vs the graph schema | `test_adapter_schema_consistency.py` |
| the schema surface fingerprints | `kg_manifest.py`, `torchcell/provenance/schema_deps.py` |

Nothing guarded the SEAMS between them, and nothing guarded the identity keys a
cross-dataset join stands on. A phenotype class can be added to `PhenotypeType`
with no graph node class, and BioCypher drops the whole class silently. A node
class can declare a property no adapter method emits, and every query against that
property returns nothing rather than erroring. A `Compound` can be name-only, and
the environment it belongs to joins to nothing. Those are the failures this module
looks for.

Everything here is pure and offline: it reads the live pydantic models, the
committed `biocypher/config/torchcell_schema_config.yaml`, and
`torchcell/adapters/cell_adapter.py` as an AST. No LMDB, no database, no network,
so the whole module runs in about 6 s and belongs in the ordinary test run rather
than in a nightly.

### Checks implemented

Grouped by the property each defends. Every check is a pydantic-returning callable
in `ontology_checks.py` plus a test; the tests below are the passing ones unless
marked.

**Representation: persistent entities stay separable from contingent observations**

1. `lane_membership` / `test_record_lanes_are_disjoint` -- the genotype, environment
   and phenotype lanes (each closed under inheritance AND composition) share zero
   classes. Measured today: 25 / 10 / 14 classes, all three pairwise intersections
   empty. If a `Media` were also reachable from a `Phenotype`, "what the cell was
   grown in" and "what was measured" would be one object, and a query for every
   record on YPD would depend on which phenotype family was measured.
2. `lane_back_edges` / `test_no_lane_composes_an_experiment` -- the composition DAG
   points down only. Node ids are the sha256 of a model dump, so a phenotype
   composing its experiment would make each id depend on the other and neither could
   be computed. Empty today.
3. `orphan_models` / `test_every_schema_model_is_reachable_or_documented` -- every
   pydantic model in `schema.py` is reachable from `Experiment` /
   `ExperimentReference` by composition or inheritance, except a named list. Exactly
   two orphans today, both listed in the test so a NEW orphan fails.
4. `test_experiment_and_reference_agree_on_their_type_tag` (14 params) -- the
   registry key, `experiment_type` and `experiment_reference_type` are one string.
   The three are written independently in `schema.py`; a drift between any two
   splits one phenotype family into two.
5. `test_experiment_and_reference_share_a_phenotype_class` (14 params) -- a record
   and its control measure the same phenotype class, else the control is not a
   control.
6. `test_every_experiment_family_uses_the_shared_environment_class` (14 params) --
   `Environment` and `ReferenceGenome` are never narrowed per family, which is what
   keeps the YPD-level aggregate expressible across families.

**Additive expansion: a new family is a leaf plus a node class plus a method**

7. `adapter_node_sites` / `test_every_adapter_node_site_is_statically_readable` --
   all 41 `BioCypherNode(...)` constructions in `cell_adapter.py` expose a literal
   label and a resolvable property dict (literal, local variable, or a helper's
   returned dict). A site written in a form the reader does not understand FAILS
   here rather than being skipped, so the coverage below cannot quietly shrink.
8. `adapter_property_mismatches` /
   `test_adapter_emits_exactly_the_declared_properties` -- emitted property set ==
   declared property set, for all 41 sites, in BOTH directions.
   `test_every_node_method_property_is_declared` pins two labels by hand; this
   derives all of them from the AST. Clean today.
9. `phenotype_label_map` /
   `test_phenotype_classes_and_node_classes_are_in_bijection` -- 13 concrete
   phenotype classes <-> 13 phenotype node classes. The mapping is derived from the
   property sets, not a naming convention, which is why
   `RNASeqExpressionPhenotype` <-> `rnaseq expression phenotype` and
   `CalMorphPhenotype` <-> `calmorph phenotype` resolve with no alias table, and why
   a field renamed on one side without the other breaks the match.
10. `test_phenotype_member_of_sources_are_exactly_the_phenotype_node_classes` -- the
    existing test checks the subset direction; a source that is not a phenotype node
    class is the other way a new family gets mis-wired.
11. `isolated_graph_node_classes` / `test_no_graph_node_class_is_isolated` -- all 25
    node classes participate in at least one of the 12 edge classes. An isolated node
    class is written and then unreachable from any traversal.
12. `test_every_declared_node_class_is_emitted_by_the_adapter` -- no declared node
    class that nothing writes.
13. `adapter_optional_traversals` /
    `test_adapter_never_walks_through_an_optional_schema_field` -- **xfail(strict),
    defect 3 below**.

**Joins: the identity keys**

14. `compound_has_identity` / `compound_has_identity_gap` /
    `test_compound_identity_predicates_discriminate` +
    `test_small_molecule_perturbation_compound_is_checkable` -- the rule every other
    compound check rests on: identified, honestly gapped, and name-only are three
    distinguishable states.
15. `media_library_compound_issues` /
    `test_shared_media_compounds_are_identified_or_gapped` -- **xfail(strict), defect
    1 below**.
16. `media_base_issues` / `test_every_media_base_resolves_to_a_library_member` --
    **xfail(strict), defect 2 below**.
17. `media_derivation_issues` /
    `test_resolvable_media_bases_survive_into_their_derivatives` -- for every medium
    naming a base that exists, each (compound name, role) of the base reappears in
    the derivative or is a declared dropout. Clean today; the single missing
    component in SC-Ura is exactly the declared uracil dropout, which is what makes
    SC-Ura a typed edit of SC rather than a lookalike recipe.
18. `test_media_base_chain_is_acyclic` + `test_library_media_names_are_unique` -- a
    root names itself (`YPD.base_medium == "YPD"`, a fixed point, not a cycle); a
    real cycle would make "the base of this medium" undefined.
19. `environment_compound_collisions` /
    `test_a_component_compound_may_not_also_be_a_perturbation` +
    `test_shipped_library_media_carry_no_self_collision` -- the same species cannot
    be both a constant ingredient and the studied edit, because encoding it twice
    double-counts the dose. Identity is the InChIKey when both sides carry one, else
    the normalized name (so `D-glucose` and `d-glucose` collide, which is the point).
20. `join_key_audit` / three tests -- the census a verifier runs over a built LMDB.
    It takes `(experiment, reference)` MAPPINGS, so a verifier reads any dataset
    without importing its loader, and reports per-record joinability on genome,
    systematic gene names, media name, media base, temperature (present vs typed gap
    vs silent None), and full compound identification. Exercised on synthetic records
    built from the schema; running it over the 36 served and 14 unserved builds is
    the verifier's job, not a unit test's.

**Provenance honesty**

21. `test_gap_mixin_rule_is_never_weakened_by_a_subclass` (16 params) -- no
    `ProvenanceGapMixin` subclass shadows `validate_provenance_gaps` or redeclares
    `provenance_gaps`. The two honesty rules live in one inherited validator; a
    subclass redefining either would keep the attribute name while dropping the
    guarantee, and every downstream reader would still treat its gaps as audited.
22. `test_gap_on_a_populated_field_is_rejected_on_each_mixin_family` +
    `test_gap_on_an_unknown_field_is_rejected` -- the rule exercised through all
    three inheritance paths (`Compound`, `Environment`, `FitnessPhenotype`).
23. `sourced_value_fields` / `test_sourced_value_fields_keep_one_name` -- every field
    carrying `SourcedValue` is called `provenance` (2 today: `Media.provenance`,
    `MediaComponent.provenance`). One name is what lets a generic auditor walk any
    model and find the quotes without a per-class table.
24. `enum_value_collisions` / `test_no_schema_enum_aliases_two_members_to_one_value`
    -- `a = "x"` then `b = "x"` does not error in an `Enum`: `b` becomes an ALIAS of
    `a`, iteration yields one member, and two vocabulary terms collapse into one with
    no diagnostic anywhere. Clean across all 11 schema enums.

### Defects found today (the four xfails)

**1. The shared media library's own compounds cannot join.**
`torchcell/datamodels/media.py` builds every compound as a bare `Compound(name=...)`
instead of going through `torchcell/datamodels/compound_identity.py:196`
`resolved_compound`, which fills identifiers or attaches a typed
`ProvenanceGap(field="inchikey")`. Measured: of 55 distinct compound names in
`MEDIA_LIBRARY`, 3 carry a structure identifier (ethanol, galactose, glycerol), 8
carry an `inchikey` gap (the Bloom YP sugars), and **44 carry neither**. Restricting
to substances that name ONE molecule (a `defined` component or a dropout; mixtures
like peptone and commercial YNB legitimately have no structure) leaves **40 distinct
substances across 110 component slots**: `D-glucose` (media.py:114, 262, 359, 554),
the 20 SC amino acids (`_SC_AMINO_ACIDS`, media.py:315), the 9 YNB vitamins
(`_YNB_VITAMINS`, media.py:304), the four SGA selection agents (media.py:110), agar
(media.py:165), adenine, ammonium sulfate, and the four dropout compounds
(media.py:201-204). None of the 44 resolve from the pinned
`compound_identity_table.json` either, so closing this needs the table to grow, not
just a call-site change.

This is the single biggest join blocker in the current ontology: the media library is
the object the cross-dataset environment join is supposed to key on, and its own
components cannot join to ChEBI/PubChem, nor to a chemogenomic dataset's compound for
the same substance. It also breaks the review brief's own rule ("if we don't know the
molecule id there is no point in including this data") at the layer where that rule
matters most.

**2. `Media.base_medium` is free text and 3 of 22 library media point at nothing.**
`schema.py:1349` types it `str | None` with no controlled vocabulary. `SD_MINIMAL`
declares base `"SD"` (media.py:356) and the two SGA selection media declare
`"SD_MSG"` (media.py:210, 221); neither names a `MEDIA_LIBRARY` member. A base label
that resolves to no object carries no components and no provenance, so "aggregate
every record on an SD base" joins nothing. The fix is either to add `SD` and `SD_MSG`
as first-class library media (they are real recipes, and `SD_MSG` is what every
Costanzo/Kuzmin record sits on) or to type `base_medium` as a key into the library.
Note the derivation check (17) passes for every base that DOES resolve, so the
mechanism works; only the two labels are missing.

**3. `Environment.temperature` is optional but the adapter dereferences it
unguarded.** `schema.py:1518` makes `temperature` `Temperature | None` so a secondary
curation layer that never carried it records a typed gap instead of a guess.
`torchcell/adapters/cell_adapter.py` then walks straight through it at lines 612,
694, 759, 768, 769, 771, 782, 792, 793, 795 and 1788 (6 distinct chains after
deduplication). A record that legally gaps temperature raises `AttributeError` during
the KG build. This is latent rather than live -- no dataset in `dataset_adapter_map`
gaps temperature today -- but it is exactly the shape of break that a new
YeastPhenome-style dataset would hit on first admission, and the schema and the
adapter currently disagree about what a valid record is. The check is generic (any
adapter chain through any Optional schema field), so it keeps working after this one
is fixed.

**4. `SOTerm` is dead.** `schema.py:61` defines it and
`test_ontology_invariants.py:378-385` tests it, but no field composes it and no
adapter method emits it: the gene-perturbation leaves carry `mechanism_so_id` and
`mechanism_so_name` as plain strings. It is an unreachable class inside the schema
surface that `kg_manifest` still fingerprints. Either type those two fields on
`SOTerm` (which would be the coherent move, and would give the SO pair one encoding
instead of two parallel strings) or delete the class. Low severity; recorded so it
does not sit there indefinitely.

`Publication` is the other orphan and is NOT a defect: it is a record the DATASET
carries rather than the experiment, read by the adapter from `data["publication"]`
and emitted as a `publication` node class. It is named in `DOCUMENTED_ORPHANS` so
the reachability check stays exact.

### Checks considered and rejected

- **"Every `Compound` field is `Media.base_medium`-style validated at the model
  level."** Rejected as a model validator, kept as an audit callable. Putting the
  identity requirement inside `Compound.__init__` would change the class's contract
  fingerprint and therefore force a FULL rebuild of every served dataset
  (`kg_manifest` blocks on a served class whose closure changed). The same rule as an
  external callable costs nothing and is what a verifier or an admission gate can run
  per dataset, which is also where the "drop the records" decision belongs.
- **"No enum member is unused."** Rejected as theater that actively fights the design.
  Measured: 12 of 67 schema enum members are never named outside `schema.py`, but
  `TemperatureUnit.celsius`/`kelvin`/`fahrenheit` are only "unused" because
  `Temperature.unit` defaults to celsius and loaders never spell it, while
  `BiologicAgentClass.antibody`, `AssayType.halo_zone` and `MeasurementType.growth_rate`
  are vocabulary deliberately provisioned ahead of the datasets that will use them.
  A rule against that is a rule against additive expansion. The value-collision check
  (24) keeps the half of the idea that is a real silent hazard.
- **"Fingerprint every enum member referenced by a served closure."** Rejected as a
  duplicate. `torchcell/provenance/schema_deps.py` already content-addresses every
  schema symbol, and `kg_manifest.check_admission` already blocks an incremental
  admission when a served dataset's closure fingerprint moved. A second fingerprint
  in the test suite would drift from the one that actually gates the build.
- **"Every class reachable from `Experiment` belongs to exactly one lane."** Rejected
  as stated, replaced by checks 1 and 2. Exactly-one-lane is wrong as a rule: a
  `Compound` legitimately belongs to both the environment lane and (once metabolite
  phenotypes key on compounds instead of `s_NNNN` strings) the phenotype lane, and
  shared identity objects across lanes is the goal, not a violation. What must hold
  is DIRECTION, which is what `lane_back_edges` checks. The disjointness assertion is
  kept as a regression lock on today's state, not as a law.
- **"Run the join-key audit over the built LMDBs in the test suite."** Rejected for
  the test suite, kept as the callable. A test that needs `$DATA_ROOT` populated is
  not a test, it is a report; the verifier owns that run and the report path.
- **Duplicating `test_schema_invariants`'s registry-coverage checks.** Rejected;
  those already prove the maps cover the unions. Checks 4-6 add only the parts that
  module does not have (tag equality against the key, phenotype-class pairing,
  environment non-narrowing).

### Notable non-findings

These were suspected and measured clean, which is worth recording because they are
the checks people assume are failing:

- All 41 `BioCypherNode` sites match the yaml exactly, in both directions. The graph
  schema and the adapter are currently in perfect agreement.
- The phenotype bijection is exact at 13 <-> 13 with no ambiguity, derived from
  property sets alone.
- All 14 experiment/reference pairs agree on tag, phenotype class and environment
  class.
- No lane overlap, no back-edge, no enum value collision, no isolated node class, no
  medium losing a base component.

## 2026.09.12 - Defects 1 and 2 closed

The media library now builds every single-substance component through `resolved_compound` (55 substances identified, 9 undefined preparations by design, one gapped builder row), and every `base_medium` resolves to a `MEDIA_LIBRARY` key with a module-level check that raises otherwise ([[torchcell.datamodels.media]]). The two strict xfails now pass; the remaining xfail is `SOTerm` (defect 4), left in place deliberately. Defect 3 (the unguarded optional temperature) was closed by the adapter guard in the same branch.

## 2026.09.13 - Gene identity: the resolver is the policy, the FASTA set is not

Found while re-verifying Hoepfner 2014 (see [[plan.serve-all-50.2026.09.12]]): the L4
`gene_containment_sgd` / `current_genome_genes` rules take their gene universe from the
R64 ORF + RNA FASTA headers, and that set lists `pseudogene`, `blocked_reading_frame` and
`transposable_element_gene` features as ORFs. The L1 `canonical_gene_names` rule, when a
resolver is supplied, asks `SCerevisiaeGenome.resolve_gene_name` and fails any name whose
status is not CURRENT. The two disagree on exactly those non-gene features, by
construction. Loader convention that follows: every loader resolves its source gene names
through the shared resolver (CURRENT kept; RENAMED kept under the current systematic name
with the source name as `perturbed_gene_name`; NON_GENE_FEATURE and RETIRED dropped to a
ledger). A loader that filters on the FASTA set alone passes L4 and fails L1. This is not
a DAG check and is not added to `ontology_checks.py`; it is enforced by running the L1
rule with a resolver on every gene-perturbation dataset (`runners.py` supplies one to
the environment-response and fitness runners; the segregant runner has no gene
perturbations to name, and the expression, morphology, metabolite, protein and RNA-seq
runners do not yet take one, which is an open gap to close when those verifiers are
next touched).
