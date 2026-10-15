---
id: 7fg6e8u39ry9bfmt2btlksb
title: Perturbation_families
desc: ''
updated: 1792086529375
created: 1792086529375
---

## 2026.10.15 - Lane R: the three audit flags that were never proposals

Lane R of the ontology unification (Path U) owns the three findings the 2026-10-15
audits flagged and that had never been fixed: the strain constant in the fingerprinted
schema surface (audit A section 5.1), the two morphology classes named backwards (audit A
section 5.2), and the hand-enumerated perturbation sets in verification (audit C section
2). This note holds the measurements and what each one became. Source of the module:
`torchcell/verification/perturbation_families.py`.

### Item 1. A strain's verbatim genotype left the schema surface

`BW25113_BACKGROUND_GENOTYPE` and `BW25113_BACKGROUND_LESIONS` now live in the genomes
tier, `torchcell/sequence/genome/registry.py`, beside the `ECOLI_K12_BW25113` set id whose
deposited GenBank `/note` they were read from. The provenance comment moved with them and
the data-gated test that re-reads the string from the real bytes moved from
`tests/torchcell/datamodels/test_schema.py` to
`tests/torchcell/sequence/genome/test_registry.py`. One loader reads the lesions
(`torchcell/datasets/ecoli/price2018.py`); the string and the six designations are
byte-identical, so no stored value changes.

Measured, and it corrects the audit: on this branch the two bindings are in **1 and 0
loader closures**, not the 84 the audit reported from main before the unification
(`torchcell.provenance.schema_deps.symbol_dependents` over `torchcell/datasets/**`:
`BW25113_BACKGROUND_LESIONS` -> `price2018.py` only). The record's
`rebuild_scope` is `none`, tool-computed, and no served dataset class holds either symbol.
The defect was never the blast radius: it is that the class-contract surface held data
about one strain, so every future host had a place in `schema.py` to put its genotype
string. Record: `database/schema_changes/2026-10-15-bw25113-background-genotype-leaves-the-schema-surface.json`.

### Item 2. `MorphologyPhenotype` is the general class; `CalMorphPhenotype` keeps its name

`BacterialMorphologyPhenotype` -> `MorphologyPhenotype`, and
`torchcell/datamodels/bacterial_morphology_features.py` ->
`torchcell/datamodels/morphology_features.py` with its test file. Verified before
changing: the renamed class takes its vocabulary from the assay a record names
(`assay` plus `MORPHOLOGY_ASSAYS`), so nothing in it is host specific, while
`CalMorphPhenotype` hard-codes 281 base and 220 CV parameter names and serves 3 datasets
(Ohya 2005, Ohnuki 2018, Ohnuki 2022). Lane F had already renamed the experiment pair
above it and left the phenotype for a later stage.

`CalMorphPhenotype` keeps its name because the name already states what the class is
specific to: CalMorph is the *S. cerevisiae* program whose fixed 501-parameter output the
hard-coded vocabulary is. `CalMorphMorphologyPhenotype` would add a word and no fact, and
`YeastMorphologyPhenotype` would be false in the same way the old name was, since a yeast
screen that is not CalMorph belongs on the general class with its own assay. Record:
`database/schema_changes/2026-10-15-morphology-phenotype-is-the-general-class-calmorph-is-the-specific-one.json`
(`rebuild_scope: dev_stores`, 1 loader, 0 served datasets).

Follow-up, not done here and named so it is not lost: the two verifier modules are
backwards for the same reason (`torchcell/verification/morphology.py` verifies CalMorph,
`torchcell/verification/bacterial_morphology.py` verifies the general class). Swapping two
module names in one commit is a readability hazard, so it is a change of its own.

### Item 2b. The called-variant vocabulary and call object drop the host prefix

Added to the lane by the coordinator after lane M's migration table found the two classes
lane D left out of P1 (they are composed call objects, not perturbation leaves):
`BacterialVariantType` -> `VariantType`, `BACTERIAL_VARIANT_TYPE_SO` -> `VARIANT_TYPE_SO`,
`BacterialVariantCall` -> `GenomeVariantCall`.

Checked against the evidence before renaming, and the evidence is conclusive: the
vocabulary is composed onto `ConstructVariantCall` as well as onto the genome call, so it
sits in **15 loader closures and three of them are yeast stores** that call variants on a
construct contig. `torchcell/datasets/scerevisiae/cachera2023.py` writes
`VariantType.substitution` by name. Its four members (snv, insertion, deletion,
substitution) map to the same four SO terms the yeast leaves already use. The call object
is a replicon named verbatim, a 1-based end-inclusive interval, the released change and
allele cells, an optional amino-acid change and codon, and a frequency with its mode and
basis: a yeast isolate resequencing releases that same shape, and the sibling vocabularies
declared in the same block (`VariantCallMode`, `VariantFrequencyBasis`) were host-neutral
from the start. Nothing in either class is bacteria-specific.

`GenomeVariantCall` names the coordinate SPACE because that is the only thing separating it
from `ConstructVariantCall`: a replicon of a deposited genome here, a construct contig
identified by its sha256 there. A bare `VariantCall` would read as the general case of
which the construct one is a special case, which is false.

Why it had to happen in this rebuild: the tool computes **`full_kg_rebuild`, 3 served
datasets** (`BetaxanthinCachera2023Dataset`, `CarotenoidOzaydin2013Dataset`,
`CrispriMormino2022Dataset`), so after 4.0 this pure naming fix would have cost a second
full rebuild. Record:
`database/schema_changes/2026-10-15-the-called-variant-vocabulary-and-call-object-drop-the-host-prefix.json`,
and the PR carries the `schema-breaking` label for it.

### Item 3. Every hand-enumerated perturbation set, and what it became

| hand-written set | where | what it is now |
|---|---|---|
| `_FULL_DELETION_TYPES` = 2 SGA tags | `verification/runners.py` | `full_deletion_perturbation_types()`, the 11 subclasses of `DeletionPerturbation` |
| `"gene_addition"` string | `verification/metabolite.py`, `verification/visual_score.py` (the deleted-ORF key) | `addition_perturbation_types()`, the 4 subclasses of `GeneAdditionPerturbation` |
| `"gene_addition"` string | the same two modules' L4 gene sets | `heterologous_identifier(perturbation, species)`, the field test the bioproduction families already used |
| `perturbation_type == "heterologous_pathway"` | `verification/product_titer.py` | `pathway_member_perturbation_types()`, the leaves declaring `pathway_name` |
| `perturbation_type == "site_variant_call"` | `verification/runners.py` | `site_keyed_perturbation_types()`, the leaves declaring `flanking_systematic_gene_names` |
| `ALLELE_BEARING_PERTURBATION_TYPES` = 5 tags | `verification/common.py` | kept explicit, made exhaustive by test: it and `PLAIN_GENE_NAME_PERTURBATION_TYPES` must partition the 38-leaf union |
| `GENOME_REFERENCE_RULES` = rule names | `verification/report.py` | kept explicit (a rule name is not a schema symbol), made exhaustive by a fleet sweep |

**The measured defect in a hand-written set.** `ADDITION_PERTURBATION_TYPES` held
`{"gene_addition", "heterologous_pathway"}` while `GeneAdditionPerturbation` has four
placed subclasses, so `promoter_driven_gene_integration` and
`variable_promoter_reporter` were missing from every copy of that set. It lives in
`torchcell/verification/construct_placement.py`, which lane T owns, so lane R did not edit
it: the derivation `addition_perturbation_types()` is in place for lane T to adopt, and the
construct-placement census skips those two leaves until it does.

**The measured defect in `GENOME_REFERENCE_RULES`.** Sweeping every literal rule name the
fleet emits (249 static names over `torchcell/verification/**` and `torchcell/datasets/**`)
for the genome-reference vocabulary found **three rules classified `gate` that say in their
own messages that they are findings**: `perturbed_locus_in_bw25113_annotation` (Li 2021),
`released_b_numbers_on_mg1655_annotation` (Baba 2006) and
`released_tags_on_kt2440_annotation` (Borchert 2023), each reading a stored or released
identifier against OUR annotation of the host and each hard-coded `passed=True`. They are
now `report`, and
`tests/torchcell/verification/test_finding_class.py::test_the_registry_is_exhaustive_over_every_rule_name_the_fleet_emits`
sweeps the fleet so the next one fails a test rather than silently gating a store on our
annotation. The nine gates whose names match the vocabulary for another reason (a pinned
COUNT, the SOURCE's annotations, a companion release) are listed with their reasons in
`DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES`.

**Behavior changes on real stores, stated plainly.** Two, both in the more-correct
direction, and both re-measured by the rebuild's own verification pass:

1. An ADDED gene is now part of the strain signature in `verification/metabolite.py`. A
   store whose addition background is constant has that constant in every key and so
   groups exactly as before; a store whose strains differ only in what they add (every
   production-chassis release) no longer reads as one strain measured many times.
2. The L4 gene universe of the metabolite and visual-score families is now "every perturbed
   name except the heterologous ones" instead of "every name except the `gene_addition`
   ones". Measured over the built stores: no store in either family carries an addition
   leaf other than `gene_addition` today, and 162 of Brunk 2016's 603 pathway
   perturbations name a NATIVE locus of their own host (`source_organism` equal to the
   record's species), which the type-string rule left out of the universe while it kept
   the heterologous names in.

### Found, measured, and deliberately NOT changed

- **`environment_response._genotype_signature`'s `_identity`** is the largest
  hand-maintained list in verification (17 discriminator fields across four hand-written
  blocks) and the audit is right that nothing compares it with the union. Measured: the
  38 genotype leaves declare **92 fields** beyond `GenePerturbation`'s 5: **17** are read
  by the three identity sites, one (`perturbation_type`) is the key's own second element,
  and the other **74** are read by no identity site at all. The three sites also
  disagree with each other: `fitness.py` keys on `strain_id` and
  `environment_response.py` does not, while `environment_response.py` keys on
  `collection` and `construction` and `fitness.py` does not. Deriving the set is not a
  shape question -- `barcode` and `library_pool` are strain discriminators while
  `description`, `state`, `mechanism_so_id` and `provenance_gaps` are not -- so making it
  exhaustive means classifying 74 fields one at a time, and adding any of them changes
  L1 duplicate accounting on every store in the family. That is its own lane with its own
  re-verification, not a line in this one.
- **`runners._bacterial_families`** and the per-family dataset registries enumerate
  DATASETS, not schema leaves. A dataset registry is explicit by nature and a missing
  entry is caught by `resolve_verifier` plus the typed `--verify` outcome, so there is
  nothing to derive.
- **`PROJECTED_PERTURBATION_CLASSES`** (`torchcell/datamodels/identity.py`) is already
  pinned to `get_args(EnvironmentPerturbationType)` by its test (PR #1104), which is the
  mechanism this note generalizes. Not broken.
- **`--verify` returning a silent pass** is already fixed: `VerifyOutcome` in
  `torchcell/database/build_dataset_lmdb.py` is a typed outcome and a dataset with no
  per-dataset entry point prints `no_per_dataset_verifier` rather than a pass. Nothing to
  do, and no verifier was found still naming a leaf the unification renamed: every
  tag-shaped literal under `torchcell/verification/**` was checked against the 43 current
  tags and the only perturbation tags present (`site_variant_call`,
  `inducible_promoter_insertion`, `heterologous_pathway`) are current.
