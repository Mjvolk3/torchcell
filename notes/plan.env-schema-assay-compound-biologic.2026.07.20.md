---
id: jjo3l7w4s9sltzt8lx9v0r8
title: '20'
desc: ''
updated: 1784597799708
created: 1784597799708
---

## Context

This is **UI-1 of a 3-unit pipeline** that came out of an environmental / chemogenomic
dataset audit of `torchcell/datamodels/schema.py`. The audit (see
`[[plan.torchcell-perturbation-ontology.2026.07.08]]` for the ontology design record,
and the env-chemogenomic ingestion memory `environmental-chemogenomic-ingestion-plan`)
surfaced three gaps in the env-response side of the schema. UI-1 lays the **additive,
non-breaking schema foundation** for all three; it adds NO validators and enforces
nothing new. UI-2 and UI-3 do the enforcement/population and are referenced but NOT
planned here.

The three audit findings and what UI-1 does about each:

1. **Assay method is being smuggled through free-text `units`.** `EnvironmentResponsePhenotype`
   (`schema.py:2690`) records WHAT a number is via `measurement_type: MeasurementType`
   (locked axis G3) and a human-readable `units` string, but there is no typed field for
   HOW the response was measured -- the experimental design (Bar-seq pooled growth vs
   colony-size array vs halo zone vs biosensor). Curators have been leaking that
   distinction into `units`, which is a category error and un-queryable. UI-1 adds a typed
   nullable `assay_type: AssayType | None`.
2. **No home for a biologic (peptide/protein/antibody/toxin) as the added agent.** The
   env-perturbation union (`schema.py:1407`) has `SmallMoleculePerturbation` and
   `EnvironmentPhysicalPerturbation` only; a proteinaceous agent added to the medium has
   nowhere typed to live. UI-1 adds `BiologicPerturbation` as a third union leaf.
3. **`Compound` (`schema.py:1081`) cannot record a documented identity gap.** When an
   InChIKey/CID is genuinely unavailable, the loader has no typed way to say "looked, not
   found" -- it silently stores `None`, indistinguishable from "not yet resolved." UI-1
   gives `Compound` the `ProvenanceGapMixin` **affordance** (no gap validator this unit).

**Follow-ups (do NOT build here):** UI-2 = compound-identity resolver + `ProvenanceGap`
enforcement on `Compound` + reconciliation of `Compound` gaps vs `Media.open_gaps`. UI-3
= `assay_type` population across loaders + loader hygiene + the L1 uniqueness-key decision.

## Relevant Files

| Path | Action | Purpose | Stance |
| --- | --- | --- | --- |
| `torchcell/datamodels/schema.py` | MODIFY | Add `AssayType`, `BiologicAgentClass`, `BiologicPerturbation`; widen `EnvironmentPerturbationType`; add `assay_type` field; `Compound` gains `ProvenanceGapMixin`; docstring edits | stable-core / additive-growth |
| `tests/torchcell/datamodels/test_ontology_all_trees.py` | MODIFY | Add `BiologicPerturbation` entry to `_ENV_FACTORY` (`:214`) -- REQUIRED or 3 ontology tests fail | stable, gate |
| `torchcell/datamodels/__init__.py` | MODIFY | Export `AssayType`, `BiologicAgentClass`, `BiologicPerturbation` (import block `~13-55`, `__all__` `~106`) | stable |
| `tests/torchcell/datamodels/test_schema.py` (or new module beside it) | MODIFY / NEW | New unit tests for the 3 additions (see Tests) | new |
| `torchcell/datamodels/media.py` + ~15 loaders | REFERENCE | Existing `Compound(name=...)` call sites -- must stay valid (they do: mixin default_factory=list) | unchanged |
| `torchcell/data/experiment_dataset.py` | REFERENCE | LMDB round-trip via pickle (`:461`, `:494`) -- union widening cannot break reads | unchanged |
| `notes/torchcell.datamodels.schema.md` | REFERENCE | Paired module note -- **STALE** (last entry 2026.01.29); authoritative design lives in the ontology plan + memory | do not trust as current |

## Key Design Decisions

1. **Additive + optional only; ZERO validators this unit.** Every new field/class is
   nullable-with-default or a new union leaf. Nothing existing is required, renamed, or
   re-validated. *Why:* the `is_essential`/`#92` break was caused by a field becoming
   REQUIRED, re-invalidating old serialized dicts. `assay_type` is the inverse -- nullable,
   defaulted -- so old records re-validate cleanly. `ModelStrict` (extra=forbid) rejects
   unknown-PRESENT keys, not missing-optional keys, so an old dict lacking `assay_type`
   passes.
2. **`assay_type` on `EnvironmentResponsePhenotype` ONLY, not the `Phenotype` base**
   (settled). *Why:* the assay concept is meaningful only for env-response readouts; the
   other phenotype subclasses (fitness, gene-interaction, protein-abundance) do not carry
   a chemogenomic assay design, and putting it on the base would create a null field on
   every phenotype for no gain.
3. **`AssayType` records HOW (method); `MeasurementType` records WHAT (number semantics);
   free-text `units` is NOT a substitute for either.** *Why:* the audit found method
   detail leaking into `units`. `assay_type` is orthogonal to the locked G3 `measurement_type`
   axis -- a pooled Bar-seq assay can yield a `log2_ratio` OR a `z_score`, so the two axes
   cross. The `EnvironmentResponsePhenotype` docstring must state this explicitly.
4. **`BiologicAgentClass` = `peptide` / `protein` / `antibody` / `toxin`** (settled). *Why
   these four:* they cover the proteinaceous agents seen in the DB and every token is
   BANNED_VOCAB-clean. Critically `toxin != toxic` -- the M1 test
   (`test_m1_env_perturbation_is_edit_only`) does exact-token matching after splitting on
   `[_\-]`, and `toxic` (banned) never appears as a token of `toxin`. `BiologicPerturbation`
   names the EDIT (the agent ADDED), never a consequence word.
5. **`Compound` gets `ProvenanceGapMixin` as AFFORDANCE ONLY -- no structure-or-gap
   validator this unit** (settled). *Why:* enabling the mixin is a base-class change that
   is safe (default_factory=list -> the validator is a no-op with zero gaps, so all
   existing `Compound(name=...)` calls stay valid). Enforcing "an unresolved identity MUST
   carry a gap" is a behavioral change that belongs in UI-2 with the resolver, not here.
6. **`Compound.provenance_gaps` vs `Media.open_gaps` -- keep BOTH, reconcile later**
   (settled). *Why:* `Media` already has its own gap mechanism (`open_gaps`); `Compound`
   getting `ProvenanceGapMixin` introduces a second, differently-named gap surface. Rather
   than unify now (risking a `Media` regression), UI-1 records a reconciliation follow-up
   and both stay affordance-only until UI-2 enforces either.
7. **L1 uniqueness key NOT touched in UI-1 -- deferred to UI-3** (settled). *Why:*
   `assay_type` is null until loaders populate it (UI-3); folding a null field into the L1
   dedup key now risks a dedup regression for zero benefit while every value is None.
8. **Keep all new classes PLAIN, top-level, non-generic (no PEP695 generics).** *Why:* the
   pickle-by-qualname hang (PR `#119`, memory `kg-adapter-pickle-hang-fixes`) is avoided
   iff `BiologicPerturbation`, `BiologicAgentClass`, and the `Compound` mixin stay plain,
   module-level, and non-generic. LMDB persists these via `pickle` by qualname, so a
   generic/local class would reintroduce the hang.

## Approach

Execute in this order; each step is self-contained and the tests only pass once the
coordinated ontology edits (step 2) are ALL present.

**Step 1 -- `AssayType` enum + the `assay_type` field (schema.py).**
Add `class AssayType(StrEnum)` immediately beside `MeasurementType` (`schema.py:2662`),
in the `MeasurementType` style: explicit `x = "x"` members + a bulleted docstring mapping
each member to the DB assays it covers. Seed the vocabulary to what the DB already holds:
`pooled_competitive_growth_barcode` (HIP/HOP, Bar-seq), `colony_size_array`
(SGA / condition-SGA), `spot_dilution`, `halo_zone`, `liquid_od_growth`,
`biosensor_readout`, `other`. Then add to `EnvironmentResponsePhenotype` (among its fields
`~2704-2746`):

```python
assay_type: AssayType | None = Field(
    default=None,
    description="HOW the response was measured (experimental design); "
    "orthogonal to measurement_type (WHAT the number is). Free-text units is NOT a "
    "substitute for this typed method axis.",
)
```

Extend the class docstring to state the HOW/WHAT/`units`-is-not-a-substitute distinction.

**Step 2 -- `BiologicPerturbation` (THREE coordinated edits -- all required or ontology
tests fail).**
(a) In `schema.py`, near the env-perturbation classes (`schema.py:1356-1409`), add
`class BiologicAgentClass(StrEnum)` (members `peptide` / `protein` / `antibody` / `toxin`)
and `class BiologicPerturbation(EnvironmentPerturbation, ModelStrict)` with:
`perturbation_type: Literal["biologic"] = "biologic"`; a `description` default;
`agent_class: BiologicAgentClass`; `name: str`; `uniprot_id: str | None = None`;
`sequence: str | None = None`; `concentration: Concentration`. The docstring names the
EDIT (the biologic agent added), never a consequence; keep it BANNED_VOCAB-clean.
(b) Widen the union at `schema.py:1407` to
`SmallMoleculePerturbation | EnvironmentPhysicalPerturbation | BiologicPerturbation`. This
is a plain pydantic-v2 smart union (no discriminator) -- the new `Literal["biologic"]`
tag keeps parse unambiguous (S5 discriminator-uniqueness).
(c) In `tests/torchcell/datamodels/test_ontology_all_trees.py:214`, add a `_ENV_FACTORY`
entry for `BiologicPerturbation` with a minimal valid kwargs dict, e.g.
`dict(agent_class=BiologicAgentClass.protein, name="nisin", concentration=Concentration(value=1.0, unit=ConcentrationUnit.millimolar))`.
The ontology suite asserts `union members == discovered leaves == _ENV_FACTORY keys`
(`test_s3_union_equals_concrete_leaves:151`, `test_env_factory_covers_every_leaf:230`,
plus `test_s6_env_round_trip` and `test_s5:177`) -- omitting any of (a)/(b)/(c) fails at
least one.

**Step 3 -- `Compound` mixin + docstring (schema.py).**
Change `class Compound(ModelStrict)` (`schema.py:1081`) to
`class Compound(ProvenanceGapMixin, ModelStrict)`. Add NO validator. Update the
`ProvenanceGapMixin` docstring (`schema.py:1412`), which currently reads "Shared by
`Phenotype` and `Environment`", to add `Compound`. All existing `Compound(name=...)` sites
in `media.py` + ~15 loaders stay valid (mixin's `provenance_gaps` has `default_factory=list`;
its validator is a no-op when the list is empty).

**Step 4 -- exports (`__init__.py`).**
Add `AssayType`, `BiologicAgentClass`, `BiologicPerturbation` to both the `from .schema
import (...)` block (`~13-55`) and `__all__` (`~106`), keeping alphabetical/grouped order.

**Step 5 -- tests** (see Tests section).

**Step 6 -- verify** (see Verification): datamodels suite, the two ontology test files,
whole-tree mypy, schema-impact dry-run.

**Step 7 -- commit (from the worktree; schema-impact classifies this BREAKING).**
Use the `PYTHONPATH="$WT" TORCHCELL_SCHEMA_ACK=1 git commit` incantation from Gotcha 1
(inspect with the dry-run first). No hash file to regenerate; NEVER `--no-verify`.

## Tests

Add to `tests/torchcell/datamodels/test_schema.py` (or a new module beside
`test_ontology_all_trees.py`), plus the mandatory `_ENV_FACTORY` entry from step 2c:

1. **`AssayType` round-trips** -- every member serializes to its string value and parses back.
2. **`assay_type` default + set value** -- `EnvironmentResponsePhenotype(...)` constructs
   with `assay_type` defaulting to `None`, and also accepts an explicit
   `AssayType.pooled_competitive_growth_barcode`.
3. **`ProvenanceGap` on `assay_type` validates when the field is `None`** -- an
   `EnvironmentResponsePhenotype` with a `ProvenanceGap(field="assay_type", ...)` and
   `assay_type=None` passes (proves the mixin invariant already resolves the inherited
   field; no new validator needed).
4. **`BiologicPerturbation` constructs, round-trips through
   `TypeAdapter(EnvironmentPerturbationType)`, and parses back to the concrete type**,
   while a `SmallMoleculePerturbation` in the same union STILL parses (no regression from
   widening).
5. **`Compound` accepts a `ProvenanceGap` on `'inchikey'` when `inchikey is None`** --
   affordance works.
6. **A `Compound` with NO structure and NO gap still constructs** -- `Compound(name="x")`
   is valid, proving NO strict structure-or-gap enforcement was added.

## Gotchas

1. **schema-impact pre-commit gate CRASHES from a worktree AND flags this BREAKING**
   (memory `schema-impact-worktree-crash`). It imports main's `torchcell` rather than the
   worktree's, and the `Compound` base-class change trips the breaking classifier. Commit
   with `WT=<worktree>; PYTHONPATH="$WT" TORCHCELL_SCHEMA_ACK=1 git commit ...`; inspect
   first with `PYTHONPATH="$WT" python scripts/schema_impact_check.py --base HEAD`. No hash
   file to regenerate. NEVER `--no-verify`.
2. **Ontology three-edit trap.** Adding `BiologicPerturbation` requires ALL of: the class,
   the union widening (`schema.py:1407`), AND the `_ENV_FACTORY` entry
   (`test_ontology_all_trees.py:214`). The suite asserts `union == leaves == factory keys`;
   any one omission fails `test_s3` / `test_env_factory_covers_every_leaf` / `test_s6`.
3. **`extra=forbid` is additive-safe here, not a hazard** (see Key Design Decision 1 for
   the mechanism). Do NOT make `assay_type` required and do NOT add a validator -- that
   would recreate the `#92` break instead of staying its inverse.
4. **M1 banned-vocab is exact-token.** `test_m1` splits names on `[_\-]` and matches
   against `{stress, sensitive, tolerant, resistant, essential, lethal, damaging, toxic,
   inhibitory, suppressor}`. `toxin` is CLEAN (`toxic != toxin`). Do not introduce a field
   or enum member whose tokens hit that set on any env leaf.
5. **CI mypy is diff-scoped** (memory `torchcell-ci-mypy-diff-scoped`) -- it only checks
   changed files, so a union-widening or base-class regression elsewhere can land
   invisibly. Run WHOLE-TREE mypy locally.
6. **Stale-LMDB rebuild is a FOLLOW-UP, not UI-1.** No dataset is rebuilt here; existing
   built LMDBs round-trip unchanged because the union widens via a plain smart union and
   pickle reads (`experiment_dataset.py:461,494`) are tolerant of the added optional field.
   Repopulating `assay_type` (and any `database/data/...` rebuild) is UI-3.

## Verification

```bash
PY=~/miniconda3/envs/torchcell/bin/python
# 1. datamodels suite (must stay green; current baseline ~395-404 tests)
$PY -m pytest tests/torchcell/datamodels/ -xvs
# 2. the ontology gate specifically
$PY -m pytest tests/torchcell/datamodels/test_ontology_all_trees.py \
              tests/torchcell/datamodels/test_ontology_invariants.py -xvs
# 3. whole-tree mypy (diff-scoped CI will NOT catch a union/base-class regression)
$PY -m mypy torchcell/datamodels/schema.py torchcell/datamodels/__init__.py
# 4. schema-impact dry-run from the worktree (inspect the BREAKING classification)
WT=/home/michaelvolk/Documents/projects/torchcell.worktrees/plan/env-schema-assay-compound-biologic
PYTHONPATH="$WT" $PY scripts/schema_impact_check.py --base HEAD
```

Confirm the change is **additive-only**: no `datamodels/` or `verification/` test that was
green before regresses, and no existing `Compound(name=...)` / `EnvironmentResponsePhenotype`
call site breaks. The four ontology invariants (S3 union==leaves, S5 discriminator
uniqueness, S6 round-trip, M1 edit-only vocab) must all pass with `BiologicPerturbation`
present.
