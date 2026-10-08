---
id: uj4ivis9npprhlthckrj5kp
title: Schema Dependency Tracking
desc: ''
updated: 1784161583113
created: 1784161583113
---

## 2026.07.15 - Design and Implementation

### Motivation

When the component-based `Media` schema landed (commit `1cf60cdc`), it added a required
`is_synthetic` field and updated 33 dataset loaders — but **missed `hoepfner2014.py`**. CI is
diff-scoped (checks only changed files), so it never re-checked the unchanged loader; the break
stayed latent until that dataset was next built and failed with a pydantic `ValidationError`.

The ask: *when a schema part a dataset depends on changes, flag that dataset for rebuild* —
without rebuilding every dataset on every schema edit, and working across machines that do not
all hold every dataset.

### Two scoping levers (why it stays tight, not cry-wolf)

1. **Closure scoping.** A loader depends only on the schema symbols reachable from its imports
   (transitive containment closure). A change to a symbol *outside* that closure never flags it.
   Empirically: a change to `MetabolitePhenotype` flags 7 datasets, `RNASeqExpressionPhenotype`
   flags 1 — not the whole fleet.
2. **Contract fingerprint.** Each symbol gets a SHA-256 of its *contract* — field shape (name /
   type / required-ness / semantic `Field(...)` kwargs), validator/serializer bodies, enum
   members, `Config`, bases — deliberately **excluding** docstrings, comments, plain methods,
   `Field` descriptions, and field ORDER. So a benign edit (reword a docstring, add a helper
   method, reorder fields) leaves the fingerprint unchanged and flags **nobody**, even for a
   universal symbol like `Media` that sits under every record.

These are orthogonal: closure handles localized changes; the fingerprint tames the universal
symbols. Only a *real contract change to a universal symbol* (e.g. adding required `is_synthetic`
to `Media`) flags the whole fleet — and that is correct, because they all genuinely need it.

Required-ness is detected through `Field(...)`: `is_synthetic: bool = Field(description=...)`
(no default) is correctly seen as **required** — the exact case that distinguishes a breaking
change from a benign one.

### Two tiers

- **Stage 1 — static impact gate** (`torchcell/provenance/schema_impact.py`). Proactive. Diffs
  the working-tree schema surface against a git ref (default `HEAD`), classifies each changed
  symbol `breaking` (construction fails / stored records invalid → must fix + rebuild) or `stale`
  (build still works but records differ → rebuild to refresh), and maps each change to the loaders
  whose closure contains it. Runs as a **pre-commit hook** (fires only when
  `torchcell/datamodels/{schema,pydant}.py` is staged); blocks on a breaking change unless
  `TORCHCELL_SCHEMA_ACK=1`. Needs no stored state — git + the recomputed graph *is* the state, so
  it runs on any machine from source alone.
- **Stage 2 — build manifests** (`torchcell/provenance/build_manifest.py`). Retroactive,
  authoritative. When a dataset's `process()` finishes, the `post_process` hook writes
  `preprocess/build_manifest.json` recording the contract fingerprint of every symbol in the
  loader's closure. `check_all` recomputes those fingerprints from the *local* schema and reports
  which built datasets are stale. Enumerates built datasets by **globbing local LMDBs**
  (`$DATA_ROOT/data/torchcell/*/processed/lmdb`), NOT the dataset registries — so it only ever
  reports datasets **this machine actually holds**.

### Multi-machine (GilaHyper is the stage-2 machine, for now)

Fingerprints are content-addressed (a hash of the contract, not a git SHA or timestamp), so a
manifest written on machine A is meaningful on machine B: staleness is judged by
`fingerprint_now(local schema) != fingerprint_stored`, computed entirely from **local git state** —
no server, no central DB. Manifests live under `DATA_ROOT` next to the LMDB they describe (never
in git), so each machine tracks its own built artifacts with zero cross-machine git contention,
and a manifest travels with its dataset if the dataset is copied.

Because not all machines hold all datasets, the split maps onto the machine reality:
**stage 1 (code gate) runs anywhere** (needs only source); **stage 2 (rebuild/staleness) is
inherently per-machine** and is meaningful where datasets live — GilaHyper holds all of them, so
`python scripts/check_dataset_staleness.py` there is the authoritative rebuild list. A machine
with a subset checks only that subset; a machine with none reports nothing. No machine is ever
told to rebuild something it does not have.

### Schema surface

`torchcell/datamodels/schema.py` (95 record classes) + `torchcell/datamodels/pydant.py`
(the `ModelStrict` base every record inherits, so a change to its `Config` correctly cascades to
all records). `media.py` / `calmorph_labels.py` export constant *instances*, not new record types,
so they are not part of the class-contract surface (extending value-fingerprints to those constants
is a possible follow-on).

### Files

- `torchcell/provenance/schema_deps.py` — AST core: `contract_spec` / `fingerprint`,
  `SchemaSurface` + `load_default_surface`, `forward_closure`, `loader_closure`.
- `torchcell/provenance/schema_impact.py` — stage 1: `classify_change`, `diff_surfaces`,
  `map_impacts`, `build_impact_report`, CLI.
- `torchcell/provenance/build_manifest.py` — stage 2: `BuildManifest`, `write_build_manifest`,
  `check_manifest`, `check_all`, CLI.
- `scripts/schema_impact_check.py` + `scripts/run-schema-impact.sh` — pre-commit entry.
- `scripts/check_dataset_staleness.py` — the staleness CLI (run on GilaHyper).
- Hook: `torchcell/data/experiment_dataset.py` `post_process` calls `write_build_manifest(self)`.
- Pre-commit hook `schema-impact` in `.pre-commit-config.yaml`.
- Tests: `tests/torchcell/provenance/` (35 tests, mypy-strict clean).

### Usage

```bash
# Stage 1: what does my staged schema change affect? (also runs automatically as a pre-commit hook)
python scripts/schema_impact_check.py                 # vs HEAD
python -m torchcell.provenance.schema_impact --base origin/main --strict

# Stage 2 (on GilaHyper): which built datasets are now stale vs the local schema?
python scripts/check_dataset_staleness.py
```

### Design decisions

- **No fallbacks.** `git` info in the manifest is best-effort provenance (returns `None` when not
  a checkout, e.g. an installed wheel) and is *never* used for the staleness decision — fingerprints
  are. This models "may not be a checkout" explicitly rather than masking an error.
- **Glob local LMDBs, not the registry** for stage-2 enumeration — the choice that makes it correct
  on machines that don't hold every dataset.
- **`StrEnum`** for `ChangeKind` (repo targets 3.13); the pre-existing `(str, Enum)` schema enums
  predate the `UP042` rule and are latent under diff-scoped CI.

Related: [[torchcell.datamodels.media-components]], [[torchcell.datasets.scerevisiae.hoepfner2014]].

## 2026.10.08 - Module-Level Vocabularies in the Contract Fingerprint (#734)

### Problem

The closure held classes only. A field annotated with a module-level `Literal` alias
(`BacterialGeneNamespace`), a validator reading a pattern map (`BACTERIAL_LOCUS_TAG_PATTERNS`),
or a validator calling a module-level helper that reads one, moved no fingerprint when that
module-level name changed. Measured on this branch: narrowing `BacterialGeneNamespace` by
`ecoli_b_rel606_locus_tag` moved 0 fingerprints under the old rule.

### Rule

`schema_deps.collect_module_bindings` collects every top-level non-class binding of the surface
modules (`x = ...`, `x: T = ...` with the annotation dropped, `type X = ...`, and functions with
the docstring stripped). `class_module_bindings` seeds from the binding names a class body
mentions and follows each binding's own references transitively; classes are not traversed,
since a class a binding names is a closure node with its own fingerprint. The reached bindings
go into `ContractSpec.module_bindings` and are appended to `canonical()` as `module::<name>::<source>`
lines, last and only when present. A class that reaches no module-level name therefore keeps its
pre-#734 canonical text byte for byte, and the closure graph (`ref_graph`) is unchanged.

`schema_impact.classify_change` reads the same field, so the pre-commit `schema-impact` hook sees
the same vocabularies: for a `Literal`, a union alias, a tuple/set display or a dict display,
`binding_members` gives a member set; lost members are BREAKING, only-gained members are stale,
a reorder is stale, and a change to a binding with no member set (a helper function, a computed
string) is BREAKING. Verified on the live schema: narrowing `BacterialGeneNamespace` in the
working tree makes `scripts/run-schema-impact.sh` exit 1 with 6 changed classes and 29 impacted
bacterial loaders; the REL606-shaped widening is stale (pinned in `test_schema_impact.py`).

### Measured fingerprint movement (2026.10.08, branch tip on #778)

Old rule vs new rule on the same tree (`schema.py` + `pydant.py` of this branch):

- 169 surface classes; 141 reach no module-level binding and are identical under both rules.
- 28 move once, by construction: `ArtifactRef`, `AssemblyReferenceGenome`, `BackgroundAllele`,
  `BacterialBackgroundAllele`, `BacterialCrisprInterferencePerturbation`,
  `BacterialDeletionPerturbation`, `BacterialStrainBackground`, `Compound`,
  `ConditionalAllelePerturbation`, `ConstructedOrf`, `CopyNumberVariantPerturbation`,
  `DerivedIdentifierMapping`, `EngineeredCopyNumberPerturbation`, `Environment`
  (`EnvironmentPerturbationType`), `EnvironmentResponsePhenotype`,
  `ExpressionModulationPerturbation`, `FitnessPhenotype` (`derive_se`), `Genotype`
  (`GenePerturbationType`), `HeterologousPathwayPerturbation`, `IntegratedCassette`,
  `PhagePerturbation`, `PresenceAbsencePerturbation`, `ProductTiterPhenotype`,
  `PromoterReplacementPerturbation`, `SOTerm`, `SequencePerturbation`, `StrainBackground`,
  `TransposonInsertionPerturbation`.
- Served store (`/scratch/projects/torchcell/database/kg_manifest.json`, read-only, 51
  datasets): all 51 closures move under the new rule, since `Compound`, `Environment` and
  `Genotype` sit in every closure. Under the OLD rule on the same tree all 51 already differ
  from the served manifest (`Publication` and `SourceType` in all 51, from #778), so the new rule
  adds zero datasets to the rebuild #778 already requires.
- Dev build manifests under `DATA_ROOT=/scratch/projects/torchcell-scratch`: old rule 6 fresh,
  101 stale, 20 unmanifested; new rule 0 fresh, 107 stale, 20 unmanifested. The 6 that flip are
  `crispr_magic_lian2019`, `growth_auc_rapp2026`, `growth_rate_schmidt2016`,
  `inhibitor_bioscreen_volk2021`, `metabolite_intensity_rapp2026`, `targeted_metabolome_rapp2026`.

### Left open

- Union aliases (`GenePerturbationType`, `EnvironmentPerturbationType`) are folded as TEXT into
  `Genotype`/`Environment`, so adding or removing a member class is caught; the member classes are
  still not closure edges of `Genotype`, so a contract change inside a member reaches a loader only
  if the loader imports that member (most do). Adding those edges grows every served closure, and
  `kg_manifest admit` counts an added closure symbol as drift, so it is a separate decision.
- `ProvenanceGap` is imported from `torchcell/verification/sourced.py`, outside the surface; its
  contract is not fingerprinted.
