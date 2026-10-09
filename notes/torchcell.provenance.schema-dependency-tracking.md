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

## 2026.10.08 - Module-Level Vocabularies as Closure Nodes (#734)

### Problem

The closure held classes only. A field annotated with a module-level `Literal` alias
(`BacterialGeneNamespace`), a validator reading a pattern map (`BACTERIAL_LOCUS_TAG_PATTERNS`),
or a validator calling a module-level helper that reads one, moved no fingerprint when that
module-level name changed. Measured on this branch: narrowing `BacterialGeneNamespace` by
`ecoli_b_rel606_locus_tag` moved 0 fingerprints under the old rule.

### Rule

`schema_deps.collect_module_bindings` collects every top-level non-class binding of the surface
modules: `x = ...`, `x: T = ...` (the annotation dropped), `type X = ...`, and functions (docstring
stripped). Each binding is a closure node with its own fingerprint (`binding_fingerprint`,
SHA-256 of `module::<normalized source>`), exactly as an enum class is:

- a class has an edge to every binding its body names (field annotation, default, validator body);
- a binding has an edge to every binding its source names, so
  `TransposonInsertionPerturbation -> _validate_bacterial_locus_tag -> BACTERIAL_LOCUS_TAG_PATTERN -> BACTERIAL_LOCUS_TAG_PATTERNS`
  is in the closure of every loader that imports the leaf;
- a binding has NO edge to a class it names. A union alias (`GenePerturbationType`) is a node whose
  membership is fingerprinted; its member classes stay in a loader's closure only through the
  loader's own imports, which keeps the closure lever tight.
- a loader importing a binding directly (`from torchcell.datamodels.schema import BACTERIAL_LOCUS_TAG_PATTERNS`)
  depends on it.

`SchemaSurface.bindings` holds them, `SchemaSurface.fingerprints` covers classes and bindings, and
`ContractSpec` and every class fingerprint are unchanged.

`schema_impact.diff_surfaces` diffs bindings after classes, and `classify_binding_change` classifies
by member set (`binding_members`: a `Literal`, a union, a tuple/set display, a dict display's
`key: value` items). Lost members are BREAKING, only-gained members stale, a reorder stale, and a
change to a binding with no member set (a helper function, a computed string) BREAKING. The
`schema-impact` pre-commit hook goes through the same `load_surface_from_sources`, so it sees the
same nodes. Verified on the live schema in the working tree, each edit reverted after: dropping
`ecoli_b_rel606_locus_tag` from `BacterialGeneNamespace` makes `scripts/run-schema-impact.sh` exit 1
with one changed symbol (`BacterialGeneNamespace`, lost members) and 29 impacted bacterial loaders;
changing the MG1655 pattern to `^b\d{5}$` exits 1 via `BACTERIAL_LOCUS_TAG_PATTERNS` with the same 29.
The REL606-shaped widening is stale (pinned in `test_schema_impact.py`).

### Why not fold the vocabulary into the class fingerprint

That was the first implementation on this branch. It moved 28 of 169 class fingerprints, including
`Genotype`, `Environment` and `Compound`, which sit in every closure. `scripts/kg_compat_page.py`
recomputes each package tag's surface with the CURRENT rule and compares it with closures recorded
under the old rule, so every historical pairing broke: `--check` failed with
`broken pair: 2026.09.21-ab6d8c5d records v1.2.1 as its package but the verdict is 'incompatible (all 51 datasets drift)'`,
while the base commit passes. A fingerprint-rule change on existing symbols is a retroactive change
to every recorded closure; new symbols are not, because every stored-closure check
(`build_manifest.check_manifest`, `releases.closure_compatibility`, the supported-query check)
iterates the STORED symbols.

### Measured movement (2026.10.08, old rule vs this rule on the same tree)

Measured on #778's branch tip and again on `origin/main` at `24a5f30e4` after #778 landed; the
numbers are the same on both.

- 169 surface classes, 41 bindings; class fingerprints moved: 0.
- Served store (`/scratch/projects/torchcell/database/kg_manifest.json`, read-only, 51 datasets):
  0 of the symbols already in a served closure change fingerprint, so the stored-symbol drift is
  identical under both rules (51 of 51 drift, from #778's `Publication`/`SourceType`). All 51
  closures GAIN binding symbols (7 to 19 each): `CHEBI_ID_PATTERN`, `INCHIKEY_PATTERN`,
  `EnvironmentPerturbationType`, `GenePerturbationType`, `SgaPerturbationType` in 51;
  `SO_ID_PATTERN`, `_validate_so_id` in 50; `_Z95`, `derive_se` in 28;
  `CATEGORICAL_MEASUREMENT_TYPES` in 13; `ALLELE_EDIT_SO`, `CASSETTE_INTEGRATION_SO`,
  `SGD_SYSTEMATIC_GENE_PATTERN`, `_require_value_or_gap` in 6; the five `ArtifactRef` bindings in 5;
  `ExperimentType` in 1.
- `kg_manifest admit` counts a symbol in the current closure but not the stored one as drift, so
  every served dataset reads as drifted until the next full rebuild records closures with the
  bindings. #778 already requires that full rebuild for all 51, so this adds no rebuild.
- `scripts/kg_compat_page.py --check`: current. Supported-query check: all ok.
- Dev build manifests under `DATA_ROOT=/scratch/projects/torchcell-scratch`: identical under both
  rules (6 fresh, 104 stale, 20 unmanifested at the time of the run), since the check reads stored
  symbols only.

### Left open

- Union-alias member classes are not closure edges (see Rule). A contract change inside a member
  reaches a loader only if the loader imports that member, as before.
- `ProvenanceGap` and `SourcedValue` are imported from `torchcell/verification/sourced.py`, outside
  the surface; their contracts are not fingerprinted.
- Bindings imported INTO `schema.py` from another module (e.g. `CALMORPH_LABELS`) are not surface
  bindings; no surface class names `CALMORPH_LABELS` today.

## 2026.10.09 - The dev-store gate is checked end to end on a vocabulary change, and 7 store manifests predate it (#734)

The closure change landed on `main`; what was not pinned was the thing issue #734 actually
measured, which is a BUILT STORE's manifest reading `is_stale=False` after a module-level
`Literal` moved. `tests/torchcell/provenance/test_build_manifest.py` now drives that
through `check_manifest` on a synthetic surface shaped like the live chain
(`Perturbation -> validate_locus_tag -> LOCUS_TAG_PATTERNS`, plus a `GeneNamespace`
`Literal` on the field):

| edit | before | now |
|---|---|---|
| namespace removed from the `Literal` | fresh | STALE, drift names `GeneNamespace` |
| namespace added to the `Literal` | fresh | STALE, drift names `GeneNamespace` |
| a namespace's regex changed | fresh | STALE, drift names `LOCUS_TAG_PATTERNS` |
| a vocabulary the closure does not reach | fresh | fresh |

The tests also pin WHERE the drift is reported: on the binding, never on the class. The
recomputed fingerprint of `Perturbation` is equal to the stored one in every case above,
which is what keeps the historical compatibility pairings readable while still staling the
store that carries the field.

On the live surface, the Rousset 2018 loader's closure is 72 symbols, 30 of them bindings,
and it holds `BacterialGeneNamespace` and `BACTERIAL_LOCUS_TAG_PATTERNS` by name, so the
two edits that were invisible when the issue was filed now move a fingerprint the store
recorded.

### What is still not covered, measured

`check_manifest` iterates the STORED closure, so a manifest written before the bindings
became closure nodes has no binding entry to compare and a vocabulary change stays
invisible to it. Counted over `$DATA_ROOT/data/torchcell/*/preprocess/build_manifest.json`
on GilaHyper (`/scratch/projects/torchcell-scratch`, 2026-10-09): **116 store manifests,
109 record binding symbols, 7 do not.** The seven are
`dmf_costanzo2016_1e5`, `dmf_costanzo2016_5e5`, `dmi_costanzo2016_1e5`,
`dmi_costanzo2016_5e5` (the four size-limited development subsets),
`env_chemgen_auesukaree2009` and `env_chemgen_vanacloig2022` (counted twice, two
directories reporting the same `dataset_name`).

This is not proposed as a new staleness condition here: flagging "the manifest records
fewer symbols than the loader now reaches" would be honest but would change the gate's
verdict for the whole fleet, which is the owner's call, and the rebuild that re-records
these seven answers it either way. The gate was left as it is and the seven are named so
the decision is made on a list rather than on a guess.
