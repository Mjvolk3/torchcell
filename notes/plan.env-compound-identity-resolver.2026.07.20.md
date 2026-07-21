---
id: cil5lqg6vg4xng9kerr24we
title: '20'
desc: ''
updated: 1784599796990
created: 1784599796990
---

## Context

The env/chemogenomic audit's **#1 systemic finding**: 11 of 12 environmental
datasets store compounds **name-only** -- a `Compound(name="furfural")` with
`inchikey`/`chebi_id`/`pubchem_cid`/`smiles` all `None`. A bare name is not a
resolvable identity (two papers spelling "H2O2" vs "hydrogen peroxide" become
distinct compounds; nothing joins to ChEBI/PubChem), which violates the
`Compound` docstring's own contract ("Identity is carried by stable, resolvable
identifiers -- not the human name alone", `torchcell/datamodels/schema.py:1125`).
The audit also flagged a DEFECT: YeastPhenome's four plant-defensin conditions
(PMID 31451498: DmAMP1/NaD1/NbD6/SBI6) are peptides mis-typed as
`SmallMoleculePerturbation`, whose `Compound`/InChIKey key cannot represent a
sequence-identified biologic.

The fix was **anticipated in the design**: `[[torchcell.datamodels.media-components]]`
(2026.07.14) states "a sourced resolver pass (ChEBI/PubChem API, never guessed)
fills them; `open_gaps` lists what is unfilled" (`notes/...media-components.md:108`).
This plan builds that resolver.

This is **UI-2 of a 3-unit audit pipeline**:

- **UI-1 LANDED** at `main ebf19aeb` -- additive schema foundation: `Compound`
  gained `ProvenanceGapMixin` (typed `provenance_gaps`, invariant "gapped field
  must be None"); `BiologicPerturbation` + `BiologicAgentClass` (peptide/protein/
  antibody/toxin) exist; `AssayType` enum exists.
- **UI-2 (this plan)** -- give every env-dataset `Compound` a **resolved
  structure where one exists, and a TYPED `ProvenanceGap` where it genuinely does
  not**, via a PINNED (sha256, no live API at build time) name->structure table +
  ONE shared resolver; retype the YeastPhenome defensins to `BiologicPerturbation`.
- **UI-3 (FOLLOW-UP, do NOT plan here)** -- `assay_type` population, loader
  hygiene, an L1 dedup key, and the full DB rebuild that picks all of this up.

## Relevant Files

| Path | Action | Purpose / Stance |
|---|---|---|
| `torchcell/datamodels/compound_identity.py` | **NEW** | The shared pure resolver: `resolve_compound_identity(...) -> CompoundIdentityResolution` + status/result pydantic models; reads the committed table, sha256 self-check, NEVER hits network. |
| `torchcell/datamodels/compound_identity_table.json` | **NEW** | The PINNED in-repo name->structure table (nested pydantic records serialized to JSON); anchored by its own sha256. |
| `scripts/build_compound_identity_table.py` | **NEW** | One-time builder: PubChem PUG REST name->InChIKey+CID; deterministic + cached; records retrieval command; run ONCE to emit the committed table. |
| `torchcell/literature/manifest.py` | **MODIFY** | Add `RetrievalMethod.pubchem_api` member (`:59` enum). NOT `schema.py` -> no gate. |
| `torchcell/datasets/scerevisiae/auesukaree2009.py` | **MODIFY** | Route `Compound` construction through resolver (fill-or-gap). ~5 static compounds -> pre-populate table. |
| `torchcell/datasets/scerevisiae/mota2024.py` | **MODIFY** | Fill-or-gap; 3 public acids -> resolved. Build-smoke candidate. |
| `torchcell/datasets/scerevisiae/vanacloig2022.py` | **MODIFY** | Fill-or-gap; data-driven GEO-token names -> deferred gaps for unseen names. |
| `torchcell/datasets/scerevisiae/costanzo2021.py` | **MODIFY** | Fill-or-gap; ~15 static compounds -> pre-populate. |
| `torchcell/datasets/scerevisiae/hillenmeyer2008.py` | **MODIFY** | Fill-or-gap; data-driven (~hundreds) -> deferred gaps, no blocking pass. |
| `torchcell/datasets/scerevisiae/wildenhain2015.py` | **MODIFY** | Fill-or-gap **by pubchem_cid** (CID->InChIKey path). |
| `torchcell/datasets/scerevisiae/hoepfner2014.py` | **MODIFY** | Fill-or-gap; ~1641 CMB proprietary codes -> `known_proprietary=True` terminal gaps. MUST NOT clobber the SMILES it already sets for its named subset (`:455`). |
| `torchcell/datasets/scerevisiae/smith2006.py` | **MODIFY** | Fill-or-gap; 3 static compounds -> pre-populate. |
| `torchcell/datasets/scerevisiae/lian2019.py` | **MODIFY** | Fill-or-gap; 1 static (furfural) -> pre-populate. |
| `torchcell/datasets/scerevisiae/mormino2022.py` | **MODIFY** | Fill-or-gap; 1 static -> pre-populate. |
| `torchcell/datasets/scerevisiae/smith2016.py` | **MODIFY** | Fill-or-gap; vendor drug-column IDs -> `known_proprietary=True` where opaque, else deferred. |
| `torchcell/datasets/scerevisiae/yeastphenome.py` | **MODIFY** | Fill-or-gap AND the peptide retype: defensin branch BEFORE the `SmallMolecule` path in `_parse_condition` (`:436`). LOADER edit, not schema. |
| `torchcell/datasets/scerevisiae/nadal_ribelles2025.py` | **MODIFY (light)** | NaCl -> pre-populate + route (it is in the static core even if not one of the 12 enumerated). |
| `torchcell/verification/sourced.py` | **REFERENCE** | `ProvenanceGap(field, reason, looked_in?, ...)` at `:147`; `ProvenanceGapReason` at `:138`. Non-generic, pickle-safe -- keep it so. |
| `torchcell/datamodels/schema.py` | **REFERENCE (do NOT edit)** | `Compound` `:1125`, validators `:1166/:1176/:1184`, `ProvenanceGapMixin` `:1081`, `BiologicPerturbation`/`BiologicAgentClass` `:1451/:1470`. |
| `notes/torchcell.datamodels.media-components.md` | **REFERENCE (stable)** | Explicitly anticipates this resolver (`:108`); do not edit. |
| `notes/plan.env-schema-assay-compound-biologic.2026.07.19.md` | **REFERENCE** | UI-1 plan (the landed foundation this builds on). |

## Key Design Decisions

**1. ONE shared resolver, not 12 per-loader tables.** Twelve ad-hoc lookup dicts
would drift and re-implement normalization 12 ways. Instead mirror the proven
`SCerevisiaeGenome.resolve_gene_name` shape (memory `genome-gene-name-resolver`):
ONE reconciler, pure and per-name, returns a **typed status enum**, and **callers
own retention** (the resolver never mutates or drops -- it reports). New module
`torchcell/datamodels/compound_identity.py` exposes a pure
`resolve_compound_identity(name=None, pubchem_cid=None, known_proprietary=False)
-> CompoundIdentityResolution`. Rationale: single source of truth, one
normalization policy, uniform fill-or-gap semantics across all loaders.

**2. PINNED table = committed in-repo JSON artifact.** The resolver reads a
committed file and NEVER touches the network at build/CI/test time -- so builds
are deterministic and offline-reproducible (provenance principle: "the stored
artifact + its sha256 is canonical, NOT the URL"). **JSON, not CSV**: each entry
is a nested pydantic record (`name`, `inchikey`, `pubchem_cid`, `chebi_id`,
`smiles`, `source_url`, `retrieval_method`, `retrieved_at`, `resolution_status`)
and JSON round-trips pydantic natively; CSV would flatten the nesting and lose
types. Home: `torchcell/datamodels/compound_identity_table.json`, next to the
resolver. **Anchored by the committed file's sha256, self-checked at load** (the
resolver computes the sha256 of the bytes it read and compares to a pinned
constant; mismatch raises, matching UI-1's honesty discipline). Rebuildable via
the recorded builder command (Decision 3).

**3. One-time builder script produces the table; it is the only thing that hits
the network.** `scripts/build_compound_identity_table.py` queries **PubChem PUG
REST** name->InChIKey+CID via
`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/<name>/property/InChIKey/TXT`
(**verified HTTP 200 on 2026-07-20**). Deterministic + caching; on lookup failure
it emits a `resolution_status` of unresolved rather than guessing (never
fabricate a structure). It records the exact retrieval command into each record
so the table is rebuildable-from-scratch. Rate-limit **<=5 req/s and <400/min**
(PubChem policy). Uses stdlib `urllib.request` (dominant in-repo retrieval
convention) or `requests` 2.32.3. Run ONCE by a human; the committed JSON is the
canonical artifact thereafter.

**4. `RetrievalMethod.pubchem_api` added to `manifest.py`.** Add one enum member
at `torchcell/literature/manifest.py:59` (precedent: `zenodo` was added the same
way). Do NOT add `chebi_api` -- v1 is **PubChem-first** and does not query ChEBI,
so `chebi_id` stays `None` and becomes its OWN future gap (honest, not
half-guessed). `manifest.py` is NOT `schema.py`, so no schema-impact gate.

**5. Resolver contract satisfies existing `Compound` validators.**
`resolve_compound_identity(...)` returns
`CompoundIdentityResolution(status, inchikey?, chebi_id?, pubchem_cid?, smiles?)`.
Every returned identifier already conforms to the live validators
(`INCHIKEY_PATTERN ^[A-Z]{14}-[A-Z]{10}-[A-Z]$` `:1166`, `CHEBI_ID_PATTERN
^CHEBI:\d+$` `:1176`, `pubchem_cid` positive int `:1184`) -- the table is built to
store only conforming values, so a caller can splat resolved fields straight into
`Compound(...)` without a second validation layer. The resolver also resolves
**by `pubchem_cid`** (the wildenhain path: CID -> InChIKey). **Additive-only on
already-populated structure fields**: it MUST NOT clobber a caller-supplied
`smiles` (hoepfner already sets SMILES for its named subset) -- callers merge
resolver output only into fields that are still `None`. **Name normalization**:
lowercase + strip + a small, documented, conservative canonical-synonym map
(e.g. "H2O2" -> "hydrogen peroxide"); **never fuzzy-guess** a structure from a
near-miss name.

**6. Gap-reason policy (fill-or-gap, uniform across loaders).**

- **Resolved** -> fill the structure fields; NO gap.
- **Unresolved PUBLIC name** -> `ProvenanceGap(field="inchikey",
  reason=deferred_pending_source_review)` -- the ONE recoverable reason (a
  growable worklist: the name is real, the table comb just is not done).
- **Known-proprietary** (hoepfner ~1641 CMB codes, smith2016 opaque vendor IDs)
  -> `ProvenanceGap(field="inchikey", reason=not_reported_by_primary)` (terminal;
  the primary never released a resolvable structure). Loaders pass
  `known_proprietary=True` on those branches.
- A gap is asserted **only when the structure field is truly `None`**
  (`ProvenanceGapMixin` invariant `:1104`: a gapped field MUST be `None`; you
  cannot both store a value and declare it missing). **Never guess a structure.**

**7. Wiring scope + table coverage: static-first, growable, non-blocking.** Wire
**all 12 loaders** to route `Compound` construction through the resolver (uniform
fill-or-gap). **Pre-populate** the table from the statically-enumerable loaders
(auesukaree ~5, costanzo2021 ~15, mota2024 3, smith2006 3, lian2019 1,
mormino2022 1, nadal NaCl) + wildenhain via CID + a hand-verified **common core**
(furfural `CHEBI:30976` + its InChIKey, acetic acid, ethanol, NaCl, H2O2,
benomyl, MMS, tamoxifen). **Data-driven loaders** (hillenmeyer ~hundreds,
vanacloig GEO tokens, yeastphenome header names, smith2016 drug column) emit
**deferred gaps** for names not yet in the table -- **NO loader blocks** on an
exhaustive resolution pass. The gap census is the worklist that grows the table
over time.

**8. YeastPhenome peptide retype (the audit DEFECT).** The four plant-defensin
conditions (PMID 31451498: DmAMP1, NaD1, NbD6, SBI6) must become
`BiologicPerturbation(agent_class=BiologicAgentClass.peptide, ...)` instead of
`SmallMoleculePerturbation` -- their identity is a sequence, not an InChIKey, so a
`Compound` cannot represent them (`BiologicPerturbation` docstring `:1470`).
Insert a defensin-name branch in `yeastphenome.py::_parse_condition` (`:436`)
**BEFORE** the `SmallMoleculePerturbation` construction path. This is a **LOADER
edit** (`yeastphenome.py`), NOT a `schema.py` edit.

**9. Strict structure-or-gap `Compound` validator DEFERRED to UI-2b.** A validator
forcing every `Compound` to carry a structure-or-gap would break ~65 name-only
sites (`media.py` 23 + ~11 test sites + every loader) -- out of scope here.
**Consequence: UI-2 touches NO `schema.py`/`pydant.py` file**, so the
schema-impact gate does NOT fire: no `TORCHCELL_SCHEMA_ACK`, no worktree-crash
incantation (memory `schema-impact-worktree-crash`). Confirmed in Gotcha 1.

## Approach

Execution order (why-before-what at each step):

**(a) Build the resolver module + models first** (`compound_identity.py`), because
every loader edit depends on its contract. Define, all plain pydantic
`BaseModel` (NON-generic, pickle-safe -- see Gotcha 6):

- `CompoundResolutionStatus(StrEnum)`: `RESOLVED`, `UNRESOLVED_PUBLIC`,
  `PROPRIETARY`.
- `CompoundIdentityRecord(BaseModel)`: one table row (`name`, `inchikey`,
  `pubchem_cid`, `chebi_id`, `smiles`, `source_url`, `retrieval_method`,
  `retrieved_at`, `resolution_status`).
- `CompoundIdentityResolution(BaseModel)`: `status`, `inchikey?`, `chebi_id?`,
  `pubchem_cid?`, `smiles?` -- what a caller splats into `Compound`.
- `resolve_compound_identity(name=None, pubchem_cid=None,
  known_proprietary=False) -> CompoundIdentityResolution`: pure; normalizes the
  name (lowercase/strip/synonym map), looks up the loaded table by normalized
  name or by CID, returns RESOLVED with structure or an UNRESOLVED_PUBLIC /
  PROPRIETARY status. Table is loaded once at import and sha256-self-checked.

Contract shape (the one disambiguating snippet):

```python
res = resolve_compound_identity(name="furfural")
# res.status == RESOLVED; res.inchikey/res.chebi_id populated
compound = Compound(
    name="furfural",
    inchikey=res.inchikey,      # only merged where the field is still None
    chebi_id=res.chebi_id,
    provenance_gaps=gaps_for(res),  # [] when RESOLVED
)
```

`gaps_for(res)` (a tiny loader-side helper, or inlined) maps status -> gap list
per Decision 6's policy.

**(b) Add `RetrievalMethod.pubchem_api`** to `manifest.py:59` (one line), so the
builder + table records can name their retrieval method.

**(c) Write the builder script and run it ONCE** to emit
`compound_identity_table.json`. Seed its input names from the static loader
enumerations + the hand-verified common core (Decision 7). It queries PubChem PUG
REST with rate-limiting + caching, writes conforming records, and records the
retrieval command per record. Hand-verify the common-core entries (furfural
`CHEBI:30976`, etc.) against the response. Commit the JSON; pin its sha256 into
the resolver.

**(d) Wire the 12 loaders fill-or-gap**, uniformly: at each `Compound(...)`
construction site, call the resolver, merge resolved structure fields **only
where currently `None`** (never clobber -- Gotcha 5), attach gaps per status.
Static loaders resolve at build; data-driven loaders emit deferred gaps for
unseen names and do NOT block. wildenhain routes by `pubchem_cid`. hoepfner +
smith2016 pass `known_proprietary=True` on their opaque-code branches.

**(e) YeastPhenome peptide retype**: in `_parse_condition` (`:436`), add a branch
matching the four defensin names -> `BiologicPerturbation(agent_class=peptide,
name=..., concentration=...)` returned BEFORE the `SmallMoleculePerturbation`
path. Keep the existing `_DOSE_RE`/`_UNIT_MAP` flow for everything else.

**(f) Tests, then verify** (next two sections).

## Gotchas

1. **No `schema.py` touched -> no schema-impact gate.** UI-2 edits only the new
   resolver/table/builder, `manifest.py`, and loaders. Since neither
   `datamodels/schema.py` nor `datamodels/pydant.py` changes, the schema-impact
   pre-commit gate does not fire -- do NOT set `TORCHCELL_SCHEMA_ACK` or use the
   `PYTHONPATH=$WT ... git commit` worktree-crash incantation (memory
   `schema-impact-worktree-crash`). If you find yourself editing `schema.py`,
   stop -- that is UI-2b, out of scope.
2. **Network lives ONLY in the one-time builder, never at build/CI/test time.**
   The resolver reads the committed JSON and self-checks sha256; it must have no
   import-time or call-time HTTP. Any test that would hit PubChem is forbidden --
   tests read the committed table. (Provenance principle: sha256 artifact
   canonical, not the URL.)
3. **Stale-LMDB tmp_path trap.** The dataset base class **skips `process()` when
   `processed/` already exists**, so a loader test pointed at an existing build
   silently reuses the OLD records and the resolver never runs. Every loader
   build-smoke MUST build into a **clean `tmp_path`** (copy the yeastphenome test
   pattern: `tmp_path_factory` + `shutil.copy` of the raw mirror into a fresh
   dir). Skip-gate on `DATA_ROOT` mirror presence.
4. **Validator-deferred blast radius.** Because no strict structure-or-gap
   validator is added (Decision 9), existing name-only `Compound(name=...)` sites
   (39 in `torchcell/`, 23 in `media.py`) stay valid -- do NOT "helpfully" add
   the validator or you break ~65 sites and drag in `schema.py` (re-triggering
   Gotcha 1). The Compound-gaps vs `Media.open_gaps` reconciliation (UI-1
   Decision 6) is likewise DEFERRED to UI-2b/UI-3 -- do not reconcile here.
5. **Do NOT clobber hoepfner's SMILES.** `hoepfner2014.py` already sets `smiles`
   for its named subset (`:455`). The resolver merge is **additive-only**: fill a
   structure field ONLY when it is still `None`. A resolved value must never
   overwrite a loader-provided one (and never turn a stored value into a gap --
   that violates the mixin invariant).
6. **Keep `ProvenanceGap` and the resolution models NON-generic.** `ProvenanceGap`
   at `sourced.py:147` is a plain `BaseModel` (pickle-safe by name); the pickle
   hangs fixed in PR `#119` were caused by generic pydantic (`SourcedValue[Any]`).
   `CompoundIdentityResolution` / `CompoundIdentityRecord` MUST be plain
   `BaseModel`, no `Generic[...]`, so loaders remain multiprocessing-safe.
7. **PubChem rate-limit.** The builder must throttle to <=5 req/s and <400/min or
   PubChem returns 503 and blocks. Cache responses so a re-run does not re-hammer.

## Verification

Run from the worktree root with the torchcell env interpreter
(`~/miniconda3/envs/torchcell/bin/python`):

```bash
# (a) fast resolver unit tests: RESOLVED / UNRESOLVED_PUBLIC / PROPRIETARY /
#     by-CID / no-clobber-smiles / name-normalization (all read committed table)
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/datamodels/test_compound_identity.py -xvs
# (b) pinned-table determinism + sha256 self-check
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/datamodels/test_compound_identity_table.py -xvs
# (c) yeastphenome peptide retype (defensin -> BiologicPerturbation, peptide)
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/datasets/scerevisiae/test_yeastphenome.py -xvs
# (d) loader build-smokes into clean tmp_path (skip-gated on DATA_ROOT mirror):
#     mota2024 public acids -> resolved; hoepfner CMB -> proprietary gap
~/miniconda3/envs/torchcell/bin/python -m pytest \
  tests/torchcell/datasets/scerevisiae/ -k "compound_resolver" -xvs

# mypy on new module + changed loaders + manifest (whole-tree locally; CI is diff-scoped)
~/miniconda3/envs/torchcell/bin/python -m mypy torchcell/datamodels/compound_identity.py \
  torchcell/literature/manifest.py torchcell/datasets/scerevisiae/
~/miniconda3/envs/torchcell/bin/python -m ruff check torchcell/ scripts/build_compound_identity_table.py

# confirm NO schema.py/pydant.py touched (must print nothing) -> no schema-impact gate
git diff --name-only main | grep -E 'datamodels/(schema|pydant)\.py' || echo "clean: no schema edit"
# confirm no test performs a live network lookup (grep the new tests)
grep -Rn "pubchem.ncbi\|urllib\|requests\." tests/torchcell/datamodels/ && echo "REVIEW" || echo "no network in tests"
```

The four test files above (a-d) are the tests to author -- all read the committed
table, never a live lookup; the loader build-smokes (d) are skip-gated on
`DATA_ROOT` mirror presence.
