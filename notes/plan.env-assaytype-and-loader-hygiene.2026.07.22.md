---
id: xv4kdvimj7hwt6qwqze2fmw
title: Env AssayType and Loader Hygiene
desc: 'UI-3a plan: populate AssayType, fold it into the L1 dedup key, and clean up the env-response loaders'
updated: 1784778360258
created: 1784778360258
---

## Context

The env/chemogenomic audit surfaced, as **finding `#1`**, that HOW an
environment-response readout was physically measured (pooled Bar-seq vs pinned colony
array vs halo zone vs liquid-OD growth ...) was being smuggled into the free-text
`units` string instead of being a typed axis. `units` answers *how normalized / what the
number literally is*; it is orthogonal to *what physical assay produced it*. Overloading
one string with both concerns is a provenance smell and — critically — it makes the L1
dedup key's discriminator load-bearing on a field that was never meant to carry method
identity.

**UI-1** landed the `AssayType` enum (`torchcell/datamodels/schema.py:2711`; members
`pooled_competitive_growth_barcode`, `colony_size_array`, `spot_dilution`, `halo_zone`,
`liquid_od_growth`, `biosensor_readout`, `other`) and the optional
`EnvironmentResponsePhenotype.assay_type` field, but **deferred populating it** (UI-1
Decision 7) precisely because folding it into the L1 dedup key looked risky.
**UI-2** landed the audit scaffolding. **UI-3** was then split two ways because it mixed
one heavy, risky dependency with three light, provenance-shaped ones. **THIS plan is
UI-3a only.**

The coupling insight that makes UI-3a a single coherent unit: populating `assay_type`
(workstream 1) is *only* provenance-safe once `assay_type` is folded into the L1 dedup
key as a **complement** to `units` (workstream 2). Adding a tuple field to the dedup key
is monotonic — it can only *split* record groups, never merge them — so it is
dedup-neutral even while every value is still `None`, which is exactly what retires
UI-1's deferral safely. Doing (1) without (2) would eventually let a future value move
method-identity *out* of `units`; if `units` were then dropped from the key we would
re-collide near-replicate screens. The two must ship together. Workstream 3 (loader
hygiene) rides along because the audit found the defects in the same files.

**UI-3b** (the deferred heavy half) is a referenced follow-up, planned separately:
route the 6 loaders that need `SCerevisiaeGenome`-based gene-name resolution through the
shared resolver, with per-process genome injection. It is isolated here because
`SCerevisiaeGenome.__init__` downloads a GFF and builds a gffutils sqlite — the single
heavy, slow, network/disk dependency in the whole audit — and per-process genome
injection is the risky plumbing. `assay_type` + L1 + hygiene share none of that weight,
so they land first and fast.

## Relevant Files

| Path | Stance | Why |
|------|--------|-----|
| `torchcell/datasets/scerevisiae/auesukaree2009.py` | MODIFY | assay `spot_dilution` (ref `:431`, record `:458`); undocumented (no paired note) |
| `torchcell/datasets/scerevisiae/vanacloig2022.py` | MODIFY | assay (ref `:314`, record `:350`); hygiene `#6a` fabricated `DoseBasis.IC30` at `:303`; solvent; undocumented |
| `torchcell/datasets/scerevisiae/hoepfner2014.py` | MODIFY | assay `pooled_competitive_growth_barcode` (ref `:431`, record `:549`) |
| `torchcell/datasets/scerevisiae/yeastphenome.py` | MODIFY | assay **per-screen** (from `units`, `:681`/`:709` region); only loader with a test; undocumented |
| `torchcell/datasets/scerevisiae/mota2024.py` | MODIFY | assay `spot_dilution` |
| `torchcell/datasets/scerevisiae/costanzo2021.py` | MODIFY | assay `colony_size_array`; solvent (DMSO); undocumented |
| `torchcell/datasets/scerevisiae/hillenmeyer2008.py` | MODIFY | assay; hygiene `#6d` irradiated-drop `:282`, `#6e` fragmented DMSO name `:303-314`; solvent; undocumented |
| `torchcell/datasets/scerevisiae/wildenhain2015.py` | MODIFY | assay `liquid_od_growth`; undocumented |
| `torchcell/datasets/scerevisiae/smith2016.py` | MODIFY | assay (confirm bar-seq); solvent (`_DMSO`/`_DMSO_PERCENT` consts exist); undocumented |
| `torchcell/datasets/scerevisiae/smith2006.py` | MODIFY | assay `halo_zone` (verify zone-of-inhibition) |
| `torchcell/datasets/scerevisiae/mormino2022.py` | MODIFY | assay `biosensor_readout` (confirm); hygiene `#6c` Media `"SD, pH 3.5"` at `:198` |
| `torchcell/datasets/scerevisiae/lian2019.py` | MODIFY | assay `biosensor_readout` (confirm scope + true readout) |
| `torchcell/verification/environment_response.py` | MODIFY | L1 key in BOTH copies: eager `_study_key`/`_l1_pair_uniqueness` (`:151-200`) + streaming inline `pkey` (`:463-467`) |
| `tests/torchcell/verification/test_environment_response_verification.py` | MODIFY | add synthetic L1-dedup-with-assay test (existing `expected_count` literals at `:107`,`:117`,`:193`,`:208`,`:307`) |
| `tests/torchcell/datasets/scerevisiae/test_yeastphenome.py` | MODIFY | extend to assert per-screen `assay_type` populated-or-gapped |
| `torchcell/datamodels/schema.py` | REFERENCE (read-only) | `AssayType:2711`, `Solvent:1224`, `DoseBasis:1046` — all already exist; **do NOT edit** |
| `$DATA_ROOT/torchcell-library/<key>/paper.md` | REFERENCE (read-only) | sha256-pinned mirrored Methods; source every `assay_type` quote here, never the publisher |

**Stance note — 8 undocumented loaders.** auesukaree2009, mota2024, vanacloig2022,
costanzo2021, hillenmeyer2008, wildenhain2015, smith2016, yeastphenome have no paired
dendron note. For each one we touch, record the sourced `assay_type` quote (or the gap)
in a short dated section of a dendron note so the provenance chain is discoverable, per
CLAUDE.md "Adding Datasets". (mota2024/smith2006/hoepfner2014/mormino2022/lian2019 have
notes; append there.)

## Key Design Decisions

1. **SPLIT UI-3 into 3a (this) and 3b (referenced).** 3a = `assay_type` population + L1
   complement + loader hygiene; 3b = gene-resolver routing of 6 loaders.
   *Rationale:* the gene resolver is the single isolable heavy dependency —
   `SCerevisiaeGenome.__init__` downloads a GFF and builds a gffutils sqlite, and 3b adds
   per-process genome injection. The 3a workstreams are light, share no heavy dep, and are
   provenance-shaped; splitting keeps this PR fast, low-risk, and reviewable while the
   risky plumbing lands on its own.

2. **L1 key: ADD `assay_type` ALONGSIDE `units`, NEVER replace it.** The dedup key gains
   a tuple field; it does not swap one.
   *Rationale:* adding a field to a uniqueness tuple is **monotonic** — it can only make
   two previously-equal keys unequal (split a group), never merge two groups. So the
   change is dedup-neutral even while every `assay_type` is `None` (all-`None` tuples
   compare exactly as before), which is what lets us retire UI-1 Decision 7's deferral
   with zero count movement. **Replacing** `units` would re-collide YeastPhenome's ~12,742
   near-replicate screens whose microarray-vs-barseq method currently lives *in* `units` —
   a silent data-loss regression. Both the eager and streaming copies must change
   identically; a one-sided edit is a latent dedup divergence between the two code paths.

3. **Assay sourcing home = inline, per loader.** A module-level `SourcedValue`-style
   record (verbatim Methods quote + sha256 + section/locator) sits next to the
   `assay_type` constant, matching how these loaders already record sourced values.
   *Rationale:* NO separate committed table. `assay_type` is a single per-dataset value
   (per-screen for yeastphenome only), so a table is unwarranted; and a new committed data
   artifact would trip the package-data *ship* requirement (the
   `compound_identity_table.json` / PR `#160` gotcha). Inline keeps the value where the
   loader reads it and where the reviewer checks it.

4. **SOURCE-OR-GAP, never guess.** Every `assay_type` is a verbatim Methods quote from
   OUR mirror (`$DATA_ROOT/torchcell-library/<key>/paper.md`, sha256-pinned). Where the
   Methods are genuinely silent after a thorough pass, emit
   `ProvenanceGap(field="assay_type", reason=deferred_pending_source_review)` and set
   `assay_type=None` (the `ProvenanceGapMixin` invariant: a gapped field must be `None`).
   *Rationale:* this is CLAUDE.md's "Adding Datasets — Sourcing Values from the SI"
   discipline, including "do not conclude not-in-SI on a first pass" (comb full Methods,
   check figure/table legends, search synonyms). A guessed assay silently corrupts the
   method axis and the dedup key.

5. **First-pass assay assignments are HYPOTHESES to confirm from Methods, not values to
   write blind.** Even the low-risk ones get a quote. *Rationale:* a plausible default is
   still a guess until the sentence is in hand; the map below marks which are near-certain
   vs which MUST be actively source-or-gapped.

6. **Hygiene defects fixed, each sourced from the paper's Methods** (details in Approach):
   (a) vanacloig fabricated `DoseBasis.IC30` -> `DoseBasis.fixed` + the paper's actual
   doses as value+unit, plus `solvent=Solvent(name="DMSO", ...)` where documented;
   (b) populate `SmallMoleculePerturbation.solvent` (DMSO) where documented
   (smith2016, hillenmeyer2008, yeastphenome, costanzo2021, vanacloig2022);
   (c) mormino Media `"SD, pH 3.5"` -> synthetic complete (SC) + `base_medium`, pH as a
   structured `EnvironmentPhysicalPerturbation(factor=ph)` or documented;
   (d) hillenmeyer irradiated branch drops the photosensitizer drug — keep BOTH the
   radiation factor AND the compound (or gap-mark the loss);
   (e) hillenmeyer DMSO %/volume baked into `Compound.name` — move % to `Solvent.percent`,
   clean the name. All target existing schema (`Solvent:1224`, `DoseBasis:1046`) — **read
   only, no schema edit**.

7. **NO `schema.py` / `pydant.py` edits.** `assay_type`, `Solvent`, `DoseBasis` all
   already exist. *Rationale:* nothing this PR needs is a new field, so the schema-impact
   pre-commit gate does NOT fire and no `TORCHCELL_SCHEMA_ACK` is required (see the
   schema-impact worktree-crash gotcha). We confirm this in Verification via
   `git diff --name-only`.

8. **Out of scope:** gene-resolver routing (UI-3b); the
   `Compound.provenance_gaps` vs `Media.open_gaps` reconciliation (deferred a 3rd time —
   not blocking assay/L1/hygiene); and the full DB rebuild, which is the KG-build user's
   job and already in flight (job `990`, shipped to Radiant `2026-07-22`). This PR changes
   loader *output*, so dev-tree builds go stale — rebuild is a KG follow-up, NOT this PR.

## Approach

Execution order per loader, then cross-cutting, then tests, then verify.

**Step 0 — source each `assay_type` from the mirror.** For every loader, open
`$DATA_ROOT/torchcell-library/<key>/paper.md` and locate the Methods sentence describing
the physical assay. Comb thoroughly (Methods + figure/table legends + SI); search
synonyms (`bar-seq`, `barcode`, `microarray`, `colony`, `spot`, `serial dilution`,
`optical density`, `OD600`, `halo`, `zone of inhibition`, `biosensor`, `fluorescence`).
Capture the verbatim quote + sha256 + locator into the inline `SourcedValue`-style record.
Where silent, prepare a `ProvenanceGap`.

**Assay hypothesis map** (CONFIRM every one from Methods — do not write blind):

- `costanzo2021` = `colony_size_array` — near-certain (condition-SGA); still quote.
- `auesukaree2009` = `spot_dilution` — likely; **CONFIRM**.
- `mota2024` = `spot_dilution` — likely; **CONFIRM**.
- `wildenhain2015` = `liquid_od_growth` — likely; **CONFIRM**.
- `hoepfner2014` = `pooled_competitive_growth_barcode` — likely; **CONFIRM** detection is
  microarray-barcode vs bar-seq matches the vocab term.
- `hillenmeyer2008` = `pooled_competitive_growth_barcode` — likely; **CONFIRM** detection
  (microarray-barcode) matches the term.
- `yeastphenome` = **per-screen** (derive from the `units`/readout string already parsed
  at `:681`/`:709`); each screen maps to its own `assay_type` or a per-screen gap.
- `smith2006` = `halo_zone` — **MUST SOURCE-OR-GAP** (verify it is a zone-of-inhibition /
  disk-diffusion assay, not a growth curve).
- `mormino2022` = `biosensor_readout` — **MUST SOURCE-OR-GAP** (confirm a biosensor/reporter
  readout).
- `lian2019` = `biosensor_readout` — **MUST SOURCE-OR-GAP** (confirm scope + the true
  readout; MAGIC modality caveats).
- `vanacloig2022` = `pooled_competitive_growth_barcode` — **MUST SOURCE-OR-GAP** (confirm
  bar-seq vs microarray).
- `smith2016` = `pooled_competitive_growth_barcode` — **MUST SOURCE-OR-GAP** (confirm bar-seq
  vs microarray).

**Step 1 — set `assay_type` at BOTH phenotype sites per loader.** Every loader builds an
`EnvironmentResponsePhenotype` twice: the `phenotype_reference` (the control/baseline) and
the per-record `phenotype`. Set `assay_type` at both (e.g. auesukaree `:431` ref + `:458`
record; vanacloig `:314` ref + `:350` record; hoepfner `:431` ref + `:549` record;
yeastphenome `:681`/`:709`). Add the inline `SourcedValue`-style record next to the
constant. If sourcing failed, pass `assay_type=None` and attach the `ProvenanceGap`.

**Step 2 — hygiene fixes** (each with its own sourced quote):

- `vanacloig2022:303` — the compound concentration currently fabricates
  `Concentration(basis=DoseBasis.IC30)` (the loader docstring even admits per-compound IC30
  molar values live in Table S1 and are unavailable to it). Replace with the paper's actual
  administered doses as value+unit under `DoseBasis.fixed` (docstring/Methods examples:
  Benomyl 10 ug/mL, MMS 0.01%, ...); where a per-compound molar value is genuinely absent,
  keep `DoseBasis.fixed` with the documented value+unit and gap the molar. Populate
  `solvent=Solvent(name="DMSO", percent=...)` where the SI documents the vehicle.
- Solvent population (DMSO) where documented: `smith2016` (reuse existing `_DMSO` /
  `_DMSO_PERCENT` consts), `hillenmeyer2008`, `yeastphenome`, `costanzo2021`,
  `vanacloig2022` — set `SmallMoleculePerturbation.solvent`.
- `mormino2022:198` — `Media(name="SD, pH 3.5", ...)` conflates medium and pH. Rename to
  synthetic complete (`SC`) and set `base_medium`; carry pH as a structured
  `EnvironmentPhysicalPerturbation(factor=ph)` (or document it), sourced from Methods.
- `hillenmeyer2008:282` — the `"irradiated"` branch maps to
  `EnvironmentPhysicalPerturbation(factor=radiation)` but DROPS the photosensitizer
  compound (angelicin/psoralen). Keep BOTH: the radiation factor AND a
  `SmallMoleculePerturbation` for the compound. If the compound identity is ambiguous per
  record, gap-mark the loss rather than silently dropping it.
- `hillenmeyer2008:303-314` — DMSO %/volume is baked into `Compound.name` fragments,
  fragmenting one molecule into name variants. Move the % to `Solvent.percent` and leave
  `Compound.name` clean.

**Step 3 — L1-key complement, in BOTH copies of `environment_response.py`.**

- Eager path: extend `_study_key` (`:151-164`) so the returned discriminator tuple
  additionally carries `record["experiment"]["phenotype"].get("assay_type")` (coerced to
  `str(... or "")`), and it flows through `_l1_pair_uniqueness` (`:167-200`). Keep `units`
  in the tuple — assay is added, not substituted.
- Streaming path: extend the inline `pkey` (`:463-467`) identically. The two tuples must be
  byte-for-byte the same shape/order.
- Because the added element is `""` for every currently-`None` `assay_type`, all existing
  keys are unchanged and no `expected_count` moves.

**Step 4 — tests** (below). **Step 5 — verify** (Verification section).

## Gotchas

- **The L1 key is DUPLICATED** — eager `_study_key`/`_l1_pair_uniqueness` (`:151-200`) and
  the streaming inline `pkey` (`:463-467`) are two independent implementations of the same
  key. Change BOTH, identically. A one-sided edit is a silent dedup divergence that only
  surfaces when the two paths disagree on a real build.
- **Complement, not replace — and monotonic.** Adding `assay_type` to the key can only
  split groups. It is safe with all values `None`. Never drop `units` from the key: method
  identity for YeastPhenome's ~12,742 near-replicate screens currently lives there, and
  dropping it re-collides them (data loss).
- **Source-or-gap, never guess.** Half the assay hypotheses are marked MUST-SOURCE-OR-GAP
  (smith2006, mormino2022, lian2019, vanacloig2022, smith2016) plus the
  hoepfner/hillenmeyer detection-term confirmation. A plausible default is still a guess
  until the Methods sentence is quoted; when silent, `assay_type=None` + `ProvenanceGap`.
- **Stale LMDB.** This changes loader OUTPUT, so dev-tree builds
  (`$DATA_ROOT/data/torchcell/<dataset>/`) go stale; the base class skips `process()` when
  `processed/` exists, so a stale build is silently reused. The full rebuild is the
  KG-build follow-up (job `990`), NOT this PR. Tests here use synthetic records or
  DATA_ROOT-skip-gated smokes, so they do not depend on a fresh build.
- **`expected_count` literals are hardcoded** at
  `test_environment_response_verification.py:107/:117/:193/:208/:307`. The synthetic
  L1-dedup test must be written so adding `assay_type` (all-distinct or all-`None` across
  its records) does NOT move any existing literal — the whole point of the monotonicity
  argument, asserted concretely.
- **hillenmeyer subtleties:** the irradiated branch's dropped photosensitizer and the
  DMSO-in-name fragmentation are easy to "fix" into new silent losses — keep the compound
  (or gap it) and clean the name into `Solvent.percent` without inventing doses.

## Verification

```bash
cd /home/michaelvolk/Documents/projects/torchcell.worktrees/plan/env-assaytype-and-loader-hygiene
PY=~/miniconda3/envs/torchcell/bin/python

# 1. env-response L1 verification tests (synthetic; no network, no build)
$PY -m pytest tests/torchcell/verification/test_environment_response_verification.py -xvs

# 2. yeastphenome loader test (per-screen assay populated-or-gapped)
$PY -m pytest tests/torchcell/datasets/scerevisiae/test_yeastphenome.py -xvs

# 3. types on changed loaders + verifier (whole-tree-ish on touched files)
$PY -m mypy torchcell/verification/environment_response.py \
  torchcell/datasets/scerevisiae/{auesukaree2009,vanacloig2022,hoepfner2014,yeastphenome,mota2024,costanzo2021,hillenmeyer2008,wildenhain2015,smith2016,smith2006,mormino2022,lian2019}.py

# 4. lint
$PY -m ruff check torchcell/verification/environment_response.py \
  torchcell/datasets/scerevisiae/{auesukaree2009,vanacloig2022,hoepfner2014,yeastphenome,mota2024,costanzo2021,hillenmeyer2008,wildenhain2015,smith2016,smith2006,mormino2022,lian2019}.py

# 5. CONFIRM no schema gate fires (must NOT list datamodels/schema.py or pydant.py)
git diff --name-only | grep -E 'datamodels/(schema|pydant)\.py' && echo "GATE WOULD FIRE — STOP" || echo "no schema edit — gate silent"
```

**Tests to author.** (a) In `test_environment_response_verification.py`, a synthetic L1
dedup test proving (i) the existing `expected_count` literals hold after `assay_type` joins
the key (all-`None` records unchanged), and (ii) two records identical except for
`assay_type` are correctly SEPARATED (not flagged as a duplicate pair) — asserted against
both the eager and streaming code paths. (b) Extend `test_yeastphenome.py` to assert each
screen's `assay_type` is populated (derived from `units`) OR carries a `ProvenanceGap`.
(c) A lightweight per-representative-loader unit assertion that `assay_type` is set or a
`ProvenanceGap` exists — construct a single phenotype without a full build where possible;
any loader build-smoke is skip-gated on `DATA_ROOT` so CI without data does not fail.
**Confirm no test hits the network** (mirror reads are local sha256-pinned files; genome
download belongs to UI-3b, not here).

**Sourcing check.** Every `assay_type` quote must trace to
`$DATA_ROOT/torchcell-library/<key>/paper.md` (verbatim + sha256 + locator); spot-check
the MUST-SOURCE-OR-GAP set (smith2006, mormino2022, lian2019, vanacloig2022, smith2016)
plus the hoepfner/hillenmeyer detection-term match before landing.

## Follow-up — UI-3b (referenced, not planned here)

UI-3b routes the 6 gene-name-normalizing loaders through the shared
`SCerevisiaeGenome.resolve_gene_name` reconciler with **per-process genome injection** —
isolated because that genome build is the audit's one heavy network/disk dependency (split
rationale in Context + Decision 1). Shares no code path with assay/L1/hygiene; planned in
its own note.
