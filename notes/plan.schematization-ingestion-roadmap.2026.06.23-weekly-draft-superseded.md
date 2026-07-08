---
id: i60zkox4gcttynvwmkb67hn
title: 23 Weekly Draft Superseded
desc: ''
updated: 1782266657376
created: 1782266657376
---

## 2026.06.23

- [ ] Roadmap: verified schematized ingestion -> rebuilt Neo4j KG (Sameith, Kemmeren, Ohya, beta-carotene, betaxanthin) -> deploy to Radiant [[plan.schematization-ingestion-roadmap.2026.06.23]]

## 2026.07.02 - Overnight autonomous session (Phase A: WS1, WS3, WS4)

Branch `plan/schematization-ingestion-roadmap` (worktree). Three Phase-A
workstreams landed, each tested + committed. No push, no merge to main.

### Done

- [x] **WS1 - schema hardening + freeze** (commit `5b1698c2`)
  - Consolidated the six identical per-subclass `validate_label_fields`
    `(cls, values)` validators into ONE instance-method validator on the
    `Phenotype` base (`type(self).model_fields`, MRO-aware). Also clears the
    pydantic 2.12 "classmethod after-validator" deprecation.
  - Fixed a **latent deserialization crash**: `MicroarrayExpressionPhenotype`
    `n_samples` validator called `v.items()` on non-dict input, raising
    `AttributeError` (which pydantic union-matching does NOT catch), so a
    `FitnessPhenotype` int `n_samples` crashed `validate_json` against the
    non-discriminated `ExperimentType` union. Now a clean `ValueError`. Would
    have bitten every real Fitness/Kuzmin record load once the freeze made int
    `n_samples` canonical.
  - Refreshed the stale `test_schema.py` (pre-refactor `label`/`label_statistic`
    API + missing required `dataset_name` -> was fully failing).
  - New `tests/torchcell/datamodels/test_schema_invariants.py`: Liskov,
    registry<->union parity, label-field invariant, round-trip, n_samples freeze.

- [x] **WS4 - new phenotypes** (commit `80857b37`)
  - `MetabolitePhenotype` (YeastGEM-keyed abundance), `ProteinAbundancePhenotype`,
    `VisualScorePhenotype` (ordinal; stored `score_min/score_max/score_description`
    so scale semantics are Neo4j-queryable per decision 7). Each with its
    Experiment/Reference wrapper + registry entries; shared dict-validation
    helpers reuse the WS1 non-dict guard. Union round-trip verified for all three.
  - YeastGEM metabolite-ID compat (decision 4) deferred to the verification
    layer, NOT a constructor validator (would load SBML on every instantiation).
    It reduces to `l1_completeness(pheno.metabolite_abundance.keys(),
    yeastgem_ids)` - the WS3 framework already covers it, no new code needed.

- [x] **WS3 - L0-L4 verification framework** (commit `cc45919a`)
  - New `torchcell/verification/` package (self-contained; `literature/manifest.py`
    does not exist in this branch yet, so sha256 is plain `hashlib`).
    `report.py`: `Level` (IntEnum gate), `Provenance`, `LevelResult`,
    `VerificationReport`, `sha256_file`. `levels.py`: `l0_structural`,
    `l1_completeness`/`l1_count`, `l2_value_fidelity`/`l2_cross_method`,
    `l3_convention`, `l4_cross_source`. 14 tests, `mypy --strict` clean.

Full datamodels + verification suite: **105 passing, 3 skipped**. Imports of
costanzo/kuzmin/sameith/kemmeren + conversion.py verified unbroken.

### Blocked / deferred

- [ ] **WS2 - Kuzmin `n_samples`/`fitness_se`** - PARTIALLY unblocked (2026.07.02).
  Costanzo derives `fitness_se = fitness_std / sqrt(n_samples)` from
  provenance-sourced constants (`N_SAMPLES_QUERY_SMF_TOTAL = 68  # 17 screens x
  4 colonies`, each traced to the SI). Per
  [[user.Mjvolk3.torchcell.tasks.weekly.2026.03.fitness-interaction-n_samples]]
  each Kuzmin `n_samples` needs a **quoted SI justification**.
  - **MinerU library found:** `$DATA_ROOT/torchcell-library/<citation_key>/`
    holds `paper.md`, `si/si1.md`, `manifest.json` (citation_key, doi, zotero
    key, per-file sha256). Only **kuzminSystematicAnalysisComplex2018** is
    processed (paper + SI). Kuzmin2020, Costanzo, Sameith, Kemmeren, Ohya,
    Ozaydin, Cachera, Zelezniak are all NOT in the library yet.
  - Kuzmin2018 SI `si1.md:29` already states the replicate design: "Every double
    mutant query strain was screened alongside its two single mutant control
    strains, **in two independent replicates**, for a total of 1,092 screens."
    -> n_samples sourcing is feasible for 2018 (colonies-per-screen TBD from SI).
  - Costanzo's existing constants cite `SI-costanzoGlobalGeneticInteraction2016.mmd`
    (Mathpix) which is NO LONGER present (0 .mmd under scratch; Costanzo not in
    torchcell-library) -> its provenance is currently un-auditable. Argues for
    pinning sha256 + keeping the source md in the library.
  - **User will drive number extraction** (stuffing PDFs into context to get the
    counts right). Do NOT autonomously bake n_samples. Kuzmin2020 still needs
    MinerU processing first.
- [ ] **WS5-WS9** - need the extracted data artifacts (Sameith/Kemmeren LMDB
  present but verification wants a build; Ozaydin Excel SI, Cachera PDF table,
  Zelezniak metabolite/protein tables). WS3 framework is ready to verify them.
- [ ] **WS10-WS14** - pan-transcriptome source TBD; Phase B needs the db env.

### Provenance / MinerU status (2026.07.02, later)

- **MinerU library exists:** `$DATA_ROOT/torchcell-library/<citation_key>/` with
  `paper.md`, `si/si1.md`, `manifest.json` (per-file `role` + `sha256`). Built
  2026-06-12. Only **kuzmin2018** (paper + SI) processed so far.
- **The literature subsystem already exists** on the unmerged
  `feat/literature-zotero-ocr` worktree/branch: `torchcell.literature` with
  `manifest.Manifest`/`FileRecord` (provenance objects), `capture_by_doi`,
  `ocr.py`/`_run_mineru.py`, born-digital `extract.py`, `zotero.ZoteroLibrary`.
  My WS3 `Provenance` partially duplicates `Manifest` -> reconcile (WS3 should
  import `literature.manifest` once that branch is in scope; sha256 = one source
  of truth from `FileRecord`).
- **Built the missing piece - `SourcedValue`** (commit `0ade7df0`,
  `torchcell/verification/sourced.py`): binds one extracted number to
  `(citation_key, path-in-library, sha256, verbatim quote)`; `.value` reads with
  zero deps; `audit_sourced_value` re-checks hash + quote on demand (skips when
  library unmounted). **No line numbers** (OCR reflow rots them; sha256 is the
  anchor) - per user insight. 21 verification tests, mypy --strict clean.
- Batch MinerU is ready to run (entrypoint `capture_by_doi`; `swanki-mineru`
  env + 4x RTX 6000 Ada present) but needs: run from the literature branch, each
  roadmap paper in the Zotero `database` collection w/ DOI. Papers to capture:
  kuzmin2020, costanzo2016, sameith2015, kemmeren2014, ohya2005, ozaydin2013,
  cachera2023, zelezniak2018 (+ SIs). **Awaiting user decision** on branch scope
  - run mode (asked; unanswered).

## 2026.07.02 - Merge-readiness assessment (CRITICAL: plan schema is stale)

User asked to consolidate everything to `main` + clean up worktrees before
fanning out per-dataset MinerU worktrees. Assessed all worktrees. **Do NOT
"merge everything" as-is** - nothing merges cleanly, and the plan branch's
schema work would REGRESS main's refactor.

### Root cause

`main` advanced **~90-102 commits** (the big mypy-strict/ci-quality refactor)
AFTER every worktree forked. That refactor rewrote `schema.py` (422 lines) in a
direction the roadmap note did NOT anticipate:

- **`main` uses `n_replicates` (dict fields), NOT `n_samples`.** The roadmap
  "decision 2" (n_samples canonical) is CONTRADICTED by main. FitnessPhenotype
  keeps scalar `n_samples: int`; MicroarrayExpressionPhenotype uses
  `n_replicates: dict`. **The roadmap decision-2 needs to be re-made against
  main's reality before any more schema work.**
- `main` kept the per-subclass `validate_label_fields` (annotated + type:ignore),
  did NOT DRY them -> my WS1 consolidation is a valid idea but must be redone on
  main, not merged.
- My WS1 union-crash fix is MOOT on main (the n_replicates rename already avoids
  the field collision).
- My WS4 phenotypes use `n_samples: Dict`; invariant test asserts
  `n_replicates not in model_fields` -> would FAIL on main.

### Merge-readiness table (merge-tree simulation onto main)

| Branch | own files | behind main | conflicts | verdict |
|---|---|---|---|---|
| feat/literature-zotero-ocr | 21 | 102 | 4 *config* only (pyproject, pre-commit, CI style, req_style) | closest; code clean |
| fitness-interaction-n_samples_2 | 10 | 102 | schema.py, data.py, kuzmin2018.py | incomplete (kuzmin pending OCR); aligns w/ main n_replicates |
| plan/schematization-ingestion-roadmap | 10 | 90 | schema.py, test_schema.py | schema STALE+wrong convention; do not merge |
| paper/figures-fig1, write/supported-datasets | - | - | - | separate (paper/docs) |

### What IS salvageable from the plan branch

`torchcell/verification/` (WS3 report/levels/sourced + tests) = **all new files,
ZERO conflict with main**. Clean to cherry-pick onto a fresh branch off main.
The `schema.py`/`test_schema.py` half (WS1/WS4) should be dropped/redone.

### Recommended sequence (NOT executed; awaiting user - questions timed out)

1. Land `feat/literature-zotero-ocr` (rebase onto main, resolve 4 ruff/tooling
   conflicts, test, merge) - enables MinerU fan-out.
2. Salvage `verification/` onto a fresh branch off main (cherry-pick 6 files).
3. Re-decide n_samples vs n_replicates vs main; then redo WS4 phenotypes on main.
4. Finish `fitness-interaction-n_samples_2` after kuzmin OCR; merge.
5. THEN clean up worktrees - only after work is merged or explicitly abandoned.
   Do NOT delete worktrees holding unmerged work without confirmation.

Nothing merged, nothing deleted this session. Lesson: diff `main` before building
on a worktree that forked long ago.
