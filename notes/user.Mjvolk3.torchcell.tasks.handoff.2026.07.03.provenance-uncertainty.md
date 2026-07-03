---
id: 9jgtpkip5rjlhhi7fbp89tq
title: provenance-uncertainty
desc: ''
updated: 1783103387623
created: 1783103387623
---

# Session handoff (2026.07.03) — provenance + uncertainty ontology

Self-contained context for a fresh session to review/merge PR #21 and continue.

## Git state
- `main` @ `5a571254`+ (pushed). This session added on main: CLAUDE.md provenance
  + pydantic-first principles; the SGA scratch note; exact-formula provenance in
  the costanzo noise-computation note.
- **PR #21 (DRAFT)** — branch `feat/provenance-and-uncertainty`, 2 commits:
  - `e7758760` provenance layer, `b8fbe3cf` fitness uncertainty ontology.
  - All tests green, mypy --strict + ruff clean. **This is what to review/merge.**
- Issue #20: Radiant VM artifact-retrieval endpoint (medium/enhancement).
- Untracked `SOM.pdf` at repo root — unidentified; decide (probably move to library).

## What PR #21 contains
1. **Provenance layer** (`torchcell/literature/`):
   - `retrieve.py`: versioned retriever fns (`springer_esm`, `direct_url`,
     `pmc_oa_api`) + `RETRIEVERS` registry. Retrieval = source code by dotted path.
   - `provenance.py`: `RetrievalMethod` (StrEnum; NO `manual_browser`;
     `radiant_endpoint` reserved), `SourceCheck`, `RetrievalRecord`,
     `ProcessingRecord`, `ArtifactRecord(FileRecord)`, `run_retriever` /
     `verify_artifact` / `check_source`. sha256 canonical, not URL; drift detected.
   - `tests/torchcell/literature/test_provenance.py` (6 tests).
   - REVIEW follow-up: type `Manifest.files` as `list[ArtifactRecord]` (move records
     into manifest.py to avoid circular import) so retrieval/processing round-trip.
2. **Uncertainty ontology** (`torchcell/datamodels/schema.py`):
   - `UncertaintyType` (StrEnum, NO `unknown` = strict), `SampleUnit`, `derive_se`.
   - `FitnessPhenotype`: + `fitness_uncertainty` + `fitness_uncertainty_type` +
     `sample_unit`; `fitness_se` auto-derived in `mode=before` (ModelStrict is
     FROZEN, so cannot assign in mode=after); strict reported<->type + n/unit
     invariants. `fitness_std` kept DEPRECATED (additive, keeps loaders green).
   - `tests/torchcell/datamodels/test_uncertainty_ontology.py` (part of 25 passing).

## Key finding (why this matters)
Costanzo/Baryshnikova interaction p-value (exact, from Supp Software 1 Matlab):
`p = sqrt( Phi(-|eps/sigma_local|) * Phi(-|log(actual/expected)/sqrt(sig_i^2+sig_j^2)|) )`
— geometric mean of a colony-local test and a pooled screen-level test. The pooled
`background_std` is UNPUBLISHED, so exact p is NOT reproducible from released data
(our best fit with local sd tops at corr 0.95). Fully sourced + sha256-pinned in
`[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]`. Ontology
mapping: SMF -> bootstrap SEM / our file's SD is `sample_sd` over screens (17/350);
DMF -> `sample_sd` over colonies (N=4-8). The old `fitness_se = std/sqrt(68|1400)`
in `costanzo2016.py` is the bug (wrong n; SMF SD is not an SE).

## Design decisions (locked)
- Provenance: stored artifact + sha256 is canonical; URL is historical metadata.
  Upstream drift/rot is DETECTED (sha256 mismatch / command fail) -> fall back to
  mirror/Zotero, never silently followed; new upstream version = new record.
- Retrieval references SOURCE CODE (retriever fn + params), not a command string.
- Zotero = PDFs ONLY (software/data live in the mirror). Raw dataset files need the
  same provenance record (a raw-data mirror) — NOT yet built.
- Ontology: Full field set; `n_samples` + `sample_unit` (NOT `n_replicates`); no
  `unknown`; scope = Fitness + MicroarrayExpression; p-value reproduction is NOT a
  gate (needs raw pipeline). See `[[torchcell.datamodels.uncertainty-statistics-ontology]]`.

## After merge — propagate (review together first)
1. Migrate fitness loaders to the ontology: costanzo (SMF->sample_sd/screen/17|350,
   DMF->sample_sd/colony/4 — fixes 68/1400), kuzmin2018/2020 (needs sourced n).
2. Apply the ontology to `MicroarrayExpressionPhenotype` (dict variant); rename its
   `n_replicates` dict -> `n_samples` for one concept schema-wide.
3. Propagate: adapters/cell_adapter.py, biocypher schema yaml, Cypher queries,
   `data/mean_experiment_deduplicate.py` (SE aggregation on dedup). Drop `fitness_std`.
4. Build the raw-data provenance mirror (same RetrievalRecord for SGA_*.txt etc.).
5. Then resume the schematization-ingestion roadmap
   `[[plan.schematization-ingestion-roadmap.2026.06.23]]` (WS2 kuzmin n_samples,
   WS5-WS9 datasets, needs MinerU capture of the roadmap papers via Zotero).

## Reference notes (all on main)
- `[[torchcell.datamodels.uncertainty-statistics-ontology]]` — the ontology decision.
- `[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]` — exact formulas + provenance.
- `notes/scratch.2026.07.03.131106-sga-normalization-sd-se-pvalue.md` — LaTeX math explainer.
- MinerU library: `$DATA_ROOT/torchcell-library/{kuzminSystematicAnalysisComplex2018,
  baryshnikovaQuantitativeAnalysisFitness2010}/`.
