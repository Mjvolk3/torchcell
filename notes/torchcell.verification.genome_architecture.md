---
id: 8n8q57jznt0818sb3hx0yfe
title: Genome_architecture
desc: ''
updated: 1791732604053
created: 1791732604053
---

## 2026.10.11 - The genome-architecture L0-L4 gate

`torchcell/verification/genome_architecture.py`, registered as `GENOME_ARCHITECTURE_DATASETS` in `runners.py` with `verify_genome_architecture_dataset` / `run_genome_architecture`.

Why a new family verifier: the two existing multi-feature verifiers (`morphology`, `bacterial_morphology`) are imaging vocabularies, and neither can express a copy number per haploid genome, a chromosome copy number, or the aggregate identity that makes this release self-checking.

What it adds over what the schema already enforces at L0 (keys belong to the named assay, values are finite and non-negative, a count is integral, and a stored ploidy-deviation aggregate equals the chromosome deviations it sums over):

- **L1 `assay_coverage`** -- every record carries the features the caller names as required and nothing outside the assay; the schema accepts any non-empty subset, so a build that dropped a released column would pass it.
- **L1 `sample_uniqueness`** -- one record per sequenced sample id, keyed on the source's own id. Two colonies of one strain must both survive; two records of one colony mean the build duplicated a row.
- **L2 `feature_fidelity`** -- per feature, a census plus the bound its declared basis implies, so a defect names the column rather than an index into a pooled list.
- **L2 `release_fidelity`** -- every stored number is re-read from the released workbook and compared cell by cell against the record carrying that sample id, at tolerance 0. This is what makes the store a faithful copy of the release rather than of the loader: 247,128 comparisons on the Puddu build.
- **L3 `vocabulary_parity`** -- the assay's symbols are distinct and every basis is one this gate bounds.
- **L3 `provenance_audit`** -- every paper-OCR quote is re-found in the sha256-pinned mirror bytes, and every workbook legend cell is still a cell of the sheet it names. A legend cell cannot be audited by `audit_sourced_value`'s substring search (an xlsx is a zip), so the caller passes the sheet's cell strings and the rule compares cell to cell.
- **L3 `reference_profile`** -- the reference states a mean, a named uncertainty kind, its sample count, and a per-key typed gap for every feature it does not cover.
- **L4 `paper_statistics`** -- the store reproduces the paper's own summary numbers: the rho0 colony count (mtDNA below one copy) and the fraction of colonies deviating from 2n. The paper counts every released colony while the store serves the knockouts, so the caller states the reference colonies' contribution (`wild_type_rho0`) and it is added back rather than hidden in a tolerance.

The family-agnostic rules (gap census, canonical gene names, uncertainty sanity, compound identity, media membership, the L4 gene universe) come from `SharedRecordRules`, so they read identically here and in the fitness verifier. Tests: `tests/torchcell/verification/test_genome_architecture.py` (11 hermetic cases, each driving one rule to a failure and asserting the count it reports).
