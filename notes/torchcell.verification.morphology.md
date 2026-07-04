---
id: 0256m5b3j2gu03vs8yp4kdp
title: Morphology
desc: ''
updated: 1783204750410
created: 1783204750410
---

## 2026.07.04 - WS6 L0-L4 verification (Ohya2005 CalMorph)

WS6 of [[plan.schematization-ingestion-roadmap.2026.06.23]]. The abstract's morphology
phenotype (r=0.619, single-KO). Unlike the expression datasets, Ohya did **not** need a
rebuild — its 4718 records already **L0-validate against the current schema**. This note
records the design of the morphology verifier + what the L0-L4 run found. Companion to
[[torchcell.verification.expression]].

### Where the data lives (important)

The built Ohya LMDB is under `$DATA_ROOT/database/data/torchcell/scmd_ohya2005`
(the KG-build tree), NOT `data/torchcell/...`. That dir is owned by the KG-build user
and is **read-only** to the dev user, so the verifier READS the LMDB there and WRITES
the `verification_report.json` to the writable `data/torchcell/scmd_ohya2005/preprocess/`
tree (source vs report paths are separate constants in the harness).

### Design principle — verify the gaps the schema leaves, not what it guarantees

`CalMorphPhenotype`'s `field_validator` already enforces, at instantiation, that every
`calmorph` key ∈ the 281 `CALMORPH_LABELS`, every CV key ∈ the 220 `CALMORPH_STATISTICS`,
and that no value is NaN. So **L0 already subsumes label-validity + NaN**. The verifier
(`torchcell/verification/morphology.py`) therefore adds only what the schema does NOT
encode:

- **L1 `calmorph_completeness`** — the schema accepts any *subset*; a correct build
  measures the FULL 281 base + 220 CV vocabulary in every record. (Confirmed: all 4718
  records carry exactly 281+220; 0 dropped.)
- **L1 `reference_populated`** — the 122 WT profiles are **aggregated into each record's
  reference phenotype** (a single mean-WT baseline with the full vocabulary), not stored
  as 122 separate records. So the count oracle is 4718 mutants + a populated WT
  reference; the "122" is not record-recoverable.
- **L2 `value_fidelity`** — finiteness (schema blocks NaN but NOT `inf`). Base values ∈
  [0, 56974], all finite.
- **L2 `cv_nonnegative`** — a coefficient of variation is non-negative by definition; a
  negative signals a computation bug the schema wouldn't catch. CV ∈ [0.026, 3.207], 0
  negatives.
- **L3 `vocabulary_parity`** — 281 + 220 == 501 == `CALMORPH_PARAMETERS`, base/CV
  disjoint (the roadmap's "501/501" parity, asserted once, not per-record).
- **L4 `gene_containment_*`** — Ohya's 4718 deletion genes should CONTAIN the deletion
  genes of the expression datasets (same yeast deletion library, consistent naming).
  This is a *directional containment*, not equality (Ohya ⊋ each expression set), so it
  is asserted directly rather than via `l4_cross_source` (which tests equality).
  Empirically: Kemmeren 0.970, SM/DM Sameith 0.988 of genes contained (threshold 0.90 —
  headroom for genes profiled for expression but absent from the morphology screen;
  a naming/format break would collapse the overlap and fail).

### Result — all L0-L4 PASS

`scripts/verify_morphology_datasets.py` → all levels green; report written next to the
(writable) `data/torchcell/scmd_ohya2005/preprocess/`. Synthetic unit tests in
`tests/torchcell/verification/test_morphology_verification.py` cover the passing case +
each failure mode (dropped parameter, negative CV, non-finite value, wrong count,
under-populated reference).

### Standing pattern (per user, 2026.07.04)

Verifiers live in `torchcell/verification/` (src, permanent gate — not throwaway). The
"why the data is shaped this way" reasoning lives in dendron module notes: the schema
note explains the object; the verifier is the machine-checkable statement of that
intent; the two reference each other.
