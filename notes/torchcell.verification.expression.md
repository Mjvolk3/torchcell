---
id: m2ugc3xrf156bgvxzexysvc
title: Expression
desc: ''
updated: 1783149461693
created: 1783149461693
---

## 2026.07.04 - WS5 rebuild + L0-L4 verification (Sameith2015 + Kemmeren2014)

WS5 of [[plan.schematization-ingestion-roadmap.2026.06.23]]. Verified the microarray
expression records against the frozen schema after rebuilding the cached LMDBs so
they carry the per-gene `expression_log2_ratio_se` + `n_replicates` that the abstract's
uncertainty depends on.

### Rebuild results (against current schema)

| Dataset | Class | Records | gene_set | Notes |
|---------|-------|---------|----------|-------|
| `dm_microarray_sameith2015` | `DmMicroarraySameith2015Dataset` | 72 | 82 | GSTF double mutants; SE/var/n_replicates populated |
| `sm_microarray_sameith2015` | `SmMicroarraySameith2015Dataset` | 82 | 82 | **was never built before**; n_replicates=4 (2 bio × 2 dye-swap) |
| `microarray_kemmeren2014`   | `MicroarrayKemmeren2014Dataset`  | 1450 | 1450 | abstract r=0.543 dataset; gene_set.json re-persisted |

### CRITICAL rebuild gotcha — genome injection is mandatory for Sameith

Sameith GEO sample titles use **common** gene names (e.g. `dot6`), not systematic
names. Rebuilding `DmMicroarraySameith2015Dataset(root=...)` with the default
`genome=None` silently collapses **72 → 2 records**: without a `SCerevisiaeGenome`,
`_extract_gene_names_from_title` cannot resolve `dot6 → YER088C`, so nearly every
sample fails the "≥2 systematic gene names" test and is dropped. A clean exit is NOT
evidence of a correct build — the record count is the oracle. The genome must be
injected exactly as the module's own `main()` does:

```python
genome = SCerevisiaeGenome(
    genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
    go_root=osp.join(DATA_ROOT, "data/go"),
    overwrite=False,
)
ds = DmMicroarraySameith2015Dataset(root=..., genome=genome)
```

Kemmeren tolerates `genome=None` because its titles carry systematic names directly
(regex-resolvable). Same base class, opposite genome-sensitivity. See also the
roadmap "silent dataset skip" gotcha. A rebuild must also clear stale derived
artifacts (`processed/lmdb` + `preprocess/{experiment_reference_index,gene_set}.json`)
so `post_process` recomputes them against the current schema instead of validating
old-schema data.

### L0-L4 verification — all three PASS

`torchcell/verification/expression.py::verify_expression_dataset` +
`scripts/verify_expression_datasets.py` (writes `verification_report.json` next to
each `experiment_reference_index.json`). Levels:

- **L0 structural** — every record validates against `ExperimentType`.
- **L1 count** — 72 / 82 / 1450 exactly; **gene_completeness** — every record measures
  the full 6169-gene platform universe (nothing silently dropped).
- **L2** — all log2 finite; SE non-negative (NaN allowed where n=1); n_replicates ≥ 1.
- **L3 reference_log2_zero** — reference phenotype log2(sample/ref) == 0 for every value.
- **L3 deletion_downregulates** — the orientation oracle: a deleted gene must be
  down-regulated, so the **median** deleted-gene log2 must be < 0.
- **L4 cross_source** — all three datasets share the identical 6169-gene universe.

The orientation oracle uses the **median** (not a fixed pass-fraction) because TF
deletions (Sameith) are legitimately noisier than a clean deletion library
(Kemmeren) — both correctly oriented, but different per-gene negative fractions:

| Dataset | median deleted-gene log2 | frac_neg | interpretation |
|---------|--------------------------|----------|----------------|
| Kemmeren | −2.474 | 0.969 | clean deletion library, strong signal |
| SM Sameith | −0.710 | 0.840 | GSTF single deletions |
| DM Sameith | −0.321 | 0.720 | GSTF double deletions, noisiest |

A sign inversion (storing `log2(reference/sample)`) would flip these medians positive
and `frac_neg` to ~0.15–0.30 — caught by the L3 check and by a dedicated unit test.

### Files

- `torchcell/verification/expression.py` — reusable verifier (uses the WS3 L0-L4
  building blocks in `torchcell/verification/levels.py`).
- `scripts/verify_expression_datasets.py` — runs it over the three datasets, writes
  the report artifacts, exits non-zero on any FAIL.
- `tests/torchcell/verification/test_expression_verification.py` — synthetic pass +
  each failure mode (sign inversion, non-zero reference, wrong count, dropped gene).
