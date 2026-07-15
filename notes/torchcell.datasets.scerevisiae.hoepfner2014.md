---
id: dm9g91o4fp9u1qnaoyo00du
title: Hoepfner2014 HIP/HOP Chemogenomic Atlas
desc: ''
updated: 1783988969952
created: 1783988969952
---

## Dataset overview

`torchcell/datasets/scerevisiae/hoepfner2014.py` — `EnvChemgenHoepfner2014Dataset`
(`env_chemgen_hoepfner2014`). The Novartis HIP-HOP chemogenomic atlas (Hoepfner et al. 2014,
Microbiol Res, doi:10.1016/j.micres.2013.11.004): ~1776 compounds profiled at IC30 against the
diploid deletion collections, stored as `env × genotype → (adjusted) MADL sensitivity_score`.
**Built ENCODABLE-ONLY** — only the 150 compounds with a released SMILES structure are kept
(the ~92% proprietary black-box `CMBxxx` are dropped at build time; see the caveat below), so
every stored record is featurisable by the cell graph transformer.

- **3,112,880 records** (L0-L4 PASS), **5,832** measured ORFs, **150** encodable compounds over
  610 sensitivity columns. (Full atlas was 29,996,238 over 1,852 compounds; recoverable by
  removing the `smiles is None` skip in the loader.)
- **HIP** (heterozygous, YSC1055): one of two copies deleted → `EngineeredCopyNumberPerturbation`
  (copy 1/2, KanMX), diploid, INCLUDES essential genes. 306 encodable experiments, 1,753,367 records.
- **HOP** (homozygous, YSC1056): both copies deleted → `KanMxDeletionPerturbation`, diploid,
  non-essential only. 304 encodable experiments, 1,359,513 records.
- Source: Dryad doi:10.5061/dryad.v5m8v (`HIP_scores.txt`, `HOP_scores.txt`, `Table_S1.xls`),
  sha256-pinned; downloader solves the Dryad Anubis proof-of-work.

## Data-quality caveats

- **HIP background mutations** — a subset of HIP strains carry undocumented secondary mutations
  (chr XI aneuploidy, WHI2 nonsense, chr V amplification) stored as clean deletions. Authoritative
  list (Table_S5) + risk audit + purge set:
  `[[torchcell.datasets.scerevisiae.hoepfner2014.background-mutations]]`. Experiment:
  `[[experiments.017-hoepfner-background-mutations.analysis]]`.
- **Compound identity / encodability — built to the 150 identifiable compounds only.** ~92% of
  the profiled compounds are PROPRIETARY Novartis `CMBxxx` black boxes whose IDENTITY is unknown,
  so there is no structure to featurise. Identity is the prerequisite; encodability follows from
  it — and here the two collapse to one number, because the compounds whose identity Novartis
  disclosed (reference + novel-MoA) were disclosed WITH their SMILES in Table S1. So
  "identity known" (151) and "has a SMILES" (150) differ by exactly one: **CMB222 "Enniatin
  derivative (Fermentation batch 1)"**, a named but structureless fermentation product, which is
  also dropped. Paper (paper.md line 110): *"In addition to 1641 proprietary compounds (named
  CMBxxx), we included 135 reference compounds with a previously reported molecular mechanism of
  action (Table S1)."* Of the 1,852 deposited compounds only **150 (8.1%) are
  identifiable + encodable**, and the loader **keeps only those** (the `smiles is None` skip in
  `_column_meta`), so every stored record is featurisable by the cell graph transformer — this is
  why the built dataset is 3,112,880 records, not 29,996,238. The full atlas is recoverable by
  removing that skip. Quantified by
  `experiments/017-hoepfner-background-mutations/scripts/hoepfner_compound_encodability.py`
  → `results/compound_encodability.json`.

## 2026.07.15 - Rebuilt encodable-only (build filter + Media schema fix); L0-L4 PASS

Rebuilt 29,996,238 → **3,112,880 records** (HIP 1,753,367 + HOP 1,359,513) by keeping only the
150 SMILES-bearing compounds. Loader changes: (1) a `smiles is None` skip in `_column_meta`
(the encodable filter); (2) both `Media(...)` calls now pass `is_synthetic=False` +
`base_medium="YPD"`, required after the component-based `Media` schema (commit `1cf60cdc`) had
left the loader incompatible — so ANY rebuild (filtered or full) failed until this fix.
Verification-runner `expected_count` updated to 3,112,880.

**L0–L4: PASS** (`preprocess/verification_report.json`): L0 3,112,880 validated; L1 count exact
and 3,112,880 unique (strain, condition) pairs; L2 value fidelity; L3 single `sensitivity_score`,
reference-zero, all env-perturbed; L4 1.000 of 5,832 measured genes are R64. Datamodels and
verification tests: 236 passed (1 pre-existing metabolite-verifier failure on `main`, unrelated).
Note: the full streaming verifier takes ~1 h for 3.1M records — it re-validates every record
through pydantic, which is expensive under the component-based `Media` schema (a verifier perf
issue, orthogonal to this dataset).
