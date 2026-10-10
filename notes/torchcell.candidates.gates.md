---
id: gykx0hsb1axij3lk48l24j0
title: Gates
desc: ''
updated: 1791617770060
created: 1791617770060
---

## 2026.10.10 - Deterministic gate evaluators

Every evaluator reads bytes already held (row fields, dataset module source, the genomes tier, the mirror manifests, a dev LMDB read-only, the committed compound table). Nothing fetches and nothing builds.

| gate | function | decides from | outcomes |
|---|---|---|---|
| G1 | `evaluate_g1(row, finding)` | a recorded `SourceKindFinding`, else `status`, `klass`, `accession_confirmed` | finding decides; a row marked aggregation without a finding, or with an unconfirmed accession, is `unmeasured` (the agentic borderline case); otherwise `primary` passes |
| G2 | `evaluate_g2(row, data_root, source_kind=...)` | host tokens (`HOSTS`) in the row text, then `registry.resolve` on every manifest member | `resolvable_readable` pass; `resolvable_not_ingestible` blocked:ingest (W3110); `absent_provisionable` blocked:provision; `absent_unnamed` fail; an aggregation naming no host is `unmeasured` |
| G3 | `inventory(key, data_root)` + `evaluate_g3` | `si/` and `data/` entries of the library and raw manifests, kept apart (#495) | pass when intact; fail on sha256 drift or a bad zip; blocked:wall for a missing `manual_browser` file; blocked:deposit for a missing scripted file; `unmeasured` with no manifest |
| G4-key | `evaluate_g4_key` | DOI, accessions (`ACCESSION`, from `check_candidate_overlap.py`) and PMID against every module under `torchcell/datasets` (parsed with `ast`, never imported) and the SynLethDB source PMIDs | fail when another module's primary DOI or a shared accession or an aggregator PMID matches; a mention in another module is recorded, not failed |
| G4-value | `read_store_values` + `match_by_value` | a long released table against a dev store's records on matched samples | `G4ValueRecord.subsumed` when every released column has a sample within tolerance; best and runner-up kept, so the nearest-wrong margin is recorded |
| G5 | `evaluate_g5` | `Phenotype` subclasses of `schema.py`; `resolve_compound_identity` offline (#726) | pass; `gap` with an issue; blocked:unfiled_gap without one |

`compose_verdict` applies the stop rule: after the first fail or blocked, later gates become `unmeasured` and drop their records.

Readable assembly sets (`READABLE_ASSEMBLY_SETS`) are the schema's `BacterialAssemblySet` vocabulary plus `SGD_S288C_R64` and `PETER2018_1011`; W3110 is resolvable and outside it, which is exactly the "in no schema vocabulary yet" state `registry.py` records.

Choices the plan did not fix, made here:

- **Host extraction is a token scan.** `HOSTS` lists the strain names rows use. A row naming only the K-12 lineage resolves to MG1655 and BW25113 (the table's own note: the exact background is a loader-time provenance item). A named public strain with no deposited set (BL21, Nissle, CEN.PK, W303) is `absent_provisionable`, so it blocks rather than refuses; `absent_unnamed` is reserved for rows naming no host.
- **An aggregation with no host is G2 `unmeasured`,** because its hosts are per source record. Refusing CeCaFDB at G2 would contradict decision 15.
- **G4 passes when G4-key is clean and no dev store was compared,** with the reason saying so. A value comparison is required only where a compendium could hide the row (the Thompson case), and it then fails the gate when subsumed.
- **A dirty dev store is refused** (`DirtyStoreError`), per gotcha 6, before any record is read; the dataset class is never instantiated, so no `process()` or download can run.
- `zlib.error` and a missing end-of-central-directory record both read as a bad archive; the first is raised by `testzip()` before the CRC check on a corrupt deflate stream (found by the bad-zip test).
