---
id: q4ftqkzrjs0pwsevtj95yg1
title: yeastphenome-ingestion-harness
desc: ''
updated: 1784103023537
created: 1784103023537
---

## 2026.07.15

- [ ] Plan a per-PMID YeastPhenome ingestion adapter (proof study on one non-already-built chemogenomic deletion screen, reusing the WS15 EnvironmentResponse family; aggregate NPV/2-hop-provenance model left as an open user decision) [[plan.yeastphenome-ingestion-harness.2026.07.15]]

## 2026.07.16

- [x] `ProvenanceGap` affordance (name settled; was "capture gap") — complement of `SourcedValue`, honest typed absence with reason + `looked_in` [[torchcell.verification.sourced]]
- [x] Encoding probe on real curated YeastPhenome data — ontology CAN encode it; 2 sharp underspecifications found (temperature required-but-uncarried; ratio-vs-signed reference convention)
- [x] Decision A: temperature (and other uncarried env fields) → Environment-level `ProvenanceGap` (shared `ProvenanceGapMixin`; `Environment.temperature` now optional)
- [x] Decision B: label = the **NPV** (mode-referenced modified z-score) — YeastPhenome's own presented label; flips plan Decision 2

## 2026.07.17

- [x] `YeastPhenomeDataset` loader built (single loader, `SCREENS`-driven; v1.0-pinned + sha256) — khozoie seed, 4228 records, L0-L4 PASS [[torchcell.datasets.scerevisiae.yeastphenome]]

## 2026.07.18

- [x] Widened to 20 screens / 38 environments / 140,264 records; column-aware parsing + drop-and-log
- [x] Fixed 2 verifier issues the multi-screen build surfaced: multi-component condition mis-parse; near-replicate screens (microarray vs barseq) falsely flagged as duplicates → uniqueness key now (study+readout, strain, condition)

## 2026.07.19

- [x] **CORRECTION** — plan Decision 4 "homozygous only" was wrong; HAPLOID screens are complete loss-of-function too (BY4741 is our dominant background). Included them: **47 screens / 83 environments / 296,777 records, L0-L4 PASS**
- [x] Overlap reconciliation vs our built datasets: 6 overlap (all excluded), 3 have no ingestible NPV (incl. Kemmeren), 23 absent; `retained ∩ built = {}`
