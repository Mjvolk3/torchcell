---
id: l1rjyl0mwua15d1jrohqdp7
title: Pseudobulk_umi_count
desc: ''
updated: 1791732533600
created: 1791732533600
---

## 2026.10.11 - A verifier family for pseudobulk single-cell UMI counts

Module: `torchcell/verification/pseudobulk_umi_count.py`, runner `runners.run_pseudobulk_umi_count` over `PSEUDOBULK_UMI_COUNT_DATASETS` (one entry, `tf_deletion_pseudobulk_jackson2020`, expected 784 records, floor `MIN_JACKSON_FEATURE_CONTAINMENT = 0.999`), per-dataset entry `runners.verify_pseudobulk_umi_count_dataset` (what the loader's `verify_build` calls), wired into `run_all` after the natural-isolate family. Results and the measured table: [[torchcell.datasets.scerevisiae.jackson2020]].

A record is the summed UMI counts of one barcoded strain in one culture, so the strongest available check is to re-sum the released matrix and compare every stored number. The verifier reads the five raw files of the store's `raw/` with the loader's own readers (strain table, barcode FASTA, the authors' settings file, the bulk header, the matrix) and the genome for the feature map, then streams the store once:

- L0 validates every record as `PseudobulkUmiCountExperiment` and every distinct reference as `PseudobulkUmiCountExperimentReference` (declared-member validation, not the whole union).
- L1 keys uniqueness on (strain label, condition), the condition identified by the canonical JSON of the stored environment against the loader's eleven; requires the stored pairs to be exactly the released pairs holding a cell; `strain_identity` requires one barcode and one gene per strain, the barcode equal to Table S2 and Source code 4, the gene the label's, and the strains-per-gene counts Table S2's.
- L2 `release_fidelity` compares every record's gene dict, marker dict, total, cell count, barcodes and per-cell totals with a fresh sum; `reference_fidelity` re-pools each condition's HO-control strains; `release_totals` requires the paper's 38,225 cells and 83,703,440 UMIs.
- L3 checks one measurement type, one feature universe equal to the loader's map, the conditions (Table 1 rows in paper.md, labels the authors'), and audits every quote (paper.md text, workbook row renderings, the two settings lines).
- L4 `feature_containment` against the SGD ORF and RNA FASTA headers; `feature_ids_corroborated` re-checks the un-mangled header against `TRIZOL_BULK.tsv`.

A raw header that no longer undoes to the bulk header raises (`FeatureMapError`) rather than reporting a row: the feature map every other rule depends on cannot be built.

Tests: `tests/torchcell/verification/test_pseudobulk_umi_count.py` builds the miniature Jackson store (`tests/torchcell/datasets/scerevisiae/_jackson2020_release.py`), passes every row, then catches an edited count, a reversed per-cell list, a dropped and a repeated record, a wrong barcode and a wrong gene, an edited environment, a reordered reference, a stray key, a second measurement type, a low containment floor and a missing quote; `tests/torchcell/verification/test_runners.py` pins the runner wiring.
