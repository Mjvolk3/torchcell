---
id: g72yvyh9ih7rcjij78ic42c
title: Rnaseq
desc: ''
updated: 1783563828999
created: 1783563828999
---

## 2026.07.08 - RNA-seq verifier: gate the WS10 Caudal pan-transcriptome per isolate, not per knockout

This verifier exists because the Caudal natural-isolate pan-transcriptome (WS10) is unlike every other family: records are keyed by natural ISOLATE rather than by a designed gene knockout, and each isolate stores ABSOLUTE per-gene `expression_tpm` + raw `expression_count` on its own genome. Its L1 identity check is therefore `strain_uniqueness` (one record per isolate, every record carrying a `strain_id`) instead of ORF uniqueness, and L2 splits into a TPM finiteness/non-negativity check plus a distinct integer-count check.

- L2 `count_value_fidelity` enforces raw counts are non-negative INTEGERS (rejecting bools), a fidelity guarantee TPM floats cannot give.
- L3 `reference_finite` checks the shared population-mean baseline TPMs are all finite -- the reference here is a population mean, not a centered zero.
- Exposes `rnaseq_gene_set` (union of measured genes) so the runner can assert L4 containment in the S288C SGD reference set; accessory/novel ORFs legitimately absent from S288C are why containment is thresholded, not exact.
- Phase A: operates purely on pydantic/LMDB records. Built from [[torchcell.verification.levels]], reported via [[torchcell.verification.report]], run by [[torchcell.verification.runners]].
