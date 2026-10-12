---
id: t0d2agaaaayhsdj89awzcy3
title: Segregant_expression
desc: ''
updated: 1791727618461
created: 1791727618462
---

## 2026.10.11 - L0-L4 verifier for segregant single-cell expression

Verifier for `SegregantExpressionExperiment` records (Boocock 2025, [[torchcell.datasets.scerevisiae.boocock2025]]). Neither the segregant growth verifier ([[torchcell.verification.segregant_growth]], Bloom's conditions and residuals) nor the expression verifiers (gene-keyed perturbations) fit a haplotype mosaic with a UMI-count readout, so this is its own family, `SEGREGANT_EXPRESSION_DATASETS` + `run_segregant_expression` in `torchcell/verification/runners.py`, called in `run_all` right after the growth family.

One streaming pass over the LMDB re-reads each panel's release from `raw/` (`segData.RDS`, and for the pooled panel `segMatch.RDS` + `gdata_42k.RData`). LMDB keys iterate in byte order ("1", "10", "100", ...), so panels interleave in the stream and every panel's release stays loaded until the pass ends (about 3 GB for the four real panels).

| level | rule | what it compares |
|---|---|---|
| L0 | `structural` | `TypeAdapter(ExperimentType)` and `experiment_type == "segregant_expression"` |
| L1 | `count`, `panel_counts`, `segregant_uniqueness` | records per panel equal the release's cells (de novo) or matched segregants (pooled); no (panel, segregant) twice |
| L1 | shared rules | gap census, compound identity, media membership, uncertainty sanity ([[torchcell.verification.common]]) |
| L2 | `value_fidelity` | counts, `total_umis`, stage and library partitions equal the release row, or the sum over the matched cells |
| L2 | `mosaic_round_trip` | de novo blocks re-expand to the released posterior (called parent exact, level within `POSTERIOR_STEP / 2`); pooled blocks to the lookup calls exactly; reports blocks per record by panel |
| L2 | `reference_pseudobulk` | each panel's reference is the sum of its cells |
| L3 | `table_s4`, `pooled_panel` | cells, analysis transcripts and median UMI per panel against Table S4; 393 segregants, median 17 cells, 277 over ten, median per-cell agreement within 0.01 of the paper's 92.5% |
| L3 | `sourced_values` | every quote still in its pinned `paper.md` |
| L4 | `parents_pinned`, `mat_orientation`, `marker_reference`, `gene_containment_sgd` | Peter ids in the member index; the MATa parent carries the MAT locus; sampled marker refs are S288C bases; analysis transcripts and the gene set are S288C genes |

`check_paper_numbers=False` skips the two printed-number rules for a synthetic release (`tests/torchcell/verification/test_segregant_expression.py`).

## 2026.10.12 - Genome-reference rows report, never gate

`marker_reference`, `mat_orientation`, `parents_pinned`, `gene_containment_sgd` compare the data with OUR genome reference (an allele, a position or an id against the R64 annotation or a genomes-tier assembly), so they are `report`-class rules ([[torchcell.verification.report]], owner 2026-10-12: "the genome is separate.. it doesn't gate anything"). Their computation is unchanged; the report lists them under the verdict with their statistic and the store's verdict reads only its gate rows.
