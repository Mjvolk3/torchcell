---
id: 5z8pfplmre459tdk75kgan8
title: Induction_timecourse
desc: ''
updated: 1791745456957
created: 1791745456957
---

## 2026.10.11 - Induction time-course verifier (Hackett 2020 IDEA)

Module: `torchcell/verification/induction_timecourse.py`; registry `INDUCTION_TIMECOURSE_DATASETS` in `torchcell/verification/runners.py` (`verify_induction_timecourse_dataset`, `run_induction_timecourse`, called right after `run_expression` in `run_all`); the loader's `verify_build` is what `build_dataset_lmdb --verify` calls. Dataset: [[torchcell.datasets.scerevisiae.hackett2020]].

Not the microarray family: `verify_expression_dataset` orients a deletion screen (`deletion_downregulates`) and its L4 compares every store's gene universe with Sameith's platform, neither of which holds for an induction time course on a different array.

| level | rule | what it asserts |
|---|---|---|
| L0 | structural | each experiment and reference validates as its concrete class; the record's `experiment_type` reconstructs the class |
| L1 | count, array_completeness | the stored (time course, minutes) set is the release's arrays minus the ledger's refused courses, each once |
| L1 | gene_completeness | every layer of every record has the same keys, a bijection with the release's gene names (`preprocess/gene_keys.json`) |
| L1 | layer_gaps | a layer is absent iff its release does not carry it, and then as a typed gap |
| L1-L3 | shared rules | `SharedRecordRules`: gap census, canonical gene names, uncertainty, compound and media identity |
| L2 | release_fidelity | every stored value equals its released cell bit for bit, re-read from `raw/` (tall zip, EV2) |
| L2 | value_fidelity, sampling_time | finite; intensities and noise non-negative; `duration_hours` x 60 is a released minute of the course |
| L3 | t0_zero, layer_nesting | log2 layers are 0 at t = 0 and in each reference; shrunken support within selected within thresholded, thresholded = cleaned where non-zero |
| L3 | inducer_timing | no estradiol at t = 0 or in a reference, exactly 1 uM after; the medium is the course's restriction |
| L3 | induction_upregulates | more than 0.9 of non-control courses have the induced gene's unprocessed log2 ratio above 0 at their last array |
| L3 | genotype_matches_release, sourced_values | the insertion names the release's TF (none for a control); every quote verbatim in its pinned bytes (text, gzip, Table EV1 row rendering) |
| L4 | shrunken_matches_dataset_ev1 | stored shrunken within 5e-08 of Dataset EV1 (EV1's rounding, measured) with the same zero pattern |
| L4 | release_matches_paper | the release's course, array, gene and signal-series counts are the paper's |
| L4 | geo_sample_coverage | every array is a GEO GSE142864 sample of its strain and time; a validation array's GEO title names its gene |
| L4 | induced_gene_containment | the induced genes are S288C genes |

The audit caches a source by (path, pinned sha256), so two quotes pinning the same file to different digests are each checked against their own pin. Result on the dev store: every rule passes; the table is in the dataset note. Run time 3.5 min, 5.9 GB peak (the tall table re-read dominates).

## 2026.10.12 - Genome-reference rows report, never gate

`induced_gene_containment` compare the data with OUR genome reference (an allele, a position or an id against the R64 annotation or a genomes-tier assembly), so they are `report`-class rules ([[torchcell.verification.report]], owner 2026-10-12: "the genome is separate.. it doesn't gate anything"). Their computation is unchanged; the report lists them under the verdict with their statistic and the store's verdict reads only its gate rows.
