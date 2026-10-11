---
id: 6we9gzaj7ydqfq1i5d5z3mq
title: Hackett2020_adapter
desc: ''
updated: 1791745464909
created: 1791745464909
---

## 2026.10.11 - Hackett 2020 IDEA adapter

`torchcell/adapters/hackett2020_adapter.py` (`InductionTimecourseHackett2020Adapter`) with `conf/induction_timecourse_hackett2020_adapter.yaml`: 17 node and 15 edge methods. The inducible promoter insertion is a first-class `perturbation` node (one per strain; none for the GEV and Z3EV control courses), beta-estradiol an `environment perturbation` node on every post-induction array, and the readout the new `induction timecourse expression phenotype` class (properties `log2_cleaned`, `log2_noise_sd`, `log2_shrunken` as JSON strings plus `timecourse_id`, `strain_label`, `experiment_date`, `release_table`; the channel intensities and the unprocessed, thresholded and selected layers travel in the experiment blob only). The typed strain background (alleles and the synthetic activator cassette) rides on the `genome` node's serialized reference genome. Map entry `InductionTimecourseHackett2020Dataset: InductionTimecourseHackett2020Adapter`; the build-time projection pin counts it among the uncalibrated served datasets. Dataset: [[torchcell.datasets.scerevisiae.hackett2020]].
