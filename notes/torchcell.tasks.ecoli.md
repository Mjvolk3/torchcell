---
id: asik4glupp26uzfij46cyqh
title: Ecoli
desc: ''
updated: 1795561775086
created: 1795561775086
---

## 2026.11.24 - The four E. coli tasks

`ECOLI_TASKS` pins dataset, assembly set and label for Campos 2018 growth and imaging, Babu 2014 and Fuhrer 2017. `match_records` resolves each record's genes against the genome; a record is a sample when it perturbs the task's number of genes and each resolves to a protein-coding gene (3,573, 3,573, 40,658 and 3,624 samples; 91, 91, 2,047 and 182 excluded records, one build on GilaHyper, 2026-11-24). `load_task` attaches the split files in `torchcell/tasks/data`, checked against `SPLIT_SHA256`.
