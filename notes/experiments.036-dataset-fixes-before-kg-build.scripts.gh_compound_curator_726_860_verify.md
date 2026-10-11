---
id: vxgt08q3ie1x3xq8ik8ldxb
title: Gh_compound_curator_726_860_verify
desc: ''
updated: 1791706880442
created: 1791706880442
---

## 2026.10.11 - Verifying the stores the build CLI did not

`build_dataset_lmdb --verify` verifies a store only when its module declares a per-dataset
entry point; for the family-runner stores and for the touched loaders whose stores were not
rebuilt, this array runs `kg4_pre_build_verification_sweep.py` for ONE dataset per task
(slurm 3872, 18 tasks), so the seven-hour Hoepfner atlas does not serialize the rest. Each
route writes its reports into the store's `preprocess/`, which
`compound_curator_726_860_verification_tables.py` renders.
