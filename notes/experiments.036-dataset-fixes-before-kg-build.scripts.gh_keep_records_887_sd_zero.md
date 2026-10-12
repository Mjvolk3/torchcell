---
id: lg8kzelyl55q7tu5dg18c08
title: Gh_keep_records_887_sd_zero
desc: ''
updated: 1791806046967
created: 1791806046967
---

## 2026.10.12 - Inventory plus read-only L0-L4 for #887 and the SD-of-zero issues

`sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_keep_records_887_sd_zero.slurm` runs [[experiments.036-dataset-fixes-before-kg-build.scripts.keep_records_887_sd_zero_inventory]], then `kg4_pre_build_verification_sweep.py --dataset <Class>` for `FattyAcidXue2025Dataset`, `DmfKuzmin2018Dataset`, `TmfKuzmin2018Dataset` and `DmfKuzmin2020Dataset` into the lane file `results/keep_records_887_sd_zero_verification.json`. Read-only on `processed/`; each verifier writes its `preprocess/verification_report.json` in the dev tree. Job 4201 (2026-10-12): COMPLETED in 16 min 51 s, max RSS 4.9 GB (the header asks 2 CPUs, 32 GB, 2 h).
