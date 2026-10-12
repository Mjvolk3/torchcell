---
id: 6air79pyz0zkn21196az1bt
title: Gh_1045_store_verify
desc: ''
updated: 1791816897833
created: 1791816897833
---

## 2026.10.12 - Verification lane for the #1045 rebuilds

Runs `kg4_pre_build_verification_sweep.py --dataset <Class>` for one rebuilt store, read-only on `processed/`, from a frozen `git clone --shared` checkout of the branch (`TORCHCELL_SRC`), writing its own lane file. Used for the Costanzo 2016 and Kuzmin 2020 stores (no per-dataset verifier, so `build_dataset_lmdb --verify` reports none for them) and Parsons 2006 (registry route), each submitted with `--dependency=afterok` on its build task. Jobs and verdicts are in [[torchcell.datasets.scerevisiae.costanzo2016]], [[torchcell.datasets.scerevisiae.kuzmin2020]] and [[torchcell.datasets.scerevisiae.parsons2006]].
