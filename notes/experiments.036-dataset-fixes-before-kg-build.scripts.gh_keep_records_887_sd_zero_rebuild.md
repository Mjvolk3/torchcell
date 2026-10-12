---
id: 2g0ueyg8hiuwfbv9be5uw69
title: Gh_keep_records_887_sd_zero_rebuild
desc: ''
updated: 1791814304328
created: 1791814304328
---

## 2026.10.12 - Rebuild the two stores PR #1054 changes

`sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_keep_records_887_sd_zero_rebuild.slurm` rebuilds, in the dev tree and one at a time, `SmfKuzmin2018Dataset` and `DmfKuzmin2018Dataset` (the query strains' Additional Data S4 SDs) with `build_dataset_lmdb --retire-existing --verify`, then censuses them with `keep_records_887_sd_zero_inventory.py --after` ([[experiments.036-dataset-fixes-before-kg-build.scripts.keep_records_887_sd_zero_inventory]]). `tmf_kuzmin2020`, whose SD this branch now labels, is not rebuilt here: `fix/audit-yeast-refusals-1045` (PR #1053) rebuilt every Kuzmin 2020 store with the retired and merged ORF rows it keeps, which this branch's tree would drop; the handover is on issue #1045.
