---
id: wvblmcswql2pb2iqloc40cx
title: Sync_igb_030_build
desc: ''
updated: 1790417073906
created: 1790417073906
---

## 2026.09.26 - GilaHyper to IGB mirror of the 030 build

rsync of `processed/` (955,268,878,336 bytes of LMDB plus the index JSONs), an empty `raw/lmdb` stub and `data_module_cache/` (the split indices the warmer writes) to `mjvolk3@biologin.igb.illinois.edu:/home/a-m/mjvolk3/scratch/torchcell/data/torchcell/experiments/030-solid-growth-multi/001-multi-build`. Resumable, no deletes; prints local and remote counts and the data.mdb byte size and exits 1 on a mismatch. Run under [[experiments.030-solid-growth-multi.scripts.gh_sync_igb_030]]; rerun after the smoke job so the cache crosses.
