---
id: pc0dd7l4lvo1y5sacnibs8n
title: Warm_split_cache_030
desc: ''
updated: 1790417014955
created: 1790417014955
---

## 2026.09.26 - Why the split cache is warmed here

`CellDataModule` names its cache `index_seed_<seed>_pin<n>-<hash>-utt_sub<n>-<hash>.json` and checks for it with a bare `osp.exists`, so four DDP ranks on IGB with no cache would each scan 13.5M records and race to write the same file. The warmer builds the dataset once (no embeddings; the split does not depend on them) and the data module per seed, under the arm's final pool, pin and unpinned-to-train rule, so the cache `sync_igb_030_build.sh` mirrors is what the launcher preflight looks for. The seed is part of the filename, so every campaign seed needs its own file (`+warm.seeds=[0,1,2]`). Runs as step 1 of [[experiments.030-solid-growth-multi.scripts.gh_smoke_dataset_token]].
