---
id: 9bvt1ddwwqc5oiai2zztg61
title: Sync_delta_store
desc: ''
updated: 1791240994278
created: 1791240994278
---

## 2026.10.05 - Push one built 019 store to Delta

`bash experiments/019-simb-multimodal/scripts/sync_delta_store.sh fig3_proteome` from
GilaHyper. Generalizes `sync_delta_fig3_core.sh`, whose destination root (`/work/hdd`)
disagreed with the root every Delta launcher reads (`/scratch/bbub/mjvolk3/torchcell`); a
store synced to the wrong root makes a compute node try to rebuild it with no Neo4j, so the
default here is the launchers' root. The `data_module_cache` travels with the store so Delta
draws the same split indices for seeds 0 to 11. Delta is Duo-gated: the user runs it and
approves one push prompt. fig3_proteome had never been synced before this date; only
fig3_core had.
