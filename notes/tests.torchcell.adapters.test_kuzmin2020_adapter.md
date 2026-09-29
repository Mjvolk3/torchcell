---
id: 1jhxc3jn8whfimqgp2vp9w3
title: Test_kuzmin2020_adapter
desc: ''
updated: 1790649306712
created: 1790649306712
---

## 2026.09.28 - Exact graph emission for every kuzmin2020 conf (Phase 9)

2 tests parametrized over the dataset confs, through `tests/torchcell/adapters/_sga_adapter_harness.py`: an in-memory two-record dataset and an independent reconstruction of every node and edge (ids, preferred ids, labels, property keys, the reference collectors' dedup by id), compared element-wise against the recorded BioCypher output; 21 + 2P nodes and 20 + 2P edges for P perturbations, 17 LMDB closes (one per chunked method: 8 node methods, 9 edge methods), 28 wandb events in order with the method table logged first; a missing conf is refused before wandb starts. The conf bodies below the header are byte-identical across the seven fitness confs and across the four interaction confs. Coverage 21% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
