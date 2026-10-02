---
id: 9ypnjr6w19kdxmambs29va2
title: Graph_reg_flags_smoke
desc: ''
updated: 1790495582864
created: 1790495582864
---

## 2026.09.27 - Real-size CPU smoke of the round-2 graph-entry flags

Builds the trainer's 6,607-gene cell graph (genome plus the nine graphs, no LMDB) and, for each round-2 variant (KL 1/2/3-hop, KL symmetric, KL on layers 1-4, mask 1-hop, directed, 2/3-hop directed, 2-hop symmetric, layers 3-4, hidden 360), constructs the model, runs one forward and backward on a two-genotype batch, and records parameters, loss finiteness, seconds and peak RSS to `results/graph_reg_flags_smoke.json`; also the allowed fraction per masked head and the count of all-zero target rows per graph. Launched by `gh_graph_reg_flags_smoke.slurm` (CPU, 64 GB, 16 cores) from the worktree. The unit tests are `tests/torchcell/models/test_cgt_graph_entry.py`; this is the plumbing check at real size before the Delta queue.
