---
id: aa8ezqcgqkrmfdyo7mao2fg
title: Pyg_probe
desc: ''
updated: 1792228685341
created: 1792228685341
---

## 2026.10.17 - Does PyG, and does torchcell's model code, run on this accelerator?

`experiments/042-hpc-smoke-tests/scripts/pyg_probe.py --machine <name> --out <json>` runs the
checks a real torchcell job depends on, one at a time, recording each as `ok` or the first line
of its exception and never stopping at the first failure: `torch_geometric.utils.scatter` for
sum, mean, max, min and mul with gradients; `global_add/mean/max_pool`, `to_dense_batch`, index
`softmax`; GCNConv, SAGEConv, GATv2Conv, GINConv and TransformerConv forward and backward on a
random 512-node graph; a two-type `HeteroConv`; bf16 autocast; then the imports of
`torch_scatter`, `torchcell.models`, `torchcell.data.graph_processor` and every module under
`torchcell/models/`, enumerated from the directory so a failing package `__init__` is a recorded
check rather than a crash. The exit code is non-zero when a PyG-core check fails; import failures
are reported only, since on XPU the `torch_scatter` imports are the expected result. W&B summary
keys `probe/<check>` in project `torchcell_042_hpc_smoke`, tag `pyg-probe`.

Runs and the per-machine results: [[experiments.042-hpc-smoke-tests]].
