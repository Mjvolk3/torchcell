---
id: lkfg1oz3ij9kvqa4q1i3vv7
title: Test_neo4j_preprocessed_cell
desc: ''
updated: 1790979675455
created: 1790979675455
---

## 2026.10.01 - Compact store against the live Lazy item

Test file: `tests/torchcell/data/test_neo4j_preprocessed_cell.py`, target [[torchcell.data.neo4j_preprocessed_cell]].

### Fixture

A hand-written `Neo4jCellDataset` build (no Neo4j; `processed/lmdb` plus `experiment_types.json`), five genes YAL001C=0 .. YAL005C=4, two gene edge types and a three-reaction metabolism bipartite graph.

- physical edge_index `[[0,1,3,0,1,2,3,4],[1,2,4,0,1,2,3,4]]` (three edges, then one self loop per node from `to_cell_data`).
- regulatory edge_index `[[2,4,0,1,2,3,4],[0,2,0,1,2,3,4]]`.
- reactions r_A (genes 0, 1), r_B (gene 3), r_C (none); GPR `[[0,1,3],[0,0,1]]`, RMR `[[0,0,1,2],[0,1,1,0]]`.
- genotypes: {1} fitness 0.9 se 0.01; {0,3} fitness 0.4 se 0.05; {4} gene_interaction -0.2 p 0.03.

### Expected values

An edge is kept iff neither endpoint is deleted; a reaction iff all its genes are kept; an RMR edge iff its reaction is kept; a GPR edge iff its gene is kept. Stored False indices:

| genotype | physical | regulatory | GPR | RMR | reactions |
|---|---|---|---|---|---|
| {1} | 0, 1, 4 | 3 | 1 | 0, 1 | 0 |
| {0,3} | 0, 2, 3, 6 | 0, 2, 5 | 0, 2 | 0, 1, 2 | 0, 1 |
| {4} | 2, 7 | 1, 6 | none | none | none |

Every key of every store of `ds[i]` equals `Neo4jCellDataset(graph_processor=LazySubgraphRepresentation()).get(i)` (dtype and value) except the two findings below. The class defines no `process`, so the PyG constructor writes nothing (`has_process` False); `processed/` holds `lmdb`, `metadata.json` = `{"length": 3}` and the lock file.

### Findings

- Reaction `node_ids` differ: the store copies the cell graph's names (`neo4j_preprocessed_cell.py:255`), live Lazy sets positions 0..R-1 (`graph_processor.py:1724`).
- GPR `num_edges` differs: copied from the cell graph, which counts unique reactions (2, `cell_data.py:557-559`); live sets the hyperedge count (3, `graph_processor.py:1691-1693`).
- No source fingerprint: an instance on a finished root serves the old build for any source (length 3 and the old genotype for a one-record source).
- Re-preprocessing a shorter source leaves stale keys; `get(1)` returns the old record after a 1-record rerun.
- A record with no statistic value (`fitness_se=None`) crashes `_extract_mask_indices` with AttributeError at line 170, because Lazy omits the `phenotype_stat_*` keys when no statistic exists (`graph_processor.py:1858-1868`).

Reach (audit 2): the two field differences are latent. The lazy collater keeps reaction `node_ids` as a per-sample list the lazy model never reads, and the model computes edge counts from `edge_index` (`hetero_cell_bipartite_dango_gi_lazy.py:1199`), not `num_edges`; they are reached only in that the 006 configs 077 (full masks) and 081/082 (compact) read these stores. The stale-store findings (no source fingerprint, stale keys) reach 077/081/082. The statistic crash is latent unless the real 006 build holds a record with no statistic (not measured).

`main_preprocess` (demo, needs genome and Neo4j build) is left uncovered.

## 2026.10.05 - Audit 2 corrections

- Line cites corrected at 4a179a2e1: `graph_processor.py:1724` (reaction ids), `1691-1693` (GPR count), `cell_data.py:557-559` (cell-graph GPR count), `graph_processor.py:1858-1868` (statistic keys).
- New Finding pinned: `ids_pert` is `list(set)` (`graph_processor.py:1550`), so its order follows `PYTHONHASHSEED`. Record 1 built in two fresh interpreters gives [YAL001C, YAL004W] under seed 0 and [YAL004W, YAL001C] under seed 1, with `perturbation_indices` [0, 3] both times. The store freezes the preprocessing process's order.
- New pin: after `close_lmdb`, the next `get` opens a new read env (a different object) and serves the same record.
