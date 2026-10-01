---
id: wbrj248zf4h2yarvugaxywz
title: Data
desc: ''
updated: 1741805889554
created: 1706126351737
---
## 2025.03.12 - Graph Types

| Graph Type          | Mathematical Representation                                | Description                                                                 | PyG Init                                   | Example in Metabolism Data                                                         |
|---------------------|------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------------------------------|------------------------------------------------------------------------------------|
| Simple Graph        | $G = (\mathcal{N}, \mathcal{E})$                           | Undirected or directed graph with single node type and single edge type.    | (`n.x`, `edge_index`)                      | protein-protein, regulatory                                                        |
| Directed Multigraph | $G = (\mathcal{N}, \mathcal{E}, \mathcal{R})$              | Graph with multiple edge types/relations between same node type.            | (`n.x`, `edge_index`, `edge_type`)         | (protein-protein, regulatory)                                                      |
| Bipartite Graph     | $G = (\mathcal{U}, \mathcal{V}, \mathcal{E})$              | Two distinct node sets with edges only between them, not within sets.       | (`n1.x`, `n2.x`, `(n1, r, n2).edge_index`) | gene-protein-reaction, reaction-metabolite-relation                                |
| Hypergraph          | $G = (\mathcal{N}, \mathcal{H})$                           | Edges (hyperedges) connect arbitrary subsets of nodes rather than pairs.    | (`n.x`, `h.x`, `(n, r, h).edge_index`)     | gene-protein-reaction, reaction-metabolite-relation                                |
| Heterogeneous Graph | $G = (\mathcal{N}, \mathcal{E}, \mathcal{T}, \mathcal{R})$ | Graph with multiple node types and edge types; generalizes all above cases. | (`n1.x`, `n2.x`, `(n1, r, n2).edge_index`) | (protein-protein, regulatory, gene-protein-reaction, reaction-metabolite-relation) |

Notation: $\mathcal{N}$ is the set of nodes, $\mathcal{E}$ is the set of edges, $\mathcal{R}$ is the set of relations/edge types, $\mathcal{T}$ is the set of node types, $\mathcal{H}$ is the set of hyperedges where $\mathcal{H} \subseteq 2^{\mathcal{N}}$, and $\mathcal{U}$, $\mathcal{V}$ are disjoint node sets in bipartite graphs.

## 2026.10.01 - ReferenceIndex refuses equal references

`ReferenceIndex.validate_data` checked only that member indices partition `range(N)`, so two entries with equal references were accepted. It now raises `DuplicateReferenceError` (a `ValueError`, wrapped by pydantic) naming both entry positions and pointing at `ExperimentReferenceIndex.combine`. References compare by class name plus key-sorted JSON dump. Before the change, 38 built dev stores under `$DATA_ROOT/data/torchcell/` (every current `preprocess/experiment_reference_index.json` under 2 MB, plus the `neo4j_query_test` and `showcase_essentiality_smf` raw indices) were read with a one-off script: none holds two entries with equal references. `ReferenceIndex` is also instantiated nowhere in `torchcell/`. Issue #541; test `test_reference_index_refuses_one_reference_split_over_two_entries`.

## 2026.10.01 - Review fix: one reference key

The key is now `reference_key` (class name plus key-sorted JSON dump), used by both the `ReferenceIndex` refusal and `ExperimentReferenceIndex.combine`. Two separately built NaN-bearing references are unequal under `==` but share the key, so they are refused together and `combine` (the remedy the refusal names) merges them. Test: `test_nan_bearing_references_are_refused_together_and_combine_merges_them`.
