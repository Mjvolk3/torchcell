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
