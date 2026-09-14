# torchcell/graph/rewire.py
# [[torchcell.graph.rewire]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/graph/rewire
# Test file: tests/torchcell/graph/test_rewire.py
"""Degree-preserving random rewiring of gene-gene graphs.

The random-graph control of the graph-regularization figure asks whether the soft KL
prior helps because of WHICH edges it carries or merely because it carries SOME
structure. The control keeps every node's degree (in and out for a directed graph, the
single degree for a symmetric one) and randomizes the wiring by repeated double-edge
swaps: two edges (a, b) and (c, d) become (a, d) and (c, b) when neither new edge is a
self-loop or already present. This is the configuration-model null of Maslov and Sneppen
(2002); degree sequence, edge count and node set are preserved exactly, and nothing else
is.

A graph stored with both directions of every edge (the STRING channels, physical
interactions) is rewired on its undirected edge set and re-symmetrized, so the result is
again symmetric; a graph with any one-directional edge (regulatory, TFLink) is rewired as
directed, preserving each node's out-degree and in-degree separately.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch_geometric.data import HeteroData


def _is_symmetric(edges: np.ndarray) -> bool:
    """True when every (u, v) has its reverse (v, u) in the edge list."""
    forward = set(map(tuple, edges.tolist()))
    return all((v, u) in forward for u, v in forward)


def _double_edge_swaps(
    edges: np.ndarray, undirected: bool, rng: np.random.Generator, n_attempts: int
) -> tuple[np.ndarray, int]:
    """Rewire ``edges`` [E, 2] in place by double-edge swaps; returns (edges, n_done)."""
    edges = edges.copy()
    n_edges = len(edges)
    present: set[tuple[int, int]] = set(map(tuple, edges.tolist()))
    done = 0
    pairs = rng.integers(0, n_edges, size=(n_attempts, 2))
    flips = rng.random(n_attempts) < 0.5
    for (i, j), flip in zip(pairs, flips):
        if i == j:
            continue
        a, b = int(edges[i, 0]), int(edges[i, 1])
        c, d = int(edges[j, 0]), int(edges[j, 1])
        if undirected and flip:
            # Either pairing of the four endpoints is a valid undirected swap.
            c, d = d, c
        # New edges (a, d) and (c, b)
        if a == d or c == b:
            continue
        e1 = (min(a, d), max(a, d)) if undirected else (a, d)
        e2 = (min(c, b), max(c, b)) if undirected else (c, b)
        if e1 in present or e2 in present or e1 == e2:
            continue
        present.discard((a, b))
        present.discard((c, d) if not (undirected and flip) else (d, c))
        present.add(e1)
        present.add(e2)
        edges[i] = e1
        edges[j] = e2
        done += 1
    return edges, done


def degree_preserving_rewire(
    edge_index: torch.Tensor, num_nodes: int, seed: int, swaps_per_edge: float = 5.0
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a rewired ``edge_index`` with the same degree sequence, plus statistics.

    Args:
        edge_index: [2, E] long tensor, no self-loops expected (self-loops are kept out
            of the swap pool and put back unchanged).
        num_nodes: Node count, for the degree checks.
        seed: RNG seed; the same seed gives the same rewiring.
        swaps_per_edge: Swap attempts per edge. Each accepted swap moves two edges, and
            a few attempts per edge is enough to lose the original wiring; the returned
            ``edge_overlap`` says how much survived.

    Returns:
        The rewired [2, E] tensor and a dict with ``n_edges``, ``symmetric`` (1.0 / 0.0),
        ``swaps_done`` and ``edge_overlap`` (fraction of original directed edges still
        present).
    """
    if edge_index.numel() == 0:
        return edge_index.clone(), {
            "n_edges": 0.0,
            "symmetric": 0.0,
            "swaps_done": 0.0,
            "edge_overlap": 1.0,
        }
    edges = edge_index.detach().cpu().numpy().T.astype(np.int64)  # [E, 2]
    loops = edges[:, 0] == edges[:, 1]
    loop_edges = edges[loops]
    edges = edges[~loops]
    symmetric = _is_symmetric(edges)
    rng = np.random.default_rng(seed)
    if symmetric:
        und = np.unique(np.sort(edges, axis=1), axis=0)  # each pair once, u < v
        rewired, done = _double_edge_swaps(
            und, True, rng, int(swaps_per_edge * len(und))
        )
        rewired = np.concatenate([rewired, rewired[:, ::-1]], axis=0)
    else:
        rewired, done = _double_edge_swaps(
            edges, False, rng, int(swaps_per_edge * len(edges))
        )
    rewired = np.concatenate([rewired, loop_edges], axis=0)

    original = set(map(tuple, np.concatenate([edges, loop_edges]).tolist()))
    kept = sum((u, v) in original for u, v in map(tuple, rewired.tolist()))
    out_deg_before = np.bincount(edge_index[0].cpu().numpy(), minlength=num_nodes)
    out_deg_after = np.bincount(rewired[:, 0], minlength=num_nodes)
    in_deg_before = np.bincount(edge_index[1].cpu().numpy(), minlength=num_nodes)
    in_deg_after = np.bincount(rewired[:, 1], minlength=num_nodes)
    assert np.array_equal(out_deg_before, out_deg_after), "out-degree changed"
    assert np.array_equal(in_deg_before, in_deg_after), "in-degree changed"
    assert len(rewired) == edge_index.shape[1], "edge count changed"
    out = torch.as_tensor(rewired.T, dtype=edge_index.dtype, device=edge_index.device)
    return out, {
        "n_edges": float(edge_index.shape[1]),
        "symmetric": 1.0 if symmetric else 0.0,
        "swaps_done": float(done),
        "edge_overlap": kept / len(rewired),
    }


def rewire_cell_graph(
    cell_graph: HeteroData, seed: int, swaps_per_edge: float = 5.0
) -> tuple[HeteroData, dict[str, dict[str, float]]]:
    """Copy ``cell_graph`` with every (gene, rel, gene) edge_index rewired.

    Each relation gets its own derived seed (``seed * 1000 + position``) so the graphs
    are rewired independently; the original object is left untouched because the
    dataset's graph processor still reads it.
    """
    num_nodes = int(cell_graph["gene"].num_nodes)
    rewired = copy.copy(cell_graph)
    stats: dict[str, dict[str, float]] = {}
    gene_gene = [
        et for et in cell_graph.edge_types if et[0] == "gene" and et[2] == "gene"
    ]
    for position, edge_type in enumerate(gene_gene):
        new_index, st = degree_preserving_rewire(
            cell_graph[edge_type].edge_index,
            num_nodes,
            seed=seed * 1000 + position,
            swaps_per_edge=swaps_per_edge,
        )
        rewired[edge_type].edge_index = new_index
        stats[edge_type[1]] = st
    return rewired, stats
