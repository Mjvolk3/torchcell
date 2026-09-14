"""Tests for degree-preserving rewiring of gene-gene graphs (torchcell.graph.rewire)."""

import torch
from torch_geometric.data import HeteroData

from torchcell.graph.rewire import degree_preserving_rewire, rewire_cell_graph


def _ring_with_chords(n: int = 200, seed: int = 0) -> torch.Tensor:
    """Symmetric graph: a ring plus random chords, both directions stored."""
    g = torch.Generator().manual_seed(seed)
    u = torch.arange(n)
    ring = torch.stack([u, (u + 1) % n])
    a = torch.randint(0, n, (300,), generator=g)
    b = torch.randint(0, n, (300,), generator=g)
    keep = a != b
    chords = torch.stack([a[keep], b[keep]])
    und = torch.cat([ring, chords], dim=1)
    und = torch.unique(torch.sort(und, dim=0).values, dim=1)
    return torch.cat([und, und.flip(0)], dim=1)


def _degrees(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.bincount(edge_index[0], minlength=n),
        torch.bincount(edge_index[1], minlength=n),
    )


def test_symmetric_graph_keeps_degrees_and_symmetry_and_loses_wiring() -> None:
    n = 200
    ei = _ring_with_chords(n)
    out, st = degree_preserving_rewire(ei, n, seed=1)
    assert out.shape == ei.shape
    assert _degrees(out, n)[0].tolist() == _degrees(ei, n)[0].tolist()
    assert _degrees(out, n)[1].tolist() == _degrees(ei, n)[1].tolist()
    # Still symmetric: every (u, v) has (v, u).
    s = set(map(tuple, out.t().tolist()))
    assert all((v, u) in s for u, v in s)
    assert not (out[0] == out[1]).any()
    assert st["symmetric"] == 1.0
    assert st["edge_overlap"] < 0.5, st
    assert st["swaps_done"] > 0


def test_directed_graph_keeps_in_and_out_degrees() -> None:
    n = 150
    g = torch.Generator().manual_seed(3)
    a = torch.randint(0, n, (600,), generator=g)
    b = torch.randint(0, n, (600,), generator=g)
    ei = torch.unique(torch.stack([a[a != b], b[a != b]]), dim=1)
    out, st = degree_preserving_rewire(ei, n, seed=7)
    assert st["symmetric"] == 0.0
    assert _degrees(out, n)[0].tolist() == _degrees(ei, n)[0].tolist()
    assert _degrees(out, n)[1].tolist() == _degrees(ei, n)[1].tolist()
    assert not (out[0] == out[1]).any()
    assert torch.unique(out, dim=1).shape[1] == out.shape[1]  # no duplicate edges
    assert st["edge_overlap"] < 0.5


def test_seed_determines_the_rewiring() -> None:
    ei = _ring_with_chords(100)
    a, _ = degree_preserving_rewire(ei, 100, seed=11)
    b, _ = degree_preserving_rewire(ei, 100, seed=11)
    c, _ = degree_preserving_rewire(ei, 100, seed=12)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_rewire_cell_graph_leaves_the_original_untouched() -> None:
    n = 120
    cg = HeteroData()
    cg["gene"].num_nodes = n
    cg["gene", "physical_interaction", "gene"].edge_index = _ring_with_chords(n, seed=1)
    cg["gene", "tflink", "gene"].edge_index = _ring_with_chords(n, seed=2)[:, :150]
    before = {et: cg[et].edge_index.clone() for et in cg.edge_types}
    new, stats = rewire_cell_graph(cg, seed=5)
    assert set(stats) == {"physical_interaction", "tflink"}
    for et in cg.edge_types:
        assert torch.equal(cg[et].edge_index, before[et])
        assert not torch.equal(new[et].edge_index, before[et])
        assert new[et].edge_index.shape == before[et].shape
