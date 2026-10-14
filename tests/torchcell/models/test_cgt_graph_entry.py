"""How the gene-gene graphs enter the cell graph transformer: reach and direction.

Round 2 of the 025 graph-regularization study added four flags (2026-09-27):
``graph_regularization.hops`` and ``.symmetrize`` for the soft KL target,
``attention_mask.hops`` and ``.symmetric`` for the hard mask. These tests pin the
targets and supports on a hand-built directed four-gene path with one extra edge,
and pin the defaults to the behavior every earlier run had: a directed one-hop KL
target without self-loops, and a symmetrized one-hop mask with self-loops.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    khop_reach,
)

# Directed graph on 4 genes: 0 -> 1 -> 2 -> 3, plus 0 -> 2.
EDGES = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 2]], dtype=torch.long)
N = 4


def _cell_graph(edges: torch.Tensor = EDGES) -> HeteroData:
    cg = HeteroData()
    cg["gene"].num_nodes = N
    cg["gene", "regulatory", "gene"].edge_index = edges
    return cg


def _model(edges: torch.Tensor = EDGES, **kwargs: Any) -> CellGraphTransformer:
    torch.manual_seed(0)
    return CellGraphTransformer(
        gene_num=N,
        hidden_channels=8,
        num_transformer_layers=2,
        num_attention_heads=2,
        cell_graph=_cell_graph(edges),
        heads_config=None,
        **kwargs,
    )


def _dense(edge_index: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    a = torch.zeros(N, N, dtype=torch.bool)
    a[edge_index[0], edge_index[1]] = True
    return a | a.T if symmetric else a


def test_khop_reach_directed() -> None:
    """One hop is the edge set; two hops adds every two-step endpoint; three hops the rest."""
    one = khop_reach(EDGES, N, 1, symmetric=False)
    assert torch.equal(one, _dense(EDGES))
    two = khop_reach(EDGES, N, 2, symmetric=False)
    # 0->1->2, 0->2->3, 1->2->3 add (0,2) [already], (0,3), (1,3)
    expect = _dense(EDGES).clone()
    expect[0, 3] = True
    expect[1, 3] = True
    assert torch.equal(two, expect)
    three = khop_reach(EDGES, N, 3, symmetric=False)
    assert torch.equal(three, expect), "no new endpoint at three hops on this DAG"
    assert not three.diagonal().any(), "a DAG has no return walks"


def test_khop_reach_symmetric_returns_home() -> None:
    """Undirected, two hops reaches back to the start, so the diagonal lights up."""
    two = khop_reach(EDGES, N, 2, symmetric=True)
    assert two.diagonal().all()
    assert two[3, 1] and two[1, 3], "3-2-1 is a two-step path once edges are undirected"


def test_kl_target_default_is_directed_one_hop_without_self_loops() -> None:
    """Defaults reproduce the original target bit for bit."""
    model = _model(
        graph_reg_lambda=1.0,
        graph_regularization_config={
            "graph_reg_lambda": 1.0,
            "regularized_heads": {"regulatory": {"layer": 1, "head": 0, "lambda": 1.0}},
        },
    )
    assert model.adjacency_matrices is not None
    a = model.adjacency_matrices["regulatory"]
    expect = _dense(EDGES).float()
    expect = expect / (expect.sum(1, keepdim=True) + 1e-10)
    assert torch.equal(a, expect)
    assert a[1, 0] == 0.0, "directed: 1 does not attend back to 0"


def test_kl_target_symmetrize_and_hops() -> None:
    """Symmetrize adds the reverse edges; hops widens the target; diagonal stays zero."""
    sym = _model(
        graph_reg_lambda=1.0,
        graph_regularization_config={
            "graph_reg_lambda": 1.0,
            "symmetrize": True,
            "regularized_heads": {"regulatory": {"layer": 1, "head": 0, "lambda": 1.0}},
        },
    ).adjacency_matrices
    assert sym is not None
    assert sym["regulatory"][1, 0] > 0.0
    assert torch.allclose(sym["regulatory"].sum(1), torch.ones(N))

    two = _model(
        graph_reg_lambda=1.0,
        graph_regularization_config={
            "graph_reg_lambda": 1.0,
            "hops": 2,
            "regularized_heads": {"regulatory": {"layer": 1, "head": 0, "lambda": 1.0}},
        },
    ).adjacency_matrices
    assert two is not None
    t = two["regulatory"]
    assert t[0, 3] > 0.0 and t[1, 3] > 0.0, "two-step endpoints are in the target"
    assert not t.diagonal().any(), "no self-loop stored, so none appears at two hops"
    assert torch.allclose(t[0].sum(), torch.tensor(1.0))
    # gene 3 has no out-edges: an all-zero row, as at one hop
    assert t[3].sum() == 0.0


def test_kl_khop_target_keeps_the_stored_self_loops() -> None:
    """The trainer's cell graph gives every gene a self-loop (add_remaining_self_loops),
    and the one-hop target of every round-1 run held it. A k-hop or symmetric target
    keeps exactly those self entries: no more (the even walks home on an undirected
    graph), no fewer (a gene with no other edge keeps its self-only row).
    """
    edges = torch.cat([EDGES, torch.tensor([[1, 3], [1, 3]])], dim=1)  # loops on 1, 3
    cfg = {
        "graph_reg_lambda": 1.0,
        "regularized_heads": {"regulatory": {"layer": 1, "head": 0, "lambda": 1.0}},
    }
    one = _model(edges, graph_reg_lambda=1.0, graph_regularization_config=cfg)
    two = _model(
        edges, graph_reg_lambda=1.0, graph_regularization_config={**cfg, "hops": 2}
    )
    sym = _model(
        edges,
        graph_reg_lambda=1.0,
        graph_regularization_config={**cfg, "symmetrize": True},
    )
    targets: dict[str, torch.Tensor] = {}
    for name, model in (("one", one), ("two", two), ("sym", sym)):
        assert model.adjacency_matrices is not None
        targets[name] = model.adjacency_matrices["regulatory"]
        assert (targets[name].diagonal() > 0).tolist() == [False, True, False, True]
    # Directed: gene 3 has no out-edge, so its row is self only at one and two hops.
    for name in ("one", "two"):
        assert targets[name][3].sum() == 1.0 and targets[name][3, 3] == 1.0
    assert targets["two"][1, 3] > 0.0, "two-step endpoint beside the kept self-loop"
    ts = targets["sym"]
    assert ts[3, 2] > 0.0 and ts[3, 3] > 0.0, "reverse edge and self-loop share row 3"


def _head_mask(**mask_kwargs: Any) -> torch.Tensor:
    model = _model(
        attention_mask_config={
            "enabled": True,
            "layers": [1],
            "head_graphs": {0: "regulatory"},
            **mask_kwargs,
        }
    )
    assert model.attention_head_mask is not None
    return model.attention_head_mask[0, 1:, 1:]  # drop the CLS row and column


def test_mask_default_is_symmetric_one_hop_with_self_loops() -> None:
    """The default mask is what every earlier mask arm trained under."""
    m = _head_mask()
    expect = _dense(EDGES, symmetric=True) | torch.eye(N, dtype=torch.bool)
    assert torch.equal(m, expect)


def test_mask_directed_and_hops() -> None:
    """Directed drops the reverse edges; hops widens the support; self-loops kept."""
    d = _head_mask(symmetric=False)
    expect = _dense(EDGES) | torch.eye(N, dtype=torch.bool)
    assert torch.equal(d, expect)
    assert not d[1, 0], "1 may not attend to 0 under the directed mask"

    two = _head_mask(symmetric=False, hops=2)
    assert two[0, 3] and two[1, 3]
    assert two.diagonal().all()
    assert not two[3, 0], "still directed at two hops"

    two_sym = _head_mask(symmetric=True, hops=2)
    assert two_sym[3, 1], "undirected two hops reaches 3 -> 2 -> 1"


def test_mask_and_prior_share_the_k_hop_support() -> None:
    """Off the diagonal, the two-hop mask support and the two-hop target support agree."""
    mask = _head_mask(symmetric=False, hops=2)
    prior = _model(
        graph_reg_lambda=1.0,
        graph_regularization_config={
            "graph_reg_lambda": 1.0,
            "hops": 2,
            "regularized_heads": {"regulatory": {"layer": 1, "head": 0, "lambda": 1.0}},
        },
    ).adjacency_matrices
    assert prior is not None
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.equal(mask & off, (prior["regulatory"] > 0) & off)
