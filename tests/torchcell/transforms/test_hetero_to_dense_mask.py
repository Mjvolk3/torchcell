# tests/torchcell/transforms/test_hetero_to_dense_mask.py
# [[tests.torchcell.transforms.test_hetero_to_dense_mask]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/transforms/test_hetero_to_dense_mask.py
"""``HeteroToDenseMask`` on a hand-written three-gene graph with one hyperedge type.

Genes 0 -> 1 and 1 -> 2 are the only edges; reactions r0 -> {m0, m1} and r1 -> {m1} the
only incidences. Padding genes to 4 nodes must extend every gene tensor by one zero row,
mark the fourth gene as padding, and leave the sparse indices untouched.
"""

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask


def _graph() -> HeteroData:
    data = HeteroData()
    data["gene"].x = torch.tensor([[1.0], [2.0], [3.0]])
    data["gene"].embedding = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    data["reaction"].num_nodes = 2
    data["metabolite"].num_nodes = 2
    data["gene", "physical", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["reaction", "rmr", "metabolite"].hyperedge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 1]]
    )
    return data


def test_adjacency_mask_has_exactly_the_two_edges() -> None:
    """A 3x3 boolean mask with True at (0, 1) and (1, 2) only."""
    out = HeteroToDenseMask()(_graph())
    adj = out["gene", "physical", "gene"].adj_mask
    expected = torch.zeros(3, 3, dtype=torch.bool)
    expected[0, 1] = expected[1, 2] = True
    assert adj.dtype == torch.bool
    assert torch.equal(adj, expected)
    # the sparse index is kept for attribute lookups
    assert torch.equal(
        out["gene", "physical", "gene"].edge_index, torch.tensor([[0, 1], [1, 2]])
    )


def test_incidence_mask_from_the_hyperedge_index() -> None:
    """r0 touches m0 and m1, r1 touches m1: True at (0,0), (0,1), (1,1)."""
    out = HeteroToDenseMask()(_graph())
    inc = out["reaction", "rmr", "metabolite"].inc_mask
    expected = torch.tensor([[True, True], [False, True]])
    assert torch.equal(inc, expected)
    assert not hasattr(out["reaction", "rmr", "metabolite"], "adj_mask")


def test_padding_genes_to_four_extends_masks_and_every_gene_tensor() -> None:
    """num_nodes_dict={"gene": 4}: 4x4 adjacency, mask [T, T, T, F], one zero row appended."""
    out = HeteroToDenseMask(num_nodes_dict={"gene": 4})(_graph())
    assert out["gene", "physical", "gene"].adj_mask.shape == (4, 4)
    assert torch.equal(out["gene"].mask, torch.tensor([True, True, True, False]))
    torch.testing.assert_close(
        out["gene"].x, torch.tensor([[1.0], [2.0], [3.0], [0.0]])
    )
    torch.testing.assert_close(
        out["gene"].embedding,
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [0.0, 0.0]]),
    )
    # node types without an override keep their size and get an all-True mask
    assert torch.equal(out["reaction"].mask, torch.tensor([True, True]))


def test_shrinking_below_the_original_node_count_is_rejected() -> None:
    """The transform never drops nodes: a smaller target fails the size assertion."""
    with pytest.raises(AssertionError):
        HeteroToDenseMask(num_nodes_dict={"gene": 2})(_graph())


def test_repr_names_the_overrides() -> None:
    """The repr is bare without overrides and carries the dict with them."""
    assert repr(HeteroToDenseMask()) == "HeteroToDenseMask()"
    assert (
        repr(HeteroToDenseMask({"gene": 4}))
        == "HeteroToDenseMask(num_nodes_dict={'gene': 4})"
    )
