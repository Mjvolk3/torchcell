# tests/torchcell/transforms/test_hetero_to_dense_mask.py
# [[tests.torchcell.transforms.test_hetero_to_dense_mask]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/transforms/test_hetero_to_dense_mask.py
"""``HeteroToDenseMask`` on a hand-written three-gene graph with one hyperedge type.

Genes 0 -> 1 and 1 -> 2 are the only edges; reactions r0 -> {m0, m1} and r1 -> {m1} the
only incidences. Padding genes to 4 nodes must extend every gene tensor by one zero row,
mark the fourth gene as padding, and leave the sparse indices untouched.

2026.09.30, Phase 16: a second hand-written graph, ``_edge_case_graph``, has the same
three genes with ``pos = [[0, 1], [2, 3], [4, 5]]``, a per-gene ``[3, 3]`` matrix
``pair = arange(9).view(3, 3)``, a length-5 tensor ``other = arange(5)``, a Python list
``names`` and a ``node_ids`` tensor; a ``physical`` edge list ``[[0, 7], [1, 1]]`` (the
second column names gene 7, which does not exist), a ``dead`` edge list ``[[7], [8]]``,
a hyperedge list ``[[5], [0]]`` naming reaction 5 of 2, and an edge type carrying only an
``edge_attr``. The validity filter is ``index < ORIGINAL count`` on both endpoints, so
with genes padded to 4 the physical mask holds only (0, 1), the dead mask and the
incidence mask are all False, and the attribute-only edge type gets neither mask. The
node-attribute padding pads ``pos`` by ``4 - 3 = 1`` zero row and the per-gene ``[3, 3]``
``pair`` (listed in ``square_attrs``) by one zero row AND one zero column, to
``[4, 4]``, and leaves everything else (``other`` of length 5, the list, and
``node_ids`` by name) as it was (issue #538 fixed the row-only padding and the
padded-count filter).

2026.09.30, issue #570: column padding is opt-in by name. ``square_attrs`` lists the
per-node ``[N, N]`` attributes; every other node tensor with ``size(0) == N`` is padded
on its rows only, so a ``[3, 3]`` feature matrix that is NOT listed comes back
``[4, 3]``. A listed name that is absent or not ``[N, N]``, or a node type the data
lacks, raises ``ValueError`` in ``forward``; a reserved name (``x``, ``pos``, ``mask``,
``num_nodes``, ``node_ids``, a leading underscore) raises at construction.
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
    assert (
        repr(HeteroToDenseMask({"gene": 4}, square_attrs={"gene": ["pair"]}))
        == "HeteroToDenseMask(num_nodes_dict={'gene': 4}, square_attrs={'gene': ['pair']})"
    )
    assert (
        repr(HeteroToDenseMask(square_attrs={"gene": ["pair"]}))
        == "HeteroToDenseMask(square_attrs={'gene': ['pair']})"
    )


def _edge_case_graph() -> HeteroData:
    data = HeteroData()
    data["gene"].x = torch.tensor([[1.0], [2.0], [3.0]])
    data["gene"].pos = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    data["gene"].pair = torch.arange(9.0).view(3, 3)
    data["gene"].other = torch.arange(5)
    data["gene"].names = ["a", "b", "c"]
    data["gene"].node_ids = torch.tensor([7, 8, 9])
    data["reaction"].num_nodes = 2
    data["metabolite"].num_nodes = 2
    data["gene", "physical", "gene"].edge_index = torch.tensor([[0, 7], [1, 1]])
    data["gene", "dead", "gene"].edge_index = torch.tensor([[7], [8]])
    data["reaction", "rmr", "metabolite"].hyperedge_index = torch.tensor([[5], [0]])
    data["gene", "attr", "gene"].edge_attr = torch.tensor([1.0, 2.0])
    return data


def test_out_of_range_indices_are_dropped_from_the_masks_but_kept_in_the_index() -> (
    None
):
    """Gene 7 and reaction 5 do not exist: only (0, 1) survives; the all-invalid lists
    give all-False masks, and every sparse index is returned exactly as given.
    """
    out = HeteroToDenseMask(num_nodes_dict={"gene": 4})(_edge_case_graph())
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[0, 1] = True
    assert torch.equal(out["gene", "physical", "gene"].adj_mask, expected)
    assert torch.equal(
        out["gene", "dead", "gene"].adj_mask, torch.zeros(4, 4, dtype=torch.bool)
    )
    assert torch.equal(
        out["reaction", "rmr", "metabolite"].inc_mask,
        torch.zeros(2, 2, dtype=torch.bool),
    )
    assert torch.equal(
        out["gene", "physical", "gene"].edge_index, torch.tensor([[0, 7], [1, 1]])
    )
    assert torch.equal(
        out["reaction", "rmr", "metabolite"].hyperedge_index, torch.tensor([[5], [0]])
    )


def test_an_edge_type_without_an_index_gets_no_mask() -> None:
    """An edge store holding only ``edge_attr`` is left with exactly that attribute."""
    out = HeteroToDenseMask()(_edge_case_graph())
    store = out["gene", "attr", "gene"]
    assert list(store.keys()) == ["edge_attr"]
    torch.testing.assert_close(store.edge_attr, torch.tensor([1.0, 2.0]))


def test_padding_extends_pos_and_node_sized_tensors_and_leaves_the_rest() -> None:
    """``pos`` gains one zero row; the gene-by-gene ``pair``, listed in
    ``square_attrs``, gains a zero row and a zero column, ``[3, 3]`` to ``[4, 4]``;
    ``other`` (length 5), the list of names and ``node_ids`` (skipped by name) are
    unchanged.

    ``pair`` used to be padded on its rows only and came back ``[4, 3]`` (issue #538).
    """
    out = HeteroToDenseMask(
        num_nodes_dict={"gene": 4}, square_attrs={"gene": ["pair"]}
    )(_edge_case_graph())
    gene = out["gene"]
    torch.testing.assert_close(
        gene.pos, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 0.0]])
    )
    torch.testing.assert_close(
        gene.pair,
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0],
                [3.0, 4.0, 5.0, 0.0],
                [6.0, 7.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(gene.other, torch.arange(5))
    assert gene.names == ["a", "b", "c"]
    assert torch.equal(gene.node_ids, torch.tensor([7, 8, 9]))


def test_an_edge_touching_the_padding_row_is_dropped_from_the_mask() -> None:
    """Validity is checked against the ORIGINAL node count on both endpoints.

    With three genes padded to four, ``3 -> 0`` and ``0 -> 3`` each name the padding
    gene 3, so neither reaches ``adj_mask``; ``0 -> 1`` does. The first used to pass
    ``3 < 4`` and set ``adj_mask[3, 0]`` on a row ``mask[3]`` marks as padding (issue
    #538). The index itself keeps all three columns. The same filter holds for an
    incidence: reaction 2 of 2 padded to 3 is dropped.
    """
    data = _graph()
    data["gene", "physical", "gene"].edge_index = torch.tensor([[3, 0, 0], [0, 3, 1]])
    data["reaction", "rmr", "metabolite"].hyperedge_index = torch.tensor(
        [[2, 1], [0, 1]]
    )
    out = HeteroToDenseMask(num_nodes_dict={"gene": 4, "reaction": 3})(data)
    expected = torch.zeros(4, 4, dtype=torch.bool)
    expected[0, 1] = True
    assert torch.equal(out["gene", "physical", "gene"].adj_mask, expected)
    assert out["gene", "physical", "gene"].edge_index.tolist() == [[3, 0, 0], [0, 3, 1]]
    assert torch.equal(out["gene"].mask, torch.tensor([True, True, True, False]))
    incidence = torch.zeros(3, 2, dtype=torch.bool)
    incidence[1, 1] = True
    assert torch.equal(out["reaction", "rmr", "metabolite"].inc_mask, incidence)


def test_an_unlisted_feature_dimension_equal_to_n_is_padded_on_rows_only() -> None:
    """A ``[3, 3]`` per-gene feature not named in ``square_attrs`` gains one zero row
    and keeps its 3 columns: ``[4, 3]``. The shape guess used to pad it to ``[4, 4]``
    because its feature dimension equals the gene count (issue #570).
    """
    out = HeteroToDenseMask(num_nodes_dict={"gene": 4})(_edge_case_graph())
    torch.testing.assert_close(
        out["gene"].pair,
        torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
        ),
    )


def test_square_attrs_apply_only_to_their_node_type() -> None:
    """Listing ``pair`` under ``reaction`` leaves the gene ``pair`` row-padded only."""
    data = _edge_case_graph()
    data["reaction"].pair = torch.ones(2, 2)
    out = HeteroToDenseMask(
        num_nodes_dict={"gene": 4, "reaction": 3}, square_attrs={"reaction": ["pair"]}
    )(data)
    assert tuple(out["gene"].pair.shape) == (4, 3)
    torch.testing.assert_close(
        out["reaction"].pair,
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_a_listed_attribute_that_is_not_square_is_refused_by_name() -> None:
    """``pair`` listed on a ``[3, 2]`` tensor raises with its shape; ``missing``, which
    the store does not hold, raises naming the store.
    """
    with pytest.raises(
        ValueError,
        match=r"^square attribute gene\.pair must be a tensor of shape \[3, 3, \.\.\.\], "
        r"got \[3, 2\]$",
    ):
        HeteroToDenseMask(square_attrs={"gene": ["pair"]})(_narrow_pair_graph())
    with pytest.raises(
        ValueError,
        match=r"^square attribute gene\.missing is missing from the gene store$",
    ):
        HeteroToDenseMask(square_attrs={"gene": ["missing"]})(_edge_case_graph())


def _narrow_pair_graph() -> HeteroData:
    """``_edge_case_graph`` with ``pair`` replaced by a ``[3, 2]`` tensor."""
    data = _edge_case_graph()
    data["gene"].pair = torch.arange(6.0).view(3, 2)
    return data


@pytest.mark.parametrize("name", ["x", "pos", "mask", "num_nodes", "node_ids", "_y"])
def test_a_reserved_name_in_square_attrs_is_refused_at_construction(name: str) -> None:
    """The transform pads ``x`` and ``pos`` on rows only and skips the other reserved
    names, so listing one would silently not be squared; it is refused up front.
    """
    with pytest.raises(ValueError) as info:
        HeteroToDenseMask(square_attrs={"gene": ["pair", name]})
    assert str(info.value) == (
        f"square attribute gene.{name} is reserved: the transform handles x, pos, "
        "mask, num_nodes, node_ids and underscore names itself, never as a square "
        "matrix"
    )


def test_an_unknown_node_type_in_square_attrs_is_refused() -> None:
    """``genes`` is not a node type of the data (``gene`` is): refused by name."""
    with pytest.raises(ValueError) as info:
        HeteroToDenseMask(square_attrs={"genes": ["pair"]})(_edge_case_graph())
    assert str(info.value) == (
        "square_attrs names node types ['genes'] absent from the data; its node "
        "types are ['gene', 'metabolite', 'reaction']"
    )
