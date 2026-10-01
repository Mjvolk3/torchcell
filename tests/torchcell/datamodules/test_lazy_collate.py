# tests/torchcell/datamodules/test_lazy_collate.py
# [[tests.torchcell.datamodules.test_lazy_collate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodules/test_lazy_collate.py
"""``lazy_collate_hetero`` / ``LazyCollater`` write PyG's batch bookkeeping (issue #572).

Each sample is a 4-gene lazy graph (two edges, an edge ``mask``) perturbing a different
gene set: {1}, {2, 3}, and nothing. The 006 lazy model reads ``gene.ptr`` and
``perturbation_indices_ptr``; before the fix the lazy collate built neither and ignored
``follow_batch``, so the follow_batch list a 006 script passed reached nothing. The
expected values are what PyG's own ``Batch.from_data_list`` writes for the same samples.
"""

import torch
from torch_geometric.data import Batch, HeteroData

from torchcell.datamodules.lazy_collate import LazyCollater, lazy_collate_hetero

EDGE = ("gene", "physical", "gene")
FOLLOW = ["x", "x_pert", "perturbation_indices"]


def _sample(perturbed: list[int]) -> HeteroData:
    data = HeteroData()
    data["gene"].x = torch.arange(8, dtype=torch.float).view(4, 2)
    data["gene"].num_nodes = 4
    data["gene"].perturbation_indices = torch.tensor(perturbed, dtype=torch.long)
    pert_mask = torch.zeros(4, dtype=torch.bool)
    pert_mask[perturbed] = True
    data["gene"].pert_mask = pert_mask
    data[EDGE].edge_index = torch.tensor([[0, 1], [1, 2]])
    data[EDGE].mask = torch.tensor([True, False])
    return data


def _samples() -> list[HeteroData]:
    return [_sample([1]), _sample([2, 3]), _sample([])]


def test_follow_batch_and_ptr_match_pyg() -> None:
    """Exact vectors, equal to PyG's: the trailing sample with no perturbation counts."""
    out = LazyCollater(_samples(), follow_batch=FOLLOW)(_samples())
    gene = out["gene"]
    assert out.num_graphs == 3
    assert gene.ptr.tolist() == [0, 4, 8, 12]
    assert gene.batch.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert gene.perturbation_indices.tolist() == [1, 2, 3]
    assert gene.perturbation_indices_batch.tolist() == [0, 1, 1]
    assert gene.perturbation_indices_ptr.tolist() == [0, 1, 3, 3]
    assert gene.x_batch.tolist() == gene.batch.tolist()
    assert gene.x_ptr.tolist() == [0, 4, 8, 12]
    assert (
        "x_pert_batch" not in gene
    )  # no node type carries x_pert: skipped, as PyG does
    assert out[EDGE].edge_index.tolist() == [[0, 1, 4, 5, 8, 9], [1, 2, 5, 6, 9, 10]]
    assert out[EDGE].mask.tolist() == [True, False] * 3

    reference = Batch.from_data_list(_samples(), follow_batch=FOLLOW)
    for key in (
        "ptr",
        "batch",
        "perturbation_indices_batch",
        "perturbation_indices_ptr",
        "x_batch",
        "x_ptr",
    ):
        assert torch.equal(gene[key], reference["gene"][key]), key
    assert torch.equal(out[EDGE].edge_index, reference[EDGE].edge_index)
    assert out.num_graphs == reference.num_graphs


def test_without_follow_batch_only_node_bookkeeping_is_built() -> None:
    """``follow_batch=None`` gives ``batch`` and ``ptr`` but no ``<key>_batch``."""
    out = LazyCollater(_samples())(_samples())
    assert sorted(out["gene"].keys()) == [
        "batch",
        "num_nodes",
        "pert_mask",
        "perturbation_indices",
        "ptr",
        "x",
    ]
    assert out["gene"].ptr.tolist() == [0, 4, 8, 12]


def test_one_sample_is_collated_like_any_other() -> None:
    """A one-sample list is no longer returned as is: it gets ``batch``, ``ptr``, follows."""
    out = lazy_collate_hetero([_sample([2, 3])], FOLLOW)
    gene = out["gene"]
    assert out.num_graphs == 1
    assert gene.ptr.tolist() == [0, 4]
    assert gene.batch.tolist() == [0, 0, 0, 0]
    assert gene.perturbation_indices_batch.tolist() == [0, 0]
    assert gene.perturbation_indices_ptr.tolist() == [0, 2]
    assert out[EDGE].edge_index.tolist() == [[0, 1], [1, 2]]
