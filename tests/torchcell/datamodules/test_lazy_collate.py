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


# ---------------------------------------------------------------------------
# 2026.10.06 - Phase 21: collate branches and verify_batch_structure
#
# Fixtures reuse ``_sample`` (4 genes, edges 0->1 and 1->2, mask [True, False]) and add
# a gene->reaction hyperedge relation with 2 reactions per sample, so offsets differ
# per type: sample i shifts gene indices by 4 i and reaction indices by 2 i.
# ---------------------------------------------------------------------------

import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from torchcell.datamodules.lazy_collate import verify_batch_structure  # noqa: E402

GPR = ("gene", "gpr", "reaction")


def _with_reactions(perturbed: list[int], label: str = "gpr") -> HeteroData:
    data = _sample(perturbed)
    data["reaction"].num_nodes = 2
    data["reaction"].names = ["R1", "R2"]
    data[GPR].hyperedge_index = torch.tensor([[0, 3], [0, 1]])
    data[GPR].mask = torch.tensor([True, True])
    data[GPR].stoich = torch.tensor([1.0, -2.0])
    data[GPR].label = label
    return data


def test_empty_list_is_refused() -> None:
    """``lazy_collate_hetero([])`` raises the exact ValueError."""
    with pytest.raises(ValueError, match=re.escape("Cannot collate empty data_list")):
        lazy_collate_hetero([])


def test_hyperedges_offset_by_their_own_types_and_extra_attributes_are_kept() -> None:
    """Row 0 (genes) shifts by 4 per sample and row 1 (reactions) by 2: [[0, 3, 4, 7],
    [0, 1, 2, 3]]. Tensor edge attributes are concatenated, non-tensors become a list,
    non-tensor node attributes stay a list per sample, num_nodes sums per type.
    """
    out = lazy_collate_hetero([_with_reactions([1]), _with_reactions([2], "other")])
    assert out[GPR].hyperedge_index.tolist() == [[0, 3, 4, 7], [0, 1, 2, 3]]
    assert "edge_index" not in out[GPR]
    assert out[GPR].mask.tolist() == [True] * 4
    assert out[GPR].stoich.tolist() == [1.0, -2.0, 1.0, -2.0]
    assert out[GPR].label == ["gpr", "other"]
    assert out["reaction"].names == [["R1", "R2"], ["R1", "R2"]]
    assert out["reaction"].ptr.tolist() == [0, 2, 4]
    assert out["reaction"].num_nodes == 4 and out["gene"].num_nodes == 8
    assert verify_batch_structure(out, expected_graphs=2) is True


def test_mixed_edge_attribute_becomes_a_list_and_a_missing_one_crashes() -> None:
    """A key that is a tensor in one sample and a string in the other is not all-tensor,
    so it is stored as the per-sample list [tensor([7]), 'gpr'] (line 142). A key that
    sample 0 has and sample 1 lacks is read from every sample (line 133) and raises
    KeyError: samples must carry identical edge keys.
    """
    first, second = _with_reactions([1]), _with_reactions([2])
    first[GPR].label = torch.tensor([7])
    out = lazy_collate_hetero([first, second])
    assert len(out[GPR].label) == 2
    assert torch.equal(out[GPR].label[0], torch.tensor([7]))
    assert out[GPR].label[1] == "gpr"
    del second[GPR].label
    with pytest.raises(KeyError, match=re.escape("'label'")):
        lazy_collate_hetero([first, second])


def test_edge_type_without_indices_loses_its_mask() -> None:
    """An edge store with a mask but no edge_index is skipped before the mask is
    collected (line 104), so the batch carries no mask and no store for it.
    """
    samples = _samples()
    for data in samples:
        data[("gene", "bare", "gene")].mask = torch.tensor([True, True])
    out = lazy_collate_hetero(samples)
    assert ("gene", "bare", "gene") not in out.edge_types
    assert out[EDGE].mask.tolist() == [True, False] * 3


def test_non_lazy_inputs_go_to_pyg() -> None:
    """HeteroData without edge masks and plain Data use PyG's Collater unchanged."""
    plain = []
    for data in _samples():
        del data[EDGE].mask
        plain.append(data)
    out = LazyCollater(plain)(plain)
    reference = Batch.from_data_list(plain)
    assert torch.equal(out[EDGE].edge_index, reference[EDGE].edge_index)
    assert torch.equal(out["gene"].ptr, reference["gene"].ptr)
    graphs = [Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0], [1]]))] * 2
    homo = LazyCollater(graphs)(graphs)
    assert homo.edge_index.tolist() == [[0, 2], [1, 3]]
    assert homo.batch.tolist() == [0, 0, 1, 1]


def _check(
    batch: HeteroData, expected: int, capsys: pytest.CaptureFixture[str]
) -> tuple[bool, str]:
    ok = verify_batch_structure(batch, expected_graphs=expected)
    return ok, capsys.readouterr().out


def test_verify_reports_each_violation_by_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each check prints its exact message and returns False; a valid batch is True."""
    prefix = "Batch structure verification failed: "
    good = lazy_collate_hetero(_samples())
    assert _check(good, 3, capsys) == (True, "")
    assert _check(good, 2, capsys) == (False, prefix + "Batch vector max 2 != 1\n")

    short = lazy_collate_hetero(_samples())
    short["gene"].num_nodes = 11
    assert _check(short, 3, capsys) == (
        False,
        prefix + "Batch vector size 12 != num_nodes 11\n",
    )

    src = lazy_collate_hetero(_samples())
    src[EDGE].edge_index = torch.tensor([[12], [0]])
    assert _check(src, 3, capsys) == (False, prefix + "Source index 12 >= 12\n")

    dst = lazy_collate_hetero(_samples())
    dst[EDGE].edge_index = torch.tensor([[0], [12]])
    assert _check(dst, 3, capsys) == (False, prefix + "Dest index 12 >= 12\n")

    mask = lazy_collate_hetero(_samples())
    mask[EDGE].mask = torch.tensor([True])
    assert _check(mask, 3, capsys) == (False, prefix + "Mask size 1 != num_edges 6\n")


def test_verify_accepts_hyperedges_and_skips_index_free_stores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """hyperedge_index is bounds-checked like edge_index (a reaction index 4 >= 4
    fails); a store with no index at all is skipped.
    """
    out = lazy_collate_hetero([_with_reactions([1]), _with_reactions([2])])
    out[("gene", "bare", "gene")].weight = torch.ones(1)
    assert _check(out, 2, capsys) == (True, "")
    out[GPR].hyperedge_index = torch.tensor([[0], [4]])
    out[GPR].mask = torch.tensor([True])
    assert _check(out, 2, capsys) == (
        False,
        "Batch structure verification failed: Dest index 4 >= 4\n",
    )


def test_verify_does_not_detect_missing_offsets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Finding: check 1 of the docstring ("Edge indices are properly offset (no overlap
    between graphs)") is not implemented; only bounds are checked.

    A 3-graph batch whose edges all point into graph 0 ([[0, 1] * 3, [1, 2] * 3], the
    shared index never offset) passes with True. Reach: latent; verify_batch_structure has no
    caller in torchcell/ or experiments/. Pinned until the verifier checks that
    each graph's edges fall inside its own ptr range.
    """
    batch = lazy_collate_hetero(_samples())
    batch[EDGE].edge_index = torch.tensor([[0, 1] * 3, [1, 2] * 3])
    assert _check(batch, 3, capsys) == (True, "")


def test_verify_raises_on_an_edge_type_with_no_edges() -> None:
    """Finding: a relation with zero edges makes the verifier raise instead of answer.

    ``edge_index[0].max()`` on an empty tensor raises RuntimeError (lazy_collate.py:268),
    which the ``except AssertionError`` does not catch. Reach: latent; verify_batch_structure has no
    caller in torchcell/ or experiments/. Pinned until empty relations are
    skipped or handled.
    """
    batch = lazy_collate_hetero(_samples())
    batch[EDGE].edge_index = torch.zeros(2, 0, dtype=torch.long)
    del batch[EDGE].mask
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "max(): Expected reduction dim to be specified for input.numel() == 0. "
            "Specify the reduction dim with the 'dim' argument."
        ),
    ):
        verify_batch_structure(batch, expected_graphs=3)


def test_verify_is_a_no_op_under_python_optimize() -> None:
    """Finding: every check is an ``assert``, so ``python -O`` strips them all and the
    verifier returns True for a batch whose batch vector is wrong.

    Reproduced in a subprocess with -O and expected_graphs=5 for a 3-graph batch (False
    without -O, above). Reach: latent; verify_batch_structure has no
    caller in torchcell/ or experiments/. Pinned until the checks raise or return explicitly.
    """
    code = (
        "import torch\n"
        "from torch_geometric.data import HeteroData\n"
        "from torchcell.datamodules.lazy_collate import verify_batch_structure\n"
        "b = HeteroData()\n"
        "b['gene'].batch = torch.tensor([0, 1, 2])\n"
        "b['gene'].num_nodes = 3\n"
        "print(verify_batch_structure(b, expected_graphs=5))\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout == "True\n"
