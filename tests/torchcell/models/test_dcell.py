# tests/torchcell/models/test_dcell.py
# [[tests.torchcell.models.test_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_dcell.py
"""``DCell`` on the three-term ontology fixture from ``tests/torchcell/conftest.py``.

Root term 0 has children 1 and 2 (stratum 1); term 1 annotates genes {0, 1}, term 2
genes {2, 3}, the root none. With ``min_subsystem_size=2`` and ``subsystem_ratio=0.5``
every subsystem has output dim max(2, ceil(0.5 * n_genes)) = 2. Input dims: a leaf sees
its 2 gene states (2); the root sees its children's outputs (2 + 2) plus the size-1
placeholder a gene-less term gets (1), so 5. Parameters: a subsystem is
Linear(in, out) + BatchNorm1d(out), i.e. in*out + out + 2*out, so the leaves have
2*2 + 2 + 4 = 10 each and the root 5*2 + 2 + 4 = 16 (36 in all); the three
Linear(2, 1) heads add 3 * 3 = 9; total 45.
"""

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.dcell import DCell, DCellSubsystem


def _model(dcell_graph: HeteroData, seed: int = 0) -> DCell:
    torch.manual_seed(seed)
    return DCell(dcell_graph, min_subsystem_size=2, subsystem_ratio=0.5, output_size=1)


def test_hierarchy_and_dimensions_follow_the_paper_formula(
    dcell_graph: HeteroData,
) -> None:
    """Child->parent map, per-term input dims {1: 2, 2: 2, 0: 5}, output dims all 2."""
    model = _model(dcell_graph)
    assert model.child_to_parents == {1: [0], 2: [0]}
    assert model.parent_to_children == {0: [1, 2]}
    assert model.term_to_genes == {1: [0, 1], 2: [2, 3]}
    assert model.term_input_dims == {0: 5, 1: 2, 2: 2}
    assert model.term_output_dims == {0: 2, 1: 2, 2: 2}
    assert set(model.subsystems.keys()) == {"0", "1", "2"}
    assert model.strata_order == [1, 0]


def test_parameter_count_is_45(dcell_graph: HeteroData) -> None:
    """36 subsystem parameters + 9 linear-head parameters."""
    counts = _model(dcell_graph).num_parameters
    assert counts == {
        "subsystems": 36,
        "dcell_linear": 9,
        "dcell": 45,
        "total": 45,
        "num_go_terms": 3,
        "num_subsystems": 3,
    }


def test_forward_returns_one_prediction_per_sample_and_every_term(
    dcell_graph: HeteroData, dcell_batch: HeteroData
) -> None:
    """Predictions are the root head's [batch] output; every term reports a linear output;
    term 1 sees the states of genes 0 and 1: [0, 1] for sample 0 (gene 0 knocked out) and
    [1, 1] for sample 1, and its activation is exactly its subsystem applied to them.
    """
    model = _model(dcell_graph)
    predictions, outputs = model(dcell_graph, dcell_batch)
    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()
    linear = outputs["linear_outputs"]
    assert set(linear) == {"GO:0", "GO:1", "GO:2", "GO:ROOT"}
    assert torch.equal(linear["GO:ROOT"], predictions)
    assert torch.equal(linear["GO:0"], predictions)
    activations = outputs["term_activations"]
    assert set(activations) == {0, 1, 2}
    states_term_1 = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    torch.testing.assert_close(activations[1], model.subsystems["1"](states_term_1))
    torch.testing.assert_close(
        linear["GO:1"], model.linear_heads["1"](activations[1]).squeeze(-1)
    )


def test_knocked_out_gene_states_reach_the_prediction(dcell_graph: HeteroData) -> None:
    """A batch differing only in gene 0's state gives a different prediction (eval mode)."""
    from tests.torchcell.conftest import make_dcell_batch  # noqa: PLC0415

    model = _model(dcell_graph).eval()
    with torch.no_grad():
        wild, _ = model(dcell_graph, make_dcell_batch([[], []]))
        knocked, _ = model(dcell_graph, make_dcell_batch([[0], []]))
    # sample 1 is untouched in both batches; sample 0 lost gene 0
    assert torch.equal(wild[1:], knocked[1:])
    assert not torch.equal(wild[:1], knocked[:1])


def test_backward_reaches_every_subsystem_and_forward_is_seeded(
    dcell_graph: HeteroData, dcell_batch: HeteroData
) -> None:
    """The prediction trains every subsystem and the root head; leaf heads only via DCellLoss."""
    model = _model(dcell_graph)
    predictions, _ = model(dcell_graph, dcell_batch)
    predictions.sum().backward()
    without_grad = sorted(n for n, p in model.named_parameters() if p.grad is None)
    # the auxiliary heads of terms 1 and 2 feed only the auxiliary loss, never the prediction
    assert without_grad == [
        "linear_heads.1.bias",
        "linear_heads.1.weight",
        "linear_heads.2.bias",
        "linear_heads.2.weight",
    ]
    again = _model(dcell_graph)
    again_predictions, _ = again(dcell_graph, dcell_batch)
    torch.testing.assert_close(again_predictions, predictions.detach())


def test_subsystem_is_linear_batchnorm_tanh_with_dcell_init() -> None:
    """Weights start uniform in [-0.001, 0.001]; a fresh eval subsystem is tanh((Wx + b) / sqrt(1 + eps))."""
    torch.manual_seed(0)
    subsystem = DCellSubsystem(3, 2)
    assert subsystem.linear.weight.abs().max().item() <= 0.001
    assert subsystem.linear.bias.abs().max().item() <= 0.001
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    subsystem.eval()  # running mean 0, running var 1: BatchNorm1d divides by sqrt(1 + eps)
    expected = torch.tanh(subsystem.linear(x) / (1 + subsystem.batch_norm.eps) ** 0.5)
    torch.testing.assert_close(subsystem(x), expected)
    assert subsystem(x).abs().max().item() < 1.0
    training = DCellSubsystem(3, 2)
    assert training(x).shape == (2, 2)  # train mode normalizes over the 2 rows


def test_default_subsystem_size_is_twenty(dcell_graph: HeteroData) -> None:
    """With the paper defaults every two-gene term gets max(20, ceil(0.3 * 2)) = 20 units."""
    torch.manual_seed(0)
    model = DCell(dcell_graph)
    assert model.term_output_dims == {0: 20, 1: 20, 2: 20}
    assert model.term_input_dims == {0: 41, 1: 2, 2: 2}


def test_stratum_to_terms_missing_the_root_raises() -> None:
    """A template whose stratum table has no stratum 0 builds but is rejected at forward time."""
    from tests.torchcell.conftest import (  # noqa: PLC0415
        make_dcell_batch,
        make_dcell_graph,
    )

    graph = make_dcell_graph()
    graph["gene_ontology"].stratum_to_terms = {1: torch.tensor([1, 2])}
    torch.manual_seed(0)
    model = DCell(graph, min_subsystem_size=2, subsystem_ratio=0.5)
    assert model.strata_order == [1]
    with pytest.raises(ValueError, match="No root terms found in stratum 0"):
        model(graph, make_dcell_batch([[0], []]))
