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


def test_root_input_is_child_activations_then_the_gene_placeholder(
    dcell_graph: HeteroData, dcell_batch: HeteroData
) -> None:
    """The root's [2, 5] input is cat(act_1, act_2, zeros[2, 1]) in child order, and its
    activation is exactly its subsystem on that input. Term 2 sees genes 2 and 3: [1, 1]
    for sample 0 and [0, 0] for sample 1 (both knocked out); the gene-less root sees [0].
    """
    model = _model(dcell_graph)
    _, outputs = model(dcell_graph, dcell_batch)
    activations = outputs["term_activations"]
    assert model._extract_gene_states_for_term(2, dcell_batch).tolist() == [
        [1.0, 1.0],
        [0.0, 0.0],
    ]
    assert model._extract_gene_states_for_term(0, dcell_batch).tolist() == [
        [0.0],
        [0.0],
    ]
    root_input = torch.cat([activations[1], activations[2], torch.zeros(2, 1)], dim=1)
    assert root_input.shape == (2, 5)
    torch.testing.assert_close(activations[0], model.subsystems["0"](root_input))
    torch.testing.assert_close(
        model._prepare_term_input(0, dcell_batch, activations), root_input
    )


def test_list_valued_stratum_to_terms_predicts_identically(
    dcell_batch: HeteroData,
) -> None:
    """Python-int term ids take the non-tensor branch and give the same predictions."""
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    tensor_graph = make_dcell_graph()
    list_graph = make_dcell_graph()
    list_graph["gene_ontology"].stratum_to_terms = {0: [0], 1: [1, 2]}
    predictions, _ = _model(tensor_graph)(tensor_graph, dcell_batch)
    list_predictions, outputs = _model(list_graph)(list_graph, dcell_batch)
    assert torch.equal(list_predictions, predictions)
    assert set(outputs["linear_outputs"]) == {"GO:0", "GO:1", "GO:2", "GO:ROOT"}


def test_empty_root_stratum_raises_at_forward(dcell_batch: HeteroData) -> None:
    """Stratum 0 present but empty builds, then fails with the root-terms-empty message."""
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    graph = make_dcell_graph()
    graph["gene_ontology"].stratum_to_terms = {
        0: torch.tensor([], dtype=torch.long),
        1: torch.tensor([1, 2]),
    }
    model = _model(graph)
    assert model.strata_order == [1, 0]
    with pytest.raises(ValueError, match="Root terms tensor is empty"):
        model(graph, dcell_batch)


def test_stratum_removed_after_construction_raises_at_forward(
    dcell_graph: HeteroData, dcell_batch: HeteroData
) -> None:
    """``strata_order`` is fixed at init, so a stratum missing from the shared dict is a bug."""
    model = _model(dcell_graph)
    del model.stratum_to_terms[1]
    assert model.strata_order == [1, 0]
    with pytest.raises(ValueError, match="Stratum 1 not found in stratum_to_terms"):
        model(dcell_graph, dcell_batch)


def test_without_child_edges_a_gene_less_root_still_runs_on_its_placeholder(
    dcell_batch: HeteroData,
) -> None:
    """Finding: the "no children and no genes" ValueError (``dcell.py:386``) is unreachable.

    ``_extract_gene_states_for_term`` returns a [batch, 1] zero placeholder for a
    gene-less term, so the root with no edges gets input dim 1 and activation
    ``subsystem_0(zeros[2, 1])``. Parameters: Linear(1, 2) + BatchNorm1d(2) = 2 + 2 + 4 = 8
    for the root, 10 per leaf, 9 for the heads: 37.
    """
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    graph = make_dcell_graph()
    del graph["gene_ontology", "is_child_of", "gene_ontology"]
    model = _model(graph)
    assert model.child_to_parents == {}
    assert model.parent_to_children == {}
    assert model.term_input_dims == {0: 1, 1: 2, 2: 2}
    assert model.num_parameters["subsystems"] == 28
    assert model.num_parameters["total"] == 37
    predictions, outputs = model(graph, dcell_batch)
    assert predictions.shape == (2,)
    torch.testing.assert_close(
        outputs["term_activations"][0], model.subsystems["0"](torch.zeros(2, 1))
    )


def test_subsystem_in_training_mode_normalizes_over_the_batch() -> None:
    """Train-mode output is tanh((z - mean_0 z) / sqrt(var_0 z + eps)) with z = Wx + b."""
    torch.manual_seed(0)
    subsystem = DCellSubsystem(3, 2)
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    z = subsystem.linear(x)
    expected = torch.tanh(
        (z - z.mean(0))
        / torch.sqrt(z.var(0, unbiased=False) + subsystem.batch_norm.eps)
    )
    torch.testing.assert_close(subsystem(x), expected)
    # two rows normalized over the batch are exact negatives of each other
    torch.testing.assert_close(subsystem(x)[0], -subsystem(x)[1])


def test_two_seeded_constructions_share_every_parameter(
    dcell_graph: HeteroData,
) -> None:
    """Same seed, same state dict, key by key."""
    first = _model(dcell_graph, seed=7)
    second = _model(dcell_graph, seed=7)
    assert list(first.state_dict()) == list(second.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name
    # per term: linear W, b + BatchNorm w, b, running_mean, running_var, num_batches_tracked
    # (7) and the head's W, b (2)
    assert len(first.state_dict()) == 3 * 7 + 3 * 2
