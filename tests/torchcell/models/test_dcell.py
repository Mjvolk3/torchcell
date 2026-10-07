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

Two more fixtures are built in this file.

* ``_dag_graph``: the same four genes and annotations, but a three-level DAG in which
  term 2 is a child of BOTH term 1 and the root (edges child -> parent (1, 0), (2, 0),
  (2, 1)), strata {0: [0], 1: [1], 2: [2]} and propagated gene counts [5, 4, 2]. With
  ``min_subsystem_size=1`` and the paper's ``subsystem_ratio=0.3`` the output rule
  max(1, ceil(0.3 * n)) gives ceil(1.5) = 2, ceil(1.2) = 2, ceil(0.6) = 1 (floor would
  give 1, 1, 0 and round 2, 1, 1, so only ceil yields {0: 2, 1: 2, 2: 1}). Input dims:
  term 2 sees its 2 genes (2); term 1 sees child 2 (1) + its 2 genes = 3; the root sees
  children 1 and 2 (2 + 1) + the gene-less placeholder (1) = 4. Parameters
  out * (in + 3) per subsystem: 1 * 5 + 2 * 6 + 2 * 7 = 31; heads out + 1: 2 + 3 + 3 = 8;
  total 39.
* ``_processed_batch``: the real data path. ``to_cell_data`` on a four-gene GO DAG
  (GO:a {g0, g1}, GO:b {g2, g3}, GO:root {g0..g3}; node order GO:a 0, GO:b 1, GO:root 2,
  so the root is term 2 and it carries four gene rows of its own), then
  ``DCellGraphProcessor`` per record and ``Batch.from_data_list(follow_batch=
  ["go_gene_strata_state"])``, which is what writes the ``go_gene_strata_state_ptr`` the
  model reads. Output dims max(2, ceil(0.5 * n)) = 2 for counts [2, 2, 4]; root input
  2 + 2 + 4 = 8; parameters 2 * (2 + 3) * 2 + 2 * (8 + 3) = 42, heads 3 * 3 = 9, total 51.
"""

import re
import sys
import types
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from sortedcontainers import SortedDict
from torch_geometric.data import Batch, HeteroData

import torchcell.models.dcell as dcell_module
from torchcell.data.cell_data import to_cell_data
from torchcell.data.graph_processor import DCellGraphProcessor
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.losses.dcell import DCellLoss
from torchcell.models.dcell import DCell, DCellSubsystem
from torchcell.sequence import GeneSet


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


def _dag_graph() -> HeteroData:
    """Term 2 is a child of both term 1 and the root; see the module docstring."""
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    graph = make_dcell_graph()
    go = graph["gene_ontology"]
    go.strata = torch.tensor([0, 1, 2])
    go.stratum_to_terms = {
        0: torch.tensor([0]),
        1: torch.tensor([1]),
        2: torch.tensor([2]),
    }
    go.term_gene_counts = torch.tensor([5, 4, 2])
    graph["gene_ontology", "is_child_of", "gene_ontology"].edge_index = torch.tensor(
        [[1, 2, 2], [0, 0, 1]]
    )
    return graph


def test_a_child_with_two_parents_feeds_both_and_sizes_follow_ceil() -> None:
    """Multi-parent DAG: 2 -> [0, 1]; dims and the 39-parameter total from the docstring."""
    torch.manual_seed(0)
    model = DCell(_dag_graph(), min_subsystem_size=1, subsystem_ratio=0.3)
    assert model.child_to_parents == {1: [0], 2: [0, 1]}
    assert model.parent_to_children == {0: [1, 2], 1: [2]}
    assert model.strata_order == [2, 1, 0]
    assert model.term_output_dims == {0: 2, 1: 2, 2: 1}
    assert model.term_input_dims == {0: 4, 1: 3, 2: 2}
    assert model.num_parameters == {
        "subsystems": 31,
        "dcell_linear": 8,
        "dcell": 39,
        "total": 39,
        "num_go_terms": 3,
        "num_subsystems": 3,
    }


def test_the_shared_child_activation_enters_both_parent_inputs(
    dcell_batch: HeteroData,
) -> None:
    """act_1 = sub_1(cat(act_2, genes of term 1)); act_0 = sub_0(cat(act_1, act_2, 0)).

    Genes of term 1 for the fixture batch (gene 0 knocked out in sample 0): [[0, 1], [1, 1]].
    """
    torch.manual_seed(0)
    model = DCell(_dag_graph(), min_subsystem_size=1, subsystem_ratio=0.3)
    predictions, outputs = model(_dag_graph(), dcell_batch)
    act = outputs["term_activations"]
    assert [tuple(act[t].shape) for t in (0, 1, 2)] == [(2, 2), (2, 2), (2, 1)]
    genes_1 = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    torch.testing.assert_close(
        act[1], model.subsystems["1"](torch.cat([act[2], genes_1], dim=1))
    )
    torch.testing.assert_close(
        act[0], model.subsystems["0"](torch.cat([act[1], act[2], torch.zeros(2, 1)], 1))
    )
    torch.testing.assert_close(predictions, model.linear_heads["0"](act[0]).squeeze(-1))


def test_a_child_scheduled_after_its_parent_is_skipped_and_the_shapes_break(
    dcell_batch: HeteroData,
) -> None:
    """Finding: ``_prepare_term_input`` drops a child with no activation yet (dcell.py:373)
    instead of failing, while ``_calculate_input_dim`` counted it; a hierarchy whose strata
    put a child above its parent therefore reaches the parent's Linear with too few
    columns. Here term 2 is declared a child of term 1 (edge (2, 1)) but both sit in
    stratum 1 and the parent runs first: term 1's input is 2 gene states against an
    expected 2 + 2 = 4 (child 2's output plus its own genes).
    """
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    graph = make_dcell_graph()
    graph["gene_ontology", "is_child_of", "gene_ontology"].edge_index = torch.tensor(
        [[1, 2, 2], [0, 0, 1]]
    )
    torch.manual_seed(0)
    model = DCell(graph, min_subsystem_size=2, subsystem_ratio=0.5)
    assert model.term_input_dims[1] == 4
    assert model._prepare_term_input(1, dcell_batch, {}).shape == (2, 2)
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        model(graph, dcell_batch)


def test_a_root_stratum_added_after_construction_is_reported_unprocessed(
    dcell_batch: HeteroData,
) -> None:
    """``strata_order`` is frozen at init, so a stratum 0 added later is never run and the
    root lookup fails with the "was not processed" ValueError (dcell.py:342-345).
    """
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    graph = make_dcell_graph()
    graph["gene_ontology"].stratum_to_terms = {1: torch.tensor([1, 2])}
    model = _model(graph)
    model.stratum_to_terms[0] = torch.tensor([0])
    with pytest.raises(ValueError, match="Root term 0 was not processed"):
        model(graph, dcell_batch)


def test_dcell_loss_on_model_outputs_trains_every_parameter(
    dcell_graph: HeteroData, dcell_batch: HeteroData
) -> None:
    """With auxiliary losses on, the heads of terms 1 and 2 (untouched by the prediction
    alone, see the backward test above) receive gradient, so every parameter does.
    The loss is exactly MSE(root) + 0.3 * (MSE(GO:1) + MSE(GO:2)), the paper's sum over
    non-root subsystems (issue #554): GO:ROOT and GO:0 are skipped because GO:0 is the
    root key the model declares in ``outputs["root_key"]``.
    """
    model = _model(dcell_graph)
    predictions, outputs = model(dcell_graph, dcell_batch)
    target = torch.tensor([0.5, -0.5])
    total, parts = DCellLoss(alpha=0.3, aux_reduction="sum")(
        predictions, outputs, target
    )
    linear = outputs["linear_outputs"]
    mse = torch.nn.functional.mse_loss
    assert outputs["root_key"] == "GO:0"
    auxiliary = mse(linear["GO:1"], target) + mse(linear["GO:2"], target)
    torch.testing.assert_close(total, mse(predictions, target) + 0.3 * auxiliary)
    torch.testing.assert_close(parts["auxiliary_loss"], auxiliary.detach())
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


# ---------------------------------------------------------------- the real data path

GENE_NAMES = ["YAL001C", "YAL002W", "YAL003W", "YAL004W"]
PHENOTYPES: list[Any] = [FitnessPhenotype]
ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))


def _cell_graph() -> HeteroData:
    genes = GeneSet(GENE_NAMES)
    base = nx.Graph()
    base.add_nodes_from(genes)
    multigraph = GeneMultiGraph(
        graphs=SortedDict(
            {"base": GeneGraph(name="base", graph=base, max_gene_set=genes)}
        )
    )
    go = nx.DiGraph()
    go.add_node("GO:root", gene_set=GENE_NAMES)
    go.add_node("GO:a", gene_set=GENE_NAMES[:2])
    go.add_node("GO:b", gene_set=GENE_NAMES[2:])
    go.add_edge("GO:a", "GO:root")
    go.add_edge("GO:b", "GO:root")
    return to_cell_data(multigraph, incidence_graphs={"gene_ontology": go})


def _record(genes: list[str], fitness: float) -> dict[str, Any]:
    genotype = Genotype(
        perturbations=[
            KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
            for g in genes
        ]
    )
    return {
        "experiment": FitnessExperiment(
            dataset_name="toy",
            genotype=genotype,
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="toy",
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="S288C"
            ),
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _processed_batch(cell_graph: HeteroData) -> Batch:
    """Sample 0 deletes YAL001C (gene 0), sample 1 deletes YAL003W and YAL004W (2, 3)."""
    processor = DCellGraphProcessor()
    samples = [
        processor.process(cell_graph, PHENOTYPES, [_record(["YAL001C"], 0.9)]),
        processor.process(
            cell_graph, PHENOTYPES, [_record(["YAL003W", "YAL004W"], 0.4)]
        ),
    ]
    return Batch.from_data_list(samples, follow_batch=["go_gene_strata_state"])


def test_model_built_from_to_cell_data_consumes_the_processor_batch() -> None:
    """The root is term 2 (stratum 0) with children [0, 1] and its own four gene rows;
    dims and the 51-parameter total follow the module docstring. The collated ptr is
    [0, 8, 16] (8 template rows per sample). The root input is cat(act_0, act_1, states
    of g0..g3): [0, 1, 1, 1] for sample 0 and [1, 1, 0, 0] for sample 1.
    """
    cell_graph = _cell_graph()
    batch = _processed_batch(cell_graph)
    assert batch["gene_ontology"].go_gene_strata_state_ptr.tolist() == [0, 8, 16]
    torch.manual_seed(0)
    model = DCell(cell_graph, min_subsystem_size=2, subsystem_ratio=0.5)
    assert model.stratum_to_terms[0].tolist() == [2]
    assert model.parent_to_children == {2: [0, 1]}
    assert model.term_input_dims == {0: 2, 1: 2, 2: 8}
    assert model.term_output_dims == {0: 2, 1: 2, 2: 2}
    assert model.num_parameters["total"] == 51
    predictions, outputs = model(cell_graph, batch)
    act = outputs["term_activations"]
    states = torch.tensor([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    assert model._extract_gene_states_for_term(2, batch).tolist() == states.tolist()
    torch.testing.assert_close(
        act[2], model.subsystems["2"](torch.cat([act[0], act[1], states], dim=1))
    )
    assert torch.equal(outputs["linear_outputs"]["GO:ROOT"], predictions)
    assert torch.equal(outputs["linear_outputs"]["GO:2"], predictions)
    assert predictions.shape == (2,)


# ---------------------------------------------------------------- the overfit script


def _main_cfg(
    lr: float, epochs: int, plot_every: int, aux_reduction: str | None = "sum"
) -> Any:
    dcell_loss: dict[str, Any] = {"alpha": 0.3, "use_auxiliary_losses": True}
    if aux_reduction is not None:
        dcell_loss["aux_reduction"] = aux_reduction
    return OmegaConf.create(
        {
            "trainer": {"accelerator": "cpu", "max_epochs": epochs},
            "data_module": {"batch_size": 2, "num_workers": 0},
            "model": {
                "subsystem_output_min": 2,
                "subsystem_output_max_mult": 0.5,
                "output_size": 1,
            },
            "regression_task": {
                "dcell_loss": dcell_loss,
                "optimizer": {"type": "AdamW", "lr": lr, "weight_decay": 0.0},
                "lr_scheduler": {
                    "type": "ReduceLROnPlateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 3,
                    "threshold": 1e-4,
                    "min_lr": 1e-9,
                },
                "clip_grad_norm": True,
                "clip_grad_norm_max_norm": 10.0,
                "plot_every_n_epochs": plot_every,
            },
        }
    )


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HeteroData:
    """``main`` imports ``load_sample_data_batch`` at call time; serve the processed batch."""
    cell_graph = _cell_graph()
    batch = _processed_batch(cell_graph)
    module = types.ModuleType("torchcell.scratch.load_batch_005")

    def load_sample_data_batch(**_: Any) -> tuple[Any, Batch, None, None]:
        return types.SimpleNamespace(cell_graph=cell_graph), batch, None, None

    module.load_sample_data_batch = load_sample_data_batch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchcell.scratch.load_batch_005", module)
    monkeypatch.setattr(dcell_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("ASSET_IMAGES_DIR", str(tmp_path))
    return cell_graph


def test_main_crashes_at_its_first_intermediate_plot(
    fake_loader: HeteroData, tmp_path: Path
) -> None:
    """Finding: ``save_intermediate_plot`` calls ``model(batch)`` (dcell.py:486) but
    ``DCell.forward`` takes ``(cell_graph, batch)``, and the plot always fires on the
    last epoch (dcell.py:706), so ``main`` can never finish. Epoch 1 of 2 with
    ``plot_every_n_epochs=1`` trains one step, creates the timestamped plot directory,
    then raises.
    """
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'batch'"
    ):
        dcell_module.main(_main_cfg(lr=1e-3, epochs=2, plot_every=1))
    plot_dirs = list(tmp_path.iterdir())
    assert [p.name.startswith("dcell_training_") for p in plot_dirs] == [True]
    assert list(plot_dirs[0].iterdir()) == []


class _OneArgPlotDCell(DCell):
    """DCell that also accepts the plot's one-argument call by reusing its template."""

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if batch is None:
            return super().forward(self.hetero_data, cell_graph)
        return super().forward(cell_graph, batch)


def test_main_with_the_plot_call_patched_saves_every_plot_and_returns_eval_outputs(
    fake_loader: HeteroData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the one-argument call tolerated, 3 epochs at ``plot_every_n_epochs=2`` plot
    epochs 2 and 3 (the interval and the last epoch) plus the final results figure. At
    lr 0 AdamW moves nothing, so every parameter equals a fresh seeded model's; the
    returned predictions are the eval-mode forward of the returned model.
    """
    monkeypatch.setattr(dcell_module, "DCell", _OneArgPlotDCell)
    torch.manual_seed(0)
    reference = DCell(fake_loader, min_subsystem_size=2, subsystem_ratio=0.5)
    torch.manual_seed(0)
    model, (predictions, outputs) = dcell_module.main(
        _main_cfg(lr=0.0, epochs=3, plot_every=2)
    )
    (plot_dir,) = list(tmp_path.iterdir())
    names = sorted(p.name.rsplit("_", 1)[0] for p in plot_dir.iterdir())
    assert names == ["dcell_epoch_0002", "dcell_epoch_0003", "dcell_results"]
    for name, value in reference.named_parameters():
        assert torch.equal(dict(model.named_parameters())[name], value), name
    assert not model.training
    with torch.no_grad():
        again, _ = model(fake_loader, _processed_batch(fake_loader))
    assert torch.equal(again, predictions)
    assert set(outputs["linear_outputs"]) == {"GO:0", "GO:1", "GO:2", "GO:ROOT"}


_EPOCH_LINE = re.compile(
    r"Epoch (\d+)/2, Total Loss: ([-\d.]+), Primary Loss: ([-\d.]+), "
    r"Aux Loss: ([-\d.]+)"
)


def test_main_reads_aux_reduction_and_logs_its_closed_form_loss(
    fake_loader: HeteroData,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` passes ``dcell_loss.aux_reduction`` to ``DCellLoss`` (issue #554).

    Epoch 1 runs the seeded initial model, so its logged loss is a closed form of that
    model's heads: with root GO:2 (MSE p) and non-root GO:0, GO:1 (MSEs a, b), "sum"
    logs p + 0.3 * (a + b) with Aux a + b, and "mean" logs p + 0.3 * (a + b) / 2 with
    Aux (a + b) / 2. At lr 1e-2 the different gradients make epoch 2 differ as well.
    """
    monkeypatch.setattr(dcell_module, "DCell", _OneArgPlotDCell)
    batch = _processed_batch(fake_loader)
    torch.manual_seed(0)
    reference = DCell(fake_loader, min_subsystem_size=2, subsystem_ratio=0.5)
    with torch.no_grad():
        _, ref_outputs = reference(fake_loader, batch)
    target = batch["gene"].phenotype_values
    assert ref_outputs["root_key"] == "GO:2"
    mse = {
        k: torch.nn.functional.mse_loss(v, target).item()
        for k, v in ref_outputs["linear_outputs"].items()
    }
    p, aux_sum = mse["GO:2"], mse["GO:0"] + mse["GO:1"]
    logged: dict[str, list[tuple[float, float, float]]] = {}
    for reduction in ("sum", "mean"):
        torch.manual_seed(0)
        dcell_module.main(
            _main_cfg(lr=1e-2, epochs=2, plot_every=5, aux_reduction=reduction)
        )
        lines = _EPOCH_LINE.findall(capsys.readouterr().out)
        assert [line[0] for line in lines] == ["1", "2"]
        logged[reduction] = [(float(t), float(q), float(a)) for _, t, q, a in lines]
    aux = {"sum": aux_sum, "mean": aux_sum / 2}
    for reduction, rows in logged.items():
        total, primary, auxiliary = rows[0]
        assert primary == pytest.approx(p, abs=1e-6)
        assert auxiliary == pytest.approx(aux[reduction], abs=1e-6)
        assert total == pytest.approx(p + 0.3 * aux[reduction], abs=1e-6)
    assert logged["sum"][1][0] != logged["mean"][1][0]


def test_main_without_aux_reduction_in_the_config_raises(
    fake_loader: HeteroData,
) -> None:
    """The key is required: a config without ``dcell_loss.aux_reduction`` fails."""
    with pytest.raises(ConfigAttributeError, match="Missing key aux_reduction"):
        dcell_module.main(
            _main_cfg(lr=1e-3, epochs=1, plot_every=5, aux_reduction=None)
        )


# ---------------------------------------------------------------- Phase 24


def test_a_term_with_no_children_and_no_gene_columns_is_refused_by_index(
    dcell_batch: HeteroData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The placeholder makes the refusal unreachable from data (see the finding above),
    so the gene-state extractor is replaced by one returning a [2, 0] tensor: the root of
    a graph without child edges then has no input at all and ``_prepare_term_input``
    raises the ValueError naming term 0 (dcell.py:386-389).
    """
    from tests.torchcell.conftest import make_dcell_graph  # noqa: PLC0415

    graph = make_dcell_graph()
    del graph["gene_ontology", "is_child_of", "gene_ontology"]
    model = _model(graph)
    monkeypatch.setattr(
        model, "_extract_gene_states_for_term", lambda term, batch: torch.zeros(2, 0)
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "GO term 0 has no children and no genes. This should not happen in a "
            "well-formed GO hierarchy."
        ),
    ):
        model._prepare_term_input(0, dcell_batch, {})


class _EmptyEvalDCell(_OneArgPlotDCell):
    """Trains normally, but its two-argument eval forward predicts nothing."""

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        predictions, outputs = super().forward(cell_graph, batch)
        if batch is not None and not self.training:
            return predictions[:0], outputs
        return predictions, outputs


def test_main_reports_no_predictions_and_skips_the_results_figure(
    fake_loader: HeteroData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One epoch with the final evaluation returning an empty prediction tensor: ``main``
    prints the no-predictions line (dcell.py:832) instead of the metrics, saves only the
    last-epoch intermediate plot (no ``dcell_results`` figure), still prints the
    closing line, and returns the empty predictions.
    """
    monkeypatch.setattr(dcell_module, "DCell", _EmptyEvalDCell)
    torch.manual_seed(0)
    _, (predictions, _) = dcell_module.main(_main_cfg(lr=0.0, epochs=1, plot_every=5))
    out = capsys.readouterr().out
    assert "No predictions were generated. Check model and data setup.\n" in out
    assert "Final Mean Squared Error" not in out
    assert out.endswith("\nDCell training demonstration complete!\n")
    assert predictions.numel() == 0
    (plot_dir,) = list(tmp_path.iterdir())
    names = sorted(p.name.rsplit("_", 1)[0] for p in plot_dir.iterdir())
    assert names == ["dcell_epoch_0001"]
