# tests/torchcell/models/test_equivariant_cell_graph_transformer.py
# [[tests.torchcell.models.test_equivariant_cell_graph_transformer]]
"""Tests for the multitask decoder heads of the Equivariant Cell Graph Transformer.

Self-contained: the synthetic ``cell_graph`` / ``batch`` fixtures (a handful of genes,
a couple of perturbed indices, a small gpr/rmr metabolic incidence) come from
``tests/torchcell/conftest.py``, so no real dataset is required.
"""

from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    MaskedMultitaskLoss,
)

# Synthetic graph sizes; a test names a dimension, never a literal (conftest CGTDims).
GENE_NUM = 8
HIDDEN = 16
NUM_LAYERS = 2
NUM_HEADS = 4
BATCH_SIZE = 3
NUM_METABOLITES = 3


def _full_heads_config() -> dict[str, Any]:
    return {
        "global": {"output_dim": 501, "use_gene_pool": True},
        "per_gene": {"output_dim": 1},
        "per_metabolite": {"output_dim": 1},
    }


def _make_model(
    cell_graph: HeteroData, heads_config: dict[str, Any] | None, seed: int = 0
) -> CellGraphTransformer:
    torch.manual_seed(seed)
    return CellGraphTransformer(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        cell_graph=cell_graph,
        heads_config=heads_config,
    )


def test_multitask_forward_shapes(cell_graph: HeteroData, batch: HeteroData) -> None:
    """All configured heads return the expected shapes."""
    model = _make_model(cell_graph, _full_heads_config())
    model.eval()
    with torch.no_grad():
        predictions, reps = model(cell_graph, batch)

    assert predictions.shape == (BATCH_SIZE, 1)
    heads = reps["head_outputs"]
    assert heads["global"].shape == (BATCH_SIZE, 501)
    assert heads["per_gene"].shape == (BATCH_SIZE, GENE_NUM)
    assert heads["per_metabolite"].shape == (BATCH_SIZE, NUM_METABOLITES)


def test_single_head_config_matches_prechange(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """heads_config=None reproduces the pre-multitask single-head model exactly.

    No extra parameters/buffers are created and the gene-interaction prediction is
    numerically identical to a model built with the same seed but multitask heads
    enabled (the heads are instantiated LAST in __init__, so the shared backbone
    init is untouched).
    """
    baseline = _make_model(cell_graph, None, seed=42)
    multitask = _make_model(cell_graph, _full_heads_config(), seed=42)

    # No head parameters/buffers leak into the single-head model.
    assert baseline.global_head is None
    assert baseline.per_gene_head is None
    assert baseline.per_metabolite_head is None
    baseline_keys = set(baseline.state_dict().keys())
    multitask_keys = set(multitask.state_dict().keys())
    assert baseline_keys.issubset(multitask_keys)
    assert baseline_keys == multitask_keys - {
        k
        for k in multitask_keys
        if k.split(".")[0].endswith("_head") and "pert" not in k
    }

    baseline.eval()
    multitask.eval()
    with torch.no_grad():
        pred_base, reps_base = baseline(cell_graph, batch)
        pred_multi, _ = multitask(cell_graph, batch)

    # Single-head model exposes an empty head_outputs dict (backward compatible).
    assert reps_base["head_outputs"] == {}
    # Backbone + GI head numerically identical -> no regression.
    assert torch.allclose(pred_base, pred_multi, atol=1e-6)


def test_masked_loss_ignores_absent_modalities(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Masked multitask loss ignores rows/heads without supervision."""
    model = _make_model(cell_graph, _full_heads_config())
    model.eval()
    with torch.no_grad():
        _, reps = model(cell_graph, batch)
    head_outputs = reps["head_outputs"]

    loss_fn = MaskedMultitaskLoss(loss_fn="mse")

    targets = {
        "global": torch.zeros(BATCH_SIZE, 501),
        "per_gene": torch.zeros(BATCH_SIZE, GENE_NUM),
        "per_metabolite": torch.zeros(BATCH_SIZE, NUM_METABOLITES),
    }
    # Sparse supervision: global only on rows {0,2}; per_gene only on {1};
    # per_metabolite on NONE.
    masks = {
        "global": torch.tensor([True, False, True]),
        "per_gene": torch.tensor([False, True, False]),
        "per_metabolite": torch.tensor([False, False, False]),
    }

    total, per_head = loss_fn(
        head_outputs, targets, masks, graph_reg_loss=reps["graph_reg_loss"]
    )

    # Head with an all-False mask contributes exactly zero.
    assert per_head["per_metabolite"].item() == pytest.approx(0.0)

    # Changing a target row that is masked OUT must not change the loss.
    targets_perturbed = {k: v.clone() for k, v in targets.items()}
    targets_perturbed["global"][1] = 999.0  # row 1 is masked out for global
    total_perturbed, _ = loss_fn(
        head_outputs, targets_perturbed, masks, graph_reg_loss=reps["graph_reg_loss"]
    )
    assert torch.allclose(total, total_perturbed, atol=1e-6)

    # Changing a target row that is masked IN must change the loss.
    targets_active = {k: v.clone() for k, v in targets.items()}
    targets_active["global"][0] = 999.0  # row 0 is masked in for global
    total_active, _ = loss_fn(
        head_outputs, targets_active, masks, graph_reg_loss=reps["graph_reg_loss"]
    )
    assert not torch.allclose(total, total_active, atol=1e-6)


def test_masked_loss_preserves_graph_reg_term() -> None:
    """The graph-regularization term is added UNCHANGED to the multitask loss."""
    loss_fn = MaskedMultitaskLoss(loss_fn="mse")
    head_outputs = {"per_gene": torch.zeros(BATCH_SIZE, GENE_NUM)}
    targets = {"per_gene": torch.zeros(BATCH_SIZE, GENE_NUM)}  # zero loss
    graph_reg = torch.tensor(0.37)
    total, _ = loss_fn(head_outputs, targets, masks=None, graph_reg_loss=graph_reg)
    assert total.item() == pytest.approx(0.37)


def test_per_metabolite_head_requires_incidence() -> None:
    """Requesting the metabolite head without gpr/rmr edges raises."""
    cg = HeteroData()
    cg["gene"].num_nodes = GENE_NUM
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    with pytest.raises(ValueError, match="per_metabolite head requested"):
        CellGraphTransformer(
            gene_num=GENE_NUM,
            hidden_channels=HIDDEN,
            num_transformer_layers=NUM_LAYERS,
            num_attention_heads=NUM_HEADS,
            cell_graph=cg,
            heads_config={"per_metabolite": {}},
        )
