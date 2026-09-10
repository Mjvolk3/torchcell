# tests/torchcell/models/test_equivariant_cell_graph_transformer.py
# [[tests.torchcell.models.test_equivariant_cell_graph_transformer]]
"""Tests for the multitask decoder heads of the Equivariant Cell Graph Transformer.

Self-contained: builds a tiny SYNTHETIC HeteroData batch + cell_graph (a handful of
genes, a couple of perturbed indices, a small gpr/rmr metabolic incidence) so no real
dataset is required.
"""

from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    MaskedMultitaskLoss,
)

GENE_NUM = 8
HIDDEN = 16
NUM_LAYERS = 2
NUM_HEADS = 4
BATCH_SIZE = 3
NUM_REACTIONS = 4
NUM_METABOLITES = 3


def _make_cell_graph() -> HeteroData:
    """Tiny cell_graph with gene-gene, gpr, and rmr edges."""
    cg = HeteroData()
    cg["gene"].num_nodes = GENE_NUM
    cg["reaction"].num_nodes = NUM_REACTIONS
    cg["metabolite"].num_nodes = NUM_METABOLITES

    # A gene-gene edge type (unused when graph_reg_lambda == 0).
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )

    # gene -> gpr -> reaction: genes {0,1}->r0, {2}->r1, {3,4}->r2, {5}->r3.
    cg["gene", "gpr", "reaction"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 3]], dtype=torch.long
    )

    # metabolite <- reaction (hyperedge): m0<-{r0,r1}, m1<-{r2}, m2<-{r3,r0}.
    cg["metabolite", "reaction", "metabolite"].edge_index = torch.tensor(
        [[0, 0, 1, 2, 2], [0, 1, 2, 3, 0]], dtype=torch.long
    )
    return cg


def _make_batch() -> HeteroData:
    """Tiny perturbation batch: 3 genotypes with varying perturbed gene counts."""
    batch = HeteroData()
    # sample 0 perturbs genes {1,2}; sample 1 perturbs {3}; sample 2 perturbs {0,4,5}
    batch["gene"].perturbation_indices = torch.tensor(
        [1, 2, 3, 0, 4, 5], dtype=torch.long
    )
    batch["gene"].perturbation_indices_batch = torch.tensor(
        [0, 0, 1, 2, 2, 2], dtype=torch.long
    )
    return batch


def _full_heads_config() -> dict[str, Any]:
    return {
        "global": {"output_dim": 501, "use_gene_pool": True},
        "per_gene": {"output_dim": 1},
        "per_metabolite": {"output_dim": 1},
    }


def _make_model(
    heads_config: dict[str, Any] | None, seed: int = 0, **kwargs: Any
) -> CellGraphTransformer:
    torch.manual_seed(seed)
    return CellGraphTransformer(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        cell_graph=_make_cell_graph(),
        heads_config=heads_config,
        **kwargs,
    )


def test_multitask_forward_shapes() -> None:
    """All configured heads return the expected shapes."""
    model = _make_model(_full_heads_config())
    model.eval()
    cg = _make_cell_graph()
    batch = _make_batch()
    with torch.no_grad():
        predictions, reps = model(cg, batch)

    assert predictions.shape == (BATCH_SIZE, 1)
    heads = reps["head_outputs"]
    assert heads["global"].shape == (BATCH_SIZE, 501)
    assert heads["per_gene"].shape == (BATCH_SIZE, GENE_NUM)
    assert heads["per_metabolite"].shape == (BATCH_SIZE, NUM_METABOLITES)


def test_single_head_config_matches_prechange() -> None:
    """heads_config=None reproduces the pre-multitask single-head model exactly.

    No extra parameters/buffers are created and the gene-interaction prediction is
    numerically identical to a model built with the same seed but multitask heads
    enabled (the heads are instantiated LAST in __init__, so the shared backbone
    init is untouched).
    """
    baseline = _make_model(None, seed=42)
    multitask = _make_model(_full_heads_config(), seed=42)

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
    cg = _make_cell_graph()
    batch = _make_batch()
    with torch.no_grad():
        pred_base, reps_base = baseline(cg, batch)
        pred_multi, _ = multitask(cg, batch)

    # Single-head model exposes an empty head_outputs dict (backward compatible).
    assert reps_base["head_outputs"] == {}
    # Backbone + GI head numerically identical -> no regression.
    assert torch.allclose(pred_base, pred_multi, atol=1e-6)


def test_masked_loss_ignores_absent_modalities() -> None:
    """Masked multitask loss ignores rows/heads without supervision."""
    model = _make_model(_full_heads_config())
    model.eval()
    cg = _make_cell_graph()
    batch = _make_batch()
    with torch.no_grad():
        _, reps = model(cg, batch)
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


def test_perturb_cls_moves_only_the_cls() -> None:
    """perturb_cls adds no parameters, leaves every gene row and the 010 interaction
    prediction bit-identical, and makes the CLS strain-specific.

    The CLS rides through the perturbation operator as one more query over the same
    deleted-gene keys; a cross-attention query's output depends on that query and the
    keys only, so the gene rows cannot change. The wild-type CLS is byte-identical
    across strains (the fact that motivates the flag); the perturbed one must not be.
    """
    cls_only_linear = {
        "global": {"output_dim": 1, "use_gene_pool": False, "linear": True}
    }
    base = _make_model(cls_only_linear, seed=7)
    pert = _make_model(cls_only_linear, seed=7, perturb_cls=True)
    assert set(base.state_dict()) == set(pert.state_dict())

    base.eval()
    pert.eval()
    cg = _make_cell_graph()
    batch = _make_batch()
    with torch.no_grad():
        pred_base, reps_base = base(cg, batch)
        pred_pert, reps_pert = pert(cg, batch)

    # 010's interaction path is untouched (perturbation_head_cls defaults to wildtype).
    assert torch.allclose(pred_base, pred_pert, atol=1e-6)
    assert torch.allclose(
        reps_base["H_genes_pert"], reps_pert["H_genes_pert"], atol=1e-6
    )
    assert reps_base["h_CLS_pert"] is None

    h_cls_pert = reps_pert["h_CLS_pert"]
    assert h_cls_pert.shape == (BATCH_SIZE, HIDDEN)
    # Three different deletion sets -> three different CLS states.
    assert h_cls_pert.std(dim=0).mean().item() > 1e-4
    # Without the flag a CLS-only readout is one number for every strain; with it the
    # readout is a linear probe of a strain-specific token.
    g_base = reps_base["head_outputs"]["global"]
    g_pert = reps_pert["head_outputs"]["global"]
    assert g_base.shape == g_pert.shape == (BATCH_SIZE, 1)
    assert g_base.std().item() < 1e-6
    assert g_pert.std().item() > 1e-6


def test_perturbation_head_cls_perturbed_changes_interaction_input() -> None:
    """Feeding the interaction head the perturbed CLS is a real change, and is refused
    when there is no perturbed CLS to feed.
    """
    heads = {"global": {"output_dim": 1, "use_gene_pool": False}}
    wild = _make_model(heads, seed=3, perturb_cls=True)
    both = _make_model(
        heads, seed=3, perturb_cls=True, perturbation_head_cls="perturbed"
    )
    wild.eval()
    both.eval()
    cg = _make_cell_graph()
    batch = _make_batch()
    with torch.no_grad():
        pred_wild, _ = wild(cg, batch)
        pred_both, _ = both(cg, batch)
    assert not torch.allclose(pred_wild, pred_both, atol=1e-6)

    with pytest.raises(ValueError, match="needs perturb_cls"):
        _make_model(heads, seed=3, perturbation_head_cls="perturbed")
