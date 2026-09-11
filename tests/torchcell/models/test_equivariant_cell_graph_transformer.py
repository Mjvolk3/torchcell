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
    heads_config: dict[str, Any] | None, seed: int = 0
) -> CellGraphTransformer:
    torch.manual_seed(seed)
    return CellGraphTransformer(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        cell_graph=_make_cell_graph(),
        heads_config=heads_config,
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


def _per_gene_model(spec: dict[str, Any], seed: int = 7) -> CellGraphTransformer:
    """A model whose only multitask head is the per-gene head with the given spec."""
    return _make_model({"per_gene": {"output_dim": 1, **spec}}, seed=seed)


def _per_gene_output(model: CellGraphTransformer) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, reps = model(_make_cell_graph(), _make_batch())
    out = reps["head_outputs"]["per_gene"]
    assert isinstance(out, torch.Tensor)
    return out


@pytest.mark.parametrize("arm", ["per_gene_weight", "context_readout"])
def test_per_gene_readout_rows_are_identity_at_init(arm: str) -> None:
    """The GEARS-form and State-form per-gene rows are zero-gated.

    At step 0 the head must return exactly the shared-MLP output, so any later movement
    is attributable to the mechanism and not to its new parameters. The rows are built
    AFTER the shared MLP, so the MLP init is identical across the two models.
    """
    base = _per_gene_model({})
    armed = _per_gene_model({arm: True})
    assert torch.allclose(_per_gene_output(base), _per_gene_output(armed), atol=0.0)
    extra = sum(p.numel() for p in armed.parameters()) - sum(
        p.numel() for p in base.parameters()
    )
    assert extra == GENE_NUM * (HIDDEN + 1) + 1


def test_linear_readout_is_wired_into_the_head() -> None:
    """``linear_readout`` must change the head that is built.

    Regression test: the spec key existed and was declared in the configs, but the model
    never passed it to ``PerGeneHead``, so every "linear" run trained the two-layer head.
    """
    mlp = _per_gene_model({})
    linear = _per_gene_model({"linear_readout": True})
    assert mlp.per_gene_head is not None and linear.per_gene_head is not None
    assert not mlp.per_gene_head.linear_readout
    assert linear.per_gene_head.linear_readout
    assert sum(p.numel() for p in linear.parameters()) < sum(
        p.numel() for p in mlp.parameters()
    )
    assert _per_gene_output(linear).shape == (BATCH_SIZE, GENE_NUM)
