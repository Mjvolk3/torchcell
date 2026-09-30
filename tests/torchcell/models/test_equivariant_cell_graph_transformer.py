# tests/torchcell/models/test_equivariant_cell_graph_transformer.py
# [[tests.torchcell.models.test_equivariant_cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_equivariant_cell_graph_transformer.py
"""Tests for the multitask decoder heads of the Equivariant Cell Graph Transformer.

Self-contained: the synthetic ``cell_graph`` / ``batch`` fixtures (a handful of genes,
a couple of perturbed indices, a small gpr/rmr metabolic incidence) come from
``tests/torchcell/conftest.py``, so no real dataset is required.

2026.09.30 (Phase 13). The two sibling files (``..._components.py``, ``..._model.py``)
already pin most of the module; the tests added here cover what the three files together
still left open, plus the Decision 12 identities the siblings do not state. Fixture: the
conftest ``cell_graph`` (8 genes, ``physical`` edges 0->1, 1->2, 2->3, 3->4, gpr
{0,1}->r0, {2}->r1, {3,4}->r2, {5}->r3, rmr m0<-{r0,r1}, m1<-{r2}, m2<-{r3,r0}) and
``batch`` (genotypes {1, 2}, {3}, {0, 4, 5}); hidden 16, 4 heads, every module in eval
or with dropout 0 so a forward is a pure function. Identities and their derivations:

* ReZero at init: HyperSAGNN with beta = 0 is pinned by the sibling; with the gates OPEN
  a singleton set has an all-masked attention row, ``nan_to_num`` makes it 0, so each
  layer adds ``beta * O.bias`` and the output is
  ``(x + b1 * O1.bias + b2 * O2.bias - relu(W x + b))^2`` in closed form.
* |S| = 1 in the perturbation transform: softmax over one key is exactly 1 and its
  backward ``y * (g - sum(g * y))`` is exactly 0, so every query row of the context is
  identical and the Q / K rows of ``in_proj`` receive a gradient of exactly 0; the null
  sink gives the softmax a second term and both stop holding.
* Propagation features in closed form on a 4-gene, two-relation graph (derivation in the
  test): ``log1p(N * r)`` with ``r`` the hop-t mass from the 1/|S|-normalized indicator.
* Parameter arithmetic: a Linear(i, o) has i*o + o parameters, a LayerNorm(d) 2d, a
  MultiheadAttention(16) 816 + 272 = 1088.
* Permuting the genotypes of a batch permutes every per-genotype output.
* Findings (pinned, not fixed): HyperSAGNN sizes its output by the number of distinct set
  ids, so ids {0, 2} index past the end; the batch size is ``max(assignment) + 1``, so a
  trailing genotype with no perturbation gets no row; ``num_parameters["total"]`` omits
  six optional modules.
"""

import math
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData

from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    CrossAttnHead,
    EquivariantPerturbationTransform,
    GraphRegularizedTransformerLayer,
    HyperSAGNN,
    MaskedMultitaskLoss,
    ObservedLabelEncoder,
    PerMetaboliteHead,
    PerturbationGraphPropagation,
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


# --- Phase 13 additions ------------------------------------------------------------- #
def _eval_model(
    cell_graph: HeteroData, seed: int = 0, **kwargs: Any
) -> CellGraphTransformer:
    """A 2-layer, hidden-16 model with dropout 0 in eval mode (a pure function)."""
    torch.manual_seed(seed)
    config: dict[str, Any] = dict(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        cell_graph=cell_graph,
        dropout=0.0,
    )
    config.update(kwargs)
    return CellGraphTransformer(**config).eval()


def _perturbation_batch(indices: list[int], assignment: list[int]) -> HeteroData:
    """A batch carrying only the two index tensors the model reads."""
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor(indices, dtype=torch.long)
    batch["gene"].perturbation_indices_batch = torch.tensor(
        assignment, dtype=torch.long
    )
    return batch


def _in_proj_weight_grad(module: EquivariantPerturbationTransform) -> torch.Tensor:
    """Layer 0's ``in_proj_weight`` gradient (rows [Q; K; V], each ``HIDDEN`` tall)."""
    attention = module.cross_attn_layers[0]
    assert isinstance(attention, nn.MultiheadAttention)
    grad = attention.in_proj_weight.grad
    assert grad is not None
    return grad


# HyperSAGNN ------------------------------------------------------------------------ #
def test_hypersagnn_open_gates_singleton_closed_form_and_set_isolation() -> None:
    """Gates beta1 = 0.5, beta2 = 2.0 on sets {rows 0, 1} and {row 2}.

    Row 2 is alone in its set, so its attention row is all -inf; softmax gives NaN and
    ``nan_to_num`` gives 0, so each layer returns ``O(0) = O.bias`` and its ReZero adds
    ``beta * O.bias``. Hence dynamic = x2 + 0.5 * O1.bias + 2.0 * O2.bias and output row 1
    is ``(dynamic - relu(W x2 + b))^2``. Set 0 is attention-coupled (its row differs from
    the attention-free value with the same bias shifts) but isolated from set 1: shifting
    row 2 leaves output row 0 exactly unchanged.
    """
    torch.manual_seed(0)
    module = HyperSAGNN(HIDDEN, num_heads=NUM_HEADS)
    x = torch.randn(3, HIDDEN)
    ids = torch.tensor([0, 0, 1])
    with torch.no_grad():
        module.beta1.fill_(0.5)
        module.beta2.fill_(2.0)
        out = module(x, ids)
        shift = 0.5 * module.O1.bias + 2.0 * module.O2.bias
        expected_singleton = (x[2] + shift - module.static_embedding(x[2])) ** 2
        attention_free_pair = (
            (x[:2] + shift - module.static_embedding(x[:2])) ** 2
        ).mean(0)
        shifted = x.clone()
        shifted[2] += 1.0
        out_shifted = module(shifted, ids)
    assert out.shape == (2, HIDDEN)
    torch.testing.assert_close(out[1], expected_singleton)
    assert not torch.allclose(out[0], attention_free_pair)
    assert torch.equal(out_shifted[0], out[0])
    assert not torch.equal(out_shifted[1], out[1])


def test_hypersagnn_non_contiguous_set_ids_index_past_the_output() -> None:
    """Finding: the output is sized by the number of DISTINCT set ids (line 246,
    ``num_batches = len(unique_batches)``) but ``scatter_mean`` indexes by the id VALUE,
    so ids {0, 2} allocate 2 rows and id 2 falls off the end. Pinned until the size is
    ``max(id) + 1`` or the ids are renumbered.
    """
    module = HyperSAGNN(HIDDEN, num_heads=NUM_HEADS)
    with pytest.raises(
        RuntimeError, match="index 2 is out of bounds for dimension 0 with size 2"
    ):
        module(torch.randn(3, HIDDEN), torch.tensor([0, 2, 2]))


# Perturbation transform ------------------------------------------------------------ #
def test_single_deletion_context_is_query_independent_and_starves_q_and_k() -> None:
    """The source's |S| = 1 claim, checked: softmax over one key is exactly 1 and its
    backward ``y * (g - sum(g * y))`` is exactly 0.

    Genotypes {1} and {3}: every gene's context row equals gene 0's bit for bit, and the
    Q / K rows ``[0, 32)`` of ``in_proj`` get a gradient of exactly 0 while V's rows do
    not. A two-gene set {1, 3} with the same weights reaches Q and K, and so does the
    null sink (bias 0), which also makes the context rows differ by gene.
    """
    torch.manual_seed(0)
    h = torch.randn(GENE_NUM, HIDDEN)
    singles = (torch.tensor([1, 3]), torch.tensor([0, 1]))
    q_and_k, v = slice(0, 2 * HIDDEN), slice(2 * HIDDEN, 3 * HIDDEN)

    plain = EquivariantPerturbationTransform(HIDDEN, num_heads=NUM_HEADS, dropout=0.0)
    _, context = plain(h, *singles)
    assert torch.equal(context, context[:, :1].expand(-1, GENE_NUM, -1))
    (context**2).sum().backward()
    grad_w = _in_proj_weight_grad(plain)
    attention = plain.cross_attn_layers[0]
    assert isinstance(attention, nn.MultiheadAttention)
    grad_b = attention.in_proj_bias.grad
    assert grad_b is not None
    assert torch.equal(grad_w[q_and_k], torch.zeros(2 * HIDDEN, HIDDEN))
    assert torch.equal(grad_b[q_and_k], torch.zeros(2 * HIDDEN))
    assert grad_w[v].abs().sum().item() > 0.0

    pair = EquivariantPerturbationTransform(HIDDEN, num_heads=NUM_HEADS, dropout=0.0)
    pair.load_state_dict(plain.state_dict())
    _, context_pair = pair(h, torch.tensor([1, 3]), torch.tensor([0, 0]))
    (context_pair**2).sum().backward()
    assert _in_proj_weight_grad(pair)[q_and_k].abs().sum().item() > 0.0

    sink = EquivariantPerturbationTransform(
        HIDDEN,
        num_heads=NUM_HEADS,
        dropout=0.0,
        null_sink=True,
        null_sink_bias_init=0.0,
    )
    assert sink.load_state_dict(plain.state_dict(), strict=False).missing_keys == [
        "null_bias"
    ]
    _, context_sink = sink(h, *singles)
    assert not torch.allclose(context_sink[0, 0], context_sink[0, 1])
    (context_sink**2).sum().backward()
    assert _in_proj_weight_grad(sink)[q_and_k].abs().sum().item() > 0.0
    assert sink.null_bias.grad is not None and sink.null_bias.grad.item() != 0.0


def test_trailing_genotype_without_perturbation_gets_no_row(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Finding: the batch size is ``int(max(assignment)) + 1`` (line 668) and the heads
    size from the transform's output (line 1323), so a batch whose LAST genotype is the
    wildtype (no entry in ``perturbation_indices_batch``) returns one row too few: here
    {1, 2}, {3}, wildtype gives 2 rows. The two rows it does return equal rows 0 and 1 of
    the fixture batch, which has the same first two genotypes. A wildtype in the middle
    keeps its row (sibling test). Pinned until the batch size is passed explicitly.
    """
    model = _eval_model(cell_graph, heads_config={"per_gene": {}})
    with torch.no_grad():
        pred, reps = model(cell_graph, _perturbation_batch([1, 2, 3], [0, 0, 1]))
        full_pred, full_reps = model(cell_graph, batch)
    assert pred.shape == (2, 1)
    assert reps["head_outputs"]["per_gene"].shape == (2, GENE_NUM)
    torch.testing.assert_close(pred, full_pred[:2])
    torch.testing.assert_close(
        reps["head_outputs"]["per_gene"], full_reps["head_outputs"]["per_gene"][:2]
    )


# Propagation ----------------------------------------------------------------------- #
def test_propagation_features_have_the_closed_form_log1p_reach() -> None:
    """Four genes, relation ``a`` = {0->1, 1->2, 0->3}, relation ``b`` = {2->0}, 2 hops.

    Row-normalized: a[0,1] = a[0,3] = 1/2 (gene 0 has out-degree 2), a[1,2] = 1,
    b[2,0] = 1. Features per gene are [hop 0, a hop 1, a hop 2, b hop 1, b hop 2] of
    ``log1p(4 * r)`` with ``r`` the mass spread from the indicator normalized by |S|.
    Genotype 0 = {0}: a hop 1 puts 1/2 on genes 1 and 3, a hop 2 moves gene 1's half to
    gene 2, and gene 0 has no ``b`` out-edge. Genotype 1 = {1, 2} at 1/2 each: a hop 1
    moves gene 1's half to gene 2, b hop 1 moves gene 2's half to gene 0, nothing reaches
    a second hop. So log1p(4 * 1) = log 5, log1p(4 * 1/2) = log 3, unreached = 0 exactly.
    The projection is set to the identity (Linear weights I, biases 0; ReLU is the
    identity on these nonnegative features) and the gate is forced on, so the output
    minus the zero input IS the feature tensor. ``gate_mode="on"`` is a fixed 1.0 held as
    a non-persistent buffer, not a parameter.
    """
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    graph["gene", "a", "gene"].edge_index = torch.tensor([[0, 1, 0], [1, 2, 3]])
    graph["gene", "b", "gene"].edge_index = torch.tensor([[2], [0]])
    torch.manual_seed(0)
    model = CellGraphTransformer(
        gene_num=4,
        hidden_channels=HIDDEN,
        num_transformer_layers=1,
        num_attention_heads=NUM_HEADS,
        cell_graph=graph,
        perturbation_propagation_config={"enabled": True, "hops": 2},
    )
    assert list(model.adjacency_T_sparse) == ["a", "b"]

    module = PerturbationGraphPropagation(
        5, ["a", "b"], hops=2, dropout=0.0, gate_mode="on"
    )
    assert module.num_features == 5
    assert "gate" not in module.state_dict()
    assert "gate" not in dict(module.named_parameters())
    assert torch.equal(module.gate, torch.ones(1))
    with torch.no_grad():
        for index in (0, 3):
            linear = module.proj[index]
            assert isinstance(linear, nn.Linear)
            linear.weight.copy_(torch.eye(5))
            linear.bias.zero_()
        out = module(
            torch.zeros(2, 4, 5),
            model.adjacency_T_sparse,
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 1]),
        )
    l3, l5 = math.log(3.0), math.log(5.0)
    expected = torch.tensor(
        [
            [
                [l5, 0.0, 0.0, 0.0, 0.0],
                [0.0, l3, 0.0, 0.0, 0.0],
                [0.0, 0.0, l3, 0.0, 0.0],
                [0.0, l3, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, l3, 0.0],
                [l3, 0.0, 0.0, 0.0, 0.0],
                [l3, l3, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    torch.testing.assert_close(out, expected)


# Observed labels ------------------------------------------------------------------- #
def test_observed_label_all_masked_offset_is_the_projection_of_zero_features() -> None:
    """Finding (the exact form of the one the components file pins): lines 980 and 991
    say a 100%-masked forward "is still an identity", but with every label masked the
    features are [0, 0] and the forced-on gate adds ``proj([0, 0]) = W2 relu(b1) + b2``,
    the same nonzero vector on every token. Pinned until the gate or the bias path
    changes. ``gate_mode="rezero"`` is a trainable zero-initialized parameter in the
    state dict and makes the encoder the exact identity even with a label observed.
    """
    torch.manual_seed(0)
    h = torch.randn(BATCH_SIZE, GENE_NUM, HIDDEN)
    values = torch.zeros(BATCH_SIZE, GENE_NUM)
    mask = torch.zeros(BATCH_SIZE, GENE_NUM)
    forced = ObservedLabelEncoder(HIDDEN, dropout=0.0).eval()
    with torch.no_grad():
        offset = forced.proj(torch.zeros(2))
        out = forced(h, values, mask)
    assert offset.abs().sum().item() > 0.0
    torch.testing.assert_close(out, h + offset)

    rezero = ObservedLabelEncoder(HIDDEN, dropout=0.0, gate_mode="rezero")
    assert isinstance(rezero.gate, nn.Parameter) and rezero.gate.requires_grad
    assert torch.equal(rezero.state_dict()["gate"], torch.zeros(1))
    values[0, 2], mask[0, 2] = 0.7, 1.0
    assert torch.equal(rezero(h, values, mask), h)


# Heads ------------------------------------------------------------------------------ #
def test_per_metabolite_head_pools_genes_then_reactions_by_the_incidence(
    cell_graph: HeteroData,
) -> None:
    """Metabolite m's input is sum_r mr[m, r] sum_n gpr[r, n] h_n, then the shared MLP.

    With the fixture incidence (r0 = mean of genes 0, 1; r1 = gene 2; r2 = mean of genes
    3, 4; r3 = gene 5; m0 = mean of r0, r1; m1 = r2; m2 = mean of r3, r0) and
    ``output_dim=2`` the head equals ``mlp(einsum(mr, gpr, h))`` as [B, 3, 2] (no squeeze).
    """
    model = _eval_model(cell_graph, heads_config={"per_metabolite": {"output_dim": 2}})
    head = model.per_metabolite_head
    gpr_t, mr = model.gpr_incidence_T, model.mr_incidence
    assert isinstance(head, PerMetaboliteHead)
    assert isinstance(gpr_t, torch.Tensor) and isinstance(mr, torch.Tensor)
    gpr_dense = torch.zeros(4, GENE_NUM)
    gpr_dense[0, 0] = gpr_dense[0, 1] = gpr_dense[2, 3] = gpr_dense[2, 4] = 0.5
    gpr_dense[1, 2] = gpr_dense[3, 5] = 1.0
    mr_dense = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.0, 0.5]]
    )
    torch.manual_seed(1)
    h = torch.randn(BATCH_SIZE, GENE_NUM, HIDDEN)
    with torch.no_grad():
        out = head(h, gpr_t, mr)
        pooled = torch.einsum("mr,rn,bnd->bmd", mr_dense, gpr_dense, h)
        expected = head.mlp(pooled)
    assert out.shape == (BATCH_SIZE, NUM_METABOLITES, 2)
    torch.testing.assert_close(out, expected)


def test_cross_attention_head_parameter_arithmetic_with_and_without_ffn() -> None:
    """F = 5 queries, hidden 16, param_dim 2.

    Without the FFN: queries 5 * 16 = 80, MultiheadAttention 816 + 272 = 1088, norm1 32,
    readout Linear(16, 2) = 34, total 1234, and no ``ffn`` / ``norm2`` attributes. With
    it: + Linear(16, 32) = 544 + Linear(32, 16) = 528 + norm2 32 = 2338. The readout is
    shared across features, so F only enters through the queries: F = 6 adds 16.
    """

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    lean = CrossAttnHead(HIDDEN, 5, num_heads=NUM_HEADS, param_dim=2, use_ffn=False)
    assert count(lean) == 1234
    assert not hasattr(lean, "ffn") and not hasattr(lean, "norm2")
    assert count(CrossAttnHead(HIDDEN, 5, num_heads=NUM_HEADS, param_dim=2)) == 2338
    assert (
        count(CrossAttnHead(HIDDEN, 6, num_heads=NUM_HEADS, param_dim=2, use_ffn=False))
        == 1250
    )


def test_per_gene_input_is_h_pert_h_i_context_pert_set_then_bilinear(
    cell_graph: HeteroData,
) -> None:
    """The per-gene head input, rebuilt from the model's own parts, in its order.

    ``concat_context`` + ``pert_set_context`` + ``bilinear_rank=2`` + FiLM, sum pooling,
    genotypes {1, 2}, wildtype, {0, 4, 5}. The input is
    [H_pert ; H_genes ; context ; z_S ; h_CLS ; bilinear(H_genes, context)], width
    16 * (3 + 2) + 2 = 82, with z_S the SUM of the perturbed H_pert rows and exactly 0
    for the wildtype (the empty-set skip at line 2908). FiLM's last layer is
    zero-initialized, so it is the identity and the output is ``mlp(input)`` squeezed. A
    swapped block order, mean pooling or a nonzero wildtype z_S all fail the comparison.
    """
    model = _eval_model(
        cell_graph,
        heads_config={
            "per_gene": {
                "concat_context": True,
                "pert_set_context": True,
                "bilinear_rank": 2,
                "film_on_pert_set": True,
            }
        },
    )
    head, bilinear = model.per_gene_head, model.bilinear
    assert head is not None and bilinear is not None
    first = head.mlp[0]
    assert isinstance(first, nn.Linear) and first.in_features == 82
    idx, assign = [1, 2, 0, 4, 5], [0, 0, 2, 2, 2]
    with torch.no_grad():
        _, reps = model(cell_graph, _perturbation_batch(idx, assign))
        h_genes, h_pert, h_cls = reps["H_genes"], reps["H_genes_pert"], reps["h_CLS"]
        _, context = model.perturbation_transform(
            h_genes, torch.tensor(idx), torch.tensor(assign)
        )
        z_s = torch.zeros(BATCH_SIZE, HIDDEN)
        z_s[0] = h_pert[0, [1, 2]].sum(0)
        z_s[2] = h_pert[2, [0, 4, 5]].sum(0)
        cond = torch.cat([z_s, h_cls.expand(BATCH_SIZE, -1)], dim=-1)
        rebuilt = torch.cat(
            [
                h_pert,
                h_genes.expand(BATCH_SIZE, -1, -1),
                context,
                cond.unsqueeze(1).expand(-1, GENE_NUM, -1),
                bilinear(h_genes, context),
            ],
            dim=-1,
        )
        expected = head.mlp(rebuilt).squeeze(-1)
    assert rebuilt.shape == (BATCH_SIZE, GENE_NUM, 82)
    torch.testing.assert_close(reps["head_outputs"]["per_gene"], expected)


# The assembled model --------------------------------------------------------------- #
def test_observed_labels_enter_before_cross_gene_then_perceiver_mixing(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """H_genes_pert = Perceiver(CrossGene(Observed(Transform(H_genes)))), in that order.

    Rebuilt from the model's own modules with gene 2 of genotype 0 observed at 0.7.
    Without ``observed_values`` the encoder is skipped and the result is
    Perceiver(CrossGene(Transform(H_genes))). The other order (observe after mixing)
    gives a different tensor, so the comparison discriminates the order.
    """
    model = _eval_model(
        cell_graph,
        observed_label_config={"enabled": True},
        cross_gene_config={"enabled": True, "rank": 4},
        post_perturbation_mixing_config={
            "enabled": True,
            "num_latents": 4,
            "gate_mode": "on",
        },
    )
    observe, cross, mix = (
        model.observed_label_encoder,
        model.cross_gene_mixing,
        model.post_perturbation_mixing,
    )
    assert observe is not None and cross is not None and mix is not None
    values = torch.zeros(BATCH_SIZE, GENE_NUM)
    mask = torch.zeros(BATCH_SIZE, GENE_NUM)
    values[0, 2], mask[0, 2] = 0.7, 1.0
    with torch.no_grad():
        _, reps = model(cell_graph, batch, observed_values=values, observed_mask=mask)
        _, reps_none = model(cell_graph, batch)
        base, _ = model.perturbation_transform(
            reps["H_genes"],
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )
        expected = mix(cross(observe(base, values, mask)))
        other_order = observe(mix(cross(base)), values, mask)
    torch.testing.assert_close(reps["H_genes_pert"], expected)
    torch.testing.assert_close(reps_none["H_genes_pert"], mix(cross(base)))
    assert not torch.allclose(expected, other_order)


def test_permuting_genotypes_permutes_every_per_genotype_output(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Genotype order [2, 0, 1]: new position k holds old genotype order[k], so old
    genotype s is renumbered argsort(order)[s] = [1, 2, 0][s]. The prediction, H_genes_pert
    and all three heads permute by ``order``; h_CLS (computed on the wildtype graph) is
    unchanged.
    """
    model = _eval_model(cell_graph, heads_config=_full_heads_config())
    order = torch.tensor([2, 0, 1])
    renumber = torch.argsort(order)
    permuted = batch.clone()
    permuted["gene"].perturbation_indices_batch = renumber[
        batch["gene"].perturbation_indices_batch
    ]
    with torch.no_grad():
        pred, reps = model(cell_graph, batch)
        pred_p, reps_p = model(cell_graph, permuted)
    torch.testing.assert_close(pred_p, pred[order])
    torch.testing.assert_close(reps_p["H_genes_pert"], reps["H_genes_pert"][order])
    assert torch.equal(reps_p["h_CLS"], reps["h_CLS"])
    for name in ("global", "per_gene", "per_metabolite"):
        torch.testing.assert_close(
            reps_p["head_outputs"][name], reps["head_outputs"][name][order]
        )
    assert not torch.allclose(pred[0], pred[1])  # the permutation is not vacuous


def test_residual_update_ratios_are_the_relative_change_per_layer(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """``return_attention=True`` records, per encoder layer, ||H_out - H_in|| /
    (||H_in|| + 1e-8), with H_in the layer's actual input (layer 1's input is layer 0's
    output), recomputed here from forward hooks. The manual-softmax path it forces gives
    the same prediction as the fused SDPA path, which returns no weights and no ratios.
    """
    model = _eval_model(cell_graph)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        seen.append((args[0].detach(), output[0].detach()))

    handles = [
        layer.register_forward_hook(record) for layer in model.transformer_layers
    ]
    with torch.no_grad():
        pred, reps = model(cell_graph, batch, return_attention=True)
    for handle in handles:
        handle.remove()
    assert len(seen) == NUM_LAYERS
    assert torch.equal(seen[1][0], seen[0][1])
    expected = [
        (torch.norm(out - inp) / (torch.norm(inp) + 1e-8)).item() for inp, out in seen
    ]
    assert reps["residual_update_ratios"] == pytest.approx(expected, rel=1e-6)
    assert all(
        isinstance(layer, GraphRegularizedTransformerLayer)
        for layer in model.transformer_layers
    )
    with torch.no_grad():
        pred_fused, reps_fused = model(cell_graph, batch)
    torch.testing.assert_close(pred_fused, pred, atol=1e-5, rtol=1e-5)
    assert reps_fused["attention_weights"] is None
    assert reps_fused["residual_update_ratios"] is None


def test_every_parameter_of_the_default_multitask_model_receives_a_gradient(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Train mode (dropout 0.1), all three default heads: one backward of the summed
    prediction and head outputs gives every parameter a gradient with a nonzero entry.
    """
    model = _make_model(cell_graph, _full_heads_config())
    model.train()
    pred, reps = model(cell_graph, batch)
    loss = pred.sum() + sum(out.sum() for out in reps["head_outputs"].values())
    loss.backward()
    starved = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or torch.count_nonzero(p.grad).item() == 0
    ]
    assert starved == []


def test_num_parameters_total_omits_the_optional_modules(
    cell_graph: HeteroData,
) -> None:
    """Finding: ``num_parameters`` (lines 2952-2985) tallies the embeddings, encoder,
    transform, interaction head and the three named heads, so its ``total`` (logged by
    ``main`` at line 3382) leaves out every optional module. 2 layers, hidden 16:

    * counted: gene_embedding 128, cls 16, encoder 2 * 3280 = 6560, transform 3280,
      interaction head 545, per-gene head Linear(16 + 2, 16) + Linear(16, 1) = 304 + 17
      = 321 (bilinear rank 2 widens its input); total 10850;
    * omitted: propagation (1 graph x 2 hops + 1 = 3 features) Linear(3, 16) 64 +
      Linear(16, 16) 272 + gate 1 = 337; cross-gene rank 4: Linear(16, 4) 68 +
      Linear(20, 16) 336 + Linear(16, 16) 272 = 676; Perceiver, 4 latents: 64 + two
      MultiheadAttention 2176 + two LayerNorms 64 + Linear(16, 32) 544 + Linear(32, 16)
      528 + gate 1 = 3377; observed-label Linear(2, 16) 48 + Linear(16, 16) 272 = 320
      (its forced-on gate is a buffer); bilinear 2 * 16 * 2 = 64; response basis rank 3:
      16 * 3 = 48 + Linear(32, 16) 528 + Linear(16, 3) 51 = 627; sum 5401;
    * so the module has 10850 + 5401 = 16251 trainable parameters. Pinned until the tally
      includes them.
    """
    model = _eval_model(
        cell_graph,
        perturbation_propagation_config={"enabled": True},
        cross_gene_config={"enabled": True, "rank": 4},
        post_perturbation_mixing_config={"enabled": True, "num_latents": 4},
        observed_label_config={"enabled": True},
        heads_config={"per_gene": {"bilinear_rank": 2, "response_basis_rank": 3}},
    )
    assert model.num_parameters == {
        "gene_embedding": 128,
        "embedding_preprocessor": 0,
        "cls_token": 16,
        "transformer_layers": 6560,
        "perturbation_transform": 3280,
        "perturbation_head": 545,
        "per_gene_head": 321,
        "total": 10850,
    }
    omitted = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in [
            ("perturbation_propagation", model.perturbation_propagation),
            ("cross_gene_mixing", model.cross_gene_mixing),
            ("post_perturbation_mixing", model.post_perturbation_mixing),
            ("observed_label_encoder", model.observed_label_encoder),
            ("bilinear", model.bilinear),
            ("response_basis", model.response_basis),
        ]
        if module is not None
    }
    assert omitted == {
        "perturbation_propagation": 337,
        "cross_gene_mixing": 676,
        "post_perturbation_mixing": 3377,
        "observed_label_encoder": 320,
        "bilinear": 64,
        "response_basis": 627,
    }
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 16251


# Constructor and graph-regularization branches the siblings leave open ------------- #
def test_preprocessor_places_a_dropout_after_every_block(
    cell_graph: HeteroData,
) -> None:
    """Dropout 0.25 and a 5-wide precomputed embedding: the default 2-layer preprocessor
    is Linear(5, 10), LayerNorm, GELU, Dropout(0.25), Linear(10, 16), LayerNorm,
    Dropout(0.25) (no GELU after the last block). Dropout adds no parameters, so the
    count is the 288 of the dropout-0 build (60 + 20 + 176 + 32).
    """
    graph = cell_graph.clone()
    graph["gene"].x = torch.zeros(GENE_NUM, 5)
    model = _eval_model(
        graph,
        node_embeddings={"toy": [Data(embeddings={"a": torch.zeros(1, 5)})]},
        dropout=0.25,
    )
    preprocessor = model.embedding_preprocessor
    assert preprocessor is not None
    layout = [
        (type(m).__name__, m.p if isinstance(m, nn.Dropout) else None)
        for m in preprocessor
    ]
    assert layout == [
        ("Linear", None),
        ("LayerNorm", None),
        ("GELU", None),
        ("Dropout", 0.25),
        ("Linear", None),
        ("LayerNorm", None),
        ("Dropout", 0.25),
    ]
    assert model.num_parameters["embedding_preprocessor"] == 288


def _regularized(
    cell_graph: HeteroData, heads: dict[str, dict[str, Any]]
) -> CellGraphTransformer:
    return _eval_model(
        cell_graph,
        graph_reg_lambda=0.5,
        graph_regularization_config={"regularized_heads": heads},
    )


def test_edgeless_regularized_graph_adds_zero_not_nan(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Without row sampling every row is kept; an edgeless graph's target rows are all 0,
    so its KL is 0 and its edge count is 0, and the ``edges_k > 0`` guard (line 2617)
    skips the 0 * KL / (0 / 8) that would be NaN. The total therefore equals the
    ``physical``-only model's loss at the same seed, bit for bit.
    """
    graph = cell_graph.clone()
    graph["gene", "empty", "gene"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    physical = {"layer": 0, "head": 1, "lambda": 0.5}
    both = _regularized(
        graph, {"empty": {"layer": 0, "head": 0, "lambda": 0.5}, "physical": physical}
    )
    alone = _regularized(graph, {"physical": physical})
    with torch.no_grad():
        loss_both = both(graph, batch)[1]["graph_reg_loss"]
        loss_alone = alone(graph, batch)[1]["graph_reg_loss"]
    assert both._edge_count_cache == {"empty": 0, "physical": 4}
    assert loss_alone.item() > 0.0
    assert torch.equal(loss_both, loss_alone)


def test_edge_count_is_computed_once_and_then_read_from_the_cache(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """``_edge_count`` counts the 4 ``physical`` nonzeros on the first forward and caches
    them. The term divides by (edges / 8), so overwriting the cached 4 with 8 halves the
    next forward's loss, which only happens if the second forward reads the cache rather
    than recounting the unchanged matrix.
    """
    model = _regularized(
        cell_graph, {"physical": {"layer": 0, "head": 1, "lambda": 0.5}}
    )
    with torch.no_grad():
        first = model(cell_graph, batch)[1]["graph_reg_loss"]
        assert model._edge_count_cache == {"physical": 4}
        model._edge_count_cache["physical"] = 8
        second = model(cell_graph, batch)[1]["graph_reg_loss"]
    assert first.item() > 0.0
    assert second.item() == pytest.approx(first.item() / 2, rel=1e-6)
