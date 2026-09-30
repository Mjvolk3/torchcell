# tests/torchcell/models/test_cell_graph_transformer_metabolism.py
# [[tests.torchcell.models.test_cell_graph_transformer_metabolism]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_cell_graph_transformer_metabolism.py
"""Tests for the CGT-Metabolism fork.

The load-bearing test is PARITY: with the metabolism heads disabled, the fork must
reproduce the parent :class:`CellGraphTransformer` bit-for-bit at a fixed seed. That is
not automatic even though the encoder is inherited -- any parameter the subclass creates
during ``__init__`` consumes the global RNG stream, so a head built in the wrong place
shifts every encoder weight and silently invalidates comparisons against Fig-3 runs.

The second half pins the readout heads and the flux wiring. The flux heads run on a real
:class:`~torchcell.metabolism.flux_layer.FluxLayer` built on the four-reaction toy network
of ``tests/torchcell/metabolism/test_flux_layer.py`` (metabolites A, B, C; r0 -> A,
r1 A -> B, r2 B -> A, r3 B ->), with the eight model genes named g0..g7 so GEM genes
g0, g1, g2 map to tokens 0, 1, 2. Closed forms: ``perturbed_gene_pool`` is a per-sample
mean over the listed tokens; a ``FluxMetaboliteHead`` is ``scale * log1p(max(omega, 0))
+ bias`` at the named metabolites (scale 1, bias 0 at init); a ``FluxScalarHead`` is
``precursor(log1p(max(omega_P, 0))) + dense(v)``.

2026.09.30 - Phase 13 additions. Exact parameter counts on the Track A config
(hidden 16, two layers): the parent tallies 128 + 16 + 6560 + 3280 + 545 = 10529; a
scalar head with both pools reads 3 * 16 = 48 features, Linear(48, 16) + Linear(16, 1)
= 784 + 17 = 801 (each of betaxanthin and beta_carotene); the 19-column vector head is
784 + Linear(16, 19) = 784 + 323 = 1107; total 10529 + 801 + 801 + 1107 = 13238, equal
to ``sum(p.numel())`` since no flux layer is attached. Structural identities on the
fixture batch: permuting the samples (and relabeling ``perturbation_indices_batch``)
permutes every output row; reordering the perturbed genes listed within a sample
changes nothing; changing sample 2's genotype leaves samples 0 and 1 unchanged; a
sample with no perturbed gene reads the zero pool, so its head output is exactly
``mlp([h_CLS, mean_i H[b, i], 0])``. Closed forms on hidden 2 with hand-set weights pin
the concatenation order ``[h_CLS, gene mean, perturbed pool]`` of ``ProductScalarHead``
(35.5 and 0.5, derived in the test) and the ``view(B, F, param_dim)`` layout of
``MetabolomeVectorHead`` (column k, parameter p is linear output ``k * param_dim + p``).
"""

import math
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from torchcell.metabolism.constraints import (
    CatalyticUnits,
    GemTensors,
    TableCoverage,
    ThermoMode,
    ThermoTable,
    independent_balance_rows,
)
from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig
from torchcell.models.cell_graph_transformer_metabolism import (
    CellGraphTransformerMetabolism,
    FluxMetaboliteHead,
    FluxScalarHead,
    MetabolomeVectorHead,
    ProductScalarHead,
    perturbed_gene_pool,
)
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

GENE_NUM = 8
HIDDEN = 16
NUM_LAYERS = 2
NUM_HEADS = 4
BATCH_SIZE = 3
MULLEDER_DIM = 19


def _make_cell_graph() -> HeteroData:
    """Tiny cell_graph with a gene-gene edge type (same shape as the WS7 test)."""
    cg = HeteroData()
    cg["gene"].num_nodes = GENE_NUM
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    return cg


def _make_batch() -> HeteroData:
    """Tiny perturbation batch: 3 genotypes with varying perturbed gene counts."""
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor(
        [1, 2, 3, 0, 4, 5], dtype=torch.long
    )
    batch["gene"].perturbation_indices_batch = torch.tensor(
        [0, 0, 1, 2, 2, 2], dtype=torch.long
    )
    return batch


def _metabolism_heads_config() -> dict[str, Any]:
    return {
        "betaxanthin": {"kind": "scalar", "output_dim": 1, "use_gene_pool": True},
        "beta_carotene": {"kind": "scalar", "output_dim": 1, "use_gene_pool": True},
        "mulleder19": {
            "kind": "vector",
            "output_dim": MULLEDER_DIM,
            "use_gene_pool": True,
        },
    }


def _build(cls: type[CellGraphTransformer], heads_config: Any, seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return cls(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        cell_graph=_make_cell_graph(),
        heads_config=heads_config,
    )


def test_encoder_and_pert_operator_parity_with_parent() -> None:
    """Heads disabled: the fork is numerically identical to the parent at one seed.

    Checks BOTH the parameters (so the RNG stream was consumed in the same order) and
    the forward outputs (so the encoder + equivariant perturbation operator are the
    parent's, unmodified).
    """
    parent = _build(CellGraphTransformer, None, seed=17)
    fork = _build(CellGraphTransformerMetabolism, None, seed=17)
    parent.eval()
    fork.eval()

    p_state = parent.state_dict()
    f_state = fork.state_dict()
    assert set(p_state) == set(f_state)
    for k in p_state:
        assert torch.equal(p_state[k], f_state[k]), f"parameter {k} differs"

    cg = _make_cell_graph()
    batch = _make_batch()
    with torch.no_grad():
        p_pred, p_reps = parent(cg, batch)
        f_pred, f_reps = fork(cg, batch)

    assert torch.allclose(p_pred, f_pred, atol=0, rtol=0)
    for key in ("h_CLS", "H_genes", "H_genes_pert", "graph_reg_loss"):
        assert torch.allclose(p_reps[key], f_reps[key], atol=0, rtol=0), key
    assert f_reps["head_outputs"] == {}


def test_encoder_parity_holds_with_metabolism_heads_active() -> None:
    """Adding the three heads must not perturb the encoder init at the same seed.

    The heads are constructed after ``super().__init__()`` returns, so every inherited
    parameter must still match the parent's -- this is what makes a metabolism run
    comparable to a Fig-3 run at the same seed.
    """
    parent = _build(CellGraphTransformer, None, seed=17)
    fork = _build(CellGraphTransformerMetabolism, _metabolism_heads_config(), seed=17)

    p_state = parent.state_dict()
    f_state = fork.state_dict()
    assert set(p_state) <= set(f_state)
    for k in p_state:
        assert torch.equal(p_state[k], f_state[k]), f"parameter {k} differs"

    cg = _make_cell_graph()
    batch = _make_batch()
    parent.eval()
    fork.eval()
    with torch.no_grad():
        p_pred, p_reps = parent(cg, batch)
        f_pred, f_reps = fork(cg, batch)
    assert torch.allclose(p_pred, f_pred, atol=0, rtol=0)
    assert torch.allclose(
        p_reps["H_genes_pert"], f_reps["H_genes_pert"], atol=0, rtol=0
    )


def test_metabolism_head_shapes() -> None:
    """The three heads emit [B, 1], [B, 1] and [B, 19]."""
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config())
    model.eval()
    with torch.no_grad():
        _, reps = model(_make_cell_graph(), _make_batch())
    heads = reps["head_outputs"]
    assert set(heads) == {"betaxanthin", "beta_carotene", "mulleder19"}
    assert heads["betaxanthin"].shape == (BATCH_SIZE, 1)
    assert heads["beta_carotene"].shape == (BATCH_SIZE, 1)
    assert heads["mulleder19"].shape == (BATCH_SIZE, MULLEDER_DIM)


def test_heads_do_not_share_parameters() -> None:
    """Betaxanthin and beta_carotene are separate modules with separate weights.

    Their units are mutually incomparable (centered fluorescence vs an ordinal -5..+5),
    so sharing a readout would force one scale onto both.
    """
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config())
    bx = {id(p) for p in model.betaxanthin_head.parameters()}
    bc = {id(p) for p in model.beta_carotene_head.parameters()}
    ml = {id(p) for p in model.mulleder19_head.parameters()}
    assert bx and bc and ml
    assert bx.isdisjoint(bc)
    assert bx.isdisjoint(ml)
    assert bc.isdisjoint(ml)


def test_scalar_head_rejects_wide_output_dim() -> None:
    """kind='scalar' with output_dim > 1 is a config error, not a silent broadcast."""
    with pytest.raises(ValueError, match="scalar head emits exactly one value"):
        _build(
            CellGraphTransformerMetabolism,
            {"betaxanthin": {"kind": "scalar", "output_dim": 19}},
        )


def test_missing_kind_is_rejected() -> None:
    """A metabolism head spec without `kind` raises rather than guessing."""
    with pytest.raises(ValueError, match=r"needs kind in \{'scalar','vector'"):
        _build(CellGraphTransformerMetabolism, {"mulleder19": {"output_dim": 19}})


def test_num_parameters_includes_metabolism_heads() -> None:
    """Exact tally (module docstring): the parent's 10529 plus heads 801, 801 and 1107
    gives 13238, which is every parameter of the module when no flux layer is attached.
    """
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config())
    parent = _build(CellGraphTransformer, None)
    counts = model.num_parameters
    assert counts == {
        **{k: v for k, v in parent.num_parameters.items() if k != "total"},
        "betaxanthin_head": 801,
        "beta_carotene_head": 801,
        "mulleder19_head": 1107,
        "total": 13238,
    }
    assert parent.num_parameters["total"] == 10529
    assert sum(p.numel() for p in model.parameters()) == 13238


def test_forward_accepts_every_parent_keyword() -> None:
    """The subclass must not pin the parent's forward signature.

    REGRESSION TEST, and the bug it guards was live on main. This override exists only to
    append heads, so it has no opinion on what arguments the parent takes -- but it used to
    NAME them, which silently made it a signature contract. When the masked-label objective
    added ``observed_values`` / ``observed_mask`` to the parent and the trainer began passing
    them unconditionally, every metabolism run died with

        TypeError: ...forward() got an unexpected keyword argument 'observed_values'

    at the FIRST TRAINING BATCH -- i.e. ~20 minutes in, after the dataset and embeddings had
    loaded, on a job that had already been queued and scheduled.

    Every other test here calls ``model(cell_graph, batch)`` positionally, which is precisely
    why none of them caught it: the trainer's call and the tests' call had diverged. This test
    calls it the way `train_cgt_multitask` does -- by keyword, with the full argument set the
    parent accepts -- so the two cannot drift apart again.
    """
    import inspect

    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config())
    model.eval()
    parent_params = inspect.signature(CellGraphTransformer.forward).parameters
    # Anything the PARENT accepts beyond (self, cell_graph, batch, return_attention) is an
    # argument the trainer may pass; assert the subclass tolerates all of them at their
    # defaults rather than enumerating a list that would itself go stale.
    extra = {
        name: p.default
        for name, p in parent_params.items()
        if name not in ("self", "cell_graph", "batch", "return_attention")
        and p.kind is not inspect.Parameter.VAR_KEYWORD
    }
    with torch.no_grad():
        _, reps = model(
            cell_graph=_make_cell_graph(),
            batch=_make_batch(),
            return_attention=False,
            **extra,
        )
    assert set(reps["head_outputs"]) >= {"betaxanthin", "beta_carotene", "mulleder19"}


# --- readout heads and the flux wiring ---------------------------------------------- #
def _toy_flux_layer() -> FluxLayer:
    """The four-reaction toy GEM of test_flux_layer.py behind an 8-gene token universe."""
    s = (
        torch.tensor(
            [[1.0, -1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
        )
        .to_sparse_coo()
        .coalesce()
    )
    rows, _ = independent_balance_rows(s)
    mask = torch.tensor([True, True, False])
    gem = GemTensors(
        s=s,
        lb=torch.zeros(4),
        ub=torch.full((4,), 10.0),
        met_ids=["A", "B", "C"],
        rxn_ids=["r0", "r1", "r2", "r3"],
        catalytic_units=CatalyticUnits(
            unit_gene_index=torch.tensor([[0, 0, 1], [0, 1, 2]]),
            unit_reaction=torch.tensor([1, 2]),
            n_units=2,
            n_multigene_units=1,
            n_reactions_with_gpr=2,
            gene_ids=["g0", "g1", "g2"],
        ),
        thermo=ThermoTable(
            met_delta_g=torch.tensor([0.0, -10.0, 0.0]),
            met_mask=mask,
            rxn_delta_g=torch.zeros(4),
            rxn_mask=torch.zeros(4, dtype=torch.bool),
            met_coverage=TableCoverage.of(mask.numpy()),
            rxn_coverage=TableCoverage.of(torch.zeros(4, dtype=torch.bool).numpy()),
            source_paths={},
            sha256={},
        ),
        independent_rows=rows,
        biomass_index=3,
        exchange_indices=torch.tensor([0, 3]),
        n_metabolites=3,
        n_reactions=4,
    )
    config = FluxLayerConfig(
        hidden_dim=HIDDEN,
        reaction_embed_dim=4,
        thermo_mode=ThermoMode.OFF,
        use_enzyme_capacity=False,
        use_protein_budget=False,
    )
    return FluxLayer(
        gem, [f"g{i}" for i in range(GENE_NUM)], config=config, kcat_per_s=torch.ones(2)
    )


def _flux_heads_config() -> dict[str, Any]:
    return {
        "aa": {"kind": "flux_metabolite", "metabolite_indices": [1, 0]},
        "bx": {"kind": "flux_scalar", "precursor_indices": [0, 1, 2]},
        "pooled": {"kind": "scalar", "output_dim": 1},
    }


def _flux_model() -> CellGraphTransformerMetabolism:
    torch.manual_seed(0)
    return CellGraphTransformerMetabolism(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=1,
        num_attention_heads=NUM_HEADS,
        cell_graph=_make_cell_graph(),
        heads_config=_flux_heads_config(),
        dropout=0.0,
        flux_layer=_toy_flux_layer(),
    )


def test_perturbed_gene_pool_is_the_per_sample_mean_over_listed_tokens() -> None:
    """H[b, i] = [10b + i, -(10b + i)]; sample 0 lists genes {0, 2}, sample 1 gene 1,
    sample 2 nothing: means [1, -1], [11, -11], and the zero vector (not NaN).
    """
    values = torch.tensor([[10.0 * b + i for i in range(3)] for b in range(3)])
    h = torch.stack([values, -values], dim=-1)  # [3, 3, 2]
    pooled = perturbed_gene_pool(h, torch.tensor([0, 2, 1]), torch.tensor([0, 0, 1]))
    assert torch.equal(pooled, torch.tensor([[1.0, -1.0], [11.0, -11.0], [0.0, 0.0]]))


def test_flux_metabolite_head_reads_log_turnover_at_the_named_metabolites() -> None:
    """Indices [2, 0]. Row 0 turnover [0, e - 1, 3]: log1p(3) = log 4 at column 0 and
    log1p(0) = 0 at column 1. Row 1 turnover [-1, 1, 0]: the negative entry clamps to 0,
    so both columns are 0. With scale [[2, 3], [1, 1]] and bias [[1, 0], [0, 0]] the
    distributional form is h * scale + bias per column.
    """
    turnover = torch.tensor([[0.0, math.e - 1.0, 3.0], [-1.0, 1.0, 0.0]])
    head = FluxMetaboliteHead(torch.tensor([2, 0]))
    assert head.output_dim == 2
    torch.testing.assert_close(
        head(turnover), torch.tensor([[math.log(4.0), 0.0], [0.0, 0.0]])
    )
    dist = FluxMetaboliteHead(torch.tensor([2, 0]), param_dim=2)
    with torch.no_grad():
        dist.scale.copy_(torch.tensor([[2.0, 3.0], [1.0, 1.0]]))
        dist.bias.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    log4 = math.log(4.0)
    torch.testing.assert_close(
        dist(turnover),
        torch.tensor(
            [[[2 * log4 + 1, 3 * log4], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]
        ),
    )


def test_flux_scalar_head_adds_the_precursor_and_dense_terms() -> None:
    """Precursor weights [1, 2], bias 0.5 on turnover [e - 1, 0] at indices [0, 1]:
    1 * 1 + 2 * 0 + 0.5 = 1.5. Dense weights [1, 1, 1], bias 0 on v = [1, 2, 3] add 6,
    so 7.5; without the dense term it is 1.5; param_dim 2 gives [B, 1, 2].
    """
    turnover = torch.tensor([[math.e - 1.0, 0.0, 5.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])

    def build(dense: bool) -> FluxScalarHead:
        head = FluxScalarHead(3, torch.tensor([0, 1]), use_dense_flux=dense)
        with torch.no_grad():
            head.precursor.weight.copy_(torch.tensor([[1.0, 2.0]]))
            head.precursor.bias.fill_(0.5)
            if dense:
                head.dense.weight.fill_(1.0)
                head.dense.bias.zero_()
        return head

    torch.testing.assert_close(build(True)(v, turnover), torch.tensor([[7.5]]))
    only_precursor = build(False)
    assert not hasattr(only_precursor, "dense")
    torch.testing.assert_close(only_precursor(v, turnover), torch.tensor([[1.5]]))
    wide = FluxScalarHead(3, torch.tensor([0, 1]), param_dim=2)
    assert wide(v, turnover).shape == (1, 1, 2)


def test_pooled_heads_widths_errors_and_distributional_shapes() -> None:
    """Input width is hidden * (1 + gene_pool + pert_pool); a pert-pool head called
    without the pool raises; a vector head narrower than 2 is rejected.

    With both pools off the head reads only h_CLS, so all three genotypes get one value.
    """
    torch.manual_seed(0)
    h_cls, h = torch.randn(HIDDEN), torch.randn(BATCH_SIZE, GENE_NUM, HIDDEN)
    pool = torch.randn(BATCH_SIZE, HIDDEN)
    first = ProductScalarHead(HIDDEN).mlp[0]
    assert isinstance(first, nn.Linear) and first.in_features == 3 * HIDDEN
    cls_only = ProductScalarHead(
        HIDDEN, use_gene_pool=False, use_pert_pool=False, dropout=0.0
    )(h_cls, h)
    assert cls_only.shape == (BATCH_SIZE, 1)
    assert torch.equal(cls_only[0], cls_only[1]) and torch.equal(
        cls_only[1], cls_only[2]
    )
    assert ProductScalarHead(HIDDEN, param_dim=2)(h_cls, h, pool).shape == (
        BATCH_SIZE,
        1,
        2,
    )
    assert MetabolomeVectorHead(HIDDEN, 4, param_dim=3)(h_cls, h, pool).shape == (
        BATCH_SIZE,
        4,
        3,
    )
    missing = "use_pert_pool=True but pert_pool was not supplied"
    with pytest.raises(ValueError, match=missing):
        ProductScalarHead(HIDDEN)(h_cls, h)
    with pytest.raises(ValueError, match=missing):
        MetabolomeVectorHead(HIDDEN, 4)(h_cls, h)
    with pytest.raises(
        ValueError, match="MetabolomeVectorHead output_dim must be >= 2, got 1;"
    ):
        MetabolomeVectorHead(HIDDEN, 1)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (
            "flux_metabolite",
            "head 'x' is kind='flux_metabolite' but no flux_layer was supplied",
        ),
        (
            "flux_scalar",
            "head 'x' is kind='flux_scalar' but no flux_layer was supplied.",
        ),
    ],
)
def test_flux_heads_require_a_flux_layer(kind: str, message: str) -> None:
    """A mechanistic head with no flux vector to read is a construction error."""
    with pytest.raises(ValueError, match=message):
        _build(CellGraphTransformerMetabolism, {"x": {"kind": kind}})


def test_flux_heads_read_the_flux_layer_turnover() -> None:
    """The model runs the flux layer on (H_genes_pert, perturbed-gene pool), stores the
    output under ``reps["flux"]``, and feeds its turnover to the flux heads: at init the
    metabolite head is exactly log1p(turnover[:, [1, 0]]), and the scalar head is its
    linear read of v and the precursor turnover. The pooled scalar head is not a flux
    head. Every parameter, the flux layer's included, receives a gradient.
    """
    model = _flux_model()
    assert model.flux_head_names == ["aa", "bx"]
    assert model.metabolism_head_names == ["aa", "bx", "pooled"]
    pred, reps = model(_make_cell_graph(), _make_batch())
    heads = reps["head_outputs"]
    v = reps["flux"]["v"]
    layer = model.flux_layer
    assert isinstance(layer, FluxLayer)
    turnover = layer.turnover(v)
    torch.testing.assert_close(
        heads["aa"], torch.log1p(turnover[:, [1, 0]].clamp(min=0))
    )
    bx = model.bx_head
    assert isinstance(bx, FluxScalarHead)
    torch.testing.assert_close(
        heads["bx"], bx.precursor(torch.log1p(turnover.clamp(min=0))) + bx.dense(v)
    )
    assert heads["pooled"].shape == (BATCH_SIZE, 1)
    assert v.shape == (BATCH_SIZE, 4)
    (pred.sum() + sum(t.sum() for t in heads.values())).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_num_parameters_leaves_out_the_flux_layer() -> None:
    """Finding: ``num_parameters`` (cell_graph_transformer_metabolism.py:574-584) adds
    each head to the parent's tally but never the flux layer, whose 386 parameters train.

    Heads: aa is scale + bias [2, 1] = 4; bx is Linear(3, 1) + Linear(4, 1) = 4 + 5 = 9;
    pooled is Linear(48, 16) + Linear(16, 1) = 784 + 17 = 801. Parent: 128 + 16 + 3280 +
    3280 + 545 = 7249. Reported total 7249 + 4 + 9 + 801 = 8063; the module holds
    8063 + 386 = 8449. The flux layer's 386, every one trainable: reaction_embedding
    4 x 4 = 16, flux_mlp Linear(20, 16) = 320 + 16, Linear(16, 1) = 16 + 1, availability
    Linear(16, 1) = 16 + 1, so 16 + 320 + 16 + 16 + 1 + 16 + 1 = 386.
    """
    model = _flux_model()
    counts = model.num_parameters
    assert (counts["aa_head"], counts["bx_head"], counts["pooled_head"]) == (4, 9, 801)
    assert counts["total"] == 8063
    assert "flux_layer" not in counts
    layer = model.flux_layer
    assert isinstance(layer, FluxLayer)
    flux = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert flux == 386
    assert [p.numel() for p in layer.parameters()] == [16, 320, 16, 16, 1, 16, 1]
    assert sum(p.numel() for p in model.parameters()) == 8449


# --- 2026.09.30 Phase 13: structural identities and closed-form head wiring ---------- #
def _track_a_outputs(
    model: CellGraphTransformerMetabolism, indices: list[int], owners: list[int]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor(indices, dtype=torch.long)
    batch["gene"].perturbation_indices_batch = torch.tensor(owners, dtype=torch.long)
    with torch.no_grad():
        pred, reps = model(_make_cell_graph(), batch)
    return pred, reps["head_outputs"]


def test_sample_permutation_permutes_every_output_row() -> None:
    """Fixture samples {1, 2}, {3}, {0, 4, 5} reordered as ({0, 4, 5}, {1, 2}, {3}):
    predictions and all three heads come back in row order [2, 0, 1].
    """
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config()).eval()
    pred, heads = _track_a_outputs(model, [1, 2, 3, 0, 4, 5], [0, 0, 1, 2, 2, 2])
    pred_p, heads_p = _track_a_outputs(model, [0, 4, 5, 1, 2, 3], [0, 0, 0, 1, 1, 2])
    order = [2, 0, 1]
    torch.testing.assert_close(pred_p, pred[order])
    assert set(heads_p) == {"betaxanthin", "beta_carotene", "mulleder19"}
    for name, out in heads.items():
        torch.testing.assert_close(heads_p[name], out[order])


def test_gene_order_within_a_sample_is_irrelevant_and_samples_are_isolated() -> None:
    """Listing each sample's genes in another order gives the same outputs; replacing
    sample 2's genotype by {6} leaves rows 0 and 1 of every head unchanged.
    """
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config()).eval()
    pred, heads = _track_a_outputs(model, [1, 2, 3, 0, 4, 5], [0, 0, 1, 2, 2, 2])
    pred_r, heads_r = _track_a_outputs(model, [2, 1, 3, 5, 0, 4], [0, 0, 1, 2, 2, 2])
    torch.testing.assert_close(pred_r, pred)
    for name, out in heads.items():
        torch.testing.assert_close(heads_r[name], out)
    _, heads_s = _track_a_outputs(model, [1, 2, 3, 6], [0, 0, 1, 2])
    for name, out in heads.items():
        torch.testing.assert_close(heads_s[name][:2], out[:2])
        assert not torch.allclose(heads_s[name][2], out[2])


def test_a_sample_without_perturbations_reads_the_zero_pool() -> None:
    """Sample 1 lists no gene (owners [0, 0, 2]), so its perturbed pool is the zero
    vector and the betaxanthin output is the head MLP on ``[h_CLS, gene mean, 0]``.
    """
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config()).eval()
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([1, 2, 5])
    batch["gene"].perturbation_indices_batch = torch.tensor([0, 0, 2])
    with torch.no_grad():
        _, reps = model(_make_cell_graph(), batch)
        h = reps["H_genes_pert"]
        head = model.betaxanthin_head
        assert isinstance(head, ProductScalarHead)
        expected = head.mlp(
            torch.cat([reps["h_CLS"], h[1].mean(dim=0), torch.zeros(HIDDEN)])
        )
    assert h.shape == (3, GENE_NUM, HIDDEN)
    torch.testing.assert_close(reps["head_outputs"]["betaxanthin"][1], expected)


def test_seeded_build_is_deterministic_and_head_order_sets_head_init() -> None:
    """Same seed and config: identical state and outputs. Listing the heads in the other
    order leaves the encoder bit-identical (heads are built after the parent) but hands
    betaxanthin a different slice of the RNG stream, so its weights differ.
    """
    config = _metabolism_heads_config()
    first = _build(CellGraphTransformerMetabolism, config, seed=3).eval()
    second = _build(CellGraphTransformerMetabolism, config, seed=3).eval()
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        assert torch.equal(a, b), name
    pred_a, heads_a = _track_a_outputs(first, [1, 2, 3, 0, 4, 5], [0, 0, 1, 2, 2, 2])
    pred_b, heads_b = _track_a_outputs(second, [1, 2, 3, 0, 4, 5], [0, 0, 1, 2, 2, 2])
    assert torch.equal(pred_a, pred_b)
    assert all(torch.equal(heads_a[k], heads_b[k]) for k in heads_a)
    swapped = _build(
        CellGraphTransformerMetabolism, dict(reversed(config.items())), seed=3
    )
    assert swapped.metabolism_head_names == [
        "mulleder19",
        "beta_carotene",
        "betaxanthin",
    ]
    assert torch.equal(swapped.gene_embedding.weight, first.gene_embedding.weight)
    first_linear = first.betaxanthin_head.mlp[0]
    swapped_linear = swapped.betaxanthin_head.mlp[0]
    assert isinstance(first_linear, nn.Linear) and isinstance(swapped_linear, nn.Linear)
    assert not torch.equal(first_linear.weight, swapped_linear.weight)


def test_track_a_gradient_reaches_every_parameter() -> None:
    """Train mode, loss = predictions plus all three heads: no parameter is left without
    a gradient, the CLS token included (every pooled head reads it).
    """
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config()).train()
    pred, reps = model(_make_cell_graph(), _make_batch())
    heads = reps["head_outputs"]
    assert torch.isfinite(pred).all()
    assert all(torch.isfinite(t).all() for t in heads.values())
    (pred.sum() + sum(t.sum() for t in heads.values())).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_inherited_head_keys_are_left_to_the_parent() -> None:
    """``global`` is built by the parent as ``global_head`` and is not a metabolism
    head; the metabolism list and the output dict hold both entries.
    """
    model = _build(
        CellGraphTransformerMetabolism,
        {"global": {"output_dim": 3}, "bx": {"kind": "scalar"}},
    ).eval()
    assert model.metabolism_head_names == ["bx"]
    assert model.flux_head_names == []
    assert isinstance(model.bx_head, ProductScalarHead)
    _, heads = _track_a_outputs(model, [1, 2, 3, 0, 4, 5], [0, 0, 1, 2, 2, 2])
    assert {k: tuple(v.shape) for k, v in heads.items()} == {
        "global": (3, 3),
        "bx": (3, 1),
    }


@pytest.mark.parametrize(
    ("spec", "shown"), [(None, "None"), ({"kind": "Scalar"}, "'Scalar'")]
)
def test_a_null_spec_or_a_miscased_kind_names_the_value(
    spec: dict[str, str] | None, shown: str
) -> None:
    """A ``None`` spec becomes ``{}`` and fails on ``kind``; kinds are case-sensitive."""
    message = (
        "metabolism head 'x' needs kind in "
        "{'scalar','vector','flux_scalar','flux_metabolite'} in its heads_config spec, "
        f"got {shown}."
    )
    with pytest.raises(ValueError) as excinfo:
        _build(CellGraphTransformerMetabolism, {"x": spec})
    assert str(excinfo.value) == message


def test_product_scalar_head_concatenates_cls_gene_mean_and_pert_pool() -> None:
    """Hidden 2, so the input is 6 wide: [h_CLS (2), gene mean (2), pert pool (2)].

    h_CLS = [1, 2]. Sample 0: genes [[1, 1], [3, 3]], mean [2, 2], pool [5, 6], input
    [1, 2, 2, 2, 5, 6]. Sample 1: genes [[0, 0], [0, -4]], mean [0, -2], pool [7, -8],
    input [1, 2, 0, -2, 7, -8]. Hidden unit 0 = x[0] + x[3] (CLS[0] + mean[1]): 3 and
    -1 -> relu 3 and 0. Hidden unit 1 = x[5] - 1 (pool[1] - 1): 5 and -9 -> 5 and 0.
    Output 10 * u0 + 1 * u1 + 0.5: sample 0 = 30 + 5 + 0.5 = 35.5, sample 1 = 0.5.
    """
    head = ProductScalarHead(2, dropout=0.0).eval()
    with torch.no_grad():
        first, last = head.mlp[0], head.mlp[3]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        first.weight.copy_(torch.tensor([[1.0, 0, 0, 1, 0, 0], [0.0, 0, 0, 0, 0, 1]]))
        first.bias.copy_(torch.tensor([0.0, -1.0]))
        last.weight.copy_(torch.tensor([[10.0, 1.0]]))
        last.bias.fill_(0.5)
        out = head(
            torch.tensor([1.0, 2.0]),
            torch.tensor([[[1.0, 1.0], [3.0, 3.0]], [[0.0, 0.0], [0.0, -4.0]]]),
            torch.tensor([[5.0, 6.0], [7.0, -8.0]]),
        )
    assert torch.equal(out, torch.tensor([[35.5], [0.5]]))


def test_vector_head_without_gene_pool_lays_out_columns_then_params() -> None:
    """Hidden 2, F = 2, param_dim = 3, gene pool off: input [h_CLS, pool] (4 wide).
    Hidden units u0 = CLS[0] = 2 and u1 = relu(pool[1]): 3 for sample 0, 0 for sample
    1. Linear output j = j * u0 + u1, so sample 0 is [3, 5, 7, 9, 11, 13] and sample 1
    [0, 2, 4, 6, 8, 10]; ``view(B, 2, 3)`` puts outputs 0-2 in column 0 and 3-5 in
    column 1. The gene tokens (all 100) are never read.
    """
    head = MetabolomeVectorHead(2, 2, use_gene_pool=False, dropout=0.0, param_dim=3)
    head.eval()
    with torch.no_grad():
        first, last = head.mlp[0], head.mlp[3]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        assert (first.in_features, last.out_features) == (4, 6)
        first.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0.0, 0, 0, 1]]))
        first.bias.zero_()
        last.weight.copy_(torch.tensor([[float(j), 1.0] for j in range(6)]))
        last.bias.zero_()
        out = head(
            torch.tensor([2.0, 9.0]),
            torch.full((2, 5, 2), 100.0),
            torch.tensor([[0.0, 3.0], [0.0, -1.0]]),
        )
    assert torch.equal(
        out,
        torch.tensor(
            [[[3.0, 5.0, 7.0], [9.0, 11.0, 13.0]], [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]]
        ),
    )


def test_vector_head_without_pert_pool_ignores_the_pool_argument() -> None:
    """``use_pert_pool=False``: the head is ``hidden * 2`` wide, runs with ``pert_pool``
    None, and gives the same output when a pool is passed anyway.
    """
    torch.manual_seed(8)
    head = MetabolomeVectorHead(HIDDEN, 4, use_pert_pool=False).eval()
    first = head.mlp[0]
    assert isinstance(first, nn.Linear) and first.in_features == 2 * HIDDEN
    h_cls, h = torch.randn(HIDDEN), torch.randn(BATCH_SIZE, GENE_NUM, HIDDEN)
    without = head(h_cls, h)
    with_pool = head(h_cls, h, torch.randn(BATCH_SIZE, HIDDEN))
    assert without.shape == (BATCH_SIZE, 4)
    torch.testing.assert_close(with_pool, without)
    expected = head.mlp(torch.cat([h_cls.expand(BATCH_SIZE, -1), h.mean(1)], dim=-1))
    torch.testing.assert_close(without, expected)
