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
    """The parameter tally reports each metabolism head and a consistent total."""
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config())
    counts = model.num_parameters
    for name in ("betaxanthin_head", "beta_carotene_head", "mulleder19_head"):
        assert counts[name] > 0
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")


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
