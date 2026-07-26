# tests/torchcell/models/test_cell_graph_transformer_metabolism.py
"""Tests for the CGT-Metabolism fork.

The load-bearing test is PARITY: with the metabolism heads disabled, the fork must
reproduce the parent :class:`CellGraphTransformer` bit-for-bit at a fixed seed. That is
not automatic even though the encoder is inherited -- any parameter the subclass creates
during ``__init__`` consumes the global RNG stream, so a head built in the wrong place
shifts every encoder weight and silently invalidates comparisons against Fig-3 runs.
"""

from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.cell_graph_transformer_metabolism import (
    CellGraphTransformerMetabolism,
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

    Their units are mutually incomparable (centred fluorescence vs an ordinal -5..+5),
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
    with pytest.raises(ValueError, match="needs kind='scalar' or kind='vector'"):
        _build(CellGraphTransformerMetabolism, {"mulleder19": {"output_dim": 19}})


def test_num_parameters_includes_metabolism_heads() -> None:
    """The parameter tally reports each metabolism head and a consistent total."""
    model = _build(CellGraphTransformerMetabolism, _metabolism_heads_config())
    counts = model.num_parameters
    for name in ("betaxanthin_head", "beta_carotene_head", "mulleder19_head"):
        assert counts[name] > 0
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")
