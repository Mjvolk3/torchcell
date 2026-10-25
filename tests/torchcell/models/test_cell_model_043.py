# tests/torchcell/models/test_cell_model_043.py
# [[tests.torchcell.models.test_cell_model_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_cell_model_043.py
"""The 043 cell model (``torchcell.models.cell_model_043``) on tiny synthetic inputs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

from torchcell.data.experiment_ontology import (
    ExperimentOntologyVocabulary,
    OntologyNode,
)
from torchcell.data.representation_043 import GapKind, PositionGap
from torchcell.models.cell_model_043 import (
    Batch043,
    CellModel043,
    CellModel043Config,
    CompositeToken,
    ContextInputs,
    MarkInputs,
    PositionTerm,
    ReadoutLoss043,
    Replicon,
    build_head_mask,
    load_vector_table,
    select_marks,
)
from torchcell.models.genotype_tokens_043 import CHANGE_TYPE_INDEX, ChangeType

G = 6
DIMS = (4, 4, 6, 4)
DEL = CHANGE_TYPE_INDEX[ChangeType.deletion]
VAR = CHANGE_TYPE_INDEX[ChangeType.variant]
ABS = CHANGE_TYPE_INDEX[ChangeType.absent]
PRES = CHANGE_TYPE_INDEX[ChangeType.present]


def _write_h5(path: Path, keys: list[str], emb: npt.NDArray[np.float32]) -> None:
    with h5py.File(path, "w") as f:
        f["sha256"] = np.asarray(keys, dtype="S64")
        f["embedding"] = emb.astype(np.float16)


def test_load_vector_table_gathers_in_order_and_raises_on_missing(
    tmp_path: Path,
) -> None:
    keys = [f"{i:064x}" for i in (5, 1, 9)]
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    _write_h5(tmp_path / "t.h5", keys, emb)
    out = load_vector_table(str(tmp_path / "t.h5"), [keys[2], keys[0], keys[2]])
    assert out.dtype == torch.float16
    assert out.tolist() == [emb[2].tolist(), emb[0].tolist(), emb[2].tolist()]
    with pytest.raises(KeyError, match="1 of 2 keys have no vector"):
        load_vector_table(str(tmp_path / "t.h5"), [keys[1], f"{7:064x}"])
    assert load_vector_table(str(tmp_path / "t.h5"), []).shape == (0, 4)


def test_select_marks_keeps_priority_then_seeded_subset() -> None:
    change = np.array([VAR] * 10 + [DEL, ABS, VAR, VAR], dtype=np.int8)
    assert select_marks(change, None, 0).tolist() == list(range(14))
    assert select_marks(change, 20, 0).tolist() == list(range(14))
    kept = select_marks(change, 5, seed=3)
    assert len(kept) == 5
    assert {10, 11} <= set(kept.tolist())  # deletion and absence always kept
    assert kept.tolist() == sorted(kept.tolist())
    assert select_marks(change, 5, seed=3).tolist() == kept.tolist()  # same each epoch
    assert select_marks(change, 1, seed=0).tolist() == [11]  # absent outranks deletion


def test_composite_token_uses_typed_gap_rows() -> None:
    tables = [torch.randn(3, d) for d in DIMS]
    comp = CompositeToken(tables, 8, DIMS)
    rows = torch.tensor([[0, 1, 2, 0], [-1, 1, 2, 0], [-1, 1, 2, 0]])
    gaps = torch.tensor(
        [
            [0, 0, 0, 0],
            [int(GapKind.LOCUS_GAP), 0, 0, 0],
            [int(GapKind.ABSENT_IN_CONTEXT), 0, 0, 0],
        ]
    )
    out = comp(rows, gaps)
    assert out.shape == (3, 8)
    # two different gap kinds give different tokens; the gap does not read table row 0
    assert not torch.allclose(out[1], out[2])
    assert not torch.allclose(out[0], out[1])
    assert comp.table_bytes() == sum(3 * d * 2 for d in DIMS)
    with pytest.raises(ValueError, match="must be"):
        CompositeToken([torch.randn(3, 5)] + tables[1:], 8, DIMS)


def test_composite_token_accepts_ambiguous_sequence_gap_on_calm_only() -> None:
    """An ambiguity code gaps one encoder (CaLM, column 3); the others keep keys."""
    tables = [torch.randn(3, d) for d in DIMS]
    comp = CompositeToken(tables, 8, DIMS)
    rows = torch.tensor([[0, 1, 2, 0], [0, 1, 2, -1], [0, 1, 2, -1]])
    gaps = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, int(GapKind.AMBIGUOUS_SEQUENCE)],
            [0, 0, 0, int(GapKind.SPLICE_UNVERIFIED)],
        ]
    )
    out = comp(rows, gaps)
    assert out.shape == (3, 8)
    assert int(GapKind.AMBIGUOUS_SEQUENCE) == len(GapKind) - 1
    assert not torch.allclose(out[1], out[2])  # its own learned row per gap kind
    assert not torch.allclose(out[0], out[1])


def test_position_term_unknown_copy_uses_gap_embedding() -> None:
    pos = PositionTerm(8, 4)
    rep = torch.tensor([3, 3, int(Replicon.ACCESSORY_ORF)])
    mid = torch.tensor([0.2, 0.2, 0.0])
    chrom = torch.tensor([1.0, float("nan"), 1.0])
    copy = torch.tensor([1.0, 1.0, float("nan")])
    pgap = torch.tensor(
        [0, int(PositionGap.PLOIDY_UNREPORTED), int(PositionGap.COPY_NUMBER_UNRELEASED)]
    )
    on = torch.tensor([True, True, False])
    out = pos(rep, mid, chrom, copy, pgap, on)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0], out[1])
    # doubling the chromosome copy changes the token
    out2 = pos(rep[:1], mid[:1], torch.tensor([2.0]), copy[:1], pgap[:1], on[:1])
    assert not torch.allclose(out[0], out2[0])


def test_build_head_mask_structure_and_unknown_relation() -> None:
    ei = torch.tensor([[0, 2], [1, 3]])
    mask = build_head_mask({"physical": ei}, {1: "physical"}, num_heads=2, n_genes=4)
    assert mask.shape == (2, 5, 5)
    assert mask[0].all()  # free head
    m = mask[1]
    assert bool(m[1, 2]) and bool(m[2, 1]) and bool(m[3, 4]) and bool(m[4, 3])
    assert not bool(m[1, 3])
    assert m.diagonal().all() and m[0].all() and m[:, 0].all()
    with pytest.raises(KeyError, match="regulatory"):
        build_head_mask({"physical": ei}, {0: "regulatory"}, 2, 4)


def _config(loss: str = "hlgauss", **kw: object) -> CellModel043Config:
    fields: dict[str, object] = {
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "mask_layers": (1,),
        "head_graphs": {0: "physical"},
        "n_parents": 3,
        "n_datasets": 3,
        "n_envs": 2,
        "exp_type_classes": ("A", "B"),
        "exp_type_mode": "dag",
        "exp_type_paths": {"A": ("root", "A"), "B": ("root", "B")},
        "head_hidden": 8,
        "loss": loss,
    }
    fields.update(kw)
    return CellModel043Config.model_validate(fields)


def _model(cfg: CellModel043Config) -> CellModel043:
    torch.manual_seed(0)
    tables = [torch.randn(5, d) for d in DIMS]
    mask = build_head_mask(
        {"physical": torch.tensor([[0, 1, 2], [1, 2, 3]])},
        cfg.head_graphs,
        cfg.num_heads,
        G,
    )
    return CellModel043(
        cfg,
        tables,
        chrom_index=torch.tensor([1, 1, 2, 2, 3, 0]),
        rel_midpoint=torch.linspace(0.1, 0.9, G),
        head_mask=mask,
        encoder_dims=DIMS,
    )


def _batch(n_marks: tuple[int, ...] = (1, 0, 2)) -> Batch043:
    """Records: 0 = deletion on context 0; 1 = isolate context 1, no mark; 2 = reporter."""
    C = 2
    key_rows = torch.randint(0, 5, (C, G, 4))
    key_rows[1, 2, 0] = -1
    key_gap = torch.zeros(C, G, 4, dtype=torch.long)
    key_gap[1, 2, 0] = int(GapKind.LOCUS_GAP)
    ctx = ContextInputs(
        key_rows=key_rows,
        key_gap=key_gap,
        chrom_copy=torch.ones(C, G),
        gene_copy=torch.ones(C, G),
        position_gap=torch.zeros(C, G, dtype=torch.long),
    )
    B, M = len(n_marks), max(n_marks)
    valid = torch.zeros(B, M, dtype=torch.bool)
    for b, n in enumerate(n_marks):
        valid[b, :n] = True
    on_graph = valid.clone()
    on_graph[2] = False
    change = torch.zeros(B, M, dtype=torch.long)
    change[0, : n_marks[0]] = DEL
    change[2, : n_marks[2]] = PRES
    marks = MarkInputs(
        valid=valid,
        on_graph=on_graph,
        gene=torch.tensor([[3, 0], [0, 0], [0, 0]])[:, :M].contiguous(),
        change=change,
        parent=torch.zeros(B, M, dtype=torch.long),
        key_rows=torch.randint(0, 5, (B, M, 4)),
        key_gap=torch.zeros(B, M, 4, dtype=torch.long),
        replicon=torch.full((B, M), int(Replicon.REPORTER_CONSTRUCT)),
        rel_midpoint=torch.zeros(B, M),
        chrom_copy=torch.ones(B, M),
        gene_copy=torch.ones(B, M),
        position_gap=torch.zeros(B, M, dtype=torch.long),
    )
    return Batch043(
        contexts=ctx,
        record_context=torch.tensor([0, 1, 0]),
        marks=marks,
        dataset_ids=torch.tensor([0, 1, 2]),
        exp_type_ids=torch.tensor([0, 0, 1]),
        env_ids=torch.tensor([0, 1, 0]),
    )


@pytest.mark.parametrize(
    ("loss", "gene_shape", "record_shape"),
    [
        ("hlgauss", (2, G, 32), (1, 32)),
        ("pinball", (2, G, 19), (1, 19)),
        ("mse", (2, G), (1,)),
    ],
)
def test_forward_shapes_per_loss(
    loss: str, gene_shape: tuple[int, ...], record_shape: tuple[int, ...]
) -> None:
    cfg = _config(loss)
    model = _model(cfg)
    out = model(_batch(), torch.tensor([0, 1]), torch.tensor([2]))
    assert tuple(out.gene.shape) == gene_shape
    assert tuple(out.record.shape) == record_shape
    assert out.h_cls.shape == (3, 16)
    assert torch.isfinite(out.gene).all() and torch.isfinite(out.record).all()


def test_attention_query_chunk_changes_memory_not_the_model() -> None:
    """Chunk 2 over the G+1 = 7 trunk queries: same outputs and parameter gradients."""
    results = []
    for chunk in (None, 2):
        model = _model(_config("mse", attention_query_chunk=chunk))
        out = model(_batch(), torch.tensor([0, 1]), torch.tensor([2]))
        (out.gene.sum() + out.record.sum()).backward()
        grads = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        results.append((out.gene.detach(), out.record.detach(), grads))
    (g0, r0, d0), (g1, r1, d1) = results
    torch.testing.assert_close(g1, g0)
    torch.testing.assert_close(r1, r0)
    assert d1.keys() == d0.keys()
    for n in d0:
        torch.testing.assert_close(d1[n], d0[n], msg=n)


def test_operator_is_the_post_norm_residual() -> None:
    cfg = _config("mse")
    model = _model(cfg).eval()
    op = model.operator
    batch = _batch()
    with torch.no_grad():
        X = model.context_tokens(batch.contexts)
        H = model.trunk(X)[batch.record_context]
        keys = model.mark_keys(batch.marks, H)
        H_pert = op(H, keys, batch.marks.valid)
        # the trunk's post-norm form, written out
        B, N, d = H.shape
        hd = d // op.num_heads
        q = op.q_proj(H).view(B, N, op.num_heads, hd).transpose(1, 2)
        k = op.k_proj(keys).view(B, -1, op.num_heads, hd).transpose(1, 2)
        v = op.v_proj(keys).view(B, -1, op.num_heads, hd).transpose(1, 2)
        k = torch.cat(
            [op.null_key.view(1, op.num_heads, 1, hd).expand(B, -1, -1, -1), k], 2
        )
        v = torch.cat([v.new_zeros(B, op.num_heads, 1, hd), v], 2)
        allowed = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool), batch.marks.valid], 1
        ).view(B, 1, 1, -1)
        att = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed
        )
        a = op.out_proj(att.transpose(1, 2).reshape(B, N, d))
        H1 = op.norm1(H + a)
        expected = op.norm2(H1 + op.ffn(H1))
    assert torch.allclose(H_pert, expected, atol=1e-5)
    # the attention branch is live from the first step and its ratio is what is logged
    norms = op.last_norms
    ratio = (a.norm(dim=-1) / H.norm(dim=-1)).mean()
    assert float(norms["attn"]) > 0.0
    assert torch.allclose(norms["attn"], ratio, atol=1e-6)
    # no statistic is computed in training mode
    op.last_norms = {}
    model.train()
    model(batch, torch.tensor([0, 1]), torch.tensor([2]))
    assert op.last_norms == {}


def test_dataset_id_reaches_only_the_head() -> None:
    cfg = _config("mse")
    model = _model(cfg).eval()
    b1 = _batch()
    b2 = Batch043(
        contexts=b1.contexts,
        record_context=b1.record_context,
        marks=b1.marks,
        dataset_ids=torch.tensor([2, 2, 2]),
        exp_type_ids=b1.exp_type_ids,
        env_ids=b1.env_ids,
    )
    with torch.no_grad():
        o1 = model(b1, torch.tensor([0]), torch.tensor([2]))
        o2 = model(b2, torch.tensor([0]), torch.tensor([2]))
    # trunk and operator outputs do not depend on the dataset id
    assert torch.allclose(o1.h_cls, o2.h_cls) and torch.allclose(o1.z_s, o2.z_s)
    assert not torch.allclose(o1.gene, o2.gene)


def test_record_without_marks_attends_to_null_key() -> None:
    cfg = _config("mse")
    model = _model(cfg)
    out = model(
        _batch((0, 0, 0)), torch.tensor([0, 1, 2]), torch.tensor([], dtype=torch.long)
    )
    assert torch.isfinite(out.gene).all()
    assert out.record.shape == (0,)
    assert torch.allclose(out.z_s, torch.zeros_like(out.z_s))


def test_readout_loss_per_kind() -> None:
    target = torch.tensor([[0.5, float("nan")], [-1.0, 2.0]])
    for kind, width in (("hlgauss", 32), ("pinball", 19), ("mse", 0)):
        loss = ReadoutLoss043(_config(kind))
        shape = (2, 2, width) if width else (2, 2)
        pred = torch.randn(*shape, requires_grad=True)
        elem, measured = loss.elementwise(pred, target)
        assert elem.shape == (2, 2)
        assert measured.tolist() == [[True, False], [True, True]]
        assert float(elem[0, 1]) == 0.0
        elem.sum().backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert loss.point(pred.detach()).shape == (2, 2)
    mse = ReadoutLoss043(_config("mse"))
    elem, _ = mse.elementwise(torch.zeros(2, 2), target)
    assert elem.tolist() == [[0.25, 0.0], [1.0, 4.0]]
    pin = ReadoutLoss043(_config("pinball"))
    q = torch.arange(19, dtype=torch.float32).expand(2, 2, 19)
    assert pin.point(q)[0, 0] == 9.0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _config(hidden_dim=15)
    with pytest.raises(ValueError, match="mask_layers"):
        _config(mask_layers=(5,))
    with pytest.raises(ValueError, match="masks nothing"):
        _config(head_graphs={})


# --------------------------------------------------------------------------- #
# Ontology conditioning (2026-10-23)
# --------------------------------------------------------------------------- #
def _tiny_vocab() -> ExperimentOntologyVocabulary:
    return ExperimentOntologyVocabulary(
        nodes=[
            OntologyNode(name="Experiment", kind="class"),
            OntologyNode(name="A", kind="class", parents=["Experiment"]),
            OntologyNode(name="B", kind="class", parents=["Experiment"]),
            OntologyNode(name="A.temp", kind="numeric", parents=["A"]),
        ]
    )


def _onto_model(loss: str = "hlgauss") -> CellModel043:
    cfg = _config(loss, conditioning="ontology")
    torch.manual_seed(0)
    tables = [torch.randn(5, d) for d in DIMS]
    mask = build_head_mask(
        {"physical": torch.tensor([[0, 1, 2], [1, 2, 3]])},
        cfg.head_graphs,
        cfg.num_heads,
        G,
    )
    return CellModel043(
        cfg,
        tables,
        chrom_index=torch.tensor([1, 1, 2, 2, 3, 0]),
        rel_midpoint=torch.linspace(0.1, 0.9, G),
        head_mask=mask,
        encoder_dims=DIMS,
        ontology_vocabulary=_tiny_vocab(),
    )


def _with_ids(b: Batch043, ids: torch.Tensor, onto: torch.Tensor | None) -> Batch043:
    return Batch043(
        contexts=b.contexts,
        record_context=b.record_context,
        marks=b.marks,
        dataset_ids=ids,
        exp_type_ids=ids,
        env_ids=ids,
        ontology=onto,
    )


ONTO = torch.tensor([[1.0, 1.0, 0.0, 0.3], [1.0, 1.0, 0.0, 0.3], [1.0, 0.0, 1.0, 0.0]])


@pytest.mark.parametrize("loss", ["hlgauss", "pinball", "mse"])
def test_ontology_conditioning_has_the_name_shapes_and_no_dataset_tables(
    loss: str,
) -> None:
    name = _model(_config(loss))
    onto = _onto_model(loss)
    b = _with_ids(_batch(), torch.tensor([0, 1, 2]), ONTO)
    o_name = name(_batch(), torch.tensor([0, 1]), torch.tensor([2]))
    o_onto = onto(b, torch.tensor([0, 1]), torch.tensor([2]))
    assert o_name.gene.shape == o_onto.gene.shape
    assert o_name.record.shape == o_onto.record.shape
    names = {n for n, _ in onto.named_parameters()}
    for leak in (
        "dataset_token",
        "env_token",
        "type_token",
        "gene_bias",
        "record_bias",
    ):
        assert not any(n.startswith(leak + ".") or n == leak for n in names), leak
    assert {
        "onto_token.out.weight",
        "gene_bias_map.weight",
        "record_bias_map.weight",
    } <= names
    # the bias maps keep PyTorch's default init: not zero, no gate
    assert onto.gene_bias_map.weight.abs().sum() > 0
    # the head input width is the name model's (token 40 at the default widths)
    assert onto.config.token_dim == name.config.token_dim


def test_ontology_forward_never_reads_a_dataset_id() -> None:
    model = _onto_model("mse").eval()
    b = _batch()
    real = _with_ids(b, torch.tensor([0, 1, 2]), ONTO)
    # ids far outside every table: any index by them raises
    poison = _with_ids(b, torch.tensor([10**9, -(10**9), 10**9]), ONTO)
    with torch.no_grad():
        o1 = model(real, torch.tensor([0, 1]), torch.tensor([2]))
        o2 = model(poison, torch.tensor([0, 1]), torch.tensor([2]))
    assert torch.equal(o1.gene, o2.gene) and torch.equal(o1.record, o2.record)
    # the annotation is what conditions: changing it changes the output
    other = ONTO.clone()
    other[0, 3] = 0.25
    with torch.no_grad():
        o3 = model(
            _with_ids(b, torch.tensor([0, 1, 2]), other),
            torch.tensor([0, 1]),
            torch.tensor([2]),
        )
    assert not torch.allclose(o1.gene[0], o3.gene[0])
    assert torch.allclose(o1.gene[1], o3.gene[1])
    # a batch without the annotation is refused, and the vocabulary is required
    with pytest.raises(ValueError, match="Batch043.ontology"):
        model(
            _with_ids(b, torch.tensor([0, 1, 2]), None),
            torch.tensor([0]),
            torch.tensor([2]),
        )
    with pytest.raises(ValueError, match="vocabulary"):
        _model(_config("mse", conditioning="ontology"))


# --------------------------------------------------------------------------- #
# 043 sweep hooks (2026-10-24): trunk_ffn_mult and the mle / binned_quantile kinds
# --------------------------------------------------------------------------- #
# Recorded at HEAD 5f29aff8d, before the hooks existed, with this file's _config,
# _model and _batch (CPU, float32): parameter count, sum of gene and record outputs.
HEAD_REFERENCE = {
    "hlgauss": (14076, 9.171564197167754, 2.805013310164213),
    "pinball": (13764, -20.97563796164468, -0.6964259445667267),
    "mse": (13332, 3.15146341919899, 0.00433029979467392),
}


@pytest.mark.parametrize("loss", sorted(HEAD_REFERENCE))
def test_default_config_is_bit_identical_to_head(loss: str) -> None:
    n_params, gene_sum, record_sum = HEAD_REFERENCE[loss]
    model = _model(_config(loss)).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch, torch.tensor([0, 1]), torch.tensor([2]))
    assert sum(p.numel() for p in model.parameters()) == n_params
    assert float(out.gene.double().sum()) == gene_sum  # exact
    assert float(out.record.double().sum()) == record_sum  # exact


def test_trunk_ffn_mult_sets_the_trunk_ffn_width_only() -> None:
    for mult in (2, 4):
        model = _model(_config("mse", trunk_ffn_mult=mult))
        trunk_ffn = [
            (m.in_features, m.out_features)
            for name, m in model.trunk.named_modules()
            if isinstance(m, torch.nn.Linear) and ".ffn." in name
        ]
        assert trunk_ffn == [(16, 16 * mult), (16 * mult, 16)] * 2  # two trunk layers
        op_ffn = [
            m.out_features
            for name, m in model.operator.named_modules()
            if isinstance(m, torch.nn.Linear) and name.startswith("ffn.")
        ]
        assert op_ffn == [16 * 4, 16]  # operator_ffn_mult, untouched
    with pytest.raises(ValueError):
        _config("mse", trunk_ffn_mult=0)


def test_param_dim_of_the_sweep_losses() -> None:
    assert _config("mle").param_dim == 2
    assert _config("binned_quantile").param_dim == 32
    model = _model(_config("mle"))
    out = model(_batch(), torch.tensor([0, 1]), torch.tensor([2]))
    assert tuple(out.gene.shape) == (2, G, 2) and tuple(out.record.shape) == (1, 2)
    with pytest.raises(ValueError, match="ReadoutLossFamily043"):
        ReadoutLoss043(_config("mle")).elementwise(out.record, torch.zeros(1))


def test_token_dropout_trains_only_and_default_draws_nothing() -> None:
    batch = _batch()
    rows = (torch.tensor([0, 1]), torch.tensor([2]))
    base = _model(_config("mse"))
    drop = _model(_config("mse", token_dropout=0.5))
    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in drop.parameters()
    )
    base.eval()
    drop.eval()
    with torch.no_grad():
        torch.testing.assert_close(drop(batch, *rows).gene, base(batch, *rows).gene)
    drop.train()
    torch.manual_seed(1)
    X = drop.context_tokens(batch.contexts)
    drop.eval()
    X_eval = drop.context_tokens(batch.contexts)
    pos_only = X_eval - drop.composite(batch.contexts.key_rows, batch.contexts.key_gap)
    dropped = torch.isclose(X, pos_only, atol=1e-6).all(-1)
    assert 0 < int(dropped.sum()) < dropped.numel()  # some genes dropped, not all
    with pytest.raises(ValueError):
        _config("mse", token_dropout=1.0)
