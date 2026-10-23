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


def test_rezero_operator_is_identity_at_init_and_learns() -> None:
    cfg = _config("mse", operator_residual="rezero")
    model = _model(cfg).eval()
    batch = _batch()
    # at init (betas 0) records sharing a context get identical gene states
    with torch.no_grad():
        X = model.context_tokens(batch.contexts)
        H = model.trunk(X)[batch.record_context]
        keys = model.mark_keys(batch.marks, H)
        H_pert = model.operator(H, keys, batch.marks.valid)
    assert torch.allclose(H_pert, H)
    # closed gates: the branches as added are exactly zero
    norms = model.operator.last_norms
    assert float(norms["attn"]) == 0.0 and float(norms["update"]) == 0.0
    # the gates receive gradient
    model.train()
    out = model(batch, torch.tensor([0, 1]), torch.tensor([2]))
    (out.gene.sum() + out.record.sum()).backward()
    assert model.operator.beta_attn.grad is not None
    assert float(model.operator.beta_attn.grad.abs()) > 0.0


def test_plain_operator_is_the_default_post_norm_residual() -> None:
    cfg = _config("mse")
    assert cfg.operator_residual == "plain"
    model = _model(cfg).eval()
    op = model.operator
    assert not hasattr(op, "beta_attn") and not hasattr(op, "beta_ffn")
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
    cfg = _config("mse", operator_residual="rezero")
    model = _model(cfg)
    with torch.no_grad():
        model.operator.beta_attn.fill_(1.0)
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
