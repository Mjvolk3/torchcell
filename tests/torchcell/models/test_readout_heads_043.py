# tests/torchcell/models/test_readout_heads_043.py
# [[tests.torchcell.models.test_readout_heads_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_readout_heads_043.py
"""The 043 output-side prototype on random tensors.

Shapes for every head in point and quantile form; the ``NaN`` mask (a missing entry
contributes neither value nor gradient, and a type with nothing measured returns a
graph-connected zero); the per-dataset affine (dataset ``k``'s scale and shift receive
gradient only from rows of dataset ``k``, and the affine is the identity at
construction); the ontology path mode (two leaves under one parent differ by exactly
their own leaf rows); and the refusals the configs make.
"""

import pytest
import torch
from pydantic import ValidationError

from torchcell.models.readout_heads_043 import (
    ExperimentTypeEmbedding,
    ExperimentTypeEmbeddingConfig,
    GlobalScalarReadout,
    GlobalScalarReadoutConfig,
    MaskedTypedLoss,
    MaskedTypedLossConfig,
    PerGeneReadout,
    PerGeneReadoutConfig,
)

CLASSES = (
    "MicroarrayExpressionPhenotype",
    "RNASeqExpressionPhenotype",
    "ProteinAbundancePhenotype",
)
PATHS = {
    "MicroarrayExpressionPhenotype": (
        "Phenotype",
        "ExpressionPhenotype",
        "MicroarrayExpressionPhenotype",
    ),
    "RNASeqExpressionPhenotype": (
        "Phenotype",
        "ExpressionPhenotype",
        "RNASeqExpressionPhenotype",
    ),
    "ProteinAbundancePhenotype": ("Phenotype", "ProteinAbundancePhenotype"),
}


def test_type_embedding_flat_shape_and_one_row_per_class() -> None:
    emb = ExperimentTypeEmbedding(
        ExperimentTypeEmbeddingConfig(classes=CLASSES, hidden_dim=8)
    )
    assert emb.table.num_embeddings == 3
    out = emb(torch.tensor([0, 2, 1, 1]))
    assert out.shape == (4, 8)
    torch.testing.assert_close(out[0], emb.table.weight[0])
    torch.testing.assert_close(out[2], out[3])


def test_type_embedding_dag_shares_the_parent_rows() -> None:
    emb = ExperimentTypeEmbedding(
        ExperimentTypeEmbeddingConfig(
            classes=CLASSES, hidden_dim=8, mode="dag", dag_paths=PATHS
        )
    )
    # Phenotype, ExpressionPhenotype and the three leaves: five node rows.
    assert emb.table.num_embeddings == 5
    out = emb(torch.tensor([0, 1, 2]))
    w = emb.table.weight
    leaf = {name: w[emb.node_index(name)] for name in CLASSES}
    root = w[emb.node_index("Phenotype")]
    expr = w[emb.node_index("ExpressionPhenotype")]
    torch.testing.assert_close(out[0], root + expr + leaf[CLASSES[0]])
    torch.testing.assert_close(out[2], root + leaf[CLASSES[2]])
    # Two expression leaves differ by their own rows only.
    torch.testing.assert_close(out[0] - out[1], leaf[CLASSES[0]] - leaf[CLASSES[1]])


def test_type_embedding_refusals() -> None:
    with pytest.raises(ValidationError, match="needs dag_paths"):
        ExperimentTypeEmbeddingConfig(classes=CLASSES, hidden_dim=8, mode="dag")
    with pytest.raises(ValidationError, match="no ontology path"):
        ExperimentTypeEmbeddingConfig(
            classes=CLASSES,
            hidden_dim=8,
            mode="dag",
            dag_paths={k: v for k, v in PATHS.items() if k != CLASSES[1]},
        )
    with pytest.raises(ValidationError, match="must end in the class"):
        ExperimentTypeEmbeddingConfig(
            classes=CLASSES[:1],
            hidden_dim=8,
            mode="dag",
            dag_paths={CLASSES[0]: ("Phenotype", "ExpressionPhenotype")},
        )
    with pytest.raises(ValidationError, match="only read in mode='dag'"):
        ExperimentTypeEmbeddingConfig(classes=CLASSES, hidden_dim=8, dag_paths=PATHS)
    emb = ExperimentTypeEmbedding(
        ExperimentTypeEmbeddingConfig(classes=CLASSES, hidden_dim=8)
    )
    with pytest.raises(IndexError):
        emb(torch.tensor([3]))


def test_per_gene_shapes_point_and_quantile() -> None:
    H = torch.randn(4, 10, 16)
    ids = torch.tensor([0, 1, 1, 2])
    point = PerGeneReadout(PerGeneReadoutConfig(hidden_dim=16, n_datasets=3))
    assert point(H, ids).shape == (4, 10)
    quant = PerGeneReadout(
        PerGeneReadoutConfig(
            hidden_dim=16, n_datasets=3, token_dim=5, head_hidden=8, param_dim=19
        )
    )
    assert quant(H, ids, torch.randn(4, 5)).shape == (4, 10, 19)


def test_per_gene_affine_is_identity_at_init_and_reads_the_gene_itself() -> None:
    torch.manual_seed(0)
    cfg = PerGeneReadoutConfig(hidden_dim=16, n_datasets=2, head_hidden=8)
    head = PerGeneReadout(cfg)
    head.eval()
    H = torch.randn(3, 7, 16)
    ids = torch.tensor([0, 1, 0])
    out = head(H, ids)
    bare = head.head(H).squeeze(-1)
    torch.testing.assert_close(out, bare)
    # Gene g's prediction depends on H[:, g] only: perturb another gene's state.
    H2 = H.clone()
    H2[:, 3] += 10.0
    out2 = head(H2, ids)
    torch.testing.assert_close(out2[:, :3], out[:, :3])
    torch.testing.assert_close(out2[:, 4:], out[:, 4:])
    assert not torch.allclose(out2[:, 3], out[:, 3])


def test_per_gene_per_dataset_affine_isolation() -> None:
    torch.manual_seed(1)
    head = PerGeneReadout(PerGeneReadoutConfig(hidden_dim=8, n_datasets=3))
    assert head.scale is not None and head.shift is not None
    H = torch.randn(5, 6, 8)
    ids = torch.tensor([0, 0, 2, 0, 2])  # dataset 1 absent from the batch
    out = head(H, ids)
    out.sum().backward()
    assert head.scale.grad is not None and head.shift.grad is not None
    assert torch.all(head.scale.grad[1] == 0)
    assert torch.all(head.shift.grad[1] == 0)
    assert torch.any(head.scale.grad[0] != 0)
    assert torch.any(head.scale.grad[2] != 0)
    # The shift gradient of dataset k is the number of (row, gene) entries of k.
    torch.testing.assert_close(head.shift.grad[0], torch.tensor([18.0]))
    torch.testing.assert_close(head.shift.grad[2], torch.tensor([12.0]))
    # Changing dataset 1's affine leaves every prediction in this batch unchanged.
    with torch.no_grad():
        head.scale[1] = 5.0
        head.shift[1] = -3.0
    torch.testing.assert_close(head(H, ids), out)


def test_per_gene_refusals() -> None:
    head = PerGeneReadout(PerGeneReadoutConfig(hidden_dim=8, n_datasets=2))
    H = torch.randn(2, 4, 8)
    with pytest.raises(IndexError):
        head(H, torch.tensor([0, 2]))
    with pytest.raises(ValueError, match="pass no tokens"):
        head(H, torch.tensor([0, 1]), torch.randn(2, 3))
    tok = PerGeneReadout(PerGeneReadoutConfig(hidden_dim=8, n_datasets=2, token_dim=3))
    with pytest.raises(ValueError, match=r"tokens must be \[B, 3\]"):
        tok(H, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="H must be"):
        head(torch.randn(2, 8), torch.tensor([0, 1]))


def test_global_scalar_shapes() -> None:
    cfg = GlobalScalarReadoutConfig(hidden_dim=16, token_dim=4, head_hidden=8)
    head = GlobalScalarReadout(cfg)
    h = torch.randn(6, 16)
    assert head(h, torch.randn(6, 16), torch.randn(6, 4)).shape == (6,)
    quant = GlobalScalarReadout(
        GlobalScalarReadoutConfig(
            hidden_dim=16, head_hidden=8, param_dim=19, use_pert_summary=False
        )
    )
    assert quant(h).shape == (6, 19)


def test_global_scalar_refusals() -> None:
    head = GlobalScalarReadout(GlobalScalarReadoutConfig(hidden_dim=16, head_hidden=8))
    h = torch.randn(2, 16)
    with pytest.raises(ValueError, match="needs z_s"):
        head(h)
    with pytest.raises(ValueError, match="pass no tokens"):
        head(h, h, torch.randn(2, 3))
    bare = GlobalScalarReadout(
        GlobalScalarReadoutConfig(hidden_dim=16, head_hidden=8, use_pert_summary=False)
    )
    with pytest.raises(ValueError, match="pass no z_s"):
        bare(h, h)


def _loss() -> MaskedTypedLoss:
    return MaskedTypedLoss(
        MaskedTypedLossConfig(
            loss_by_type={"expression": "mse", "protein": "pinball"},
            weight_by_type={"protein": 0.5},
            num_quantiles=3,
        )
    )


def test_masked_loss_mse_measured_only_and_no_nan_gradient() -> None:
    loss = _loss()
    pred = torch.zeros(2, 3, requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 3.0], [float("nan"), float("nan"), 2.0]])
    total, per_type = loss({"expression": pred}, {"expression": target})
    # Measured squared errors 1, 9, 4 over three entries.
    torch.testing.assert_close(total, torch.tensor(14.0 / 3))
    torch.testing.assert_close(per_type["expression"], total)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    nan_entries = torch.isnan(target)
    assert torch.all(pred.grad[nan_entries] == 0)
    assert torch.all(pred.grad[~nan_entries] != 0)


def test_masked_loss_pinball_over_the_quantile_axis() -> None:
    loss = _loss()
    taus = loss.taus
    torch.testing.assert_close(taus, torch.tensor([0.25, 0.5, 0.75]))
    pred = torch.zeros(1, 2, 3, requires_grad=True)
    target = torch.tensor([[2.0, float("nan")]])
    total, per_type = loss({"protein": pred}, {"protein": target})
    # u = 2 at every knot: rho = tau * u, mean over knots (0.5 + 1 + 1.5) / 3 = 1.
    torch.testing.assert_close(per_type["protein"], torch.tensor(1.0))
    torch.testing.assert_close(total, torch.tensor(0.5))  # weight 0.5
    total.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad[0, 1] == 0)
    assert torch.all(pred.grad[0, 0] != 0)


def test_masked_loss_two_types_sum_with_weights() -> None:
    loss = _loss()
    preds = {"expression": torch.zeros(2, 2), "protein": torch.zeros(2, 2, 3)}
    targets = {
        "expression": torch.tensor([[1.0, 1.0], [1.0, float("nan")]]),
        "protein": torch.full((2, 2), 2.0),
    }
    total, per_type = loss(preds, targets)
    torch.testing.assert_close(per_type["expression"], torch.tensor(1.0))
    torch.testing.assert_close(per_type["protein"], torch.tensor(1.0))
    torch.testing.assert_close(total, torch.tensor(1.5))


def test_masked_loss_nothing_measured_is_a_connected_zero() -> None:
    loss = _loss()
    pred = torch.randn(2, 2, requires_grad=True)
    target = torch.full((2, 2), float("nan"))
    total, _ = loss({"expression": pred}, {"expression": target})
    assert total.item() == 0.0
    assert total.requires_grad
    total.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad == 0)


def test_masked_loss_refusals() -> None:
    loss = _loss()
    with pytest.raises(KeyError, match="different types"):
        loss({"expression": torch.zeros(1, 1)}, {"protein": torch.zeros(1, 1)})
    with pytest.raises(KeyError, match="no loss configured"):
        loss.elementwise("morphology", torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError, match="pinball needs pred"):
        loss({"protein": torch.zeros(1, 2)}, {"protein": torch.zeros(1, 2)})
    with pytest.raises(ValueError, match="mse needs pred"):
        loss({"expression": torch.zeros(1, 2, 3)}, {"expression": torch.zeros(1, 2)})
    with pytest.raises(ValidationError, match="types without a loss"):
        MaskedTypedLossConfig(
            loss_by_type={"expression": "mse"}, weight_by_type={"protein": 1.0}
        )
