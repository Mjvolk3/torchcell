"""CellGraphTransformer.dataset_token: per-entry readouts conditioned on the source."""

from typing import Any, cast

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

GENE_NUM = 8
HIDDEN = 16
VOCAB = 3


def _cell_graph() -> HeteroData:
    cg = HeteroData()
    cg["gene"].num_nodes = GENE_NUM
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    return cg


def _batch() -> HeteroData:
    # genotype 0 perturbs {1, 2}, genotype 1 perturbs {3}, genotype 2 perturbs {0, 4, 5}
    b = HeteroData()
    b["gene"].perturbation_indices = torch.tensor([1, 2, 3, 0, 4, 5])
    b["gene"].perturbation_indices_batch = torch.tensor([0, 0, 1, 2, 2, 2])
    return b


def _model(seed: int = 0, **kwargs: Any) -> CellGraphTransformer:
    torch.manual_seed(seed)
    return CellGraphTransformer(
        gene_num=GENE_NUM,
        hidden_channels=HIDDEN,
        num_transformer_layers=1,
        num_attention_heads=4,
        cell_graph=_cell_graph(),
        heads_config={
            "global": {"output_dim": 1, "use_gene_pool": False, "linear": True}
        },
        perturb_cls=True,
        perturbation_head_cls="perturbed",
        **kwargs,
    )


def test_disabled_token_is_the_prechange_model() -> None:
    plain = _model(seed=3)
    off = _model(seed=3, dataset_token={"enabled": False})
    assert plain.state_dict().keys() == off.state_dict().keys()
    assert "dataset_token" not in plain.num_parameters
    plain.eval()
    off.eval()
    with torch.no_grad():
        p1, r1 = plain(_cell_graph(), _batch())
        p2, r2 = off(_cell_graph(), _batch())
    assert torch.equal(p1, p2)
    assert torch.equal(r1["head_outputs"]["global"], r2["head_outputs"]["global"])
    with pytest.raises(ValueError, match="dataset_token is disabled"):
        plain(
            _cell_graph(),
            _batch(),
            entry_batch=torch.tensor([0]),
            entry_dataset=torch.tensor([0]),
        )


def test_enabled_token_gives_one_row_per_entry_and_differs_only_by_token() -> None:
    model = _model(dataset_token={"enabled": True, "vocab_size": VOCAB, "dim": 4})
    model.eval()
    # five entries over three genotypes: genotype 0 measured by tokens 0 and 1
    entry_batch = torch.tensor([0, 0, 1, 2, 2])
    entry_dataset = torch.tensor([0, 1, 2, 0, 0])
    with torch.no_grad():
        pred, reps = model(
            _cell_graph(),
            _batch(),
            entry_batch=entry_batch,
            entry_dataset=entry_dataset,
        )
    assert pred.shape == (5, 1)
    assert reps["head_outputs"]["global"].shape == (5, 1)
    # Same genotype, same token: identical rows. Same genotype, other token: different.
    assert torch.allclose(pred[3], pred[4], atol=1e-6)
    assert not torch.allclose(pred[0], pred[1], atol=1e-6)
    assert model.num_parameters["dataset_token"] == VOCAB * 4
    assert cast(Any, model.perturbation_head).mlp[0].in_features == 2 * HIDDEN + 4
    assert cast(Any, model.global_head).mlp.in_features == HIDDEN + 4


def test_entry_readout_reproduces_forward_and_rejects_bad_indices() -> None:
    model = _model(dataset_token={"enabled": True, "vocab_size": VOCAB, "dim": 4})
    model.eval()
    entry_batch = torch.tensor([0, 1, 2])
    entry_dataset = torch.tensor([1, 1, 1])
    with torch.no_grad():
        pred, reps = model(
            _cell_graph(),
            _batch(),
            entry_batch=entry_batch,
            entry_dataset=entry_dataset,
        )
        again, fit = model.entry_readout(reps, _batch(), entry_batch, entry_dataset)
    assert torch.allclose(pred, again)
    assert fit is not None and torch.allclose(fit, reps["head_outputs"]["global"])
    with pytest.raises(ValueError, match="vocabulary has 3 names"):
        model(
            _cell_graph(),
            _batch(),
            entry_batch=entry_batch,
            entry_dataset=torch.tensor([0, 1, 3]),
        )
    with pytest.raises(ValueError, match="needs entry_batch"):
        model(_cell_graph(), _batch())


def test_input_mode_is_the_deferred_ablation() -> None:
    with pytest.raises(NotImplementedError):
        _model(dataset_token={"enabled": True, "vocab_size": VOCAB, "mode": "input"})
