"""Per-entry rows of RegressionTask: loss over entries, policy-reduced validation."""

from typing import Any

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.trainers.int_transformer_cell import (
    RegressionTask,
    token_precedence_rank,
)

GENE_NUM = 8
VOCAB = [
    "DmiCostanzo2016Dataset",
    "DmiKuzmin2018Dataset",
    "DmiKuzmin2020Dataset",
    "GeneEssentialitySgdDataset",
    "SmfCostanzo2016Dataset",
]


def _cell_graph() -> HeteroData:
    cg = HeteroData()
    cg["gene"].num_nodes = GENE_NUM
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    return cg


def _task(**kwargs: Any) -> tuple[RegressionTask, list[tuple[str, float]]]:
    torch.manual_seed(0)
    model = CellGraphTransformer(
        gene_num=GENE_NUM,
        hidden_channels=16,
        num_transformer_layers=1,
        num_attention_heads=4,
        cell_graph=_cell_graph(),
        heads_config={
            "global": {"output_dim": 1, "use_gene_pool": False, "linear": True}
        },
        perturb_cls=True,
        perturbation_head_cls="perturbed",
        dataset_token={"enabled": True, "vocab_size": len(VOCAB), "dim": 4},
    )
    task = RegressionTask(
        model=model,
        cell_graph=_cell_graph(),
        optimizer_config={"type": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
        lr_scheduler_config=None,
        device="cpu",
        loss_func=nn.MSELoss(),
        fitness_lambda=1.0,
        per_order_metrics=True,
        per_entry=True,
        dataset_vocabulary=VOCAB,
        **kwargs,
    )
    logged: list[tuple[str, float]] = []
    task.log = lambda name, value, **kw: logged.append((name, float(value)))  # type: ignore[method-assign]
    return task, logged


def _batch() -> HeteroData:
    """Three genotypes; entry rows carry (row, type, value, token).

    genotype 0 (double {1,2}): gi from Costanzo (0) and Kuzmin 2018 (1), fitness Costanzo
    genotype 1 (single {3}): fitness from Costanzo smf (4) and the SGD converted 0 (3)
    genotype 2 (triple {0,4,5}): gi Kuzmin 2020 (2) twice, fitness Costanzo
    """
    b = HeteroData()
    b["gene"].perturbation_indices = torch.tensor([1, 2, 3, 0, 4, 5])
    b["gene"].perturbation_indices_batch = torch.tensor([0, 0, 1, 2, 2, 2])
    b["gene"].phenotype_types = ["fitness", "gene_interaction"]
    rows = [0, 0, 0, 1, 1, 2, 2, 2]
    types = [1, 1, 0, 0, 0, 1, 1, 0]
    vals = [0.10, 0.30, 0.90, 0.95, 0.00, -0.20, -0.40, 0.70]
    toks = [0, 1, 0, 4, 3, 2, 2, 0]
    b["gene"].phenotype_values_batch = torch.tensor(rows)
    b["gene"].phenotype_type_indices = torch.tensor(types)
    b["gene"].phenotype_values = torch.tensor(vals)
    b["gene"].phenotype_dataset_indices = torch.tensor(toks)
    return b


def test_precedence_ranks_follow_the_label_policy() -> None:
    assert [token_precedence_rank(n) for n in VOCAB] == [2, 0, 1, 3, 2]
    assert token_precedence_rank("SyntheticOffsetDataset") == 4
    with pytest.raises(ValueError):
        token_precedence_rank("MysteryDataset")


def test_entry_rows_and_policy_reduction() -> None:
    task, _ = _task()
    rows, vals, orig, toks, sel = task._entry_rows(_batch(), "gene_interaction")
    assert rows.tolist() == [0, 0, 2, 2]
    assert toks.tolist() == [0, 1, 2, 2]
    assert int(sel.sum()) == 4
    preds = torch.tensor([1.0, 2.0, 3.0, 5.0])
    target, pred, present = task._policy_reduce(rows, vals, preds, toks, batch_size=3)
    # genotype 0 keeps the Kuzmin 2018 row (rank 0) over Costanzo (rank 2)
    assert target[0].item() == pytest.approx(0.30)
    assert pred[0].item() == pytest.approx(2.0)
    # genotype 1 has no interaction row
    assert present.tolist() == [True, False, True]
    # genotype 2 averages its two same-source rows
    assert target[2].item() == pytest.approx(-0.30)
    assert pred[2].item() == pytest.approx(4.0)
    # fitness: the measured Costanzo single beats the converted 0
    rows_f, vals_f, _, toks_f, _ = task._entry_rows(_batch(), "fitness")
    t_f, _, p_f = task._policy_reduce(
        rows_f, vals_f, torch.zeros_like(vals_f), toks_f, 3
    )
    assert t_f[1].item() == pytest.approx(0.95)
    assert p_f.tolist() == [True, True, True]


def test_per_entry_step_trains_on_every_row_and_logs_counts() -> None:
    task, logged = _task()
    loss, pred, target = task._shared_step_per_entry(_batch(), 0, "train")
    assert loss.requires_grad and torch.isfinite(loss)
    assert pred is not None and target is not None
    assert pred.shape == (4, 1) and target.shape == (4, 1)  # four interaction entries
    assert task._entry_counts["train"] == {"gene_interaction": 4, "fitness": 4}
    # per-order rows counted per entry in training: doubles 2, triples 2
    assert task._order_counts["train_order_metrics"] == {1: 0, 2: 2, 3: 2}
    assert task._order_counts["train_order_fitness_metrics"] == {1: 2, 2: 1, 3: 1}
    task._log_per_entry_epoch_metrics("train")
    names = [n for n, _ in logged]
    for name in VOCAB:
        assert f"train/token/{name}/gene_interaction/Pearson" in names
        assert f"train/token/{name}/fitness/n_entries" in names
    assert ("train/n_entries/gene_interaction", 4.0) in logged
    assert task._entry_counts["train"] == {"gene_interaction": 0, "fitness": 0}


def test_validation_reduces_to_one_row_per_genotype() -> None:
    task, _ = _task()
    task.eval()
    with torch.no_grad():
        _, pred, target = task._shared_step_per_entry(_batch(), 0, "val")
    assert pred is not None and target is not None
    # genotypes 0 and 2 carry interaction rows; genotype 1 does not
    assert pred.shape == (2, 1)
    assert target.view(-1).tolist() == pytest.approx([0.30, -0.30])
    assert task._order_counts["val_order_metrics"] == {1: 0, 2: 1, 3: 1}
    # the essentiality loader is scored on singles under the two tokens
    ess = {
        "token_smf": "SmfCostanzo2016Dataset",
        "token_sgd": "GeneEssentialitySgdDataset",
        "released": {"3": 1, "6": 0},
        "matched": {"3": 1},
    }
    task2, logged = _task(essentiality_eval=ess)
    task2.eval()
    single = HeteroData()
    single["gene"].perturbation_indices = torch.tensor([3, 6])
    single["gene"].perturbation_indices_batch = torch.tensor([0, 1])
    assert task2.validation_step(single, 0, dataloader_idx=1) is None
    assert task2._ess_buffer["node"][0].tolist() == [3, 6]
    task2._log_essentiality_epoch()
    names = [n for n, _ in logged]
    for key in ("released_smf", "released_sgd", "matched_smf", "matched_sgd"):
        assert f"val_ess/auroc_{key}" in names
    with pytest.raises(ValueError, match="single-deletion"):
        task2.validation_step(_batch(), 0, dataloader_idx=1)
