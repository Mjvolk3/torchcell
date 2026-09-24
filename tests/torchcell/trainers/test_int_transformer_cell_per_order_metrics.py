"""Per-perturbation-order metrics of RegressionTask (torchcell.trainers.int_transformer_cell)."""

from typing import Any

import torch
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.trainers.int_transformer_cell import RegressionTask

GENE_NUM = 8


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
    )
    task = RegressionTask(
        model=model,
        cell_graph=_cell_graph(),
        optimizer_config={"type": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
        lr_scheduler_config=None,
        device="cpu",
        **kwargs,
    )
    logged: list[tuple[str, float]] = []
    task.log = lambda name, value, **kw: logged.append((name, float(value)))  # type: ignore[method-assign]
    return task, logged


def _batch() -> HeteroData:
    # row 0 perturbs {1, 2} (order 2), row 1 perturbs {3} (order 1), row 2 {0, 4, 5} (order 3)
    b = HeteroData()
    b["gene"].perturbation_indices = torch.tensor([1, 2, 3, 0, 4, 5])
    b["gene"].perturbation_indices_batch = torch.tensor([0, 0, 1, 2, 2, 2])
    return b


def test_per_order_metrics_split_rows_by_perturbation_count() -> None:
    task, logged = _task(per_order_metrics=True)
    assert task._perturbation_order(_batch(), 3).tolist() == [2, 1, 3]
    mask = torch.ones(3, 1, dtype=torch.bool)
    for preds, targets in [
        (torch.tensor([[0.1], [0.5], [0.9]]), torch.tensor([[0.2], [0.4], [1.0]])),
        (torch.tensor([[0.3], [0.7], [0.2]]), torch.tensor([[0.3], [0.6], [0.1]])),
    ]:
        task._update_order_metrics(
            "train", "train_order_metrics", _batch(), 3, preds, targets, mask
        )
    assert task._order_counts["train_order_metrics"] == {1: 2, 2: 2, 3: 2}
    task._log_order_epoch_metrics("train")
    names = {n for n, _ in logged}
    for k in (1, 2, 3):
        assert f"train/gene_interaction/order{k}/MSE" in names
        assert f"train/gene_interaction/order{k}/Pearson" in names
        assert f"train/n_records/gene_interaction/order{k}" in names
    counts = {n: v for n, v in logged if n.startswith("train/n_records/")}
    assert counts == {
        f"train/n_records/gene_interaction/{k}": 2.0
        for k in ("order1", "order2", "order3", "dmi", "tmi")
    }
    assert task._order_counts["train_order_metrics"] == {1: 0, 2: 0, 3: 0}


def test_single_gene_fitness_is_logged_without_an_interaction_label() -> None:
    """A single carries fitness but no interaction; its fitness metrics must still log.

    Before 2026-09-18 the fitness collection was gated on the interaction record count,
    so order-1 fitness was computed every epoch and silently discarded (S3 run
    kj03xx8y logged no ``train/fitness/order1/*`` despite 5,694 singles in the pool).
    """
    task, logged = _task(per_order_metrics=True, fitness_lambda=1.0)
    gi_mask = torch.tensor([[True], [False], [True]])  # the single has no interaction
    fit_mask = torch.ones(3, 1, dtype=torch.bool)  # every row carries fitness
    for preds, targets in [
        (torch.tensor([[0.1], [0.5], [0.9]]), torch.tensor([[0.2], [0.4], [1.0]])),
        (torch.tensor([[0.3], [0.7], [0.2]]), torch.tensor([[0.3], [0.6], [0.1]])),
    ]:
        task._update_order_metrics(
            "train", "train_order_metrics", _batch(), 3, preds, targets, gi_mask
        )
        task._update_order_metrics(
            "train",
            "train_order_fitness_metrics",
            _batch(),
            3,
            preds,
            targets,
            fit_mask,
        )
    assert task._order_counts["train_order_metrics"] == {1: 0, 2: 2, 3: 2}
    assert task._order_counts["train_order_fitness_metrics"] == {1: 2, 2: 2, 3: 2}
    task._log_order_epoch_metrics("train")
    names = {n for n, _ in logged}
    assert "train/fitness/order1/Pearson" in names
    assert "train/gene_interaction/order1/Pearson" not in names
    counts = {n: v for n, v in logged if n.startswith("train/n_records/")}
    assert counts["train/n_records/fitness/order1"] == 2.0
    assert counts["train/n_records/gene_interaction/order1"] == 0.0


def test_per_order_values_are_also_logged_under_phenotype_names() -> None:
    """smf/dmf/tmf and dmi/tmi carry the same values as order1/2/3, counts included."""
    task, logged = _task(per_order_metrics=True, fitness_lambda=1.0)
    gi_mask = torch.tensor([[True], [False], [True]])
    fit_mask = torch.ones(3, 1, dtype=torch.bool)
    for preds, targets in [
        (torch.tensor([[0.1], [0.5], [0.9]]), torch.tensor([[0.2], [0.4], [1.0]])),
        (torch.tensor([[0.3], [0.7], [0.2]]), torch.tensor([[0.3], [0.6], [0.1]])),
    ]:
        task._update_order_metrics(
            "train", "train_order_metrics", _batch(), 3, preds, targets, gi_mask
        )
        task._update_order_metrics(
            "train",
            "train_order_fitness_metrics",
            _batch(),
            3,
            preds,
            targets,
            fit_mask,
        )
    task._log_order_epoch_metrics("train")
    values = dict(logged)
    for order, name in [(1, "smf"), (2, "dmf"), (3, "tmf")]:
        for metric in ("MSE", "RMSE", "Pearson"):
            assert (
                values[f"train/fitness/{name}/{metric}"]
                == values[f"train/fitness/order{order}/{metric}"]
            )
        assert values[f"train/n_records/fitness/{name}"] == 2.0
    for order, name in [(2, "dmi"), (3, "tmi")]:
        for metric in ("MSE", "RMSE", "Pearson"):
            assert (
                values[f"train/gene_interaction/{name}/{metric}"]
                == values[f"train/gene_interaction/order{order}/{metric}"]
            )
        assert values[f"train/n_records/gene_interaction/{name}"] == 2.0
    # a single has no interaction label, so no interaction name exists for order 1
    assert not any(n.startswith("train/gene_interaction/smf") for n in values)
    assert "train/n_records/gene_interaction/order1" in values


def test_orders_without_samples_are_not_logged() -> None:
    task, logged = _task(per_order_metrics=True)
    # A label-present mask that drops the order-1 row.
    mask = torch.tensor([[True], [False], [True]])
    for _ in range(2):
        task._update_order_metrics(
            "val",
            "val_order_metrics",
            _batch(),
            3,
            torch.tensor([[0.1], [0.5], [0.9]]),
            torch.tensor([[0.2], [0.4], [1.0]]),
            mask,
        )
    task._log_order_epoch_metrics("val")
    names = {n for n, _ in logged}
    assert "val/gene_interaction/order2/MSE" in names
    assert "val/gene_interaction/order3/MSE" in names
    assert "val/gene_interaction/order1/MSE" not in names


def test_flag_off_adds_no_modules_and_logs_nothing() -> None:
    task, logged = _task(per_order_metrics=False)
    assert not hasattr(task, "train_order_metrics")
    task._update_order_metrics(
        "train",
        "train_order_metrics",
        _batch(),
        3,
        torch.zeros(3, 1),
        torch.zeros(3, 1),
        torch.ones(3, 1, dtype=torch.bool),
    )
    task._log_order_epoch_metrics("train")
    assert logged == []
