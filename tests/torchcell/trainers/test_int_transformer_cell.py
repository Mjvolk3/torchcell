# tests/torchcell/trainers/test_int_transformer_cell.py
# [[tests.torchcell.trainers.test_int_transformer_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_transformer_cell.py
"""``RegressionTask`` (the live CGT trainer) on CPU with a tiny model and two batches.

Lightning 2.5 ``fast_dev_run=True`` runs one training and one validation batch, disables
loggers and checkpoint callbacks, and skips the sanity check. Every plotting frequency is
0, which ``_is_scheduled`` reads as "never" (the ZeroDivisionError fix its docstring
records), so nothing reaches ``wandb.log``; a recorder on it proves that. The trainer
uses manual optimization, so the test also checks that the optimizer step moved the
weights and that the logged losses are finite.
"""

from typing import Any, cast

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData

from torchcell.losses.logcosh import LogCoshLoss
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.trainers.int_transformer_cell import RegressionTask

N, D, HEADS = 8, 16, 4


def _labelled(batch: HeteroData, values: list[float]) -> HeteroData:
    out = batch.clone()
    out["gene"].phenotype_values = torch.tensor(values).unsqueeze(1)
    return out


def _task(cell_graph: HeteroData, **overrides: Any) -> RegressionTask:
    torch.manual_seed(0)
    model = CellGraphTransformer(
        gene_num=N,
        hidden_channels=D,
        num_transformer_layers=1,
        num_attention_heads=HEADS,
        cell_graph=cell_graph,
        dropout=0.0,
    )
    kwargs: dict[str, Any] = dict(
        optimizer_config={"type": "AdamW", "learning_rate": 1e-2},
        lr_scheduler_config=None,
        plot_every_n_epochs=0,
        plot_transformer_diagnostics_every_n_epochs=0,
        plot_edge_recovery_every_n_epochs=0,
        loss_func=LogCoshLoss(),
        device="cpu",
    )
    kwargs.update(overrides)
    return RegressionTask(model=model, cell_graph=cell_graph, **kwargs)


def _trainer(tmp_path: Any) -> L.Trainer:
    return L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )


@pytest.fixture
def loader(batch: HeteroData) -> DataLoader[HeteroData]:
    """Two pre-collated batches; batch_size=None hands each through uncollated."""
    batches = [_labelled(batch, [0.1, -0.2, 0.3]), _labelled(batch, [0.0, 0.5, -0.1])]
    return DataLoader(cast("Dataset[HeteroData]", batches), batch_size=None)


def test_fast_dev_run_trains_one_step_without_touching_wandb(
    cell_graph: HeteroData,
    loader: DataLoader[HeteroData],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One optimizer step moves every weight, logs finite train/val losses, calls wandb never."""
    calls: list[Any] = []
    monkeypatch.setattr("wandb.log", lambda *a, **k: calls.append((a, k)))
    task = _task(cell_graph)
    before = {n: p.detach().clone() for n, p in task.model.named_parameters()}

    _trainer(tmp_path).fit(task, train_dataloaders=loader, val_dataloaders=loader)

    assert task.trainer.global_step == 1
    moved = sorted(
        n for n, p in task.model.named_parameters() if not torch.equal(p, before[n])
    )
    assert moved == sorted(before)  # every parameter has a gradient and AdamW moves it
    metrics = task.trainer.callback_metrics
    assert torch.isfinite(metrics["train/loss"]) and torch.isfinite(metrics["val/loss"])
    assert metrics["learning_rate"].item() == pytest.approx(1e-2)
    assert calls == []


def test_forward_is_the_model_forward_on_the_cloned_cell_graph(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """task(batch) equals model(task.cell_graph, batch); the clone is made once and reused."""
    task = _task(cell_graph)
    labelled = _labelled(batch, [0.1, -0.2, 0.3])
    predictions, representations = task(labelled)
    expected, _ = task.model(task.cell_graph, labelled)
    torch.testing.assert_close(predictions, expected)
    assert predictions.shape == (3, 1)
    assert representations["attention_weights"] is None
    assert task.cell_graph is not cell_graph  # cloned in __init__
    graph_before = task.cell_graph
    task(labelled)
    assert task.cell_graph is graph_before
    assert task._cell_graph_device == torch.device("cpu")


def test_batch_size_is_the_number_of_genotypes(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Perturbation batches count distinct batch indices: 3, not the 6 perturbed genes."""
    task = _task(cell_graph)
    assert task._get_batch_size(_labelled(batch, [0.1, -0.2, 0.3])) == 3
    dense = HeteroData()
    dense["gene"].x = torch.zeros(5, 2)
    assert task._get_batch_size(dense) == 5


@pytest.mark.parametrize(
    ("freq", "epoch", "expected"),
    [
        (0, 0, False),
        (None, 0, False),
        (1, 0, True),
        (2, 0, False),
        (2, 1, True),
        (3, 5, True),
    ],
)
def test_is_scheduled_treats_zero_and_none_as_never(
    cell_graph: HeteroData, freq: int | None, epoch: int, expected: bool
) -> None:
    """(epoch + 1) % freq == 0, with freq 0 or None meaning never (the documented fix)."""
    task = _task(cell_graph)
    task.trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    task.trainer.fit_loop.epoch_progress.current.completed = epoch
    assert task.current_epoch == epoch
    assert task._is_scheduled(freq) is expected


def test_configure_optimizers_builds_the_named_optimizer_and_scheduler(
    cell_graph: HeteroData,
) -> None:
    """AdamW at lr 1e-2 alone; CosineAnnealingLR when asked; ReduceLROnPlateau by default."""
    alone = _task(cell_graph).configure_optimizers()
    assert isinstance(alone, torch.optim.AdamW)
    assert alone.param_groups[0]["lr"] == pytest.approx(1e-2)

    cosine = _task(
        cell_graph, lr_scheduler_config={"type": "CosineAnnealingLR", "T_max": 5}
    ).configure_optimizers()
    assert isinstance(
        cosine["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert cosine["lr_scheduler"]["interval"] == "epoch"

    plateau = _task(
        cell_graph, lr_scheduler_config={"factor": 0.5}
    ).configure_optimizers()
    assert isinstance(
        plateau["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert plateau["lr_scheduler"]["monitor"] == "val/gene_interaction/MSE"


def test_missing_loss_function_is_an_error_at_step_time(
    cell_graph: HeteroData,
    batch: HeteroData,
    loader: DataLoader[HeteroData],
    tmp_path: Any,
) -> None:
    """A task built without loss_func fails the first shared step, not silently."""
    task = _task(cell_graph, loss_func=None)
    with pytest.raises(ValueError, match="No loss function provided"):
        _trainer(tmp_path).fit(task, train_dataloaders=loader)
