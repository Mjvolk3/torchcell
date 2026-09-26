# tests/torchcell/trainers/test_neo_regression.py
# [[tests.torchcell.trainers.test_neo_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_neo_regression.py
"""``neo_regression.RegressionTask`` and its two helpers on CPU with a two-layer toy model.

Exact values: ``MSEListMLELoss(alpha=0.5)`` on predictions [1, 2, 3] against [3, 2, 1] is
MSE (4 + 0 + 4) / 3 = 2.6666666667 plus 0.5 * 4.2228178933 = 4.7780756133, where the
implemented ListMLE is -sum log_softmax([1, 2, 3]) = 3 ln(e + e^2 + e^3) - 6 (see
``test_list_mle.py``). ``ListMLEMetric`` weights each update by its batch
size: one list at 4.2228178933 and a two-list batch at (4.2228178933 + 3 log 3) / 2 give
(4.2228178933 + 2 * 3.7593273797) / 3 = 3.9138242176.

The fit test runs one training and one validation batch. ``on_validation_epoch_end`` bins
the predictions into a wandb table and logs a box-plot image every ``boxplot_every_n_epochs``
epochs, including epoch 0, so the wandb calls are recorded at the boundary and asserted.
The model-artifact branch is inert here because ``ModelCheckpoint`` saves in
``on_validation_end``, after the module's ``on_validation_epoch_end`` runs; the branch
therefore sees the PREVIOUS epoch's best path, and the final epoch's checkpoint is never
logged as an artifact (a latent defect, pinned by ``last_logged_best_step is None``).
"""

import math
from typing import Any, cast

import lightning as L
import pytest
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool

from torchcell.losses.list_mle import ListMLELoss
from torchcell.trainers.neo_regression import (
    ListMLEMetric,
    MSEListMLELoss,
    RegressionTask,
)

Y_PRED = torch.tensor([[1.0, 2.0, 3.0]])
Y_TRUE = torch.tensor([[3.0, 2.0, 1.0]])
LIST_MLE = 4.2228178933


def test_mse_list_mle_loss_is_mse_plus_alpha_times_list_mle() -> None:
    """8/3 + 0.5 * 4.2228178933 = 4.7780756133."""
    loss = MSEListMLELoss(alpha=0.5)(Y_PRED, Y_TRUE)
    assert loss.item() == pytest.approx(8 / 3 + 0.5 * LIST_MLE, abs=1e-6)
    assert MSEListMLELoss(alpha=0.0)(Y_PRED, Y_TRUE).item() == pytest.approx(
        8 / 3, abs=1e-6
    )


def test_list_mle_metric_is_the_sample_weighted_mean_over_updates() -> None:
    """(1 * 4.2228178933 + 2 * 3.7593273797) / 3 = 3.9138242176."""
    metric = ListMLEMetric()
    metric.update(Y_PRED, Y_TRUE)
    two = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    metric.update(two, Y_TRUE.repeat(2, 1))
    expected = (LIST_MLE + 2 * (LIST_MLE + 3 * math.log(3)) / 2) / 3
    assert metric.compute().item() == pytest.approx(expected, abs=1e-6)
    assert metric.total.item() == 3
    metric.reset()
    assert metric.total.item() == 0


class _Main(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 4)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.lin(x))
        return h, global_mean_pool(h, batch)


def _model() -> nn.ModuleDict:
    torch.manual_seed(0)
    return nn.ModuleDict({"main": _Main(), "top": nn.Linear(4, 1)})


def _batch(seed: int) -> HeteroData:
    torch.manual_seed(seed)
    data = HeteroData()
    data["gene"].x = torch.randn(6, 3)
    data["gene"].label_value = torch.tensor([[0.5], [1.0]])
    data["gene"].batch = torch.tensor([0, 0, 0, 1, 1, 1])
    return data


@pytest.mark.parametrize(
    ("loss", "cls"),
    [("mse", nn.MSELoss), ("list_mle", ListMLELoss), ("mse+list_mle", MSEListMLELoss)],
)
def test_loss_name_selects_the_module(loss: str, cls: type[nn.Module]) -> None:
    """The three loss names build the matching module; alpha reaches the combined one."""
    task = RegressionTask(_model(), target="fitness", loss=loss, alpha=0.25)
    assert isinstance(task.loss, cls)
    if isinstance(task.loss, MSEListMLELoss):
        assert task.loss.alpha == 0.25


def test_unknown_loss_name_raises() -> None:
    """Anything else fails at construction."""
    with pytest.raises(ValueError, match="Loss type 'huber' is not valid"):
        RegressionTask(_model(), target="fitness", loss="huber")


def test_configure_optimizers_is_adam_with_the_given_lr_and_decay() -> None:
    """Adam, lr 3e-3, weight_decay 1e-4, over the model parameters."""
    task = RegressionTask(
        _model(), target="fitness", learning_rate=3e-3, weight_decay=1e-4
    )
    optimizer = task.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_forward_is_top_of_main_pooled_per_graph() -> None:
    """Two graphs of three genes give two predictions: top(mean over genes of tanh(lin(x)))."""
    model = _model()
    task = RegressionTask(model, target="fitness")
    batch = _batch(1)
    y_hat = task(batch["gene"].x, batch["gene"].batch)
    h = torch.tanh(cast(_Main, model["main"]).lin(batch["gene"].x))
    expected = model["top"](torch.stack([h[:3].mean(0), h[3:].mean(0)]))
    torch.testing.assert_close(y_hat, expected)
    assert y_hat.shape == (2, 1)


def test_one_epoch_fit_logs_losses_and_the_validation_box_plot(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One train and one val batch: weights move, losses log, wandb sees the table and the plot."""
    logged: list[dict[str, Any]] = []

    class _Table:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns
            self.rows: list[list[Any]] = []

        def add_data(self, *row: Any) -> None:
            self.rows.append(list(row))

    monkeypatch.setattr(wandb, "Table", _Table)
    monkeypatch.setattr(wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(wandb, "log", lambda payload: logged.append(payload))

    model = _model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    task = RegressionTask(model, target="fitness", loss="mse", boxplot_every_n_epochs=1)
    batches = cast("Dataset[HeteroData]", [_batch(1), _batch(2)])
    loader: DataLoader[HeteroData] = DataLoader(batches, batch_size=None)
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)

    assert trainer.global_step == 1
    moved = sorted(
        n for n, p in model.named_parameters() if not torch.equal(p, before[n])
    )
    assert moved == sorted(before)
    metrics = trainer.callback_metrics
    assert torch.isfinite(metrics["train/loss"]) and torch.isfinite(metrics["val/loss"])
    assert torch.isfinite(metrics["val/MSE"]) and torch.isfinite(metrics["val/ListMLE"])
    keys = [k for payload in logged for k in payload]
    assert keys == ["val/Prediction_Stats_0", "binned_values_box_plot"]
    table = logged[0]["val/Prediction_Stats_0"]
    assert table.columns == ["Range", "Mean", "StdDev"]
    # the artifact branch never ran: ModelCheckpoint had not saved when the hook fired
    assert task.last_logged_best_step is None
