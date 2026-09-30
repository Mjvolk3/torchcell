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

2026.09.30 (Phase 12). A parameter-free main (``x`` pooled by graph mean, one feature)
under an identity ``Linear(1, 1)`` top (weight 1, bias 0) makes every prediction the
graph's mean feature. Training batch: graph means m = [1, 2], targets [2, 3], so
y_hat = [1, 2], MSE = MAE = RMSE = 1.0, Pearson = Spearman = 1.0; the gradients are
dL/dw = mean(2 (y_hat - y) m) = (-2 - 4) / 2 = -3 and dL/db = mean(2 (y_hat - y)) = -2,
so Adam's first step (lr 0.1, no weight decay) moves both by +lr: w = 1.1, b = 0.1 (up to
eps / |g|, below 1e-8). The validation pass on the same batch then predicts [1.2, 2.3]:
MSE (0.64 + 0.49) / 2 = 0.565, MAE 0.75, RMSE sqrt(0.565), both correlations 1.0. The
test batch has four graphs with means [-0.5, 0.05, 1.5, 2.0] against [0, 0, 1, 2]:
MSE (0.25 + 0.0025 + 0.25 + 0) / 4 = 0.125625, MAE 0.2625, Spearman sqrt(0.9) (ranks
[1, 2, 3, 4] against [1.5, 1.5, 3, 4]), Pearson 0.9478749590 (the sample formula on the
four pairs); the binned table has rows "-inf - 0.0" (-0.5), "0.0 - 0.1" (0.05) and
"1.2 - inf" (mean 1.75, sample std sqrt(0.125)). ListMLE on ``[B, 1]`` lists of length
one is exactly 0. Findings: the test stage logs ``test_pearson``/``test_spearman``
(underscore) against ``train/``/``val/`` elsewhere; the mandated ``fast_dev_run`` fit with
``enable_checkpointing=False`` raises in ``on_validation_epoch_end``; ``list_mle`` never
trains a single-output model; an unknown target raises ``UnboundLocalError``; a skipped
box-plot epoch keeps its stored predictions; a one-prediction bin has StdDev NaN.
"""

import math
from typing import Any, cast

import lightning as L
import matplotlib.pyplot as plt
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
from torchcell.viz import genetic_interaction_score

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


class _MeanMain(nn.Module):
    """Parameter-free main: node features and their per-graph mean."""

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, global_mean_pool(x, batch)


def _identity_model() -> nn.ModuleDict:
    top = nn.Linear(1, 1)
    nn.init.ones_(top.weight)
    nn.init.zeros_(top.bias)
    return nn.ModuleDict({"main": _MeanMain(), "top": top})


def _graph_batch(means: list[float], targets: list[float]) -> HeteroData:
    """Two nodes per graph, both at the graph's mean, so pooling returns it exactly."""
    data = HeteroData()
    data["gene"].x = torch.tensor([[m] for m in means for _ in range(2)])
    data["gene"].label_value = torch.tensor([[t] for t in targets])
    data["gene"].batch = torch.arange(len(means)).repeat_interleave(2)
    return data


TRAIN = _graph_batch([1.0, 2.0], [2.0, 3.0])
TEST = _graph_batch([-0.5, 0.05, 1.5, 2.0], [0.0, 0.0, 1.0, 2.0])


class _Table:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[list[Any]] = []

    def add_data(self, *row: Any) -> None:
        self.rows.append(list(row))


class _Artifact:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class _Wandb:
    """Records every wandb call the task makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[_Artifact] = []
        monkeypatch.setattr(wandb, "Table", _Table)
        monkeypatch.setattr(wandb, "Image", lambda fig: fig)
        monkeypatch.setattr(wandb, "log", self.logged.append)
        monkeypatch.setattr(wandb, "Artifact", _Artifact)
        monkeypatch.setattr(wandb, "log_artifact", self.artifacts.append)

    @property
    def keys(self) -> list[str]:
        return [k for payload in self.logged for k in payload]


def _loader(data: HeteroData) -> DataLoader[HeteroData]:
    return DataLoader(cast("Dataset[HeteroData]", [data]), batch_size=None)


def _trainer(tmp_path: Any, **kw: Any) -> L.Trainer:
    settings: dict[str, Any] = {
        "fast_dev_run": True,
        "logger": False,
        "enable_checkpointing": False,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": tmp_path,
    }
    settings.update(kw)
    return L.Trainer(**settings)


def _task(**kw: Any) -> RegressionTask:
    settings: dict[str, Any] = {
        "target": "fitness",
        "loss": "mse",
        "learning_rate": 0.1,
        "weight_decay": 0.0,
    }
    settings.update(kw)
    return RegressionTask(_identity_model(), **settings)


def _metrics(trainer: L.Trainer) -> dict[str, float]:
    return {k: float(v) for k, v in trainer.callback_metrics.items()}


def test_test_stage_logs_exact_metrics_and_the_binned_table(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Predictions [-0.5, 0.05, 1.5, 2.0] against [0, 0, 1, 2]: every value in closed form.

    Finding: the correlations are logged as ``test_pearson`` and ``test_spearman``
    (``neo_regression.py:355,361``) while train and validation use ``train/pearson`` and
    ``val/pearson``; a one-prediction bin reports StdDev NaN (unbiased std of one value).
    Pinned until the keys share the slash form.
    """
    record = _Wandb(monkeypatch)
    task = _task()
    trainer = _trainer(tmp_path)
    trainer.test(task, dataloaders=_loader(TEST), verbose=False)
    metrics = _metrics(trainer)
    assert sorted(metrics) == [
        "test/ListMLE",
        "test/MAE",
        "test/MSE",
        "test/RMSE",
        "test/loss",
        "test_pearson",
        "test_spearman",
    ]
    assert metrics["test/loss"] == pytest.approx(0.125625, abs=1e-6)
    assert metrics["test/MSE"] == pytest.approx(0.125625, abs=1e-6)
    assert metrics["test/RMSE"] == pytest.approx(math.sqrt(0.125625), abs=1e-6)
    assert metrics["test/MAE"] == pytest.approx(0.2625, abs=1e-6)
    assert metrics["test/ListMLE"] == 0.0
    assert metrics["test_spearman"] == pytest.approx(math.sqrt(0.9), abs=1e-6)
    assert metrics["test_pearson"] == pytest.approx(0.9478749590, abs=1e-6)
    assert record.keys == ["test/Prediction_Stats_0", "test_binned_values_box_plot"]
    table = record.logged[0]["test/Prediction_Stats_0"]
    assert [row[0] for row in table.rows] == ["-inf - 0.0", "0.0 - 0.1", "1.2 - inf"]
    assert table.rows[0][1] == -0.5 and math.isnan(table.rows[0][2])
    assert table.rows[1][1] == pytest.approx(0.05) and math.isnan(table.rows[1][2])
    assert table.rows[2][1:] == pytest.approx([1.75, math.sqrt(0.125)])
    assert task.true_values == [] and task.predictions == []


def test_fast_dev_run_fit_without_checkpointing_raises(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: with ``enable_checkpointing=False`` the trainer has no checkpoint callback,
    and ``on_validation_epoch_end`` reads ``ckpt.best_model_path`` on ``None``
    (``neo_regression.py:324-325``), so the campaign's standard fit configuration cannot
    finish an epoch. Pinned until the branch checks for a callback.
    """
    record = _Wandb(monkeypatch)
    with pytest.raises(
        AttributeError, match="^'NoneType' object has no attribute 'best_model_path'$"
    ):
        _trainer(tmp_path).fit(
            _task(), train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN)
        )
    # the table and the box plot were logged before the crash
    assert record.keys == ["val/Prediction_Stats_0", "binned_values_box_plot"]


def test_one_step_logs_exact_train_and_validation_values(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train at w = 1, b = 0 (loss 1.0); Adam moves both by +0.1; validate at [1.2, 2.3]."""
    _Wandb(monkeypatch)
    task = _task()
    trainer = _trainer(tmp_path, enable_checkpointing=True)
    trainer.fit(task, train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN))
    top = cast(nn.Linear, cast(nn.ModuleDict, task.model)["top"])
    assert top.weight.item() == pytest.approx(1.1, abs=1e-6)
    assert top.bias.item() == pytest.approx(0.1, abs=1e-6)
    metrics = _metrics(trainer)
    expected = {
        "model/parameters_size": 2.0,
        "train/loss": 1.0,
        "train/pearson": 1.0,
        "train/spearman": 1.0,
        "train/MSE": 1.0,
        "train/RMSE": 1.0,
        "train/MAE": 1.0,
        "train/ListMLE": 0.0,
        "val/loss": 0.565,
        "val/pearson": 1.0,
        "val/spearman": 1.0,
        "val/MSE": 0.565,
        "val/RMSE": math.sqrt(0.565),
        "val/MAE": 0.75,
        "val/ListMLE": 0.0,
    }
    assert sorted(metrics) == sorted(expected)
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, abs=1e-5), key


def test_clip_grad_norm_clips_the_step_gradient(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clip_grad_norm=True``: called once with max_norm 0.25 on the norm sqrt(9 + 4).

    Adam's first step is invariant to a gradient's scale, so the weights still move by
    +0.1 (up to eps over the clipped gradient, about 1e-7).
    """
    _Wandb(monkeypatch)
    calls: list[tuple[float, float]] = []
    real_clip = nn.utils.clip_grad_norm_

    def clip(parameters: Any, max_norm: float, **kw: Any) -> torch.Tensor:
        norm = real_clip(parameters, max_norm=max_norm, **kw)
        calls.append((max_norm, float(norm)))
        return norm

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", clip)
    task = _task(clip_grad_norm=True, clip_grad_norm_max_norm=0.25)
    trainer = _trainer(tmp_path, enable_checkpointing=True)
    trainer.fit(task, train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN))
    assert len(calls) == 1
    assert calls[0][0] == 0.25
    assert calls[0][1] == pytest.approx(math.sqrt(13), abs=1e-6)
    top = cast(nn.Linear, cast(nn.ModuleDict, task.model)["top"])
    assert top.weight.item() == pytest.approx(1.1, abs=1e-5)


def test_list_mle_never_moves_a_single_output_model(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: predictions are ``[B, 1]``, so each ListMLE list has one element,
    ``log_softmax`` of it is 0 and the loss is exactly 0 (``neo_regression.py:131-132``);
    with no weight decay the Adam step is exactly zero. Pinned until the task ranks
    across the batch.
    """
    _Wandb(monkeypatch)
    task = _task(loss="list_mle")
    trainer = _trainer(tmp_path, enable_checkpointing=True)
    trainer.fit(task, train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN))
    top = cast(nn.Linear, cast(nn.ModuleDict, task.model)["top"])
    assert (top.weight.item(), top.bias.item()) == (1.0, 0.0)
    assert float(trainer.callback_metrics["train/loss"]) == 0.0


def test_interaction_target_uses_its_box_plot(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``genetic_interaction_score`` routes validation (targets [2, 3], predictions
    [1.2, 2.3]) and then the test stage to its own box plot.
    """
    record = _Wandb(monkeypatch)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def box_plot(true_values: torch.Tensor, predictions: torch.Tensor) -> str:
        seen.append((true_values, predictions))
        return "gi-figure"

    monkeypatch.setattr(genetic_interaction_score, "box_plot", box_plot)
    monkeypatch.setattr(plt, "close", lambda fig: None)
    task = _task(target="genetic_interaction_score")
    trainer = _trainer(tmp_path, enable_checkpointing=True)
    trainer.fit(task, train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN))
    assert len(seen) == 1
    torch.testing.assert_close(seen[0][0], torch.tensor([[2.0], [3.0]]))
    torch.testing.assert_close(
        seen[0][1], torch.tensor([[1.2], [2.3]]), atol=1e-5, rtol=0
    )
    assert record.logged[1] == {"binned_values_box_plot": "gi-figure"}
    trainer.test(task, dataloaders=_loader(TRAIN), verbose=False)
    assert len(seen) == 2
    assert record.logged[-1] == {"test_binned_values_box_plot": "gi-figure"}


def test_unknown_target_raises_at_the_box_plot(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: a target other than fitness or genetic_interaction_score leaves ``fig``
    unbound (``neo_regression.py:312-316``), so validation ends in ``UnboundLocalError``
    after the table was logged. Pinned until the constructor validates the target.
    """
    record = _Wandb(monkeypatch)
    with pytest.raises(UnboundLocalError, match="fig"):
        _trainer(tmp_path, enable_checkpointing=True).fit(
            _task(target="growth_rate"),
            train_dataloaders=_loader(TRAIN),
            val_dataloaders=_loader(TRAIN),
        )
    assert record.keys == ["val/Prediction_Stats_0"]


def test_skipped_box_plot_epoch_keeps_its_predictions(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every 2 epochs over 2 epochs: epoch 0 plots and clears, epoch 1 returns early.

    Finding: the early return (``neo_regression.py:302-305``) skips the clearing, so the
    epoch-1 validation targets stay in ``true_values`` and would join the next plotted
    epoch's table. Pinned until the lists are cleared every epoch.
    """
    record = _Wandb(monkeypatch)
    task = _task(boxplot_every_n_epochs=2)
    trainer = _trainer(
        tmp_path,
        fast_dev_run=False,
        max_epochs=2,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )
    trainer.fit(task, train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN))
    assert record.keys == ["val/Prediction_Stats_0", "binned_values_box_plot"]
    assert len(task.true_values) == 1
    torch.testing.assert_close(task.true_values[0], torch.tensor([[2.0], [3.0]]))


def test_second_epoch_logs_the_first_epoch_checkpoint_as_an_artifact(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Epoch 0 saves ``epoch=0-step=1.ckpt`` after its hook ran; epoch 1's hook (global
    step 2) logs that path as ``model-global_step-2`` with empty hparams metadata.
    """
    record = _Wandb(monkeypatch)
    task = _task()
    trainer = _trainer(
        tmp_path,
        fast_dev_run=False,
        max_epochs=2,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )
    trainer.fit(task, train_dataloaders=_loader(TRAIN), val_dataloaders=_loader(TRAIN))
    assert len(record.artifacts) == 1
    artifact = record.artifacts[0]
    assert artifact.kwargs == {
        "name": "model-global_step-2",
        "type": "model",
        "description": "Model on validation epoch end step - 2",
        "metadata": {},
    }
    assert artifact.files == [str(tmp_path / "checkpoints" / "epoch=0-step=1.ckpt")]
    assert task.last_logged_best_step == 2
    assert record.keys == [
        "val/Prediction_Stats_0",
        "binned_values_box_plot",
        "val/Prediction_Stats_1",
        "binned_values_box_plot",
    ]


def test_optimizer_is_plain_adam_over_the_model_parameters() -> None:
    """Defaults lr 1e-3 and weight_decay 1e-5, Adam's betas (0.9, 0.999), eps 1e-8,
    no amsgrad, one group holding exactly the model's parameters, no scheduler.
    """
    model = _identity_model()
    task = RegressionTask(model, target="fitness")
    optimizer = task.configure_optimizers()
    assert type(optimizer) is torch.optim.Adam
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert [id(p) for p in group["params"]] == [id(p) for p in model.parameters()]
    assert (group["lr"], group["weight_decay"]) == (1e-3, 1e-5)
    assert (group["betas"], group["eps"], group["amsgrad"]) == (
        (0.9, 0.999),
        1e-8,
        False,
    )
