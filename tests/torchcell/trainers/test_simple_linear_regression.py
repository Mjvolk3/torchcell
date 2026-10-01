# tests/torchcell/trainers/test_simple_linear_regression.py
# [[tests.torchcell.trainers.test_simple_linear_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_simple_linear_regression.py
"""``SimpleLinearRegressionTask`` on CPU with a hand-set ``SimpleLinearModel``.

Fixture: ``SimpleLinearModel(2, 1, scatter="add")`` with weight [1, -1] and bias 0.5,
and one batch of five perturbed-gene rows in three sets (``x_pert_batch`` [0, 0, 1, 2,
2]). Summed per set the features are [1, 1], [2, 1], [1, 2], so the predictions are
y_hat = [0.5, 1.5, -0.5] against ``fitness`` y = [1.0, 0.0, 0.25]:

* residual e = y_hat - y = [-0.5, 1.5, -0.75];
* MSE = (0.25 + 2.25 + 0.5625) / 3 = 49/48 = 1.0208333, RMSE = 7 / sqrt(48) = 1.0103630,
  MAE = (0.5 + 1.5 + 0.75) / 3 = 11/12 = 0.9166667;
* Pearson: y_hat deviates [0, 1, -1], y deviates [7/12, -5/12, -1/6], cov -1/4 over
  norms sqrt(2) * sqrt(78) / 12, so r = -3 / sqrt(156) = -0.2401922;
* Spearman: ranks 2, 3, 1 against 3, 1, 2, sum d^2 = 6, rho = 1 - 36/24 = -0.5;
* MSE gradients: dL/dw = 2/3 * e @ [[1, 1], [2, 1], [1, 2]] = [7/6, -1/3],
  dL/db = 2/3 * sum(e) = 1/6, total norm sqrt(49 + 4 + 1) / 6 = 1.2247449.

None of the MSE gradients is zero, so the first Adam step moves w to [0.999, -0.999] and
b to 0.499 (-lr * sign(grad); eps 1e-8 and weight decay 1e-5 perturb it below 1e-7).
Metric values from torchmetrics carry float32 error of about 1e-6, hence ``abs=1e-6``.
"""

import math
from pathlib import Path
from typing import Any, cast

import lightning as L
import matplotlib.pyplot as plt
import pytest
import torch
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

import torchcell.viz.fitness as viz_fitness
import torchcell.viz.genetic_interaction_score as viz_gi
from torchcell.models.linear import SimpleLinearModel
from torchcell.trainers.simple_linear_regression import SimpleLinearRegressionTask

Y = [1.0, 0.0, 0.25]
Y_HAT = [0.5, 1.5, -0.5]
MSE = 49 / 48
MAE = 11 / 12
PEARSON = -3 / math.sqrt(156)
SPEARMAN = -0.5
GRAD_NORM = math.sqrt(54) / 6


def _model() -> SimpleLinearModel:
    model = SimpleLinearModel(2, 1, scatter="add")
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, -1.0]]))
        model.linear.bias.fill_(0.5)
    return model


def _batch() -> Data:
    return Data(
        x_pert=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [0.0, 2.0], [1.0, 0.0]]
        ),
        x_pert_batch=torch.tensor([0, 0, 1, 2, 2]),
        fitness=torch.tensor(Y),
    )


def _loader() -> DataLoader[Data]:
    return DataLoader(cast("Dataset[Data]", [_batch()]), batch_size=None)


def _trainer(tmp_path: Path, **kwargs: Any) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        **kwargs,
    )


class _BoxPlots:
    """Stands in for a ``box_plot``: records its inputs, returns a real figure."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[float], list[float]]] = []

    def __call__(self, true_values: torch.Tensor, predictions: torch.Tensor) -> Figure:
        self.calls.append((true_values.tolist(), predictions.tolist()))
        return plt.figure()


def _record_wandb(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "log", lambda payload: logged.append(payload))
    monkeypatch.setattr(wandb, "Image", lambda fig: fig)
    return logged


@pytest.mark.parametrize(("loss", "cls"), [("mse", nn.MSELoss), ("mae", nn.L1Loss)])
def test_loss_name_selects_the_criterion(loss: str, cls: type[nn.Module]) -> None:
    """``"mse"`` builds ``MSELoss`` and ``"mae"`` builds ``L1Loss``; both give exact values."""
    task = SimpleLinearRegressionTask(_model(), target="fitness", loss=loss)
    assert type(task.loss) is cls
    value = task.loss(torch.tensor(Y_HAT), torch.tensor(Y)).item()
    assert value == pytest.approx({"mse": MSE, "mae": MAE}[loss], abs=1e-6)


def test_unknown_loss_name_raises_with_the_run_together_message() -> None:
    """Finding: the two message literals concatenate with no space (lines 74 to 75).

    The text reads "...is not valid.Currently, supports...". Pinned until a space is
    added.
    """
    with pytest.raises(ValueError) as excinfo:
        SimpleLinearRegressionTask(_model(), target="fitness", loss="huber")
    assert str(excinfo.value) == (
        "Loss type 'huber' is not valid.Currently, supports 'mse' or 'mae' loss."
    )


def test_configure_optimizers_is_adam_over_the_model_parameters() -> None:
    """One Adam group with the given lr and weight decay over [weight, bias]."""
    model = _model()
    task = SimpleLinearRegressionTask(
        model, target="fitness", learning_rate=3e-3, weight_decay=1e-4
    )
    optimizer = task.configure_optimizers()
    assert type(optimizer) is torch.optim.Adam
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (3e-3, 1e-4)
    assert [id(p) for p in group["params"]] == [
        id(model.linear.weight),
        id(model.linear.bias),
    ]


def test_forward_squeezes_the_set_dimension_even_for_one_set() -> None:
    """Three sets give ``[3]``; a single set squeezes all the way to a 0-d tensor.

    The 0-d case is a quirk of the bare ``.squeeze()`` (line 118): against a ``[1]``
    target, ``MSELoss`` broadcasts and warns. Pinned as the current behavior.
    """
    task = SimpleLinearRegressionTask(_model(), target="fitness")
    batch = _batch()
    three = task(batch.x_pert, batch.x_pert_batch)
    assert (three.shape, three.tolist()) == (torch.Size([3]), Y_HAT)
    one = task(batch.x_pert[:2], torch.tensor([0, 0]))
    assert (one.shape, one.item()) == (torch.Size([]), 0.5)


def test_one_training_step_logs_closed_form_values_and_takes_one_adam_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_loss 49/48 and the module docstring metrics; w -> [0.999, -0.999], b -> 0.499.

    ``training_step`` returns the loss (callbacks see ``{"loss": 49/48}``), the train
    start logs 3 parameters, and the epoch-0 validation logs one box plot.
    """
    logged = _record_wandb(monkeypatch)
    step_outputs: list[dict[str, float]] = []

    class _Outputs(Callback):
        def on_train_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            step_outputs.append({k: v.item() for k, v in outputs.items()})

    model = _model()
    task = SimpleLinearRegressionTask(model, target="fitness")
    trainer = _trainer(tmp_path, fast_dev_run=True, callbacks=[_Outputs()])
    trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())

    assert step_outputs == [{"loss": pytest.approx(MSE, abs=1e-6)}]
    metrics = {
        k: v.item()
        for k, v in trainer.callback_metrics.items()
        if not k.startswith("val_")
    }
    assert metrics == {
        "model/parameters_size": 3.0,
        "train_loss": pytest.approx(MSE, abs=1e-6),
        "train_pearson": pytest.approx(PEARSON, abs=1e-6),
        "train_spearman": pytest.approx(SPEARMAN, abs=1e-6),
        "train_MSE": pytest.approx(MSE, abs=1e-6),
        "train_RMSE": pytest.approx(math.sqrt(MSE), abs=1e-6),
        "train_MAE": pytest.approx(MAE, abs=1e-6),
    }
    assert model.linear.weight.flatten().tolist() == [
        pytest.approx(0.999, abs=1e-7),
        pytest.approx(-0.999, abs=1e-7),
    ]
    assert model.linear.bias.item() == pytest.approx(0.499, abs=1e-7)
    assert [list(payload) for payload in logged] == [["binned_values_box_plot"]]


@pytest.mark.parametrize("clip", [True, False])
def test_clip_grad_norm_flag_rescales_gradients_to_the_max_norm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clip: bool
) -> None:
    """With the flag, the model gradients (norm 1.2247449) are clipped to norm 0.25.

    ``clip_grad_norm_`` receives both model parameters and ``max_norm=0.25``, returns
    the pre-clip norm sqrt(54) / 6, and leaves gradients of norm 0.25 * 1.2247449 /
    (1.2247449 + 1e-6). Without the flag it is never called.
    """
    _record_wandb(monkeypatch)
    calls: list[dict[str, Any]] = []
    original = nn.utils.clip_grad_norm_

    def _clip(parameters: Any, max_norm: float) -> torch.Tensor:
        params = list(parameters)
        total = original(params, max_norm=max_norm)
        after = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in params))
        calls.append(
            {
                "n": len(params),
                "max_norm": max_norm,
                "total": total.item(),
                "after": after,
            }
        )
        return total

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", _clip)
    task = SimpleLinearRegressionTask(
        _model(), target="fitness", clip_grad_norm=clip, clip_grad_norm_max_norm=0.25
    )
    _trainer(tmp_path, fast_dev_run=True).fit(
        task, train_dataloaders=_loader(), val_dataloaders=_loader()
    )
    expected = [
        {
            "n": 2,
            "max_norm": 0.25,
            "total": pytest.approx(GRAD_NORM, abs=1e-6),
            "after": pytest.approx(0.25 * GRAD_NORM / (GRAD_NORM + 1e-6), abs=1e-6),
        }
    ]
    assert calls == (expected if clip else [])


def test_validate_logs_loss_metrics_and_plots_the_raw_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation with ``loss="mae"``: val_loss 11/12, the MSE-family metrics unchanged.

    The fitness box plot receives y and y_hat exactly, the buffers are emptied after it,
    and ``wandb.log`` sees one payload.
    """
    logged = _record_wandb(monkeypatch)
    box_plots = _BoxPlots()
    monkeypatch.setattr(viz_fitness, "box_plot", box_plots)
    task = SimpleLinearRegressionTask(_model(), target="fitness", loss="mae")
    results = _trainer(tmp_path).validate(task, dataloaders=_loader(), verbose=False)
    assert results == [
        {
            "val_loss": pytest.approx(MAE, abs=1e-6),
            "val_pearson": pytest.approx(PEARSON, abs=1e-6),
            "val_spearman": pytest.approx(SPEARMAN, abs=1e-6),
            "val_MSE": pytest.approx(MSE, abs=1e-6),
            "val_RMSE": pytest.approx(math.sqrt(MSE), abs=1e-6),
            "val_MAE": pytest.approx(MAE, abs=1e-6),
        }
    ]
    assert box_plots.calls == [(Y, Y_HAT)]
    assert [list(payload) for payload in logged] == [["binned_values_box_plot"]]
    assert (task.true_values, task.predictions) == ([], [])


def test_test_logs_loss_metrics_and_correlations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``trainer.test`` returns the six closed-form values and never calls ``wandb.log``."""
    logged = _record_wandb(monkeypatch)
    task = SimpleLinearRegressionTask(_model(), target="fitness")
    results = _trainer(tmp_path).test(task, dataloaders=_loader(), verbose=False)
    assert results == [
        {
            "test_loss": pytest.approx(MSE, abs=1e-6),
            "test_pearson": pytest.approx(PEARSON, abs=1e-6),
            "test_spearman": pytest.approx(SPEARMAN, abs=1e-6),
            "test_MSE": pytest.approx(MSE, abs=1e-6),
            "test_RMSE": pytest.approx(math.sqrt(MSE), abs=1e-6),
            "test_MAE": pytest.approx(MAE, abs=1e-6),
        }
    ]
    assert logged == []


def test_genetic_interaction_target_reads_that_field_and_its_box_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``target`` names the batch field regressed and the box plot used.

    With ``genetic_interaction_score`` = [0.0, 1.0, -1.0] (y_hat minus 0.5, so the MSE
    loss is exactly 0.25) the gi box plot, not the fitness one, receives it.
    """
    _record_wandb(monkeypatch)
    fitness_plots, gi_plots = _BoxPlots(), _BoxPlots()
    monkeypatch.setattr(viz_fitness, "box_plot", fitness_plots)
    monkeypatch.setattr(viz_gi, "box_plot", gi_plots)
    batch = _batch()
    batch.genetic_interaction_score = torch.tensor([0.0, 1.0, -1.0])
    task = SimpleLinearRegressionTask(_model(), target="genetic_interaction_score")
    loader = DataLoader(cast("Dataset[Data]", [batch]), batch_size=None)
    results = _trainer(tmp_path).validate(task, dataloaders=loader, verbose=False)
    assert results[0]["val_loss"] == pytest.approx(0.25, abs=1e-6)
    assert fitness_plots.calls == []
    assert gi_plots.calls == [([0.0, 1.0, -1.0], Y_HAT)]


def test_unknown_target_is_rejected_at_construction() -> None:
    """A target without a box plot raises ``ValueError`` naming it, in ``__init__``.

    Issue #516: the ``if``/``elif`` on the target had no ``else``, so a batch field that
    exists but is not ``fitness``/``genetic_interaction_score`` trained and validated,
    then ``wandb.Image(fig)`` raised ``UnboundLocalError``.
    """
    with pytest.raises(ValueError) as excinfo:
        SimpleLinearRegressionTask(_model(), target="growth_rate")
    assert str(excinfo.value) == (
        "Unknown target 'growth_rate': expected one of "
        "('fitness', 'genetic_interaction_score')."
    )


def test_sanity_check_predictions_stay_out_of_the_first_box_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The epoch-0 box plot holds only the real validation pass, after one Adam step.

    Issue #516: the sanity pass used to be collected and plotted with the real one (six
    values). After the step w = [0.999, -0.999] and b = 0.499, so the summed features
    [1, 1], [2, 1], [1, 2] predict [0.499, 1.498, -0.500].
    """
    _record_wandb(monkeypatch)
    box_plots = _BoxPlots()
    monkeypatch.setattr(viz_fitness, "box_plot", box_plots)
    trainer = _trainer(
        tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=1,
        enable_checkpointing=False,
    )
    task = SimpleLinearRegressionTask(_model(), target="fitness")
    trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())
    expected_calls: list[Any] = [
        (Y, [pytest.approx(v, abs=1e-6) for v in (0.499, 1.498, -0.5)])
    ]
    assert box_plots.calls == expected_calls
    assert (task.true_values, task.predictions) == ([], [])


def test_artifact_of_the_current_best_is_logged_on_every_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each epoch logs the checkpoint ``ModelCheckpoint`` just saved, plotting or not.

    Issue #516: the artifact branch sat behind the plotting early return in
    ``on_validation_epoch_end``, which runs before ``ModelCheckpoint`` saves, so it
    logged the previous epoch's checkpoint and only on plotting epochs. With
    validation every epoch the checkpoint is saved in ``on_train_epoch_end`` after the
    module hooks, so the task now logs from the next ``on_train_epoch_start`` and from
    ``on_train_end``. One manual-optimization batch per epoch makes
    the global step 1 after epoch 0 and 2 after epoch 1; val_loss falls each step, so
    each epoch's checkpoint is the best. ``boxplot_every_n_epochs`` 1 and 2 give the
    same artifacts.
    """
    _record_wandb(monkeypatch)
    monkeypatch.setattr(viz_fitness, "box_plot", _BoxPlots())
    names: dict[int, tuple[list[tuple[str, list[str]]], int | None]] = {}
    for every in (1, 2):
        records: list[tuple[str, list[str]]] = []

        class _Artifact:
            def __init__(self, name: str, **kwargs: Any) -> None:
                self.name = name
                self.files: list[str] = []

            def add_file(self, path: str) -> None:
                self.files.append(Path(path).name)

        monkeypatch.setattr(wandb, "Artifact", _Artifact)
        monkeypatch.setattr(
            wandb, "log_artifact", lambda a: records.append((a.name, a.files))
        )
        checkpoint = ModelCheckpoint(dirpath=tmp_path / str(every), monitor="val_loss")
        trainer = _trainer(
            tmp_path,
            max_epochs=2,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            callbacks=[checkpoint],
        )
        task = SimpleLinearRegressionTask(
            _model(), target="fitness", boxplot_every_n_epochs=every
        )
        trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())
        names[every] = (records, task.last_logged_best_step)
    per_epoch = [
        ("model-global_step-1", ["epoch=0-step=1.ckpt"]),
        ("model-global_step-2", ["epoch=1-step=2.ckpt"]),
    ]
    assert names == {1: (per_epoch, 2), 2: (per_epoch, 2)}
