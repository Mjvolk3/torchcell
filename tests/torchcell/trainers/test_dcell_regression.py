# tests/torchcell/trainers/test_dcell_regression.py
# [[tests.torchcell.trainers.test_dcell_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_dcell_regression.py
"""``DCellRegressionTask`` on CPU with a weight-free DCell stand-in over the conftest GO.

Fixture (``tests/torchcell/conftest.py``): ``make_dcell_regression_batch()`` is three
samples of the three-term DCell hierarchy with knockouts {0, 1, 2, 3}, {0} and {2};
``DCellCountSubsystems`` + ``DCellIdentityHeads`` predict intact-gene counts, so
``GO:1`` = [0, 1, 2], ``GO:2`` = [0, 2, 1], ``GO:ROOT`` = ``GO:1 - GO:2`` = [0, -1, 1],
against ``fitness`` y = [1.0, 0.0, 0.5].

The trainer calls the loss it constructs, ``torchcell.losses.DCellLoss(predictions,
outputs, target)``, with predictions = the squeezed root head and
``outputs["linear_outputs"]`` = every squeezed head (the loss skips ``GO:ROOT``):

* loss = MSE(root) + 0.3 * mean(MSE(GO:1), MSE(GO:2))
  = (1 + 1 + 0.25) / 3 + 0.3 * ((1 + 1 + 2.25) / 3 + (1 + 4 + 0.25) / 3) / 2
  = 0.75 + 0.3 * 9.5 / 6 = 0.75 + 0.475 = 1.225;
* subsystem mean m = mean over the three terms = [0, 2/3, 4/3];
* Pearson(m, y) = -0.5 and Pearson(root, y) = +0.5 (deviations [-2/3, 0, 2/3] and
  [0, -1, 1] against [0.5, -0.5, 0]: cov -1/3 over 2/3, cov 1/2 over 1); Spearman
  equals Pearson here (ranks of m are 1, 2, 3; of root 2, 1, 3; of y 3, 1, 2);
* RMSE/MSE/MAE score the root, the prediction DCell reports (Ma et al. 2018 read the
  root term's output as the phenotype; the subsystem mean is not a prediction of
  anything): residual e = root - y = [-1, -1, 0.5], MSE = 2.25 / 3 = 0.75,
  MAE = 2.5 / 3 = 5/6, RMSE = sqrt(0.75) = 0.8660254.

Gradients of that loss: root head weight 1.0 and bias -1.0 (2/3 * sum e * h and
2/3 * sum e); GO:1 head weight 0.4 and bias 0.15; GO:2 head weight 0.45 and bias 0.15
(each auxiliary head carries alpha / 2 = 0.15 of its MSE gradient); ``scale``
[1.0, 0.4, 0.45] (every feature and head weight is 1). None is zero, so the first Adam
step moves each parameter by exactly -lr * sign(grad) (m_hat / sqrt(v_hat) = g / |g|
on step one; eps 1e-8 and weight decay 1e-5 perturb it below 1e-7). After that step the
root prediction is 0.999 * 0.999 * [0, -1, 1] + 0.001 = [0.001, -0.997001, 0.999001].

Metric values from torchmetrics carry float32 error of about 1e-6, hence ``abs=1e-6``.
"""

import math
import tracemalloc
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
from torch_geometric.data import HeteroData

import torchcell.viz.fitness as viz_fitness
import torchcell.viz.genetic_interaction_score as viz_gi
from tests.torchcell.conftest import (
    DCellCountSubsystems,
    DCellIdentityHeads,
    make_dcell_regression_batch,
)
from torchcell.losses.dcell import DCellLoss
from torchcell.trainers.dcell_regression import DCellRegressionTask

Y = [1.0, 0.0, 0.5]
ROOT = [0.0, -1.0, 1.0]
LOSS = 1.225
ROOT_METRICS = {"MSE": 0.75, "MAE": 5 / 6, "RMSE": math.sqrt(0.75)}


def _models() -> dict[str, nn.Module]:
    return {"dcell": DCellCountSubsystems(), "dcell_linear": DCellIdentityHeads()}


def _task(**kwargs: Any) -> DCellRegressionTask:
    return DCellRegressionTask(_models(), target="fitness", **kwargs)


def _loader() -> DataLoader[HeteroData]:
    batches = cast("Dataset[HeteroData]", [make_dcell_regression_batch()])
    return DataLoader(batches, batch_size=None)


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
    """Stands in for ``fitness.box_plot``: records its inputs, returns a real figure."""

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


def test_init_registers_submodels_metrics_and_the_current_dcell_loss() -> None:
    """Submodels become child modules; three prefixed metric collections; loss alpha 0.3.

    Construction leaves process-wide allocation tracing alone: the memory HACK that
    called ``tracemalloc.start()`` in ``__init__`` is gone (issue #516).
    """
    assert tracemalloc.is_tracing() is False
    models = _models()
    task = DCellRegressionTask(models, target="fitness")
    assert dict(task.named_children())["dcell"] is models["dcell"]
    assert dict(task.named_children())["dcell_linear"] is models["dcell_linear"]
    assert task.automatic_optimization is False
    assert type(task.loss) is DCellLoss
    assert (task.loss.alpha, task.loss.use_auxiliary_losses) == (0.3, True)
    assert sorted(map(str, task.train_metrics.keys())) == [
        "train_MAE",
        "train_MSE",
        "train_RMSE",
    ]
    assert sorted(map(str, task.val_metrics.keys())) == [
        "val_MAE",
        "val_MSE",
        "val_RMSE",
    ]
    assert sorted(map(str, task.test_metrics.keys())) == [
        "test_MAE",
        "test_MSE",
        "test_RMSE",
    ]
    assert tracemalloc.is_tracing() is False


def test_configure_optimizers_is_adam_over_dcell_then_linear_parameters() -> None:
    """One Adam group: lr and weight decay as given, dcell parameters then the heads."""
    models = _models()
    task = DCellRegressionTask(
        models, target="fitness", learning_rate=3e-3, weight_decay=1e-4
    )
    optimizer = task.configure_optimizers()
    assert type(optimizer) is torch.optim.Adam
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (3e-3, 1e-4)
    expected = list(models["dcell"].parameters()) + list(
        models["dcell_linear"].parameters()
    )
    assert [id(p) for p in group["params"]] == [id(p) for p in expected]


def test_forward_feeds_dcell_outputs_through_the_linear_heads() -> None:
    """Root [0, -1, 1], GO:1 [0, 1, 2], GO:2 [0, 2, 1], each ``[3, 1]``, in that key order."""
    task = _task()
    out = task(make_dcell_regression_batch())
    assert list(out) == ["GO:ROOT", "GO:1", "GO:2"]
    assert {k: v.tolist() for k, v in out.items()} == {
        "GO:ROOT": [[0.0], [-1.0], [1.0]],
        "GO:1": [[0.0], [1.0], [2.0]],
        "GO:2": [[0.0], [2.0], [1.0]],
    }


@pytest.mark.parametrize(("auxiliary", "expected"), [(True, LOSS), (False, 0.75)])
def test_loss_feeds_the_root_as_prediction_and_every_head_as_auxiliary(
    auxiliary: bool, expected: float
) -> None:
    """``_loss`` calls ``DCellLoss(predictions, outputs, target)`` in that order.

    With auxiliary losses the value is 1.225 (root MSE 0.75 plus 0.3 times the mean of
    the GO:1 and GO:2 MSEs); without them it is the root MSE alone, 0.75, so the root
    is what lands in ``predictions`` (issue #516: the steps used to pass the deprecated
    ``(outputs, target, weights)`` order and raised on a parameter generator).
    """
    task = _task()
    task.loss.use_auxiliary_losses = auxiliary
    batch = make_dcell_regression_batch()
    loss = task._loss(task(batch), batch.fitness)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_one_training_step_logs_closed_form_values_and_takes_one_adam_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_loss 1.225, correlations -0.5 (subsystems) / +0.5 (root), root metrics.

    ``training_step`` returns the loss (seen by callbacks as ``{"loss": 1.225}``), the
    parameter count logged at train start is 3 + 3 * 2 = 9, and every parameter of both
    submodels moves by exactly -1e-3 * sign(grad) (signs from the module docstring).
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

    task = _task()
    before = {n: p.detach().clone() for n, p in task.named_parameters()}
    trainer = _trainer(tmp_path, fast_dev_run=True, callbacks=[_Outputs()])
    trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())

    assert step_outputs == [{"loss": pytest.approx(LOSS, abs=1e-6)}]
    metrics = {
        k: v.item()
        for k, v in trainer.callback_metrics.items()
        if not k.startswith("val_")
    }
    assert metrics == {
        "model/parameters_size": 9.0,
        "train_loss": pytest.approx(LOSS, abs=1e-6),
        "train_pearson_subsystems": pytest.approx(-0.5, abs=1e-6),
        "train_spearman_subsystems": pytest.approx(-0.5, abs=1e-6),
        "train_pearson_root": pytest.approx(0.5, abs=1e-6),
        "train_spearman_root": pytest.approx(0.5, abs=1e-6),
        "train_MSE": pytest.approx(ROOT_METRICS["MSE"], abs=1e-6),
        "train_MAE": pytest.approx(ROOT_METRICS["MAE"], abs=1e-6),
        "train_RMSE": pytest.approx(ROOT_METRICS["RMSE"], abs=1e-6),
    }
    delta = {
        n: (p.detach() - before[n]).flatten().tolist()
        for n, p in task.named_parameters()
    }
    lr = 1e-3
    assert delta == {
        "dcell.scale": [pytest.approx(-lr, abs=1e-7)] * 3,
        "dcell_linear.heads.0.weight": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.0.bias": [pytest.approx(lr, abs=1e-7)],
        "dcell_linear.heads.1.weight": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.1.bias": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.2.weight": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.2.bias": [pytest.approx(-lr, abs=1e-7)],
    }
    # epoch 0 is a plotting epoch, so the fit logs exactly one box plot
    assert [list(payload) for payload in logged] == [["binned_values_box_plot"]]


def test_validate_logs_root_suffixed_correlations_like_train_and_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation names the root correlations ``val_pearson_root``/``val_spearman_root``.

    The three stages now agree on the ``_root`` suffix (issue #516; no config or
    sweep in the repo reads the old ``val_pearson``/``val_spearman`` of this task).
    The box plot receives y and the root prediction, the stored buffers are emptied
    afterwards, and every step-level log call carries batch size 3
    (``batch.batch[-1] + 1``).
    """
    logged = _record_wandb(monkeypatch)
    box_plots = _BoxPlots()
    monkeypatch.setattr(viz_fitness, "box_plot", box_plots)
    task = _task()
    batch_sizes: dict[str, int | None] = {}
    original_log = task.log

    def _log(name: str, value: Any, **kwargs: Any) -> None:
        batch_sizes[name] = kwargs.get("batch_size")
        original_log(name, value, **kwargs)

    monkeypatch.setattr(task, "log", _log)
    results = _trainer(tmp_path).validate(task, dataloaders=_loader(), verbose=False)

    assert results == [
        {
            "val_loss": pytest.approx(LOSS, abs=1e-6),
            "val_pearson_subsystems": pytest.approx(-0.5, abs=1e-6),
            "val_spearman_subsystems": pytest.approx(-0.5, abs=1e-6),
            "val_pearson_root": pytest.approx(0.5, abs=1e-6),
            "val_spearman_root": pytest.approx(0.5, abs=1e-6),
            "val_MSE": pytest.approx(ROOT_METRICS["MSE"], abs=1e-6),
            "val_MAE": pytest.approx(ROOT_METRICS["MAE"], abs=1e-6),
            "val_RMSE": pytest.approx(ROOT_METRICS["RMSE"], abs=1e-6),
        }
    ]
    assert batch_sizes == {
        "val_loss": 3,
        "val_pearson_subsystems": 3,
        "val_spearman_subsystems": 3,
        "val_pearson_root": 3,
        "val_spearman_root": 3,
        # ``on_validation_epoch_end``'s ``log_dict`` passes no batch size
        "val_MSE": None,
        "val_MAE": None,
        "val_RMSE": None,
    }
    assert box_plots.calls == [(Y, ROOT)]
    assert [list(payload) for payload in logged] == [["binned_values_box_plot"]]
    assert (task.true_values.tolist(), task.predictions.tolist()) == ([], [])


def test_test_logs_root_suffixed_correlations_and_pooled_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``trainer.test`` returns the eight values of the module docstring, root suffixed.

    RMSE/MSE/MAE score the root alone (issue #516: they used to pool the subsystem mean
    and the root, six predictions per batch of three).
    """
    logged = _record_wandb(monkeypatch)
    results = _trainer(tmp_path).test(_task(), dataloaders=_loader(), verbose=False)
    assert results == [
        {
            "test_loss": pytest.approx(LOSS, abs=1e-6),
            "test_pearson_subsystems": pytest.approx(-0.5, abs=1e-6),
            "test_spearman_subsystems": pytest.approx(-0.5, abs=1e-6),
            "test_pearson_root": pytest.approx(0.5, abs=1e-6),
            "test_spearman_root": pytest.approx(0.5, abs=1e-6),
            "test_MSE": pytest.approx(ROOT_METRICS["MSE"], abs=1e-6),
            "test_MAE": pytest.approx(ROOT_METRICS["MAE"], abs=1e-6),
            "test_RMSE": pytest.approx(ROOT_METRICS["RMSE"], abs=1e-6),
        }
    ]
    assert logged == []


def test_genetic_interaction_target_uses_its_own_box_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``target="genetic_interaction_score"`` routes to that module's box plot only.

    The target name does not change what is regressed: the step still reads
    ``batch.fitness``, so the plot sees y = [1.0, 0.0, 0.5].
    """
    _record_wandb(monkeypatch)
    fitness_plots, gi_plots = _BoxPlots(), _BoxPlots()
    monkeypatch.setattr(viz_fitness, "box_plot", fitness_plots)
    monkeypatch.setattr(viz_gi, "box_plot", gi_plots)
    task = DCellRegressionTask(_models(), target="genetic_interaction_score")
    _trainer(tmp_path).validate(task, dataloaders=_loader(), verbose=False)
    assert fitness_plots.calls == []
    assert gi_plots.calls == [(Y, ROOT)]
    assert (task.true_values.tolist(), task.predictions.tolist()) == ([], [])


def test_unknown_target_is_rejected_at_construction() -> None:
    """A target without a box plot raises ``ValueError`` naming it, in ``__init__``.

    Issue #516: the ``if``/``elif`` on the target had no ``else``, so the first plotting
    validation epoch raised ``UnboundLocalError`` on ``fig``.
    """
    with pytest.raises(ValueError) as excinfo:
        DCellRegressionTask(_models(), target="growth_rate")
    assert str(excinfo.value) == (
        "Unknown target 'growth_rate': expected one of "
        "('fitness', 'genetic_interaction_score')."
    )


def test_each_epoch_best_checkpoint_is_logged_once_at_its_own_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two epochs log ``epoch=0-step=1`` as step 1 and ``epoch=1-step=2`` as step 2.

    ``ModelCheckpoint`` saves in ``on_train_epoch_end`` after the module hooks, so the
    task logs from the next ``on_train_epoch_start`` and from ``on_train_end``; one
    manual-optimization batch per epoch makes the global step 1 after epoch 0 and 2
    after epoch 1, and val_loss falls each step, so each epoch's checkpoint is the best
    (issue #516: the artifact used to hold the previous epoch's checkpoint).
    ``metadata`` is ``dict(self.hparams)``, empty because the task never calls
    ``save_hyperparameters``. A further call at the same global step is skipped by
    ``last_logged_best_step``.
    """
    _record_wandb(monkeypatch)
    monkeypatch.setattr(viz_fitness, "box_plot", _BoxPlots())
    records: list[dict[str, Any]] = []

    class _Artifact:
        def __init__(self, **kwargs: Any) -> None:
            self.record = dict(kwargs, files=[])

        def add_file(self, path: str) -> None:
            self.record["files"].append(Path(path).name)

    monkeypatch.setattr(wandb, "Artifact", _Artifact)
    monkeypatch.setattr(wandb, "log_artifact", lambda a: records.append(a.record))
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "ckpt", monitor="val_loss")
    task = _task()
    trainer = _trainer(
        tmp_path,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint],
    )
    trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())
    task._log_best_checkpoint()
    assert records == [
        {
            "name": f"model-global_step-{step}",
            "type": "model",
            "description": f"Best model checkpoint at step - {step}",
            "metadata": {},
            "files": [f"epoch={step - 1}-step={step}.ckpt"],
        }
        for step in (1, 2)
    ]
    assert task.last_logged_best_step == 2


def test_training_without_a_checkpoint_callback_logs_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enable_checkpointing=False`` trains and validates cleanly with no artifact.

    Issue #516: the artifact branch read ``None.best_model_path`` and raised.
    """
    _record_wandb(monkeypatch)
    monkeypatch.setattr(viz_fitness, "box_plot", _BoxPlots())
    logged_artifacts: list[Any] = []
    monkeypatch.setattr(wandb, "log_artifact", logged_artifacts.append)
    task = _task()
    trainer = _trainer(
        tmp_path,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())
    assert trainer.current_epoch == 2
    assert logged_artifacts == []
    assert task.last_logged_best_step is None


def test_sanity_check_predictions_stay_out_of_the_first_box_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The epoch-0 box plot holds only the real validation pass, after one Adam step.

    Issue #516: the sanity pass used to be collected and plotted with the real one (six
    values). Now ``validation_step`` skips collection during the sanity check, so the
    plot sees y and the post-step root [0.001, -0.997001, 0.999001] (module docstring).
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
    trainer.fit(_task(), train_dataloaders=_loader(), val_dataloaders=_loader())
    expected_calls: list[Any] = [
        (Y, [pytest.approx(v, abs=1e-6) for v in (0.001, -0.997001, 0.999001)])
    ]
    assert box_plots.calls == expected_calls
