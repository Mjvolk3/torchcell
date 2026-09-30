# tests/torchcell/trainers/test_dcell_regression.py
# [[tests.torchcell.trainers.test_dcell_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_dcell_regression.py
"""``DCellRegressionTask`` on CPU with a weight-free DCell stand-in over the conftest GO.

Fixture (``tests/torchcell/conftest.py``): ``make_dcell_regression_batch()`` is three
samples of the three-term DCell hierarchy with knockouts {0, 1, 2, 3}, {0} and {2};
``DCellCountSubsystems`` + ``DCellIdentityHeads`` predict intact-gene counts, so
``GO:1`` = [0, 1, 2], ``GO:2`` = [0, 2, 1], ``GO:ROOT`` = ``GO:1 - GO:2`` = [0, -1, 1],
against ``fitness`` y = [1.0, 0.0, 0.5].

The trainer calls ``self.loss(y_hat, y, dcell.parameters())``, the argument order of the
deprecated ``torchcell.losses.dcell_DEPRECATED.DCellLoss(outputs, target, weights)``.
The loss it constructs is the current ``torchcell.losses.DCellLoss(predictions,
outputs, target)``, so every step raises (Finding 1). The remaining tests swap in the
deprecated loss the trainer was written against, which gives

* loss = MSE(root) + 0.3 * (MSE(GO:1) + MSE(GO:2))
  = (1 + 1 + 0.25) / 3 + 0.3 * ((1 + 1 + 2.25) / 3 + (1 + 4 + 0.25) / 3)
  = 0.75 + 0.3 * (4.25 + 5.25) / 3 = 0.75 + 0.95 = 1.7;
* subsystem mean m = mean over the three terms = [0, 2/3, 4/3];
* Pearson(m, y) = -0.5 and Pearson(root, y) = +0.5 (deviations [-2/3, 0, 2/3] and
  [0, -1, 1] against [0.5, -0.5, 0]: cov -1/3 over 2/3, cov 1/2 over 1); Spearman
  equals Pearson here (ranks of m are 1, 2, 3; of root 2, 1, 3; of y 3, 1, 2);
* the RMSE/MSE/MAE collection is updated twice per step, with m and then with root, so
  it pools six predictions (Finding 2): MSE = (1 + 4/9 + 25/36 + 1 + 1 + 1/4) / 6
  = 158/216 = 0.7314815, MAE = (1 + 2/3 + 5/6 + 1 + 1 + 1/2) / 6 = 5/6,
  RMSE = sqrt(158/216) = 0.8552669.

Gradients of that loss: root head weight 1.0 and bias -1.0 (2/3 * sum e * h and
2/3 * sum e for residual e = [-1, -1, 0.5]); GO:1 head weight 0.8 and bias 0.3; GO:2 head
weight 0.9 and bias 0.3; ``scale`` [1.0, 0.8, 0.9] (every feature and head weight is 1).
None is zero, so the first Adam step moves each parameter by exactly -lr * sign(grad)
(m_hat / sqrt(v_hat) = g / |g| on step one; eps 1e-8 and weight decay 1e-5 perturb it
below 1e-7).

Metric values from torchmetrics carry float32 error of about 1e-6, hence ``abs=1e-6``.
"""

import math
import tracemalloc
from collections.abc import Iterator
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
from torchcell.losses.dcell_DEPRECATED import DCellLoss as DeprecatedDCellLoss
from torchcell.trainers.dcell_regression import DCellRegressionTask

Y = [1.0, 0.0, 0.5]
SUBSYSTEM_MEAN = [0.0, 2 / 3, 4 / 3]
LOSS = 1.7
POOLED = {"MSE": 158 / 216, "MAE": 5 / 6, "RMSE": math.sqrt(158 / 216)}


@pytest.fixture(autouse=True)
def _stop_tracemalloc() -> Iterator[None]:
    """Every construction starts ``tracemalloc``; never let it leak into later tests."""
    yield
    if tracemalloc.is_tracing():
        tracemalloc.stop()


def _models() -> dict[str, nn.Module]:
    return {"dcell": DCellCountSubsystems(), "dcell_linear": DCellIdentityHeads()}


def _task(**kwargs: Any) -> DCellRegressionTask:
    task = DCellRegressionTask(_models(), target="fitness", **kwargs)
    task.register_module("loss", DeprecatedDCellLoss())
    return task


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

    Finding: ``__init__`` calls ``tracemalloc.start()`` process-wide (line 106), so
    constructing the task turns on allocation tracing for the whole interpreter until a
    plotting validation epoch stops it. Pinned until the HACK is removed.
    """
    tracemalloc.stop()
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
    assert tracemalloc.is_tracing() is True


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


def test_the_constructed_loss_rejects_the_trainer_argument_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: every step raises with the loss ``__init__`` builds (lines 148, 200, 290).

    ``self.loss(y_hat, y, dcell.parameters())`` lands in ``DCellLoss.forward(predictions,
    outputs, target)`` as predictions = the output dict and target = a parameter
    generator, and ``nn.MSELoss`` fails reading ``target.size()``. Pinned until the
    trainer calls the current loss signature.
    """
    logged = _record_wandb(monkeypatch)
    task = DCellRegressionTask(_models(), target="fitness")
    trainer = _trainer(tmp_path, fast_dev_run=True)
    with pytest.raises(
        AttributeError, match="^'generator' object has no attribute 'size'$"
    ):
        trainer.fit(task, train_dataloaders=_loader())
    with pytest.raises(
        AttributeError, match="^'generator' object has no attribute 'size'$"
    ):
        trainer.validate(task, dataloaders=_loader())
    assert logged == []


def test_one_training_step_logs_closed_form_values_and_takes_one_adam_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_loss 1.7, correlations -0.5 (subsystems) / +0.5 (root), pooled metrics.

    ``training_step`` returns the loss (seen by callbacks as ``{"loss": 1.7}``), the
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
        "train_MSE": pytest.approx(POOLED["MSE"], abs=1e-6),
        "train_MAE": pytest.approx(POOLED["MAE"], abs=1e-6),
        "train_RMSE": pytest.approx(POOLED["RMSE"], abs=1e-6),
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


def test_validate_logs_root_correlations_without_the_root_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validation logs ``val_pearson``/``val_spearman`` for the root, unlike train and test.

    Finding: ``training_step`` and ``test_step`` name the root correlations
    ``*_pearson_root``/``*_spearman_root`` but ``validation_step`` logs them as
    ``val_pearson``/``val_spearman`` (lines 224, 230). Pinned until the names agree.

    The box plot receives y and the subsystem mean m (not the root), the stored buffers
    are emptied afterwards, the memory HACK prints its banner and stops ``tracemalloc``,
    and every step-level log call carries batch size 3 (``batch.batch[-1] + 1``).
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
    capsys.readouterr()
    results = _trainer(tmp_path).validate(task, dataloaders=_loader(), verbose=False)

    assert results == [
        {
            "val_loss": pytest.approx(LOSS, abs=1e-6),
            "val_pearson_subsystems": pytest.approx(-0.5, abs=1e-6),
            "val_spearman_subsystems": pytest.approx(-0.5, abs=1e-6),
            "val_pearson": pytest.approx(0.5, abs=1e-6),
            "val_spearman": pytest.approx(0.5, abs=1e-6),
            "val_MSE": pytest.approx(POOLED["MSE"], abs=1e-6),
            "val_MAE": pytest.approx(POOLED["MAE"], abs=1e-6),
            "val_RMSE": pytest.approx(POOLED["RMSE"], abs=1e-6),
        }
    ]
    assert batch_sizes == {
        "val_loss": 3,
        "val_pearson_subsystems": 3,
        "val_spearman_subsystems": 3,
        "val_pearson": 3,
        "val_spearman": 3,
        # ``on_validation_epoch_end``'s ``log_dict`` passes no batch size
        "val_MSE": None,
        "val_MAE": None,
        "val_RMSE": None,
    }
    expected_calls: list[Any] = [
        (Y, [pytest.approx(v, abs=1e-6) for v in SUBSYSTEM_MEAN])
    ]
    assert box_plots.calls == expected_calls
    assert [list(payload) for payload in logged] == [["binned_values_box_plot"]]
    assert (task.true_values.tolist(), task.predictions.tolist()) == ([], [])
    assert tracemalloc.is_tracing() is False
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "======"
    assert printed[1].startswith("Current memory usage is ")
    assert printed[2] == "======"


def test_test_logs_root_suffixed_correlations_and_pooled_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``trainer.test`` returns the eight values of the module docstring, root suffixed."""
    logged = _record_wandb(monkeypatch)
    results = _trainer(tmp_path).test(_task(), dataloaders=_loader(), verbose=False)
    assert results == [
        {
            "test_loss": pytest.approx(LOSS, abs=1e-6),
            "test_pearson_subsystems": pytest.approx(-0.5, abs=1e-6),
            "test_spearman_subsystems": pytest.approx(-0.5, abs=1e-6),
            "test_pearson_root": pytest.approx(0.5, abs=1e-6),
            "test_spearman_root": pytest.approx(0.5, abs=1e-6),
            "test_MSE": pytest.approx(POOLED["MSE"], abs=1e-6),
            "test_MAE": pytest.approx(POOLED["MAE"], abs=1e-6),
            "test_RMSE": pytest.approx(POOLED["RMSE"], abs=1e-6),
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
    task.register_module("loss", DeprecatedDCellLoss())
    _trainer(tmp_path).validate(task, dataloaders=_loader(), verbose=False)
    assert fitness_plots.calls == []
    expected_calls: list[Any] = [
        (Y, [pytest.approx(v, abs=1e-6) for v in SUBSYSTEM_MEAN])
    ]
    assert gi_plots.calls == expected_calls
    assert (task.true_values.tolist(), task.predictions.tolist()) == ([], [])


def test_unknown_target_leaves_the_figure_unbound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: any other target reaches ``wandb.Image(fig)`` with ``fig`` unassigned.

    The ``if``/``elif`` on the target (lines 251 to 254) has no ``else``, so the first
    plotting validation epoch raises ``UnboundLocalError`` instead of rejecting the
    target at construction. Pinned until the target is validated in ``__init__``.
    """
    logged = _record_wandb(monkeypatch)
    task = DCellRegressionTask(_models(), target="growth_rate")
    task.register_module("loss", DeprecatedDCellLoss())
    with pytest.raises(UnboundLocalError, match="'fig'"):
        _trainer(tmp_path).validate(task, dataloaders=_loader(), verbose=False)
    assert logged == []


def test_best_checkpoint_is_logged_once_per_global_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A best checkpoint path produces one ``model-global_step-0`` artifact, not two.

    ``metadata`` is ``dict(self.hparams)``, which is empty because the task never calls
    ``save_hyperparameters``. A second validation at the same global step is skipped by
    ``last_logged_best_step``.
    """
    _record_wandb(monkeypatch)
    monkeypatch.setattr(viz_fitness, "box_plot", _BoxPlots())
    artifacts: list[dict[str, Any]] = []
    logged_artifacts: list[dict[str, Any]] = []

    class _Artifact:
        def __init__(self, **kwargs: Any) -> None:
            self.record = dict(kwargs, files=[])
            artifacts.append(self.record)

        def add_file(self, path: str) -> None:
            self.record["files"].append(path)

    monkeypatch.setattr(wandb, "Artifact", _Artifact)
    monkeypatch.setattr(
        wandb, "log_artifact", lambda a: logged_artifacts.append(a.record)
    )
    best = tmp_path / "best.ckpt"
    best.write_bytes(b"")
    checkpoint = ModelCheckpoint(dirpath=tmp_path)
    checkpoint.best_model_path = str(best)
    task = _task()
    trainer = _trainer(tmp_path, callbacks=[checkpoint])
    trainer.validate(task, dataloaders=_loader(), verbose=False)
    trainer.validate(task, dataloaders=_loader(), verbose=False)
    expected = {
        "name": "model-global_step-0",
        "type": "model",
        "description": "Model on validation epoch end step - 0",
        "metadata": {},
        "files": [str(best)],
    }
    assert artifacts == [expected]
    assert logged_artifacts == [expected]
    assert task.last_logged_best_step == 0


def test_sanity_check_predictions_leak_into_the_first_box_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the sanity-check early return (line 246) skips clearing the buffers.

    With one sanity batch and one real validation batch, the epoch-0 box plot receives
    six values: the sanity pass (pre-step, so exactly m) followed by the real pass.
    Pinned until the buffers are cleared on every return path.
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
    )
    trainer.fit(_task(), train_dataloaders=_loader(), val_dataloaders=_loader())
    ((true_values, predictions),) = box_plots.calls
    assert true_values == Y + Y
    assert predictions[:3] == [pytest.approx(v, abs=1e-6) for v in SUBSYSTEM_MEAN]
    assert len(predictions) == 6
