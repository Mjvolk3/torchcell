# tests/torchcell/trainers/test_dcell_regression_slim.py
# [[tests.torchcell.trainers.test_dcell_regression_slim]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_dcell_regression_slim.py
"""``DCellRegressionSlimTask`` on CPU with the weight-free DCell stand-in of the conftest.

Fixture and loss as in ``test_dcell_regression.py``: ``make_dcell_regression_batch()``
gives ``GO:ROOT`` = [0, -1, 1], ``GO:1`` = [0, 1, 2], ``GO:2`` = [0, 2, 1] against
y = [1.0, 0.0, 0.5]. The task calls the ``DCellLoss(predictions, outputs, target)`` it
constructs with the root as predictions and every head as ``linear_outputs``, so at alpha
0.3 the loss is root MSE plus alpha times the SUM of the auxiliary MSEs (Ma et al. 2018,
issue #554): loss = 0.75 + 0.3 * (4.25 + 5.25) / 3 = 0.75 + 0.95 = 1.7. The pre-fix mean
gave 0.75 + 0.3 * ((4.25 + 5.25) / 3) / 2 = 1.225.

Unlike the full task, the slim task keeps separate collections for the subsystem mean
m = [0, 2/3, 4/3] and for the root, each with Pearson and Spearman inside:

* subsystem: MSE = (1 + 4/9 + 25/36) / 3 = 77/108 = 0.7129630, RMSE = 0.8443713,
  MAE = (1 + 2/3 + 5/6) / 3 = 5/6, Pearson = Spearman = -0.5;
* root: MSE = (1 + 1 + 1/4) / 3 = 0.75, RMSE = sqrt(0.75) = 0.8660254,
  MAE = (1 + 1 + 1/2) / 3 = 5/6, Pearson = Spearman = +0.5.

Pearson: m deviates [-2/3, 0, 2/3] and root [0, -1, 1] from their means, y deviates
[0.5, -0.5, 0]; cov/norms give -1/3 / (2/3) and 1/2 / 1. Spearman equals Pearson on
these ranks. Metric values from torchmetrics carry float32 error of about 1e-6, hence
``abs=1e-6``.
"""

import math
from pathlib import Path
from typing import Any, cast

import lightning as L
import pytest
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData

from tests.torchcell.conftest import (
    DCellCountSubsystems,
    DCellIdentityHeads,
    make_dcell_regression_batch,
)
from torchcell.losses.dcell import DCellLoss
from torchcell.trainers.dcell_regression_slim import DCellRegressionSlimTask

LOSS = 1.7
SUBSYSTEM = {
    "MSE": 77 / 108,
    "RMSE": math.sqrt(77 / 108),
    "MAE": 5 / 6,
    "Pearson": -0.5,
    "Spearman": -0.5,
}
ROOT = {
    "MSE": 0.75,
    "RMSE": math.sqrt(0.75),
    "MAE": 5 / 6,
    "Pearson": 0.5,
    "Spearman": 0.5,
}


def _expected(prefix: str, values: dict[str, float]) -> dict[str, Any]:
    return {f"{prefix}{k}": pytest.approx(v, abs=1e-6) for k, v in values.items()}


def _models() -> dict[str, nn.Module]:
    return {"dcell": DCellCountSubsystems(), "dcell_linear": DCellIdentityHeads()}


def _task(**kwargs: Any) -> DCellRegressionSlimTask:
    return DCellRegressionSlimTask(
        _models(), target="fitness", aux_reduction="sum", **kwargs
    )


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


def _record_wandb(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "log", lambda payload: logged.append(payload))
    return logged


def test_init_passes_alpha_to_the_loss_and_builds_six_metric_collections() -> None:
    """``alpha`` and ``aux_reduction`` reach ``DCellLoss``; collections per split."""
    models = _models()
    task = DCellRegressionSlimTask(
        models, target="fitness", alpha=0.7, aux_reduction="mean"
    )
    assert dict(task.named_children())["dcell"] is models["dcell"]
    assert dict(task.named_children())["dcell_linear"] is models["dcell_linear"]
    assert task.automatic_optimization is False
    assert type(task.loss) is DCellLoss
    assert (task.loss.alpha, task.loss.aux_reduction) == (0.7, "mean")
    names = ["MAE", "MSE", "Pearson", "RMSE", "Spearman"]
    collections = {
        "train_": task.train_metrics,
        "val_": task.val_metrics,
        "test_": task.test_metrics,
        "train_root_": task.train_metrics_root,
        "val_root_": task.val_metrics_root,
        "test_root_": task.test_metrics_root,
    }
    assert {
        prefix: sorted(map(str, c.keys())) for prefix, c in collections.items()
    } == {prefix: [prefix + n for n in names] for prefix in collections}


def test_configure_optimizers_is_adam_over_dcell_then_linear_parameters() -> None:
    """One Adam group: lr and weight decay as given, dcell parameters then the heads."""
    models = _models()
    task = DCellRegressionSlimTask(
        models,
        target="fitness",
        learning_rate=3e-3,
        weight_decay=1e-4,
        aux_reduction="sum",
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


@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.3, LOSS), (0.7, 0.75 + 0.7 * 19 / 6)]
)
def test_loss_feeds_the_root_as_prediction_and_every_head_as_auxiliary(
    alpha: float, expected: float
) -> None:
    """``_loss`` calls ``DCellLoss(predictions, outputs, target)`` in that order.

    The summed auxiliary MSE is (4.25 + 5.25) / 3 = 19/6, so alpha 0.3 gives
    0.75 + 0.95 = 1.7 and alpha 0.7 gives 0.75 + 0.7 * 19/6 = 2.9666667 (the pre-fix mean,
    19/12, gave 1.225 and 1.8583333); ``alpha`` reaches the loss the steps
    call (issue #516: they passed the deprecated ``(outputs, target, weights)`` order).
    """
    task = _task(alpha=alpha)
    batch = make_dcell_regression_batch()
    loss = task._loss(task(batch), batch.fitness)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_one_training_step_logs_subsystem_and_root_metrics_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_loss 1.7, ``train_*`` from m, ``train_root_*`` from the root, 9 parameters.

    Gradients (root head weight 1.0, bias -1.0; GO:1 head 0.8, 0.3; GO:2 head 0.9,
    0.3; ``scale`` [1.0, 0.8, 0.9]) are all nonzero, so the one Adam step moves every
    parameter by exactly -1e-3 * sign(grad).

    The slim task never calls ``wandb.log`` (it has no box plot), and with no best
    checkpoint the artifact branch stays inert, so the recorder stays empty.
    """
    logged = _record_wandb(monkeypatch)
    task = _task()
    before = {n: p.detach().clone() for n, p in task.named_parameters()}
    trainer = _trainer(tmp_path, fast_dev_run=True)
    trainer.fit(task, train_dataloaders=_loader(), val_dataloaders=_loader())
    metrics = {
        k: v.item()
        for k, v in trainer.callback_metrics.items()
        if not k.startswith("val_")
    }
    assert metrics == {
        "model/parameters_size": 9.0,
        "train_loss": pytest.approx(LOSS, abs=1e-6),
        **_expected("train_", SUBSYSTEM),
        **_expected("train_root_", ROOT),
    }
    # first Adam step: -lr * sign(grad); only the root bias has a negative gradient
    lr = 1e-3
    delta = {
        n: (p.detach() - before[n]).flatten().tolist()
        for n, p in task.named_parameters()
    }
    assert delta == {
        "dcell.scale": [pytest.approx(-lr, abs=1e-7)] * 3,
        "dcell_linear.heads.0.weight": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.0.bias": [pytest.approx(lr, abs=1e-7)],
        "dcell_linear.heads.1.weight": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.1.bias": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.2.weight": [pytest.approx(-lr, abs=1e-7)],
        "dcell_linear.heads.2.bias": [pytest.approx(-lr, abs=1e-7)],
    }
    assert logged == []


def test_validate_logs_subsystem_and_root_metrics(tmp_path: Path) -> None:
    """``trainer.validate`` returns val_loss plus both five-metric collections."""
    results = _trainer(tmp_path).validate(_task(), dataloaders=_loader(), verbose=False)
    assert results == [
        {
            "val_loss": pytest.approx(LOSS, abs=1e-6),
            **_expected("val_", SUBSYSTEM),
            **_expected("val_root_", ROOT),
        }
    ]


def test_test_epoch_end_logs_and_resets_the_root_metrics(tmp_path: Path) -> None:
    """``on_test_epoch_end`` logs ``test_root_*`` beside ``test_*`` and resets both.

    Issue #516: it used to log and reset only ``test_metrics``, so ``test_root_*`` never
    appeared and the root state carried over into the next test run; after the epoch
    end the root collection holds no updates.
    """
    task = _task()
    trainer = _trainer(tmp_path)
    expected = [
        {
            "test_loss": pytest.approx(LOSS, abs=1e-6),
            **_expected("test_", SUBSYSTEM),
            **_expected("test_root_", ROOT),
        }
    ]
    assert trainer.test(task, dataloaders=_loader(), verbose=False) == expected
    assert task.test_metrics_root["MSE"].update_called is False


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
