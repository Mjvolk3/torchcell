# tests/torchcell/trainers/test_dcell_regression_slim.py
# [[tests.torchcell.trainers.test_dcell_regression_slim]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_dcell_regression_slim.py
"""``DCellRegressionSlimTask`` on CPU with the weight-free DCell stand-in of the conftest.

Fixture and loss as in ``test_dcell_regression.py``: ``make_dcell_regression_batch()``
gives ``GO:ROOT`` = [0, -1, 1], ``GO:1`` = [0, 1, 2], ``GO:2`` = [0, 2, 1] against
y = [1.0, 0.0, 0.5]. The slim task has the same call-order defect with the current
``DCellLoss`` (Finding 1), so the value tests swap in the deprecated loss at alpha 0.3:
loss = 0.75 + 0.3 * (4.25 + 5.25) / 3 = 1.7.

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
from torchcell.losses.dcell_DEPRECATED import DCellLoss as DeprecatedDCellLoss
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


def _task() -> DCellRegressionSlimTask:
    task = DCellRegressionSlimTask(_models(), target="fitness")
    task.register_module("loss", DeprecatedDCellLoss(alpha=0.3))
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


def _record_wandb(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "log", lambda payload: logged.append(payload))
    return logged


def test_init_passes_alpha_to_the_loss_and_builds_six_metric_collections() -> None:
    """``alpha`` reaches ``DCellLoss``; subsystem and root collections per split."""
    models = _models()
    task = DCellRegressionSlimTask(models, target="fitness", alpha=0.7)
    assert dict(task.named_children())["dcell"] is models["dcell"]
    assert dict(task.named_children())["dcell_linear"] is models["dcell_linear"]
    assert task.automatic_optimization is False
    assert type(task.loss) is DCellLoss
    assert task.loss.alpha == 0.7
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


def test_the_constructed_loss_rejects_the_trainer_argument_order(
    tmp_path: Path,
) -> None:
    """Finding: every step raises with the loss ``__init__`` builds (lines 158, 188, 227).

    ``self.loss(y_hat, y, dcell.parameters())`` binds a parameter generator to
    ``DCellLoss.forward``'s ``target``. Pinned until the call matches the signature.
    """
    task = DCellRegressionSlimTask(_models(), target="fitness")
    with pytest.raises(
        AttributeError, match="^'generator' object has no attribute 'size'$"
    ):
        _trainer(tmp_path).test(task, dataloaders=_loader(), verbose=False)


def test_one_training_step_logs_subsystem_and_root_metrics_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_loss 1.7, ``train_*`` from m, ``train_root_*`` from the root, 9 parameters.

    Gradients (root head weight 1.0, bias -1.0; GO:1 head 0.8, 0.3; GO:2 head 0.9, 0.3;
    ``scale`` [1.0, 0.8, 0.9]) are all nonzero, so the one Adam step moves every
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


def test_test_epoch_end_drops_the_root_metrics(tmp_path: Path) -> None:
    """Finding: ``on_test_epoch_end`` logs and resets only ``test_metrics`` (line 240).

    ``test_step`` updates ``test_metrics_root`` (line 236) but nothing computes, logs or
    resets it, so ``test_root_*`` never appear and the root state carries over into the
    next test run. Pinned until the epoch end mirrors train and validation.
    """
    task = _task()
    results = _trainer(tmp_path).test(task, dataloaders=_loader(), verbose=False)
    assert results == [
        {"test_loss": pytest.approx(LOSS, abs=1e-6), **_expected("test_", SUBSYSTEM)}
    ]
    carried = {k: v.item() for k, v in task.test_metrics_root.compute().items()}
    assert carried == _expected("test_root_", ROOT)


def test_validation_epoch_end_requires_a_checkpoint_callback(tmp_path: Path) -> None:
    """Finding: with checkpointing disabled the epoch end reads ``None.best_model_path``.

    ``on_validation_epoch_end`` casts ``trainer.checkpoint_callback`` to
    ``ModelCheckpoint`` (line 208) without checking it exists, so
    ``enable_checkpointing=False`` makes every validation epoch raise. Pinned until the
    artifact branch tolerates a missing callback.
    """
    trainer = _trainer(tmp_path, enable_checkpointing=False)
    with pytest.raises(
        AttributeError, match="^'NoneType' object has no attribute 'best_model_path'$"
    ):
        trainer.validate(_task(), dataloaders=_loader(), verbose=False)


def test_best_checkpoint_is_logged_once_per_global_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A best checkpoint path produces one ``model-global_step-0`` artifact, not two."""
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
