# tests/torchcell/trainers/test_int_dcell.py
# [[tests.torchcell.trainers.test_int_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_dcell.py
"""``int_dcell.RegressionTask._shared_step`` hands ``DCellLoss`` matching ``[B]`` shapes.

The task reshapes the root prediction and the target to ``[B, 1]`` for its metrics and
plots, while the model's heads are ``[B]``. Before issue #554 it passed those ``[B, 1]``
tensors to the loss, so every auxiliary MSE broadcast over a ``[B, B]`` grid (and the
root head was counted, issue #578). The loss now refuses mismatched shapes and the task
passes ``[B]`` tensors.

Fixture: a weight-free stand-in whose heads are a scalar parameter w = 1 times fixed
values: root GO:0 = [0, -1, 1], GO:1 = [0, 1, 2], GO:2 = [0, 2, 1]; target
y = [1, 0, 0.5]. Paired MSEs: root (1 + 1 + 0.25) / 3 = 0.75, GO:1 (1 + 1 + 2.25) / 3 =
4.25 / 3, GO:2 (1 + 4 + 0.25) / 3 = 5.25 / 3. Hence

* ``"sum"``: 0.75 + 0.3 * 9.5 / 3 = 0.75 + 0.95 = 1.7;
* ``"mean"``: 0.75 + 0.3 * 9.5 / 6 = 0.75 + 0.475 = 1.225.

The old path gave neither value: each broadcast term is the mean of (o_j - y_i)^2 over
all nine pairs, var(o) + var(y) + (mean o - mean y)^2 = 2/3 + 1/6 + 0.25 = 1.0833333
for every head here (all three are permutations of {0, 1, 2} shifted alike), and with
GO:0 counted the old "mean" loss was 0.75 + 0.3 * 1.0833333 = 1.075.
"""

import logging
import math
import re
from pathlib import Path
from typing import Any, Literal

import lightning as L
import pytest
import torch
from lightning.pytorch.core.optimizer import _init_optimizers_and_lr_schedulers
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torchmetrics import Metric, MetricCollection

from tests.torchcell.conftest import make_dcell_batch, make_dcell_graph
from torchcell.losses.dcell import DCellLoss
from torchcell.models.dcell import DCell
from torchcell.models.dcell_opt import DCellOpt
from torchcell.trainers.int_dcell import RegressionTask

ROOT = [0.0, -1.0, 1.0]
GO1 = [0.0, 1.0, 2.0]
GO2 = [0.0, 2.0, 1.0]
Y = [1.0, 0.0, 0.5]


class _FixedHeads(nn.Module):
    """``DCell``-shaped outputs: ``[B]`` heads, the root aliased and declared."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        root = self.w * torch.tensor(ROOT)
        heads = {
            "GO:0": root,
            "GO:1": self.w * torch.tensor(GO1),
            "GO:2": self.w * torch.tensor(GO2),
            "GO:ROOT": root,
        }
        return root, {"linear_outputs": heads, "root_key": "GO:0"}


def _batch() -> HeteroData:
    batch = HeteroData()
    batch["gene"].phenotype_values = torch.tensor(Y)
    return batch


@pytest.mark.parametrize(("reduction", "loss"), [("sum", 1.7), ("mean", 1.225)])
def test_shared_step_loss_is_the_paired_closed_form(
    reduction: Literal["sum", "mean"], loss: float
) -> None:
    """``_shared_step`` returns 1.7 under "sum" and 1.225 under "mean", with ``[B, 1]``
    predictions and targets returned for the metrics and plots.
    """
    task = RegressionTask(
        model=_FixedHeads(),
        cell_graph=HeteroData(),
        optimizer_config={"type": "AdamW", "lr": 1e-3},
        lr_scheduler_config={},
        loss_func=DCellLoss(alpha=0.3, aux_reduction=reduction),
        device="cpu",
    )
    total, predictions, target = task._shared_step(_batch(), 0, "val")
    assert total.item() == pytest.approx(loss, abs=1e-6)
    assert predictions.tolist() == [[v] for v in ROOT]
    assert target.tolist() == [[v] for v in Y]


# ---------------------------------------------------------------------------
# 2026.10.01 (phase 19): the rest of ``RegressionTask``.
#
# The same ``_FixedHeads`` stand-in (w = 1) and target y = [1, 0, 0.5]. Under
# ``DCellLoss(alpha=0.3, aux_reduction="sum")``: primary 0.75, auxiliary
# (4.25 + 5.25) / 3 = 3.1666667, weighted 0.3 * 3.1666667 = 0.95, total 1.7.
#
# Gradient of the total in w at w = 1 (the loss is quadratic in w; o = w * values):
# dL/dw = 2/3 * sum_i (o_i - y_i) * v_i per head. Root: residuals [-1, -1, 0.5] times
# [0, -1, 1] sum to 1.5, so 1.0. GO:1: [-1, 1, 1.5] . [0, 1, 2] = 4; GO:2:
# [-1, 2, 0.5] . [0, 2, 1] = 4.5; the auxiliary part is 0.3 * 2/3 * 8.5 = 1.7. Total 2.7.
#
# Root metrics on [0, -1, 1] vs [1, 0, 0.5]: MSE 0.75, RMSE sqrt(0.75) = 0.8660254;
# Pearson: centered r = [0, -1, 1], centered y = [0.5, -0.5, 0]; cov sum 0.5,
# sum r^2 = 2, sum y^2 = 0.5, so 0.5 / sqrt(2 * 0.5) = 0.5.
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]
SUM_LOGS = [
    ("primary_loss", 0.75),
    ("auxiliary_loss", 9.5 / 3),
    ("weighted_auxiliary_loss", 0.95),
]


class _Log:
    """Stands in for ``LightningModule.log``: (name, float value, kwargs) in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, dict[str, Any]]] = []

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        scalar = value.detach() if isinstance(value, torch.Tensor) else value
        self.calls.append((name, float(scalar), kwargs))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


def _task(
    model: nn.Module | None = None,
    cell_graph: HeteroData | None = None,
    loss_func: nn.Module | None = None,
    **kwargs: Any,
) -> RegressionTask:
    return RegressionTask(
        model=model if model is not None else _FixedHeads(),
        cell_graph=cell_graph if cell_graph is not None else HeteroData(),
        optimizer_config=kwargs.pop("optimizer_config", {"type": "AdamW", "lr": 1e-3}),
        lr_scheduler_config=kwargs.pop("lr_scheduler_config", {}),
        loss_func=(
            loss_func
            if loss_func is not None
            else DCellLoss(alpha=0.3, aux_reduction="sum")
        ),
        device="cpu",
        **kwargs,
    )


def _recorded(monkeypatch: pytest.MonkeyPatch, task: RegressionTask) -> _Log:
    recorder = _Log()
    monkeypatch.setattr(task, "log", recorder)
    return recorder


def _record_updates(
    monkeypatch: pytest.MonkeyPatch, collection: MetricCollection
) -> list[tuple[list[float], list[float]]]:
    """Record what a ``MetricCollection.update`` receives, then forward the call."""
    seen: list[tuple[list[float], list[float]]] = []
    original = collection.update

    def update(preds: torch.Tensor, target: torch.Tensor) -> None:
        seen.append((preds.tolist(), target.tolist()))
        original(preds, target)

    monkeypatch.setattr(collection, "update", update)
    return seen


def _mse_total(collection: MetricCollection) -> float:
    """The summed squared error state of the collection's MSE metric (0 after reset)."""
    total = getattr(collection["MSE"], "total")
    assert isinstance(total, torch.Tensor)
    return float(total.item())


def _attach(task: RegressionTask, tmp_path: Path, epoch: int = 0) -> None:
    task.trainer = L.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    task.trainer.fit_loop.epoch_progress.current.completed = epoch


def test_shared_step_logs_each_loss_component_then_the_loss_with_batch_size_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keys ``val/<component>`` for the three ``DCellLoss`` components, then
    ``val/loss``; every call carries ``batch_size=3`` (the genotype count, the size of
    the [B, 1] prediction) and ``sync_dist=True``. Issue 596 / PR 587 concern trainers
    that logged a node count; here the size comes from ``predictions.size(0)``.
    """
    task = _task()
    log = _recorded(monkeypatch, task)
    task._shared_step(_batch(), 0, "val")
    expected = [(f"val/{k}", v) for k, v in SUM_LOGS] + [("val/loss", 1.7)]
    assert log.names() == [k for k, _ in expected]
    for (name, value, kwargs), (_, want) in zip(log.calls, expected):
        assert value == pytest.approx(want, abs=1e-6), name
        assert kwargs == {"batch_size": 3, "sync_dist": True}, name


def test_shared_step_metrics_receive_root_predictions_and_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no inverse transform and no ``phenotype_values_original`` both collections
    receive the root [0, -1, 1] against y [1, 0, 0.5]; computed MSE 0.75, RMSE
    0.8660254, Pearson 0.5 (module comment above).
    """
    task = _task()
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    trans = _record_updates(monkeypatch, task._metrics("val_transformed_metrics"))
    task._shared_step(_batch(), 0, "val")
    assert orig == [(ROOT, Y)]
    assert trans == [(ROOT, Y)]
    for prefix in ("val/gene_interaction/", "val/transformed/gene_interaction/"):
        name = (
            "val_metrics" if "transformed" not in prefix else "val_transformed_metrics"
        )
        computed = {k: v.item() for k, v in task._metrics(name).compute().items()}
        assert computed == pytest.approx(
            {
                f"{prefix}MSE": 0.75,
                f"{prefix}RMSE": math.sqrt(0.75),
                f"{prefix}Pearson": 0.5,
            },
            abs=1e-6,
        )
    # the train and test collections are untouched
    assert _mse_total(task._metrics("train_metrics")) == 0


class _Affine(nn.Module):
    """Inverse transform stand-in: ``gene.gene_interaction`` -> 2 * x + 1."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[float]] = []

    def forward(self, data: HeteroData) -> HeteroData:
        x = data["gene"]["gene_interaction"]
        self.seen.append(x.tolist())
        out = HeteroData()
        out["gene"].gene_interaction = 2 * x + 1
        return out


def test_inverse_transform_feeds_original_scale_metrics_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transform sees the squeezed [B] root [0, -1, 1]; original-scale metrics get
    2 * root + 1 = [1, -1, 3] against ``phenotype_values_original`` [3, 1, 2]
    (MSE (4 + 4 + 1) / 3 = 3); transformed metrics keep root vs y (MSE 0.75). The loss
    stays on the transformed scale (1.7) and the returned prediction is the
    TRANSFORMED [B, 1] root while the returned target is the ORIGINAL scale.
    """
    inverse = _Affine()
    task = _task(inverse_transform=inverse)
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    trans = _record_updates(monkeypatch, task._metrics("val_transformed_metrics"))
    batch = _batch()
    batch["gene"].phenotype_values_original = torch.tensor([3.0, 1.0, 2.0])
    loss, predictions, target = task._shared_step(batch, 0, "val")
    assert inverse.seen == [ROOT]
    assert orig == [([1.0, -1.0, 3.0], [3.0, 1.0, 2.0])]
    assert trans == [(ROOT, Y)]
    assert loss.item() == pytest.approx(1.7, abs=1e-6)
    assert predictions.tolist() == [[v] for v in ROOT]
    assert target.tolist() == [[3.0], [1.0], [2.0]]
    mse = task._metrics("val_metrics")["MSE"].compute().item()
    assert mse == pytest.approx(3.0, abs=1e-6)


def test_each_metric_scale_masks_by_its_own_target_nans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transformed metrics mask by the NaNs of ``phenotype_values`` and the
    original-scale metrics by the NaNs of ``phenotype_values_original``, independently
    (int_dcell.py:257 and :288). Transformed y = [NaN, 0, 0.5] keeps rows 1 and 2:
    root [-1, 1] against [0, 0.5]. Original [3, 1, NaN] with the affine inverse
    2x + 1 (root -> [1, -1, 3]) keeps rows 0 and 1: [1, -1] against [3, 1]. A mask
    taken from the other target would hand over the wrong rows.
    """
    task = _task(inverse_transform=_Affine())
    _recorded(monkeypatch, task)
    trans = _record_updates(monkeypatch, task._metrics("val_transformed_metrics"))
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    batch = HeteroData()
    batch["gene"].phenotype_values = torch.tensor([float("nan"), 0.0, 0.5])
    batch["gene"].phenotype_values_original = torch.tensor([3.0, 1.0, float("nan")])
    task._shared_step(batch, 0, "val")
    assert trans == [([-1.0, 1.0], [0.0, 0.5])]
    assert orig == [([1.0, -1.0], [3.0, 1.0])]


class _OneSample(nn.Module):
    """One genotype: root prediction [0.5], no heads."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.w * torch.tensor([0.5]), {}


def test_a_batch_of_one_reaches_the_inverse_transform_as_a_0_dim_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B = 1: the [1, 1] prediction is squeezed to shape () before the inverse
    transform (int_dcell.py:269), so the transform sees a 0-dim tensor, not [1]; the
    0-dim result is unsqueezed back to [1, 1] (:280-281) and the original metrics get
    2 * 0.5 + 1 = 2.0 against the original target 4.0.
    """
    seen: list[tuple[int, ...]] = []

    class _Shape(nn.Module):
        def forward(self, data: HeteroData) -> HeteroData:
            x = data["gene"]["gene_interaction"]
            seen.append(tuple(x.shape))
            out = HeteroData()
            out["gene"].gene_interaction = 2 * x + 1
            return out

    task = _task(
        model=_OneSample(), loss_func=_PlainLoss(False), inverse_transform=_Shape()
    )
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    batch = HeteroData()
    batch["gene"].phenotype_values = torch.tensor([1.0])
    batch["gene"].phenotype_values_original = torch.tensor([4.0])
    _, predictions, target = task._shared_step(batch, 0, "val")
    assert seen == [()]
    assert orig == [([2.0], [4.0])]
    assert predictions.tolist() == [[0.5]]
    assert target.tolist() == [[4.0]]


def test_a_non_tensor_inverse_result_is_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: when the inverse transform's ``gene_interaction`` is not a tensor (here a
    list) the ``isinstance`` check at int_dcell.py:279 skips it without error and the
    original-scale metrics compare the TRANSFORMED root [0, -1, 1] against the original
    targets [3, 1, 2] (MSE (9 + 4 + 1) / 3 = 4.6666667 instead of 3). Latent: the 006
    scripts pass ``inverse_transform=None``. Pinned until a non-tensor result is
    refused.
    """

    class _ListInverse(nn.Module):
        def forward(self, data: HeteroData) -> HeteroData:
            out = HeteroData()
            out["gene"].gene_interaction = (
                2 * data["gene"]["gene_interaction"] + 1
            ).tolist()
            return out

    task = _task(inverse_transform=_ListInverse())
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    batch = _batch()
    batch["gene"].phenotype_values_original = torch.tensor([3.0, 1.0, 2.0])
    task._shared_step(batch, 0, "val")
    assert orig == [(ROOT, [3.0, 1.0, 2.0])]
    mse = task._metrics("val_metrics")["MSE"].compute().item()
    assert mse == pytest.approx(14 / 3, abs=1e-6)


def test_a_multi_column_original_target_is_cut_to_its_first_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``phenotype_values_original`` [[3, 9], [1, 9], [2, 9]] keeps column 0 only
    (int_dcell.py:201-203): the original metrics and the returned target see [3, 1, 2];
    the 9s are dropped without a message.
    """
    task = _task()
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    batch = _batch()
    batch["gene"].phenotype_values_original = torch.tensor(
        [[3.0, 9.0], [1.0, 9.0], [2.0, 9.0]]
    )
    _, _, target = task._shared_step(batch, 0, "val")
    assert orig == [(ROOT, [3.0, 1.0, 2.0])]
    assert target.tolist() == [[3.0], [1.0], [2.0]]


def test_a_nan_target_is_masked_for_metrics_but_reaches_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``_shared_step`` drops NaN targets before both metric updates
    (int_dcell.py:257-262 and :288-293) but hands the unmasked target to ``DCellLoss``
    (:221-223), so one NaN label makes the step loss NaN (and ``training_step`` would
    backpropagate it). y = [1, NaN, 0.5]: metrics receive the root at samples 0 and 2,
    [0, 1] against [1, 0.5]; the logged primary loss and total are NaN. Not measured:
    whether any 005/006 Kuzmin TMI label is NaN. Pinned until the loss applies the same
    mask (or the dataset guarantees finite labels and the mask is removed).
    """
    task = _task()
    log = _recorded(monkeypatch, task)
    trans = _record_updates(monkeypatch, task._metrics("val_transformed_metrics"))
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    batch = HeteroData()
    batch["gene"].phenotype_values = torch.tensor([1.0, float("nan"), 0.5])
    loss, _, _ = task._shared_step(batch, 0, "val")
    assert trans == [([0.0, 1.0], [1.0, 0.5])]
    assert orig == [([0.0, 1.0], [1.0, 0.5])]
    assert math.isnan(loss.item())
    logged = dict((name, value) for name, value, _ in log.calls)
    assert math.isnan(logged["val/primary_loss"])
    assert math.isnan(logged["val/loss"])


class _PlainLoss(nn.Module):
    """A non-DCell loss: MSE of [B, 1] tensors, returned as a tuple or a tensor."""

    def __init__(self, as_tuple: bool) -> None:
        super().__init__()
        self.as_tuple = as_tuple
        self.shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.shapes.append((tuple(pred.shape), tuple(target.shape)))
        loss = ((pred - target) ** 2).mean()
        return (loss, {"ignored": loss}) if self.as_tuple else loss


@pytest.mark.parametrize("as_tuple", [False, True])
def test_a_non_dcell_loss_gets_b_by_1_tensors_and_only_the_loss_is_logged(
    monkeypatch: pytest.MonkeyPatch, as_tuple: bool
) -> None:
    """Outside ``DCellLoss`` the [B, 1] prediction and target go in as they are (a
    tuple return keeps element 0), no component is logged, and the loss is the root
    MSE 0.75. A [B, 2] target keeps its first column only (int_dcell.py:185-187), so
    y2 = [[1, 9], [0, 9], [0.5, 9]] gives the same 0.75.
    """
    loss_func = _PlainLoss(as_tuple)
    task = _task(loss_func=loss_func)
    log = _recorded(monkeypatch, task)
    batch = HeteroData()
    batch["gene"].phenotype_values = torch.tensor([[1.0, 9.0], [0.0, 9.0], [0.5, 9.0]])
    loss, _, target = task._shared_step(batch, 0, "test")
    assert loss_func.shapes == [((3, 1), (3, 1))]
    assert loss.item() == pytest.approx(0.75, abs=1e-6)
    assert log.names() == ["test/loss"]
    assert log.calls[0][2] == {"batch_size": 3, "sync_dist": True}
    assert target.tolist() == [[1.0], [0.0], [0.5]]


def test_no_loss_function_is_refused() -> None:
    """``loss_func=None`` raises the exact message before anything is logged."""
    task = RegressionTask(
        model=_FixedHeads(),
        cell_graph=HeteroData(),
        optimizer_config={"type": "AdamW", "lr": 1e-3},
        lr_scheduler_config={},
        loss_func=None,
        device="cpu",
    )
    with pytest.raises(ValueError, match=re.escape("No loss function provided")):
        task._shared_step(_batch(), 0, "val")


class _Opt:
    """``self.optimizers()`` stand-in that records step and zero_grad into ``events``."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.param_groups = [{"lr": 0.01}]

    def step(self) -> None:
        self.events.append("step")

    def zero_grad(self) -> None:
        self.events.append("zero_grad")


def _manual(
    monkeypatch: pytest.MonkeyPatch, task: RegressionTask
) -> tuple[list[Any], _Log]:
    """Patch ``optimizers``, ``manual_backward``, ``clip_grad_norm_`` and ``log``."""
    events: list[Any] = []
    opt = _Opt(events)
    monkeypatch.setattr(task, "optimizers", lambda: opt)

    def backward(loss: torch.Tensor) -> None:
        events.append(("backward", loss.item()))
        loss.backward()

    monkeypatch.setattr(task, "manual_backward", backward)

    def clip(params: Any, max_norm: float) -> None:
        events.append(("clip", [p.shape for p in params], max_norm))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    return events, _recorded(monkeypatch, task)


def _w(task: RegressionTask) -> nn.Parameter:
    model = task.model
    assert isinstance(model, _FixedHeads)
    return model.w


@pytest.mark.parametrize("clip", [False, True])
def test_training_step_backward_clip_step_zero_grad_and_learning_rate_log(
    monkeypatch: pytest.MonkeyPatch, clip: bool
) -> None:
    """No accumulation schedule: backward of the full 1.7, then (if enabled) clipping of
    ``task.parameters()`` (only w, shape ()) at ``max_norm=0.25``, then ``step`` and
    ``zero_grad``. w.grad is the hand gradient 2.7 (the stand-in ``zero_grad`` does
    not clear it). After the train/* losses ``learning_rate`` = 0.01 is logged with
    ``batch_size`` = ``phenotype_values.size(0)`` = 3.
    """
    task = _task(clip_grad_norm=clip, clip_grad_norm_max_norm=0.25)
    events, log = _manual(monkeypatch, task)
    loss = task.training_step(_batch(), 0)
    assert loss.item() == pytest.approx(1.7, abs=1e-6)
    assert events[0][0] == "backward"
    assert events[0][1] == pytest.approx(1.7, abs=1e-6)
    clipping = [("clip", [torch.Size([])], 0.25)] if clip else []
    assert events[1:] == [*clipping, "step", "zero_grad"]
    grad = _w(task).grad
    assert grad is not None
    assert grad.item() == pytest.approx(2.7, abs=1e-5)
    assert log.names() == [
        "train/primary_loss",
        "train/auxiliary_loss",
        "train/weighted_auxiliary_loss",
        "train/loss",
        "learning_rate",
    ]
    assert log.calls[-1] == (
        "learning_rate",
        0.01,
        {"batch_size": 3, "sync_dist": True},
    )


def test_accumulation_divides_by_the_counter_and_steps_on_its_multiples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a schedule present and ``current_accumulation_steps`` = 2: batch 0
    backpropagates 1.7 / 2 = 0.85 and does not step; batch 1 does the same and steps.
    The gradient accumulates to 2 * 2.7 / 2 = 2.7.
    """
    task = _task(grad_accumulation_schedule={0: 2})
    task.current_accumulation_steps = 2
    events, _ = _manual(monkeypatch, task)
    first = task.training_step(_batch(), 0)
    second = task.training_step(_batch(), 1)
    assert first.item() == pytest.approx(0.85, abs=1e-6)
    assert second.item() == pytest.approx(0.85, abs=1e-6)
    assert [e if isinstance(e, str) else e[0] for e in events] == [
        "backward",
        "backward",
        "step",
        "zero_grad",
    ]
    grad = _w(task).grad
    assert grad is not None
    assert grad.item() == pytest.approx(2.7, abs=1e-5)


def test_the_accumulation_schedule_values_are_never_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``grad_accumulation_schedule`` is only tested for ``None``
    (int_dcell.py:383, :388); nothing in the module sets ``current_accumulation_steps``
    from it (it stays the 1 of :59; no ``on_train_epoch_start`` reads the schedule, unlike
    ``fit_int_cell_diffpool_dense_regression.py:170``). A schedule ``{0: 4}`` therefore
    divides by 1 and steps on every batch. Every 005/006 DCell config sets the schedule to
    null, so no recorded run is affected. Pinned until the schedule is applied or the
    argument is removed.
    """
    task = _task(grad_accumulation_schedule={0: 4})
    events, _ = _manual(monkeypatch, task)
    task.on_train_epoch_start()
    loss = task.training_step(_batch(), 0)
    assert task.current_accumulation_steps == 1
    assert loss.item() == pytest.approx(1.7, abs=1e-6)
    assert [e if isinstance(e, str) else e[0] for e in events] == [
        "backward",
        "step",
        "zero_grad",
    ]


def test_validation_and_test_steps_return_the_shared_step_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each returns 1.7 and logs under its own stage prefix."""
    task = _task()
    log = _recorded(monkeypatch, task)
    assert task.validation_step(_batch(), 0).item() == pytest.approx(1.7, abs=1e-6)
    assert task.test_step(_batch(), 0).item() == pytest.approx(1.7, abs=1e-6)
    assert [n for n in log.names() if n.endswith("/loss")] == ["val/loss", "test/loss"]


class _WithLatents(_FixedHeads):
    """``_FixedHeads`` that also returns ``subsystem_outputs["root"]`` latents."""

    LATENT = [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        root, outputs = super().forward(cell_graph, batch)
        outputs["subsystem_outputs"] = {"root": torch.tensor(self.LATENT)}
        return root, outputs


def test_samples_are_collected_per_stage_with_the_train_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plot_every_n_epochs=1`` makes epoch 0 a collection epoch. Train, ceiling 4 with
    two batches of B = 3: the first batch fits (count 0, remaining 4 >= 3) and is kept
    whole; the second sees count 3, remaining 4 - 3 = 1 < 3, so it keeps the
    ``randperm(3)[:1]`` row (the oracle draws the same permutation from the same seed),
    4 rows in all. Target, prediction and latent rows are subsampled together. Val and
    test keep every row.
    """
    task = _task(model=_WithLatents(), plot_every_n_epochs=1, plot_sample_ceiling=4)
    _recorded(monkeypatch, task)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        idx = torch.randperm(3)[:1].tolist()
        torch.manual_seed(0)
        task._shared_step(_batch(), 0, "train")
        task._shared_step(_batch(), 1, "train")
    train = task.train_samples
    assert [t.tolist() for t in train["true_values"]] == [
        [[v] for v in Y],
        [[Y[i]] for i in idx],
    ]
    assert [t.tolist() for t in train["predictions"]] == [
        [[v] for v in ROOT],
        [[ROOT[i]] for i in idx],
    ]
    assert [t.tolist() for t in train["latents"]["subsystem_outputs"]] == [
        _WithLatents.LATENT,
        [_WithLatents.LATENT[i] for i in idx],
    ]
    assert sum(t.size(0) for t in train["true_values"]) == 4
    # a third batch adds nothing: count 4 is not below the ceiling 4
    task._shared_step(_batch(), 2, "train")
    assert len(task.train_samples["true_values"]) == 2
    for stage in ("val", "test"):
        task._shared_step(_batch(), 0, stage)
        samples = getattr(task, f"{stage}_samples")
        assert [t.tolist() for t in samples["true_values"]] == [[[v] for v in Y]]
        assert [t.tolist() for t in samples["predictions"]] == [[[v] for v in ROOT]]
        assert [t.tolist() for t in samples["latents"]["subsystem_outputs"]] == [
            _WithLatents.LATENT
        ]


def test_off_epochs_collect_train_and_val_samples_never_and_test_samples_always(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``plot_every_n_epochs=10`` at epoch 0: (0 + 1) % 10 != 0, so train and
    val collect nothing; test ignores the epoch.
    """
    task = _task()
    _recorded(monkeypatch, task)
    for stage in ("train", "val", "test"):
        task._shared_step(_batch(), 0, stage)
    assert task.train_samples["true_values"] == []
    assert task.val_samples["true_values"] == []
    assert [t.tolist() for t in task.test_samples["true_values"]] == [[[v] for v in Y]]
    assert task.test_samples["latents"] == {}


@pytest.mark.parametrize("model_class", [DCell, DCellOpt])
def test_dcell_models_never_supply_the_latents_the_trainer_collects(
    monkeypatch: pytest.MonkeyPatch, model_class: type[nn.Module]
) -> None:
    """Finding: the trainer reads ``representations["subsystem_outputs"]["root"]``
    (int_dcell.py:206, :318-376) but neither ``DCell`` (what every 005/006 config
    builds) nor ``DCellOpt`` returns that key: DCell returns ``linear_outputs``,
    ``root_key``, ``term_activations``, ``stratum_outputs``, DCellOpt those plus
    ``all_activations_tensor`` and ``activation_mask``. With the real model no latent is
    ever collected and the oversmoothing log in ``_plot_samples`` (:489-491) never
    fires in any run. Pinned until the trainer reads ``term_activations`` (or the dead
    branch is removed).
    """
    graph = make_dcell_graph()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = model_class(graph, min_subsystem_size=2, subsystem_ratio=0.5)
    model.eval()
    task = _task(model=model, cell_graph=graph)
    _recorded(monkeypatch, task)
    batch = make_dcell_batch([[0], [2, 3], [1]])
    batch["gene"].phenotype_values = torch.tensor(Y)
    _, outputs = model(graph, batch)
    keys = ["linear_outputs", "root_key", "stratum_outputs", "term_activations"]
    if model_class is DCellOpt:
        keys += ["activation_mask", "all_activations_tensor"]
    assert sorted(outputs) == sorted(keys)
    task._shared_step(batch, 0, "test")
    assert len(task.test_samples["true_values"]) == 1
    assert task.test_samples["latents"] == {}


def _stub_plots(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    visual: list[Any] = []
    logged: list[Any] = []

    class _Vis:
        def __init__(self, base_dir: str, max_points: int) -> None:
            visual.append(("init", base_dir, max_points))

        def visualize_model_outputs(self, *args: Any, **kwargs: Any) -> None:
            visual.append((args, kwargs))

    monkeypatch.setattr("torchcell.trainers.int_dcell.Visualization", _Vis)
    monkeypatch.setattr(
        "torchcell.trainers.int_dcell.genetic_interaction_score.box_plot",
        lambda true, pred: ("fig", true.tolist(), pred.tolist()),
    )
    monkeypatch.setattr("wandb.Image", lambda fig: ("image", fig))
    monkeypatch.setattr("wandb.log", lambda payload: logged.append(payload))
    monkeypatch.setattr("torchcell.trainers.int_dcell.plt.close", lambda fig: None)
    return visual, logged


def test_plot_samples_hands_the_concatenated_samples_and_latent_smoothness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two collected chunks are concatenated: true [[1], [0]] + [[0.5]], predictions
    [[0], [-1]] + [[1]], latents the three rows of ``_WithLatents.LATENT``.
    ``Visualization(base_dir=<root dir>, max_points=1000)`` gets (predictions, true,
    {"subsystem_outputs": latents}, "DCellLoss", epoch 0, None, stage). Smoothness is
    the Frobenius norm of the centered latents: mean (1, 1), deviations
    [[-1, -1], [1, -1], [0, 2]], sqrt(8) = 2.8284271. The box plot gets column 0.
    """
    visual, logged = _stub_plots(monkeypatch)
    task = _task()
    _attach(task, tmp_path)
    latent = torch.tensor(_WithLatents.LATENT)
    samples = {
        "true_values": [torch.tensor([[1.0], [0.0]]), torch.tensor([[0.5]])],
        "predictions": [torch.tensor([[0.0], [-1.0]]), torch.tensor([[1.0]])],
        "latents": {"subsystem_outputs": [latent[:2], latent[2:]]},
    }
    task._plot_samples(samples, "val_sample")
    assert visual[0] == ("init", str(tmp_path), 1000)
    args, kwargs = visual[1]
    predictions, true_values, latents, loss_name, epoch, stamp = args
    assert predictions.tolist() == [[0.0], [-1.0], [1.0]]
    assert true_values.tolist() == [[1.0], [0.0], [0.5]]
    assert list(latents) == ["subsystem_outputs"]
    assert latents["subsystem_outputs"].tolist() == _WithLatents.LATENT
    assert (loss_name, epoch, stamp, kwargs) == (
        "DCellLoss",
        0,
        None,
        {"stage": "val_sample"},
    )
    assert len(visual) == 2
    assert len(logged) == 2
    assert list(logged[0]) == ["val_sample/oversmoothing_subsystem"]
    assert logged[0]["val_sample/oversmoothing_subsystem"] == pytest.approx(
        math.sqrt(8.0), abs=1e-6
    )
    assert logged[1] == {
        "val_sample/gene_interaction_box_plot": (
            "image",
            ("fig", [1.0, 0.0, 0.5], [0.0, -1.0, 1.0]),
        )
    }


def test_plot_samples_over_the_ceiling_subsamples_rows_jointly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ceiling 2 with 3 rows: ``randperm(3)[:2]`` (seed 0, reproduced by the oracle
    call) picks the same rows of true, prediction and latent; an all-NaN target skips
    the box plot; no latents means no smoothness log.
    """
    visual, logged = _stub_plots(monkeypatch)
    task = _task(plot_sample_ceiling=2)
    _attach(task, tmp_path)
    true = [10.0, 20.0, 30.0]
    pred = [1.0, 2.0, 3.0]
    with torch.random.fork_rng():
        torch.manual_seed(0)
        idx = torch.randperm(3)[:2].tolist()
        torch.manual_seed(0)
        task._plot_samples(
            {
                "true_values": [torch.tensor(true)],
                "predictions": [torch.tensor(pred)],
                "latents": {"subsystem_outputs": []},
            },
            "train_sample",
        )
    args, _ = visual[1]
    assert visual[0] == ("init", str(tmp_path), 2)
    assert args[0].tolist() == [[pred[i]] for i in idx]
    assert args[1].tolist() == [[true[i]] for i in idx]
    assert args[2] == {}
    assert list(logged[0]) == ["train_sample/gene_interaction_box_plot"]
    visual.clear()
    logged.clear()
    nan = float("nan")
    task._plot_samples(
        {
            "true_values": [torch.tensor([[nan]])],
            "predictions": [torch.tensor([[1.0]])],
        },
        "test_sample",
    )
    assert len(visual) == 2
    assert logged == []
    task._plot_samples({"true_values": [], "predictions": []}, "test_sample")
    assert len(visual) == 2


class _Raises(Metric):
    """A metric whose ``compute`` raises ``ValueError(message)``."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def update(self) -> None:
        pass

    def compute(self) -> torch.Tensor:
        raise ValueError(self.message)


class _Const(Metric):
    """A metric whose ``compute`` returns a constant."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def update(self) -> None:
        pass

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.value)


def test_compute_metrics_safely_skips_only_the_two_known_messages() -> None:
    """The two "too few samples" messages are skipped, any other ``ValueError``
    propagates unchanged.
    """
    task = _task()
    collection = MetricCollection(
        {
            "a": _Raises("Needs at least two samples to calculate r"),
            "b": _Raises("No samples to concatenate"),
            "c": _Const(4.0),
        },
        prefix="p/",
    )
    assert {
        k: v.item() for k, v in task._compute_metrics_safely(collection).items()
    } == {"p/c": 4.0}
    other = MetricCollection({"d": _Raises("shape mismatch")})
    with pytest.raises(ValueError, match=re.escape("shape mismatch")):
        task._compute_metrics_safely(other)


def test_compute_metrics_safely_on_empty_torchmetrics_returns_nan() -> None:
    """With torchmetrics 1.8 an un-updated MSE, RMSE and Pearson return NaN instead
    of raising, so the skip branch never fires for the real collections and an empty
    epoch logs NaN.
    """
    task = _task()
    computed = task._compute_metrics_safely(task._metrics("val_metrics"))
    assert sorted(computed) == [
        "val/gene_interaction/MSE",
        "val/gene_interaction/Pearson",
        "val/gene_interaction/RMSE",
    ]
    assert all(math.isnan(v.item()) for v in computed.values())


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_epoch_end_logs_both_collections_resets_them_and_plots_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    """After one batch of the stage at epoch 0 with ``plot_every_n_epochs=1``, the hook
    logs the original collection then the transformed one (MSE 0.75, Pearson 0.5, RMSE
    0.8660254, ``sync_dist=True`` and no batch size), resets both, plots the samples
    under ``<stage>_sample`` once and clears them.
    """
    visual, logged = _stub_plots(monkeypatch)
    task = _task(plot_every_n_epochs=1)
    _attach(task, tmp_path)
    log = _recorded(monkeypatch, task)
    task._shared_step(_batch(), 0, stage)
    log.calls.clear()
    getattr(task, f"on_{'validation' if stage == 'val' else stage}_epoch_end")()
    values = {"MSE": 0.75, "Pearson": 0.5, "RMSE": math.sqrt(0.75)}
    expected = [
        (f"{stage}/{part}gene_interaction/{m}", v)
        for part in ("", "transformed/")
        for m, v in values.items()
    ]
    assert log.names() == [k for k, _ in expected]
    for (name, value, kwargs), (_, want) in zip(log.calls, expected):
        assert value == pytest.approx(want, abs=1e-6), name
        assert kwargs == {"sync_dist": True}, name
    for name in (f"{stage}_metrics", f"{stage}_transformed_metrics"):
        assert _mse_total(task._metrics(name)) == 0
    stages = [kw["stage"] for v in visual if v[0] != "init" for kw in [v[1]]]
    assert stages == [f"{stage}_sample"]
    assert logged == [
        {f"{stage}_sample/gene_interaction_box_plot": ("image", ("fig", Y, ROOT))}
    ]
    assert getattr(task, f"{stage}_samples") == {
        "true_values": [],
        "predictions": [],
        "latents": {},
    }


def test_validation_epoch_end_does_not_plot_or_clear_during_sanity_checking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With ``trainer.sanity_checking`` the six metrics are still logged (original
    collection then transformed: MSE 0.75, Pearson 0.5, RMSE 0.8660254, each with
    ``sync_dist=True``) and reset, but the collected val samples are neither plotted
    nor cleared.
    """
    visual, logged = _stub_plots(monkeypatch)
    task = _task(plot_every_n_epochs=1)
    _attach(task, tmp_path)
    log = _recorded(monkeypatch, task)
    task._shared_step(_batch(), 0, "val")
    log.calls.clear()
    monkeypatch.setattr(type(task.trainer), "sanity_checking", property(lambda s: True))
    task.on_validation_epoch_end()
    values = {"MSE": 0.75, "Pearson": 0.5, "RMSE": math.sqrt(0.75)}
    expected = [
        (f"val/{part}gene_interaction/{m}", v)
        for part in ("", "transformed/")
        for m, v in values.items()
    ]
    assert log.names() == [k for k, _ in expected]
    for (name, value, kwargs), (_, want) in zip(log.calls, expected):
        assert value == pytest.approx(want, abs=1e-6), name
        assert kwargs == {"sync_dist": True}, name
    assert _mse_total(task._metrics("val_metrics")) == 0
    assert visual == [] and logged == []
    assert len(task.val_samples["true_values"]) == 1


def test_epoch_start_hooks_clear_samples_only_on_collection_epochs(
    tmp_path: Path,
) -> None:
    """Train and val clear at epochs where (epoch + 1) % 2 == 0 (epoch 1), not at
    epoch 0; test always clears.
    """
    task = _task(plot_every_n_epochs=2)
    marker = {"true_values": ["x"], "predictions": ["x"], "latents": {}}
    for epoch, cleared in ((0, False), (1, True)):
        _attach(task, tmp_path, epoch=epoch)
        task.train_samples = dict(marker)
        task.val_samples = dict(marker)
        task.test_samples = dict(marker)
        task.on_train_epoch_start()
        task.on_validation_epoch_start()
        task.on_test_epoch_start()
        empty: dict[str, Any] = {"true_values": [], "predictions": [], "latents": {}}
        assert (task.train_samples == empty) is cleared
        assert (task.val_samples == empty) is cleared
        assert task.test_samples == empty


class _Recorder(nn.Module):
    """Records the (cell_graph, batch) it is called with; one parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.calls: list[tuple[HeteroData, HeteroData]] = []

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls.append((cell_graph, batch))
        return self.w.sum().reshape(1), {}


def test_configure_optimizers_builds_adamw_and_plateau_from_the_configs() -> None:
    """``learning_rate`` is renamed to ``lr``; the optimizer holds exactly
    ``task.parameters()``; the scheduler is ``ReduceLROnPlateau`` with the config's
    keys minus ``type``; the returned dict names ``val/gene_interaction/MSE`` as the
    monitor per epoch, though under the task's manual optimization Lightning ignores
    the monitor and never steps the scheduler
    (``test_the_plateau_scheduler_of_a_dcell_run_is_never_stepped``). The dummy batch (cell graph without ``go_gene_strata_state``) is the
    one-row fallback [[0, 0, 0, 0]] with ptr [0, 1].
    """
    model = _Recorder()
    task = _task(
        model=model,
        optimizer_config={"type": "AdamW", "learning_rate": 0.02, "weight_decay": 0.5},
        lr_scheduler_config={
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.2,
            "patience": 3,
            "min_lr": 1e-5,
        },
    )
    config = task.configure_optimizers()
    optimizer = config["optimizer"]
    assert type(optimizer) is torch.optim.AdamW
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (0.02, 0.5)
    assert [id(p) for p in group["params"]] == [id(p) for p in task.parameters()]
    assert [id(p) for p in task.parameters()] == [id(model.w)]
    scheduler_config = config["lr_scheduler"]
    assert isinstance(scheduler_config, dict)
    scheduler = scheduler_config["scheduler"]
    assert type(scheduler) is ReduceLROnPlateau
    assert (
        scheduler.mode,
        scheduler.factor,
        scheduler.patience,
        scheduler.min_lrs,
    ) == ("min", 0.2, 3, [1e-5])
    assert scheduler.optimizer is optimizer
    assert {k: v for k, v in scheduler_config.items() if k != "scheduler"} == {
        "monitor": "val/gene_interaction/MSE",
        "interval": "epoch",
        "frequency": 1,
    }
    assert sorted(config) == ["lr_scheduler", "optimizer"]
    ((graph, dummy),) = model.calls
    go = dummy["gene_ontology"]
    assert go.go_gene_strata_state.tolist() == [[0, 0, 0, 0]]
    assert go.go_gene_strata_state_ptr.tolist() == [0, 1]
    assert torch.equal(go.mutant_state, torch.zeros(1, 3))
    assert torch.equal(dummy["gene"].x, torch.zeros(1, 1))
    assert dummy["gene"].batch.tolist() == [0]
    assert dummy["gene"].phenotype_values.tolist() == [0.0]
    assert dummy["gene"].perturbation_indices.tolist() == [0]
    assert dummy.num_graphs == 1


def test_configure_optimizers_dummy_batch_zeroes_the_template_states() -> None:
    """With the DCell template (4 rows) in the cell graph, the dummy batch is the
    template with column 3 set to 0 and ptr [0, 4]; the task's own copy is untouched.
    """
    model = _Recorder()
    graph = make_dcell_graph()
    task = _task(model=model, cell_graph=graph)
    task.configure_optimizers()
    ((_, dummy),) = model.calls
    template = graph["gene_ontology"].go_gene_strata_state
    zeroed = template.clone()
    zeroed[:, 3] = 0
    assert torch.equal(dummy["gene_ontology"].go_gene_strata_state, zeroed)
    assert dummy["gene_ontology"].go_gene_strata_state_ptr.tolist() == [0, 4]
    assert task.cell_graph["gene_ontology"].go_gene_strata_state[:, 3].tolist() == [
        1,
        1,
        1,
        1,
    ]


def test_scheduler_type_is_ignored_and_always_plateau() -> None:
    """Finding: ``lr_scheduler_config["type"]`` is dropped (int_dcell.py:701-704) and
    ``ReduceLROnPlateau`` is always built, so ``{"type": "StepLR"}`` yields a plateau
    scheduler and ``{"type": "CosineAnnealingLR", "T_max": 2}`` fails on the foreign
    keyword. Every 005/006 DCell config names ReduceLROnPlateau, so no recorded run is
    affected. Pinned until the type is honored or refused when it is not
    ReduceLROnPlateau.
    """
    step = _task(model=_Recorder(), lr_scheduler_config={"type": "StepLR"})
    scheduler_config = step.configure_optimizers()["lr_scheduler"]
    assert isinstance(scheduler_config, dict)
    assert type(scheduler_config["scheduler"]) is ReduceLROnPlateau
    cosine = _task(
        model=_Recorder(), lr_scheduler_config={"type": "CosineAnnealingLR", "T_max": 2}
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'T_max'"):
        cosine.configure_optimizers()


class _Broken(nn.Module):
    """A model whose dummy forward raises."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, cell_graph: HeteroData, batch: HeteroData) -> Any:
        raise RuntimeError("boom")


class _NoParams(nn.Module):
    def forward(self, cell_graph: HeteroData, batch: HeteroData) -> Any:
        return torch.zeros(1), {}


def test_a_failing_dummy_forward_is_logged_and_adds_model_dummy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exception is swallowed with the warning "Error during model initialization:
    boom" and a zero parameter ``model.dummy`` (shape [1]) joins the optimizer after w.
    """
    model = _Broken()
    task = _task(model=model)
    with caplog.at_level(logging.INFO, logger="torchcell.trainers.int_dcell"):
        config = task.configure_optimizers()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["Error during model initialization: boom"]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos == [
        "Setting up optimizer and initializing DCellModel parameters",
        "Added dummy parameter to allow optimizer creation",
        "Total trainable parameters: 2",
    ]
    assert [n for n, _ in task.named_parameters()] == ["model.w", "model.dummy"]
    dummy = dict(model.named_parameters())["dummy"]
    assert torch.equal(dummy, torch.zeros(1))
    assert config["optimizer"].param_groups[0]["params"][1] is dummy


def test_a_model_without_parameters_gets_a_task_level_dummy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Zero trainable parameters after the dummy forward: the warning "No parameters
    found after initialization, adding dummy parameter" and a task-level ``dummy``.
    """
    task = _task(model=_NoParams())
    with caplog.at_level(logging.WARNING, logger="torchcell.trainers.int_dcell"):
        config = task.configure_optimizers()
    assert [r.getMessage() for r in caplog.records] == [
        "No parameters found after initialization, adding dummy parameter"
    ]
    assert [n for n, _ in task.named_parameters()] == ["dummy"]
    assert config["optimizer"].param_groups[0]["params"] == [task.dummy]


@pytest.mark.parametrize("model_class", [DCell, DCellOpt])
def test_dcell_in_training_mode_always_gets_a_dummy_that_breaks_checkpoint_reload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, model_class: type[nn.Module]
) -> None:
    """Finding: the dummy forward in ``configure_optimizers`` (int_dcell.py:664-675)
    feeds ONE sample to the real model, whose BatchNorm1d refuses a batch of one in
    training mode (the module default, and the mode Lightning trains in). The
    exception is swallowed, so every DCell training run registers ``model.dummy``
    (the comment's premise that the model creates parameters in its first forward is
    false: ``DCell``, which every 005/006 config builds, and ``DCellOpt`` both build every
    module in ``__init__``). Reproduced for both: the state dict carries
    ``model.dummy``, and the strict state-dict load that ``load_from_checkpoint`` performs
    fails on a freshly built model; the failed forward also increments ``num_batches_tracked`` of term 1's
    BatchNorm (the first subsystem run, stratum 1) from 0 to 1. In eval mode the same
    call succeeds and adds nothing. Reach: real checkpoints carry the key
    (``experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile.py:139-145``
    deletes a ``dummy`` key and attributes it to an older model); the resume paths
    (006 dcell.py:410, 005 dcell.py:363) are latent, since all seven DCell configs set
    ``checkpoint_path: null``. Pinned until the dummy forward is removed (or run in eval
    mode) and the exception is not swallowed.
    """
    graph = make_dcell_graph()

    def build() -> nn.Module:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            return model_class(graph, min_subsystem_size=2, subsystem_ratio=0.5)

    model = build()
    assert model.training
    before = {k: v.clone() for k, v in model.state_dict().items()}
    task = _task(model=model, cell_graph=graph)
    with caplog.at_level(logging.WARNING, logger="torchcell.trainers.int_dcell"):
        task.configure_optimizers()
    assert [r.getMessage() for r in caplog.records] == [
        "Error during model initialization: Expected more than 1 value per channel "
        "when training, got input size torch.Size([1, 2])"
    ]
    assert "model.dummy" in task.state_dict()
    changed = [k for k in before if not torch.equal(before[k], model.state_dict()[k])]
    assert changed == ["subsystems.1.batch_norm.num_batches_tracked"]
    assert model.state_dict()["subsystems.1.batch_norm.num_batches_tracked"].item() == 1
    checkpoint = tmp_path / "dcell.ckpt"
    torch.save(
        {
            "state_dict": task.state_dict(),
            "hyper_parameters": dict(task.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        checkpoint,
    )
    # The strict load is done directly: it is the step ``load_from_checkpoint`` ends
    # with, and newer Lightning releases (the CI runner's) unpickle the checkpoint with
    # ``weights_only=True`` first and stop earlier on the hyperparameters.
    saved = torch.load(checkpoint, weights_only=False)["state_dict"]
    assert "model.dummy" in saved
    fresh = _task(model=build(), cell_graph=graph)
    with pytest.raises(
        RuntimeError, match=re.escape('Unexpected key(s) in state_dict: "model.dummy"')
    ):
        fresh.load_state_dict(saved)
    evaluated = build()
    evaluated.eval()
    eval_before = {k: v.clone() for k, v in evaluated.state_dict().items()}
    eval_task = _task(model=evaluated, cell_graph=graph)
    eval_task.configure_optimizers()
    assert "model.dummy" not in eval_task.state_dict()
    for k, v in eval_before.items():
        assert torch.equal(evaluated.state_dict()[k], v), k


def _shipped(path: str) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(REPO / path).regression_task)
    assert isinstance(config, dict)
    return {str(k): v for k, v in config.items()}


def _run_epochs_with_lightning_schedulers(
    task: RegressionTask,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    epochs: int = 4,
) -> tuple[list[tuple[Any, ...]], torch.optim.Optimizer]:
    """Drive ``epochs`` epochs of the task's hooks and Lightning's own epoch-end
    scheduler update, recording every ``ReduceLROnPlateau.step`` call.

    The optimizers and scheduler configs come from Lightning's
    ``_init_optimizers_and_lr_schedulers`` (what ``Trainer.fit`` uses) and are installed
    on the attached trainer's strategy; per epoch: ``training_step``,
    ``validation_step``, ``on_validation_epoch_end``, ``on_train_epoch_end``, then
    ``fit_loop.epoch_loop._update_learning_rates("epoch", update_plateau_schedulers=True)``,
    the call Lightning 2.5.5 makes after validation (training_epoch_loop.py:454-468; it
    returns at once when ``automatic_optimization`` is False).
    """
    steps: list[tuple[Any, ...]] = []
    original = ReduceLROnPlateau.step

    def step(self: ReduceLROnPlateau, *args: Any, **kwargs: Any) -> None:
        steps.append(args)
        original(self, *args, **kwargs)

    monkeypatch.setattr(ReduceLROnPlateau, "step", step)
    _attach(task, tmp_path)
    optimizers, configs = _init_optimizers_and_lr_schedulers(task)
    task.trainer.strategy.optimizers = optimizers
    task.trainer.strategy.lr_scheduler_configs = configs
    task.trainer.strategy.connect(task)
    task.trainer._logger_connector._callback_metrics["val/gene_interaction/MSE"] = (
        torch.tensor(1.0)
    )
    _recorded(monkeypatch, task)
    monkeypatch.setattr(task, "optimizers", lambda: optimizers[0])
    monkeypatch.setattr(task, "manual_backward", lambda loss: loss.backward())
    for epoch in range(epochs):
        task.trainer.fit_loop.epoch_progress.current.completed = epoch
        task.training_step(_batch(), 0)
        task.validation_step(_batch(), 0)
        task.on_validation_epoch_end()
        task.on_train_epoch_end()
        task.trainer.fit_loop.epoch_loop._update_learning_rates(
            "epoch", update_plateau_schedulers=True
        )
    return steps, optimizers[0]


@pytest.mark.parametrize(
    "path",
    [
        "experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi.yaml",
        "experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi_mmli_000.yaml",
        "experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi_mmli_001.yaml",
        "experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi_mmli_002.yaml",
        "experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi.yaml",
    ],
)
def test_the_plateau_scheduler_of_a_dcell_run_is_never_stepped(
    path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding: ``RegressionTask`` sets ``automatic_optimization = False``
    (int_dcell.py:100) and never calls a scheduler's ``step``, and under manual
    optimization Lightning neither steps it (``_update_learning_rates`` returns early)
    nor keeps its monitor (the config comes back with ``reduce_on_plateau`` False and
    ``monitor`` None). With each shipped config, four epochs of the task's hooks record
    NO ``ReduceLROnPlateau.step`` call and the learning rate stays at the configured
    1e-3: every DCell run trained at a constant learning rate whatever its scheduler
    says. The same harness with ``automatic_optimization`` switched on steps the
    scheduler every epoch (control below), so the empty record is the task's, not the
    harness's. Moot secondary fact: these configs also set ``min_lr`` equal to ``lr``
    (1e-3), which would have kept a stepped scheduler at 1e-3 too. Not checked against
    the wandb run configs. Pinned until the task steps its scheduler (or drops it).
    """
    regression = _shipped(path)
    assert regression["optimizer"] == {
        "type": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-6,
    }
    plateau = regression["lr_scheduler"]
    assert (plateau["type"], plateau["min_lr"], plateau["factor"]) == (
        "ReduceLROnPlateau",
        1e-3,
        0.2,
    )
    task = _task(optimizer_config=regression["optimizer"], lr_scheduler_config=plateau)
    assert task.automatic_optimization is False
    steps, optimizer = _run_epochs_with_lightning_schedulers(
        task, monkeypatch, tmp_path
    )
    assert steps == []
    assert optimizer.param_groups[0]["lr"] == 1e-3
    config = task.trainer.lr_scheduler_configs[0]
    assert (config.reduce_on_plateau, config.monitor) == (False, None)

    # control: the same harness under automatic optimization steps the plateau once per
    # epoch with the monitored value (1.0, never improving after the first); with
    # min_lr 1e-5 and patience 0 it would lower 1e-3 to 2e-4 at the 2nd step.
    control = _task(
        optimizer_config=regression["optimizer"],
        lr_scheduler_config={**plateau, "patience": 0, "cooldown": 0, "min_lr": 1e-5},
    )
    control.automatic_optimization = True
    control_steps, control_optimizer = _run_epochs_with_lightning_schedulers(
        control, monkeypatch, tmp_path / "control", epochs=2
    )
    assert [[float(v) for v in call] for call in control_steps] == [[1.0], [1.0]]
    assert control_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)


@pytest.mark.parametrize(
    "path",
    [
        "experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_cpu.yaml",
        "experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_maxing_resources.yaml",
    ],
)
def test_a_spaced_exponent_threshold_reaches_the_scheduler_as_a_string_never_stepped(
    path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Config observation: ``threshold: 1e -4`` loads as the STRING "1e -4" and
    ``configure_optimizers`` hands it to ``ReduceLROnPlateau`` unchanged, which would
    raise ``TypeError`` at its first ``step``; but the task never steps its scheduler
    (``test_the_plateau_scheduler_of_a_dcell_run_is_never_stepped``), so a run with
    these configs trains without error. Latent; pinned until the two configs write 1e-4.
    """
    regression = _shipped(path)
    assert regression["lr_scheduler"]["threshold"] == "1e -4"
    task = _task(
        optimizer_config=regression["optimizer"],
        lr_scheduler_config=regression["lr_scheduler"],
    )
    steps, optimizer = _run_epochs_with_lightning_schedulers(
        task, monkeypatch, tmp_path
    )
    scheduler = task.trainer.lr_scheduler_configs[0].scheduler
    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.threshold == "1e -4"
    assert steps == []
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_forward_always_moves_the_batch_because_hetero_data_has_no_gene_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``forward`` looks for the batch device via ``hasattr(batch, "gene")``
    (int_dcell.py:131), but node stores of a ``HeteroData`` are not attributes
    (``hasattr`` is False even when ``batch["gene"]`` exists), and ``HeteroData`` has no
    ``device`` either; so ``batch_device`` stays None and every call moves the batch
    with ``batch.to(model_device)``, even when ``gene.perturbation_indices`` already sits
    on that device (line 134 cannot run). On one device the move is a no-op that returns
    the same object. The cell graph is moved once and cached. Pinned until the device
    probe reads ``batch["gene"]`` (or the dead probe is removed).
    """
    model = _Recorder()
    task = _task(model=model)
    moves: list[torch.device] = []
    original = HeteroData.to

    def to(self: HeteroData, device: torch.device, *args: Any, **kwargs: Any) -> Any:
        moves.append(device)
        return original(self, device, *args, **kwargs)

    monkeypatch.setattr(HeteroData, "to", to)
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([0])
    assert hasattr(batch, "gene") is False
    task(batch)
    task(batch)
    cpu = torch.device("cpu")
    # first call: the batch, then the cell graph; second call: the batch only
    assert moves == [cpu, cpu, cpu]
    assert model.calls[-1][1] is batch
    assert task._cell_graph_device == cpu


# ---------------------------------------------------------------------------
# 2026.10.06 (phase 21): the 0-dim reshapes, a 2-D inverse result, a first train
# collection that is already over the ceiling, and latents subsampled in the plot.
# ---------------------------------------------------------------------------


class _Scalar(nn.Module):
    """A model returning a 0-dim prediction w * 0.5 and no heads."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.w * torch.tensor(0.5), {}


def test_zero_dim_prediction_and_targets_become_one_by_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0-dim prediction 0.5, target 1.0 and original 4.0 are each unsqueezed twice
    (int_dcell.py:170, :182, :198): the plain loss sees ((1, 1), (1, 1)) and returns
    (0.5 - 1)^2 = 0.25; original metrics get [0.5] vs [4.0].
    """
    loss_func = _PlainLoss(False)
    task = _task(model=_Scalar(), loss_func=loss_func)
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    batch = HeteroData()
    batch["gene"].phenotype_values = torch.tensor(1.0)
    batch["gene"].phenotype_values_original = torch.tensor(4.0)
    loss, predictions, target = task._shared_step(batch, 0, "val")
    assert loss_func.shapes == [((1, 1), (1, 1))]
    assert loss.item() == pytest.approx(0.25)
    assert predictions.tolist() == [[0.5]]
    assert target.tolist() == [[4.0]]
    assert orig == [([0.5], [4.0])]


def test_a_two_dimensional_inverse_result_is_used_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inverse transform returning [B, 1] (here 10 * root as a column) is taken
    unchanged (int_dcell.py:285): original metrics get [0, -10, 10] vs y.
    """

    class _Column(nn.Module):
        def forward(self, data: HeteroData) -> HeteroData:
            out = HeteroData()
            out["gene"].gene_interaction = 10 * data["gene"]["gene_interaction"].view(
                -1, 1
            )
            return out

    task = _task(inverse_transform=_Column())
    _recorded(monkeypatch, task)
    orig = _record_updates(monkeypatch, task._metrics("val_metrics"))
    task._shared_step(_batch(), 0, "val")
    assert orig == [([0.0, -10.0, 10.0], Y)]


def test_a_first_train_batch_over_the_ceiling_is_subsampled_with_its_latents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ceiling 2, B = 3, nothing collected yet: remaining 2 < 3, so the FIRST batch is
    already subsampled to ``randperm(3)[:2]`` (oracle: same seed), and the latent list
    is created on that path (int_dcell.py:315-316) with the same two rows.
    """
    task = _task(model=_WithLatents(), plot_every_n_epochs=1, plot_sample_ceiling=2)
    _recorded(monkeypatch, task)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        idx = torch.randperm(3)[:2].tolist()
        torch.manual_seed(0)
        task._shared_step(_batch(), 0, "train")
    train = task.train_samples
    assert [t.tolist() for t in train["true_values"]] == [[[Y[i]] for i in idx]]
    assert [t.tolist() for t in train["predictions"]] == [[[ROOT[i]] for i in idx]]
    assert [t.tolist() for t in train["latents"]["subsystem_outputs"]] == [
        [_WithLatents.LATENT[i] for i in idx]
    ]


def test_plot_samples_subsamples_latents_with_the_same_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ceiling 2 with three latent rows: ``randperm(3)[:2]`` selects the same rows of
    true, prediction and latent (int_dcell.py:454), and the smoothness log is the
    Frobenius norm of the two centered latent rows: for rows a, b it is |a - b| / sqrt 2.
    """
    visual, logged = _stub_plots(monkeypatch)
    task = _task(plot_sample_ceiling=2)
    _attach(task, tmp_path)
    latent = torch.tensor(_WithLatents.LATENT)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        idx = torch.randperm(3)[:2].tolist()
        torch.manual_seed(0)
        task._plot_samples(
            {
                "true_values": [torch.tensor(Y)],
                "predictions": [torch.tensor(ROOT)],
                "latents": {"subsystem_outputs": [latent]},
            },
            "val_sample",
        )
    args, _ = visual[1]
    assert args[0].tolist() == [[ROOT[i]] for i in idx]
    assert args[1].tolist() == [[Y[i]] for i in idx]
    assert args[2]["subsystem_outputs"].tolist() == [
        _WithLatents.LATENT[i] for i in idx
    ]
    a, b = latent[idx[0]], latent[idx[1]]
    expected = ((a - b).norm() / math.sqrt(2.0)).item()
    assert logged[0]["val_sample/oversmoothing_subsystem"] == pytest.approx(
        expected, abs=1e-6
    )
