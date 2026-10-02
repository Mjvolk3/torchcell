# tests/torchcell/trainers/test_int_hetero_cell.py
# [[tests.torchcell.trainers.test_int_hetero_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_hetero_cell.py
"""Batch sizing in the 006 hetero trainers (``RegressionTask``, ``DiffusionRegressionTask``).

Both tasks size a perturbation batch (no ``gene.x``) by the collated batch's
``num_graphs``. Reading ``max(perturbation_indices_batch) + 1`` instead drops a trailing
genotype with no perturbed gene (issue #567). The ladder below that is unchanged: no
``perturbation_indices_batch`` counts perturbed genes, then phenotype values, then 1.

2026.10.01, Phase 19: exact behavior on scripted stand-ins. ``_Fixed`` returns the
predictions p = [1, 3, -2] times one trainable ``scale`` (1.0) and the representations
``{"z_p": [[3, 4], [0, 0], [6, 8]]}`` (row norms 5, 0, 10, mean 5); ``_coo`` is a
collated COO batch of three genotypes with targets y = [2, 5, -1], one
``gene_interaction`` value per genotype in order, ``num_graphs`` 3. So the squared
error (the generic stand-in loss, and the diffusion task's ``F.mse_loss`` at
validation and test) is (1 + 4 + 1) / 3 = 2.0, the mean log cosh is
(2 log cosh 1 + log cosh 2) / 3 = 0.7308548, and Pearson(p, y) = 15 / sqrt(228) =
0.9933993. ``log`` is replaced by a recorder of every call; the optimizer, backward,
gradient clipping and schedulers are stand-ins that record their calls, except in the
three ``fast_dev_run`` tests. The real 006 inverse is a ``COOInverseCompose`` over a
standard ``COOLabelNormalizationTransform`` fitted on [0, 4] (mean 2, sd 2), so it maps
v to 2 v + 2. Derivations are in each test docstring; tests whose docstring starts
``Finding:`` pin current behavior that contradicts the code's names or callers.
"""

import math
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer.states import RunningStage
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torchmetrics import MeanSquaredError

from torchcell.losses.isomorphic_cell_loss import ICLoss
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchcell.trainers.int_hetero_cell import DiffusionRegressionTask, RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelNormalizationTransform,
)

TASKS = [RegressionTask, DiffusionRegressionTask]


def _task(cls: Any, **overrides: Any) -> Any:
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    kwargs: dict[str, Any] = dict(
        model=nn.Linear(1, 1),
        cell_graph=graph,
        optimizer_config={"type": "AdamW", "learning_rate": 1e-2},
        lr_scheduler_config={},
        device="cpu",
    )
    kwargs.update(overrides)
    return cls(**kwargs)


def _wild_type_last() -> HeteroData:
    """Genotypes perturb {1, 2}, {3}, and nothing: the batch vector is [0, 0, 1]."""
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([1, 2, 3])
    batch["gene"].perturbation_indices_batch = torch.tensor([0, 0, 1])
    batch["gene"].phenotype_values = torch.tensor([0.1, -0.2, 0.3])
    batch.num_graphs = 3  # a collated PyG Batch carries it
    return batch


@pytest.mark.parametrize("cls", TASKS)
def test_trailing_genotype_without_a_perturbation_is_counted(cls: Any) -> None:
    """``num_graphs`` = 3 is the size, not max([0, 0, 1]) + 1 = 2."""
    assert _task(cls)._get_batch_size(_wild_type_last()) == 3


@pytest.mark.parametrize("cls", TASKS)
def test_profiling_step_logs_the_genotype_count(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataloader-profiling step logs 3.0 under ``batch_size=3`` for that batch."""
    task = _task(cls, execution_mode="dataloader_profiling")
    logged: dict[str, tuple[float, int]] = {}

    def record(name: str, value: Any, **kwargs: Any) -> None:
        logged[name] = (float(value), kwargs["batch_size"])

    monkeypatch.setattr(task, "log", record)
    loss, predictions, targets = task._shared_step(_wild_type_last(), 0, "val")
    assert (predictions, targets) == (None, None)
    assert loss.item() == 0.0
    assert logged == {
        "val/dataloader_profile_loss": (0.0, 3),
        "val/dataloader_profile_batch_size": (3.0, 3),
    }


@pytest.mark.parametrize("cls", TASKS)
def test_batch_size_ladder_below_num_graphs(cls: Any) -> None:
    """``x`` rows, then perturbed-gene count, then value count, then 1."""
    task = _task(cls)
    dense = HeteroData()
    dense["gene"].x = torch.zeros(5, 2)
    perts = HeteroData()
    perts["gene"].perturbation_indices = torch.tensor([4, 5, 6])
    values = HeteroData()
    values["gene"].phenotype_values = torch.tensor([2.0, 5.0])
    empty = HeteroData()
    empty["gene"].num_nodes = 0
    sizes = [task._get_batch_size(b) for b in (dense, perts, values, empty)]
    assert sizes == [5, 3, 2, 1]


# --------------------------------------------- 2026.10.01, Phase 19: scripted stand-ins


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Epoch hooks free and synchronize CUDA when it is available; keep them on CPU."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


P = [1.0, 3.0, -2.0]
Y = [2.0, 5.0, -1.0]
Z_P = [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]
LOGCOSH = (2 * math.log(math.cosh(1.0)) + math.log(math.cosh(2.0))) / 3  # 0.7308548
PEARSON = float(np.corrcoef(P, Y)[0, 1])  # 15 / sqrt(228) = 0.9933993


def _column(values: list[float]) -> list[list[float]]:
    return [[v] for v in values]


class _Fixed(nn.Module):
    """Hand-set predictions times one trainable ``scale`` (1.0) and a fixed reps dict.

    A representation passed as ``None`` is left out of the dict, so ``z_p=None`` models
    a network that emits no ``z_p``.
    """

    def __init__(
        self, predictions: list[float] | float | None = None, **reps: Any
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.predictions = torch.tensor(P if predictions is None else predictions)
        self.reps: dict[str, Any] = {"z_p": torch.tensor(Z_P)}
        self.reps.update(reps)
        self.calls: list[tuple[Any, HeteroData]] = []
        self.outputs: list[tuple[torch.Tensor, dict[str, Any]]] = []

    def forward(
        self, cell_graph: Any, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls.append((cell_graph, batch))
        reps = {k: v for k, v in self.reps.items() if v is not None}
        out = (self.predictions * self.scale, reps)
        self.outputs.append(out)
        return out


class _SquaredError(nn.Module):
    """A loss outside every ``isinstance`` branch: the mean squared residual.

    ``mode`` "bare" returns the tensor, "single" a one-tuple, "pair" ``(loss,
    components)``. Every call's positional and keyword arguments are recorded.
    """

    def __init__(self, mode: str = "bare", components: Any = None) -> None:
        super().__init__()
        self.mode = mode
        self.components = components
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        loss = ((args[0] - args[1]) ** 2).mean()
        if self.mode == "bare":
            return loss
        if self.mode == "single":
            return (loss,)
        return (loss, self.components)


class _ScriptedPointDist(PointDistGraphReg):
    """``PointDistGraphReg`` by type only: returns ``(0.5, components)``, records calls."""

    def __init__(self, components: Any) -> None:
        nn.Module.__init__(self)
        self.components = components
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return torch.tensor(0.5), self.components


class _ScriptedMleDist(MleDistSupCR):
    """``MleDistSupCR`` by type only: returns ``(0.5, components)``, records calls."""

    def __init__(self, components: Any) -> None:
        nn.Module.__init__(self)
        self.components = components
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return torch.tensor(0.5), self.components


class _ScriptedMleWass(MleWassSupCR):
    """``MleWassSupCR`` by type only: returns ``(0.5, components)``, records calls."""

    def __init__(self, components: Any) -> None:
        nn.Module.__init__(self)
        self.components = components
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return torch.tensor(0.5), self.components


def _coo(
    values: list[float] | list[list[float]] | float | None = None,
    original: list[float] | None = None,
    types: list[str] | None = None,
    x_nodes: int | None = None,
) -> HeteroData:
    """A collated COO batch: one ``gene_interaction`` value per genotype, in order.

    ``num_graphs`` is the number of genotypes; ``x_nodes`` adds a ``gene.x`` with that
    many node rows (a dense node-feature batch).
    """
    v = torch.tensor(Y if values is None else values)
    n = 1 if v.dim() == 0 else v.size(0)
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.arange(n)
    batch["gene"].perturbation_indices_batch = torch.arange(n)
    batch["gene"].phenotype_values = v
    batch["gene"].phenotype_type_indices = torch.zeros(n, dtype=torch.long)
    batch["gene"].phenotype_sample_indices = torch.arange(n)
    batch["gene"].phenotype_types = ["gene_interaction"] if types is None else types
    if original is not None:
        batch["gene"].phenotype_values_original = torch.tensor(original)
    if x_nodes is not None:
        batch["gene"].x = torch.zeros(x_nodes, 2)
    batch.num_graphs = n  # a collated PyG Batch carries it
    return batch


def _make(cls: Any, model: nn.Module | None = None, **overrides: Any) -> Any:
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    kwargs: dict[str, Any] = dict(
        optimizer_config={"type": "AdamW", "learning_rate": 1e-2},
        lr_scheduler_config=None,
        loss_func=_SquaredError(),
        device="cpu",
    )
    kwargs.update(overrides)
    return cls(model=_Fixed() if model is None else model, cell_graph=graph, **kwargs)


class _Log:
    """Stands in for ``LightningModule.log``: every call, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append((name, value, kwargs))

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    @property
    def values(self) -> dict[str, float]:
        return {name: float(value) for name, value, _ in self.calls}

    @property
    def batch_sizes(self) -> dict[str, Any]:
        return {name: kwargs.get("batch_size") for name, _, kwargs in self.calls}

    @property
    def sync(self) -> set[Any]:
        return {kwargs.get("sync_dist") for _, _, kwargs in self.calls}


def _record(monkeypatch: pytest.MonkeyPatch, task: Any) -> _Log:
    log = _Log()
    monkeypatch.setattr(task, "log", log)
    return log


def _attach(task: Any, tmp_path: Any, epoch: int = 0, **trainer_kwargs: Any) -> None:
    kwargs: dict[str, Any] = dict(
        accelerator="cpu",
        devices=1,
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    kwargs.update(trainer_kwargs)
    task.trainer = L.Trainer(**kwargs)
    task.trainer.fit_loop.epoch_progress.current.completed = epoch
    assert task.current_epoch == epoch


def _spy(
    monkeypatch: pytest.MonkeyPatch, collection: Any
) -> list[tuple[list[float], list[float]]]:
    """Record what a metric collection's ``update`` receives, then forward it."""
    seen: list[tuple[list[float], list[float]]] = []
    original = collection.update

    def update(preds: torch.Tensor, target: torch.Tensor) -> None:
        seen.append((preds.tolist(), target.tolist()))
        original(preds, target)

    monkeypatch.setattr(collection, "update", update)
    return seen


def _normalizer() -> COOInverseCompose:
    """The 006 inverse: a real standard normalization fitted on [0, 4] (mean 2, sd 2).

    ``denormalize(v) = v * 2 + 2`` (population sd, no eps on the way back).
    """
    dataset: Any = SimpleNamespace(
        label_df=pd.DataFrame({"index": [0, 1], "gene_interaction": [0.0, 4.0]})
    )
    norm = COOLabelNormalizationTransform(
        dataset, {"gene_interaction": {"strategy": "standard"}}
    )
    assert (
        norm.stats["gene_interaction"]["mean"],
        norm.stats["gene_interaction"]["std"],
    ) == (2.0, 2.0)
    return COOInverseCompose([norm])


# ------------------------------------------------------------------- forward, dummy loss


@pytest.mark.parametrize("cls", TASKS)
def test_forward_hands_the_model_the_cloned_cell_graph_and_returns_its_output(
    cls: Any,
) -> None:
    """``task(batch)`` is ``model(task.cell_graph, batch)``, returned as the same object.

    The cell graph was cloned in ``__init__`` (equal content, different object) and
    the device is cached from the batch. A batch with none of ``x``,
    ``perturbation_indices``, ``phenotype_values`` takes the device of the model's
    parameters.
    """
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    model = _Fixed()
    task = cls(
        model=model,
        cell_graph=graph,
        optimizer_config={"type": "AdamW", "learning_rate": 1e-2},
        lr_scheduler_config=None,
        device="cpu",
    )
    batch = _coo()
    out = task(batch)
    assert out is model.outputs[0]
    passed_graph, passed_batch = model.calls[0]
    assert passed_graph is task.cell_graph and passed_graph is not graph
    assert passed_graph["gene"].num_nodes == 4
    assert passed_batch is batch
    assert task._cell_graph_device == torch.device("cpu")
    empty = HeteroData()
    empty["gene"].num_nodes = 0
    task._cell_graph_device = torch.device("meta")
    task(empty)
    assert task._cell_graph_device == torch.device("cpu")
    assert model.calls[1][1] is empty


@pytest.mark.parametrize("cls", TASKS)
def test_unused_parameter_loss_is_zero_and_reaches_only_gradless_parameters(
    cls: Any,
) -> None:
    """``0 * sum(param)`` over trainable parameters whose ``.grad`` is None.

    Two parameters without gradients: the dummy is a tensor equal to 0.0 and backward
    gives each an exactly-zero gradient. With ``scale`` already holding a gradient of
    7, only ``extra`` is tied in, and ``scale``'s gradient stays 7. Frozen or
    all-gradient models give the integer 0 (no graph).
    """
    model = _Fixed()
    extra = nn.Parameter(torch.tensor([1.0, 2.0]))
    model.register_parameter("extra", extra)
    task = _make(cls, model)
    dummy = task._ensure_no_unused_params_loss()
    assert isinstance(dummy, torch.Tensor) and dummy.item() == 0.0
    dummy.backward()
    assert model.scale.grad is not None and model.scale.grad.item() == 0.0
    assert extra.grad is not None and extra.grad.tolist() == [0.0, 0.0]

    fresh = _Fixed()
    fresh_extra = nn.Parameter(torch.tensor([1.0, 2.0]))
    fresh.register_parameter("extra", fresh_extra)
    fresh.scale.grad = torch.tensor(7.0)
    partial = _make(cls, fresh)._ensure_no_unused_params_loss()
    partial.backward()
    assert fresh.scale.grad.item() == 7.0
    assert fresh_extra.grad is not None and fresh_extra.grad.tolist() == [0.0, 0.0]

    fresh_extra.grad = torch.zeros(2)
    assert _make(cls, fresh)._ensure_no_unused_params_loss() == 0
    frozen = _Fixed()
    frozen.scale.requires_grad_(False)
    assert _make(cls, frozen)._ensure_no_unused_params_loss() == 0


@pytest.mark.parametrize("cls", TASKS)
def test_profiling_loss_touches_every_trainable_parameter(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataloader-profiling loss is 0.0 and backward gives ``scale`` a zero
    gradient (DDP sees every parameter used); the model is never called.
    """
    model = _Fixed()
    task = _make(cls, model, execution_mode="dataloader_profiling")
    _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    loss.backward()
    assert loss.item() == 0.0
    assert model.scale.grad is not None and model.scale.grad.item() == 0.0
    assert model.calls == []


# -------------------------------------------------- RegressionTask._shared_step: losses


def test_regression_without_a_loss_function_raises_after_the_forward() -> None:
    """``loss_func=None`` raises ``ValueError("No loss function provided")``; the model
    has already run once by then.
    """
    model = _Fixed()
    task = _make(RegressionTask, model, loss_func=None)
    with pytest.raises(ValueError, match=r"^No loss function provided$"):
        task._shared_step(_coo(), 0, "train")
    assert len(model.calls) == 1


def test_logcosh_branch_returns_mean_log_cosh_and_logs_loss_and_z_p_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residuals p - y = [-1, -2, -1]: loss (2 log cosh 1 + log cosh 2) / 3 = 0.7308548.

    The z_p rows (3, 4), (0, 0), (6, 8) have norms 5, 0, 10, mean 5. Exactly two log
    calls, each with ``batch_size=3`` (the genotype count) and ``sync_dist=True``. The
    returned predictions and targets are ``[3, 1]`` columns.
    """
    task = _make(RegressionTask, loss_func=LogCoshLoss())
    log = _record(monkeypatch, task)
    loss, predictions, original = task._shared_step(_coo(), 0, "train")
    assert loss.item() == pytest.approx(LOGCOSH, rel=1e-6)
    assert predictions.tolist() == _column(P)
    assert original.tolist() == _column(Y)
    assert log.names == ["train/loss", "train/z_p_norm"]
    assert log.values == pytest.approx({"train/loss": LOGCOSH, "train/z_p_norm": 5.0})
    assert log.batch_sizes == {"train/loss": 3, "train/z_p_norm": 3}
    assert log.sync == {True}


def test_generic_loss_receives_z_p_positionally_and_no_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A loss outside the named classes is called ``(pred, target, z_p)`` with no
    keyword, even at epoch 4; without ``z_p`` it is called ``(pred, target)``. A bare
    tensor and a one-tuple are both the loss (MSE 2.0) and log no component.
    """
    loss_func = _SquaredError()
    model = _Fixed()
    task = _make(RegressionTask, model, loss_func=loss_func)
    _attach(task, tmp_path, epoch=4)
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "val")
    assert loss.item() == 2.0
    args, kwargs = loss_func.calls[0]
    assert kwargs == {}
    assert [a.tolist() for a in args] == [_column(P), _column(Y), Z_P]
    assert args[2] is model.outputs[0][1]["z_p"]
    assert log.names == ["val/loss", "val/z_p_norm"]

    single = _SquaredError(mode="single")
    no_z = _make(RegressionTask, _Fixed(z_p=None), loss_func=single)
    no_z_log = _record(monkeypatch, no_z)
    loss, _, _ = no_z._shared_step(_coo(), 0, "val")
    assert loss.item() == 2.0
    assert [len(a) for a, _ in single.calls] == [2]
    assert no_z_log.names == ["val/loss"]


def test_generic_tuple_components_log_per_key_and_per_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``(loss, dict)``: a one-element tensor logs ``.item()``, a number logs as is, a
    two-element tensor logs ``_0`` and ``_1``; an empty tensor and a string log
    nothing; a non-dict tail logs nothing. Component logs carry ``batch_size=3``.
    """
    components = {
        "one": torch.tensor(0.1),
        "vec": torch.tensor([1.0, 2.0]),
        "count": 3,
        "rate": 0.5,
        "empty": torch.tensor([]),
        "note": "text",
    }
    task = _make(RegressionTask, loss_func=_SquaredError("pair", components))
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == 2.0
    assert log.names == [
        "train/one",
        "train/vec_0",
        "train/vec_1",
        "train/count",
        "train/rate",
        "train/loss",
        "train/z_p_norm",
    ]
    assert log.values == pytest.approx(
        {
            "train/one": 0.1,
            "train/vec_0": 1.0,
            "train/vec_1": 2.0,
            "train/count": 3.0,
            "train/rate": 0.5,
            "train/loss": 2.0,
            "train/z_p_norm": 5.0,
        }
    )
    assert set(log.batch_sizes.values()) == {3}

    listed = _make(RegressionTask, loss_func=_SquaredError("pair", [1.0, 2.0]))
    listed_log = _record(monkeypatch, listed)
    listed._shared_step(_coo(), 0, "train")
    assert listed_log.names == ["train/loss", "train/z_p_norm"]


@pytest.mark.parametrize("loss_cls", [_ScriptedMleDist, _ScriptedMleWass])
def test_mle_losses_receive_the_trainer_epoch_and_z_p(
    loss_cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``MleDistSupCR`` / ``MleWassSupCR`` get ``(pred, target, z_p, epoch=4)`` at epoch
    4 and their components log like the generic branch (``vec`` element-wise).
    """
    loss_func = loss_cls({"vec": torch.tensor([1.0, 2.0]), "w": 0.25})
    task = _make(RegressionTask, loss_func=loss_func)
    _attach(task, tmp_path, epoch=4)
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == 0.5
    args, kwargs = loss_func.calls[0]
    assert kwargs == {"epoch": 4}
    assert [a.tolist() for a in args] == [_column(P), _column(Y), Z_P]
    assert log.values == pytest.approx(
        {
            "train/vec_0": 1.0,
            "train/vec_1": 2.0,
            "train/w": 0.25,
            "train/loss": 0.5,
            "train/z_p_norm": 5.0,
        }
    )


def test_real_mle_dist_supcr_components_are_all_logged_at_the_trainer_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The 006 ``mle_dist_supcr`` loss (defaults, ``embedding_dim=2``) at epoch 50.

    With 3 samples the buffered dist and SupCR terms are below their minimum sample
    counts, so the total is the weighted MSE (1 + 4 + 1) / 3 = 2.0. The epoch reaches
    the loss: the warmup is int(1000 * 0.1) = 100 epochs, so the buffer weight is
    0.1 + 0.2 * 50 / 100 = 0.2 and the exponential temperature is
    1.0 * 0.1 ** (50 / 1000) = 0.8912509. Every component of the loss's dict logs once
    under ``train/``; the one-element ``*_dim_losses`` tensors log as scalars, not
    ``_0``.
    """
    task = _make(RegressionTask, loss_func=MleDistSupCR(embedding_dim=2))
    _attach(task, tmp_path, epoch=50)
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == pytest.approx(2.0)
    components = {
        "buffer_weight",
        "temperature",
        "mse_loss",
        "mse_dim_losses",
        "weighted_mse",
        "dist_loss",
        "dist_dim_losses",
        "weighted_dist",
        "supcr_loss",
        "supcr_dim_losses",
        "weighted_supcr",
        "norm_weighted_mse",
        "norm_weighted_dist",
        "norm_weighted_supcr",
        "total_weighted",
        "total_loss",
        "norm_unweighted_mse",
        "norm_unweighted_dist",
        "norm_unweighted_supcr",
    }
    assert sorted(log.names) == sorted(
        [f"train/{k}" for k in components] + ["train/loss", "train/z_p_norm"]
    )
    values = log.values
    assert values["train/buffer_weight"] == pytest.approx(0.2)
    assert values["train/temperature"] == pytest.approx(0.1 ** (50 / 1000))
    assert values["train/mse_loss"] == pytest.approx(2.0)
    assert values["train/mse_dim_losses"] == pytest.approx(2.0)
    assert values["train/total_loss"] == pytest.approx(2.0)
    assert set(log.batch_sizes.values()) == {3}


def test_point_dist_graph_reg_gets_representations_and_drops_vector_components(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Finding (latent): the ``PointDistGraphReg`` branch logs only one-element
    tensors and numbers.

    It is called ``(pred, target, representations, epoch=3)`` with the model's own
    representations dict. Its total 0.5 is the loss: the model's ``graph_reg_loss``
    0.25 is not added again, and the task's own ``train/graph_reg_loss`` log is
    skipped (the real loss logs that key itself, see the next test). Its component
    logger (lines 302-317) skips a multi-element tensor, while the generic branch of
    the same function logs it as ``vec_0``, ``vec_1`` (lines 343-370). Latent: the
    real ``PointDistGraphReg`` returns only Python floats, so no shipped config loses
    a component; a stand-in returning a vector shows the drift. The CGT trainer
    fixed the same drift under issue #534. Pinned until the two component loggers
    are one function.
    """
    loss_func = _ScriptedPointDist(
        {"one": torch.tensor(0.1), "vec": torch.tensor([1.0, 2.0]), "count": 3}
    )
    model = _Fixed(graph_reg_loss=torch.tensor(0.25))
    task = _make(RegressionTask, model, loss_func=loss_func)
    _attach(task, tmp_path, epoch=3)
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == 0.5
    args, kwargs = loss_func.calls[0]
    assert kwargs == {"epoch": 3}
    assert args[2] is model.outputs[0][1]
    assert log.names == ["train/one", "train/count", "train/loss", "train/z_p_norm"]
    assert log.values == pytest.approx(
        {"train/one": 0.1, "train/count": 3.0, "train/loss": 0.5, "train/z_p_norm": 5.0}
    )


def test_real_point_dist_graph_reg_logs_its_fourteen_float_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real ``PointDistGraphReg`` (MSE point estimator, lambda 1; distribution term
    off with lambda 0; graph regularization lambda 1) on a model emitting
    ``graph_reg_loss`` 0.25.

    point = (1 + 4 + 1) / 3 = 2.0, graph 0.25, total 2.25; the normalized shares are
    2 / 2.25 = 8/9 and 0.25 / 2.25 = 1/9 (weighted and unweighted agree at lambda 1).
    All 14 components are Python floats and log once each under ``train/``, in the
    loss's order, then ``train/loss`` 2.25 (the graph term counted once, inside the
    loss) and ``train/z_p_norm`` 5; every log has ``batch_size=3``.
    """
    loss_func = PointDistGraphReg(
        point_estimator={"type": "mse", "lambda": 1.0},
        distribution_loss={"type": "dist", "lambda": 0.0},
        graph_regularization={"lambda": 1.0},
    )
    task = _make(
        RegressionTask, _Fixed(graph_reg_loss=torch.tensor(0.25)), loss_func=loss_func
    )
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == pytest.approx(2.25)
    expected = {
        "train/point_loss": 2.0,
        "train/weighted_point": 2.0,
        "train/dist_loss": 0.0,
        "train/weighted_dist": 0.0,
        "train/graph_reg_loss": 0.25,
        "train/weighted_graph_reg": 0.25,
        "train/norm_weighted_point": 8 / 9,
        "train/norm_weighted_dist": 0.0,
        "train/norm_weighted_graph_reg": 1 / 9,
        "train/total_loss": 2.25,
        "train/total_weighted": 2.25,
        "train/norm_unweighted_point": 8 / 9,
        "train/norm_unweighted_dist": 0.0,
        "train/norm_unweighted_graph_reg": 1 / 9,
        "train/loss": 2.25,
        "train/z_p_norm": 5.0,
    }
    assert log.names == list(expected)
    assert log.values == pytest.approx(expected, rel=1e-6)
    assert [type(v) for _, v, _ in log.calls[:14]] == [float] * 14
    assert set(log.batch_sizes.values()) == {3}


def test_real_icloss_passes_z_p_and_logs_every_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 006 ``icloss`` (``ICLoss(lambda_dist=0.1, lambda_supcr=0.001)``) through the task.

    It falls in the generic branch and is called ``(pred, target, z_p)``. The MSE
    term is (1 + 4 + 1) / 3 = 2.0; the dist and SupCR terms are taken from an
    independent call of the same stateless loss on the same three tensors, and the
    task's loss must be 2.0 + 0.1 * dist + 0.001 * supcr. All 17 components (the
    one-element ``*_dim_losses`` as scalars) log once under ``train/`` in the loss's
    order, then ``train/loss`` and ``train/z_p_norm`` 5.
    """
    loss_func = ICLoss(lambda_dist=0.1, lambda_supcr=0.001, weights=torch.ones(1))
    task = _make(RegressionTask, loss_func=loss_func)
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    _, oracle = loss_func(
        torch.tensor(_column(P)), torch.tensor(_column(Y)), torch.tensor(Z_P)
    )
    dist, supcr = float(oracle["dist_loss"]), float(oracle["supcr_loss"])
    assert float(oracle["mse_loss"]) == 2.0
    assert loss.item() == pytest.approx(2.0 + 0.1 * dist + 0.001 * supcr, rel=1e-6)
    expected = {f"train/{k}": float(v) for k, v in oracle.items()}
    expected["train/loss"] = loss.item()
    expected["train/z_p_norm"] = 5.0
    assert log.names == list(expected)
    assert len(oracle) == 17
    assert log.values == pytest.approx(expected, rel=1e-6)


def test_plain_mse_loss_cannot_take_the_z_p_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: a stock ``nn.MSELoss`` works only when the model emits no ``z_p``.

    The generic branch passes ``z_p`` as a third positional argument to any loss it
    does not name, so ``nn.MSELoss`` fails with ``TypeError`` (exact message below)
    on a model with ``z_p``, and returns the MSE 2.0 on one without. Pinned until the
    third argument is passed only to losses that declare it.
    """
    task = _make(RegressionTask, loss_func=nn.MSELoss())
    _record(monkeypatch, task)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "MSELoss.forward() takes 3 positional arguments but 4 were given"
        ),
    ):
        task._shared_step(_coo(), 0, "train")
    bare = _make(RegressionTask, _Fixed(z_p=None), loss_func=nn.MSELoss())
    _record(monkeypatch, bare)
    assert bare._shared_step(_coo(), 0, "train")[0].item() == 2.0


def test_graph_reg_loss_is_added_and_logged_for_other_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a ``graph_reg_loss`` of 0.25 the generic loss 2.0 becomes 2.25; the term is
    logged as ``train/graph_reg_loss`` before ``train/loss``.
    """
    task = _make(RegressionTask, _Fixed(graph_reg_loss=torch.tensor(0.25)))
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == 2.25
    assert log.names == ["train/graph_reg_loss", "train/loss", "train/z_p_norm"]
    assert log.values == pytest.approx(
        {"train/graph_reg_loss": 0.25, "train/loss": 2.25, "train/z_p_norm": 5.0}
    )


# ------------------------------------------- _shared_step, both tasks: logs and metrics


@pytest.mark.parametrize("cls", TASKS)
def test_gate_weights_log_their_batch_means(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asymmetric gate columns [0.1, 0.2, 0.9] and [0.9, 0.6, 0.0] average to 0.4
    (global) and 0.5 (local); their medians would be 0.2 and 0.6, so a median or
    a row-pick cannot pass. A one-column gate [0.1, 0.2, 0.9] logs only its global
    mean 0.4. Every gate log carries ``batch_size=3``.
    """
    two = _make(
        cls, _Fixed(gate_weights=torch.tensor([[0.1, 0.9], [0.2, 0.6], [0.9, 0.0]]))
    )
    log = _record(monkeypatch, two)
    two._shared_step(_coo(), 0, "test")
    gates = {k: v for k, v in log.values.items() if "gate" in k}
    assert gates == pytest.approx(
        {"test/gate_weight_global": 0.4, "test/gate_weight_local": 0.5}
    )
    assert {log.batch_sizes[k] for k in gates} == {3}
    one = _make(cls, _Fixed(gate_weights=torch.tensor([[0.1], [0.2], [0.9]])))
    one_log = _record(monkeypatch, one)
    one._shared_step(_coo(), 0, "test")
    assert {k: v for k, v in one_log.values.items() if "gate" in k} == pytest.approx(
        {"test/gate_weight_global": 0.4}
    )


@pytest.mark.parametrize("cls", TASKS)
def test_transformed_metrics_see_model_units_and_metrics_see_inverted_units(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real 006 inverse (standard normalization, mean 2, sd 2) on a val batch.

    Normalized targets y_n = [0, 1.5, -1.5] (originals 2 y_n + 2 = [2, 5, -1]) and
    predictions p = [1, 3, -2] (inverted 2 p + 2 = [4, 8, -2]). The transformed
    collection gets (p, y_n): MSE (1 + 2.25 + 0.25) / 3 = 7 / 6; the original-unit
    collection gets ([4, 8, -2], [2, 5, -1]): MSE (4 + 9 + 1) / 3 = 14 / 3, Pearson by
    numpy. The loss is computed in model units: 7 / 6 for both tasks (the generic
    squared error for ``RegressionTask``, ``F.mse_loss`` for the diffusion task's
    validation). The returned pair is the untransformed predictions next to the
    ORIGINAL-unit targets.
    """
    task = _make(cls, inverse_transform=_normalizer())
    _record(monkeypatch, task)
    transformed = _spy(monkeypatch, task.val_transformed_metrics)
    original = _spy(monkeypatch, task.val_metrics)
    loss, predictions, targets = task._shared_step(
        _coo([0.0, 1.5, -1.5], original=Y), 0, "val"
    )
    assert loss.item() == pytest.approx(7 / 6)
    assert transformed == [(P, [0.0, 1.5, -1.5])]
    assert original == [([4.0, 8.0, -2.0], Y)]
    assert predictions.tolist() == _column(P)
    assert targets.tolist() == _column(Y)
    computed = task._compute_metrics_safely(task.val_metrics)
    assert {k: v.item() for k, v in computed.items()} == pytest.approx(
        {
            "val/gene_interaction/MSE": 14 / 3,
            "val/gene_interaction/RMSE": math.sqrt(14 / 3),
            "val/gene_interaction/Pearson": float(np.corrcoef([4, 8, -2], Y)[0, 1]),
        },
        rel=1e-6,
    )
    computed = task._compute_metrics_safely(task.val_transformed_metrics)
    assert computed["val/transformed/gene_interaction/MSE"].item() == pytest.approx(
        7 / 6
    )


class _InverseSpy(nn.Module):
    """Records the COO object the task hands its inverse, then applies ``fn``."""

    def __init__(self, fn: Any) -> None:
        super().__init__()
        self.fn = fn
        self.seen: list[dict[str, Any]] = []

    def forward(self, data: HeteroData) -> HeteroData:
        gene = data["gene"]
        self.seen.append(
            {
                "phenotype_values": gene.phenotype_values.tolist(),
                "phenotype_type_indices": gene.phenotype_type_indices.tolist(),
                "phenotype_sample_indices": gene.phenotype_sample_indices.tolist(),
                "phenotype_types": gene.phenotype_types,
            }
        )
        gene.phenotype_values = self.fn(gene.phenotype_values)
        return data


@pytest.mark.parametrize("cls", TASKS)
def test_inverse_gets_a_one_type_coo_object_and_every_output_rank_is_used(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse receives the squeezed predictions [1, 3, -2], type indices [0, 0, 0],
    sample indices [0, 1, 2] and types ["gene_interaction"].

    A [3, 1] output is used as is: times ten, [10, 30, -20] against originals
    [20, 50, -10] is MSE (100 + 400 + 100) / 3 = 200. A single genotype squeezes to
    0-dim and the real normalizer returns 0-dim: 2 * 1 + 2 = 4 against 2 is MSE 4.
    """
    spy = _InverseSpy(lambda v: (v * 10).unsqueeze(1))
    column = _make(cls, inverse_transform=spy)
    _record(monkeypatch, column)
    column._shared_step(_coo(original=[20.0, 50.0, -10.0]), 0, "val")
    assert spy.seen == [
        {
            "phenotype_values": P,
            "phenotype_type_indices": [0, 0, 0],
            "phenotype_sample_indices": [0, 1, 2],
            "phenotype_types": ["gene_interaction"],
        }
    ]
    mse = column._compute_metrics_safely(column.val_metrics)["val/gene_interaction/MSE"]
    assert mse.item() == pytest.approx(200.0)

    single = _make(cls, _Fixed([1.0], z_p=None), inverse_transform=_normalizer())
    _record(monkeypatch, single)
    seen = _spy(monkeypatch, single.val_metrics)
    single._shared_step(_coo([0.0], original=[2.0]), 0, "val")
    assert seen == [([4.0], [2.0])]


@pytest.mark.parametrize("cls", TASKS)
def test_unit_mismatches_reach_the_original_unit_metrics_silently(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: model-unit predictions are scored against original-unit targets twice over.

    (a) An inverse whose ``phenotype_values`` is not a tensor is ignored (lines
    456 and 1264 test ``isinstance`` and fall through), so the metrics get the transformed
    predictions [1, 3, -2] against originals [20, 50, -10]: MSE (361 + 2209 + 64) / 3
    = 878. (b) A batch that carries ``phenotype_values_original`` (the forward
    normalization stores it) on a task built without ``inverse_transform`` scores
    the same transformed predictions against those originals. Neither raises or
    logs. The CGT trainer raises ``TypeError`` for (a) since issue #534. Pinned until
    a non-tensor inverse output and an original-unit batch without an inverse both
    raise.
    """
    listed = _make(cls, inverse_transform=_InverseSpy(lambda v: (v * 10).tolist()))
    _record(monkeypatch, listed)
    seen = _spy(monkeypatch, listed.val_metrics)
    listed._shared_step(_coo(original=[20.0, 50.0, -10.0]), 0, "val")
    assert seen == [(P, [20.0, 50.0, -10.0])]
    mse = listed._compute_metrics_safely(listed.val_metrics)["val/gene_interaction/MSE"]
    assert mse.item() == pytest.approx(878.0)

    bare = _make(cls)
    _record(monkeypatch, bare)
    bare_seen = _spy(monkeypatch, bare.val_metrics)
    bare._shared_step(_coo([0.0, 1.5, -1.5], original=Y), 0, "val")
    assert bare_seen == [(P, Y)]


@pytest.mark.parametrize("cls", TASKS)
def test_nan_targets_are_masked_per_collection_and_the_loss_is_not(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NaN in the model-unit target drops row 1 from BOTH collections (the originals
    default to the same tensor) while the loss itself is NaN: the mask is applied to
    metrics only. A NaN only in the originals drops row 1 from the original-unit
    collection alone. All-NaN targets update neither collection.
    """
    task = _make(cls)
    log = _record(monkeypatch, task)
    transformed = _spy(monkeypatch, task.val_transformed_metrics)
    original = _spy(monkeypatch, task.val_metrics)
    loss, _, _ = task._shared_step(_coo([2.0, float("nan"), -1.0]), 0, "val")
    assert math.isnan(loss.item()) and math.isnan(log.values["val/loss"])
    assert transformed == [([1.0, -2.0], [2.0, -1.0])]
    assert original == [([1.0, -2.0], [2.0, -1.0])]

    split = _make(cls, inverse_transform=_normalizer())
    _record(monkeypatch, split)
    split_t = _spy(monkeypatch, split.val_transformed_metrics)
    split_o = _spy(monkeypatch, split.val_metrics)
    split._shared_step(
        _coo([0.0, 1.5, -1.5], original=[2.0, float("nan"), -1.0]), 0, "val"
    )
    assert split_t == [(P, [0.0, 1.5, -1.5])]
    assert split_o == [([4.0, -2.0], [2.0, -1.0])]

    empty = _make(cls)
    _record(monkeypatch, empty)
    empty._shared_step(_coo([float("nan")] * 3), 0, "val")
    assert empty.val_metrics["MSE"].update_count == 0
    assert empty.val_transformed_metrics["MSE"].update_count == 0


@pytest.mark.parametrize("cls", TASKS)
def test_dense_column_and_scalar_targets_are_read_like_coo_values(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dense [3, 1] ``phenotype_values`` is used as is (loss 2.0, same metric inputs
    as the flat COO vector). A 0-dim prediction and 0-dim target both become [1, 1]:
    (1 - 2)^2 = 1.0, logged with ``batch_size=1``.
    """
    dense = _make(cls)
    _record(monkeypatch, dense)
    seen = _spy(monkeypatch, dense.test_metrics)
    loss, _, targets = dense._shared_step(_coo(_column(Y)), 0, "test")
    assert loss.item() == 2.0
    assert targets.tolist() == _column(Y)
    assert seen == [(P, Y)]

    scalar = _make(cls, _Fixed(1.0, z_p=None))
    log = _record(monkeypatch, scalar)
    loss, predictions, targets = scalar._shared_step(_coo(2.0), 0, "test")
    assert (predictions.tolist(), targets.tolist(), loss.item()) == (
        [[1.0]],
        [[2.0]],
        1.0,
    )
    assert log.batch_sizes["test/loss"] == 1


@pytest.mark.parametrize("cls", TASKS)
def test_coo_layout_is_read_positionally_not_by_sample_or_type(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``phenotype_types`` and ``phenotype_sample_indices`` are never read.

    A batch whose only phenotype is ``fitness`` is trained and scored as gene
    interaction: its values reach ``val/gene_interaction/*`` (MSE 2.0) with no error.
    The values are aligned to the prediction rows by position only, so a COO batch in
    which genotype 1 has no label (values for samples [0, 2]) fails with a broadcast
    error in the loss rather than skipping that genotype. Pinned until the task
    selects the ``gene_interaction`` entries by type index and scatters them by
    sample index.
    """
    task = _make(cls)
    _record(monkeypatch, task)
    task._shared_step(_coo(types=["fitness"]), 0, "val")
    computed = task._compute_metrics_safely(task.val_metrics)
    assert computed["val/gene_interaction/MSE"].item() == 2.0

    missing = _coo([2.0, -1.0])
    missing["gene"].phenotype_sample_indices = torch.tensor([0, 2])
    gapped = _make(cls)
    _record(monkeypatch, gapped)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The size of tensor a (3) must match the size of tensor b (2) at "
            "non-singleton dimension 0"
        ),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # F.mse_loss's broadcast note
            gapped._shared_step(missing, 0, "val")


# --------------------------------------------------------------- sample buffers


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize("stage", ["train", "val"])
def test_train_and_val_buffers_fill_on_plot_epochs_up_to_the_ceiling(
    cls: Any, stage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``plot_every_n_epochs=2``, ``plot_sample_ceiling=5``, three genotypes per batch.

    Epoch 0: (0 + 1) % 2 = 1, nothing is kept. Epoch 1: the first batch is kept whole
    (3 < 5); the second has room for 5 - 3 = 2 and keeps ``randperm(3)[:2]`` (seed 7)
    of the original targets, the inverted predictions and ``z_p``, row-aligned; the
    third finds 5 rows and keeps nothing.
    """
    task = _make(cls, plot_every_n_epochs=2, plot_sample_ceiling=5)
    _record(monkeypatch, task)
    _attach(task, tmp_path, epoch=0)
    task._shared_step(_coo(), 0, stage)
    buffer = getattr(task, f"{stage}_samples")
    assert buffer == {"true_values": [], "predictions": [], "latents": {}}

    _attach(task, tmp_path, epoch=1)
    task._shared_step(_coo(), 0, stage)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        task._shared_step(_coo(), 1, stage)
        task._shared_step(_coo(), 2, stage)
        torch.manual_seed(7)
        idx = torch.randperm(3)[:2].tolist()
    buffer = getattr(task, f"{stage}_samples")
    assert [t.tolist() for t in buffer["true_values"]] == [
        _column(Y),
        [[Y[i]] for i in idx],
    ]
    assert [t.tolist() for t in buffer["predictions"]] == [
        _column(P),
        [[P[i]] for i in idx],
    ]
    assert [t.tolist() for t in buffer["latents"]["z_p"]] == [
        Z_P,
        [Z_P[i] for i in idx],
    ]
    assert all(not t.requires_grad for t in buffer["predictions"])


@pytest.mark.parametrize("cls", TASKS)
def test_test_buffer_keeps_every_batch_regardless_of_epoch_and_ceiling(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test samples are always kept: three batches at epoch 0 with
    ``plot_every_n_epochs=2`` and ``plot_sample_ceiling=5`` give 9 rows (the ceiling
    is applied only when ``_plot_samples`` subsamples). Without ``z_p`` no latent
    list is created; train and val buffers stay empty.
    """
    task = _make(cls, _Fixed(z_p=None), plot_every_n_epochs=2, plot_sample_ceiling=5)
    _record(monkeypatch, task)
    for batch_idx in range(3):
        task._shared_step(_coo(), batch_idx, "test")
    assert [t.tolist() for t in task.test_samples["true_values"]] == [_column(Y)] * 3
    assert [t.tolist() for t in task.test_samples["predictions"]] == [_column(P)] * 3
    assert task.test_samples["latents"] == {}
    empty: dict[str, Any] = {"true_values": [], "predictions": [], "latents": {}}
    assert (task.train_samples, task.val_samples) == (empty, empty)


# ---------------------------------------------------------------- training_step


class _Opt:
    """Stand-in ``LightningOptimizer``: records ``step`` and ``zero_grad``."""

    def __init__(self, events: list[Any], lr: float = 0.05) -> None:
        self.events = events
        self.param_groups = [{"lr": lr}]

    def step(self) -> None:
        self.events.append("step")

    def zero_grad(self) -> None:
        self.events.append("zero_grad")


def _manual(monkeypatch: pytest.MonkeyPatch, task: Any) -> tuple[list[Any], _Log]:
    """Stub the optimizer, ``manual_backward`` and gradient clipping; record the order."""
    events: list[Any] = []
    opt = _Opt(events)
    monkeypatch.setattr(task, "optimizers", lambda: opt)
    monkeypatch.setattr(
        task, "manual_backward", lambda loss: events.append(("backward", loss.item()))
    )

    def clip(params: Any, max_norm: float) -> None:
        events.append(("clip", len(list(params)), max_norm))

    monkeypatch.setattr("torch.nn.utils.clip_grad_norm_", clip)
    return events, _record(monkeypatch, task)


@pytest.mark.parametrize("cls", TASKS)
def test_training_step_backward_step_zero_grad_and_learning_rate(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No accumulation: backward on the loss 2.0, one step, one zero_grad, no clip;
    ``learning_rate`` 0.05 logged with ``batch_size=3`` (``num_graphs``) and no
    ``effective_batch_size``. The returned loss is the undivided 2.0.
    """
    task = _make(cls)
    events, log = _manual(monkeypatch, task)
    loss = task.training_step(_coo(), 0)
    assert loss.item() == 2.0
    assert events == [("backward", 2.0), "step", "zero_grad"]
    assert log.names[-1] == "learning_rate"
    assert (log.values["learning_rate"], log.batch_sizes["learning_rate"]) == (0.05, 3)
    assert "effective_batch_size" not in log.names


@pytest.mark.parametrize("cls", TASKS)
def test_training_step_clips_every_parameter_before_the_step(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clip_grad_norm=True`` clips the task's one parameter at ``max_norm=0.3`` between
    backward and step.
    """
    task = _make(cls, clip_grad_norm=True, clip_grad_norm_max_norm=0.3)
    events, _ = _manual(monkeypatch, task)
    task.training_step(_coo(), 0)
    assert len(list(task.parameters())) == 1
    assert events == [("backward", 2.0), ("clip", 1, 0.3), "step", "zero_grad"]


@pytest.mark.parametrize("cls", TASKS)
def test_gradient_accumulation_divides_the_loss_and_steps_every_k_batches(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Schedule {0: 2}: k = 2 from ``__init__``. Each batch backpropagates 2.0 / 2 = 1.0
    and returns it; batch 0 does not step ((0 + 1) % 2 = 1), batch 1 does. The
    effective batch size logged is 3 * 2 * 1 = 6 with ``batch_size=3``.
    """
    task = _make(cls, grad_accumulation_schedule={0: 2})
    assert task.current_accumulation_steps == 2
    _attach(task, tmp_path)
    events, log = _manual(monkeypatch, task)
    assert task.training_step(_coo(), 0).item() == 1.0
    assert events == [("backward", 1.0)]
    assert task.training_step(_coo(), 1).item() == 1.0
    assert events == [("backward", 1.0), ("backward", 1.0), "step", "zero_grad"]
    assert log.values["effective_batch_size"] == 6.0
    assert log.batch_sizes["effective_batch_size"] == 3


@pytest.mark.parametrize("cls", TASKS)
def test_logged_batch_size_is_the_node_count_for_a_batch_with_gene_x(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Finding (issue #596, open): a batch with ``gene.x`` is sized by its node rows.

    Three genotypes of four nodes each (``gene.x`` has 12 rows): ``_shared_step`` logs
    the loss with ``batch_size=3`` (prediction rows), but ``training_step`` logs
    ``learning_rate`` with ``batch_size=12`` and an effective batch size of
    12 * 2 = 24 instead of 6 (``_get_batch_size``, lines 148-149 and 1021-1022).
    Pinned until the x branch counts graphs.
    """
    task = _make(cls, grad_accumulation_schedule={0: 2})
    _attach(task, tmp_path)
    _, log = _manual(monkeypatch, task)
    task.training_step(_coo(x_nodes=12), 1)
    assert log.batch_sizes["train/loss"] == 3
    assert log.batch_sizes["learning_rate"] == 12
    assert log.values["effective_batch_size"] == 24.0


@pytest.mark.parametrize("cls", TASKS)
def test_effective_batch_size_never_counts_ddp_ranks(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Finding: the DDP world-size factor is unreachable on Lightning 2.5.

    ``training_step`` multiplies by the world size only when
    ``trainer.strategy._strategy_name == "ddp"``; no Lightning strategy has that
    attribute (``DDPStrategy`` included), so with a DDP trainer and an initialized
    process group of 4 ranks the logged effective batch size is still 3 * 2 * 1 = 6,
    not 24. In the 006 runs this is swamped by the node-row sizing above: their
    batches carry ``gene.x``, so the logged value is node rows times the
    accumulation factor, with no rank factor. Pinned until the world size is read
    from ``trainer.world_size``.
    """
    task = _make(cls, grad_accumulation_schedule={0: 2})
    _attach(task, tmp_path, devices=2, strategy="ddp")
    assert isinstance(task.trainer.strategy, DDPStrategy)
    assert not hasattr(task.trainer.strategy, "_strategy_name")
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 4)
    _, log = _manual(monkeypatch, task)
    task.training_step(_coo(), 1)
    assert log.values["effective_batch_size"] == 6.0


@pytest.mark.parametrize("cls", TASKS)
def test_model_profiling_returns_the_loss_without_an_optimizer_step(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``execution_mode="model_profiling"``: the shared step runs (loss 2.0 logged), no
    backward, no step, no learning rate.
    """
    task = _make(cls, execution_mode="model_profiling")
    events, log = _manual(monkeypatch, task)
    assert task.training_step(_coo(), 0).item() == 2.0
    assert events == []
    assert "learning_rate" not in log.names and log.values["train/loss"] == 2.0


@pytest.mark.parametrize("cls", TASKS)
def test_validation_and_test_steps_return_the_shared_loss(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both return the stage loss 2.0 and log under their stage. Validation frees the
    CUDA cache when ``batch_idx > 0 and batch_idx % 50 == 0``: after batches 0, 49, 50
    and 100 the running count is [0, 0, 1, 2]; the test step never does.
    """
    task = _make(cls)
    log = _record(monkeypatch, task)
    emptied: list[int] = []
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: emptied.append(1))
    counts = []
    for batch_idx in (0, 49, 50, 100):
        assert task.validation_step(_coo(), batch_idx).item() == 2.0
        counts.append(len(emptied))
    assert counts == [0, 0, 1, 2]
    assert task.test_step(_coo(), 50).item() == 2.0
    assert len(emptied) == 2
    assert (log.values["val/loss"], log.values["test/loss"]) == (2.0, 2.0)


# -------------------------------------------------------- _compute_metrics_safely


class _Raises:
    """A metric whose ``compute`` raises the given exception."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def compute(self) -> torch.Tensor:
        raise self.error


@pytest.mark.parametrize("cls", TASKS)
def test_compute_metrics_safely_on_data_and_on_an_empty_epoch(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After one batch: MSE 2, RMSE sqrt 2, Pearson 15 / sqrt 228 (numpy), keyed by the
    collection prefix.

    Finding: on an epoch with no update, MSE, RMSE and Pearson in torchmetrics 1.8.2
    return NaN (with a UserWarning) instead of raising. Both swallowed messages do
    exist in torchmetrics 1.8.2 ("Needs at least two samples" in
    ``functional/regression/r2.py``, "No samples to concatenate" in
    ``utilities/data.py``), but these three metrics never reach them, so the skip
    never fires here and the epoch-end hooks log three NaNs. Pinned until empty
    metrics are skipped by ``update_count``.
    """
    task = _make(cls)
    _record(monkeypatch, task)
    task._shared_step(_coo(), 0, "train")
    computed = task._compute_metrics_safely(task.train_metrics)
    assert {k: v.item() for k, v in computed.items()} == pytest.approx(
        {
            "train/gene_interaction/MSE": 2.0,
            "train/gene_interaction/RMSE": math.sqrt(2.0),
            "train/gene_interaction/Pearson": PEARSON,
        },
        rel=1e-6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        empty = task._compute_metrics_safely(task.val_metrics)
    assert sorted(empty) == [
        "val/gene_interaction/MSE",
        "val/gene_interaction/Pearson",
        "val/gene_interaction/RMSE",
    ]
    assert all(math.isnan(v.item()) for v in empty.values())


@pytest.mark.parametrize("cls", TASKS)
def test_compute_metrics_safely_skips_two_messages_and_reraises_the_rest(
    cls: Any,
) -> None:
    """Finding: a ``ValueError`` containing "Needs at least two samples" or "No samples
    to concatenate" drops that metric from the result with no log or warning (lines
    626-637 and 1433-1444); any other ``ValueError`` and any other exception type propagate
    unchanged. Pinned until the skip is removed or logged (no-fallback rule).
    """
    task = _make(cls)
    good = MeanSquaredError()
    good.update(torch.tensor([1.0]), torch.tensor([3.0]))
    metrics = {
        "a": _Raises(ValueError("Needs at least two samples to calculate r")),
        "b": good,
        "c": _Raises(ValueError("prefix: No samples to concatenate")),
    }
    result = task._compute_metrics_safely(metrics)
    assert list(result) == ["b"] and result["b"].item() == 4.0
    with pytest.raises(ValueError, match=r"^bad input$"):
        task._compute_metrics_safely({"x": _Raises(ValueError("bad input"))})
    with pytest.raises(RuntimeError, match=r"^Needs at least two samples$"):
        task._compute_metrics_safely(
            {"x": _Raises(RuntimeError("Needs at least two samples"))}
        )


# -------------------------------------------------------------------- epoch hooks


class _Scheduler:
    def __init__(self) -> None:
        self.steps: list[tuple[Any, ...]] = []

    def step(self, *args: Any) -> None:
        self.steps.append(args)


def _metric_logs(log: _Log) -> dict[str, float]:
    return {k: v for k, v in log.values.items() if "gene_interaction" in k}


@pytest.mark.parametrize("cls", TASKS)
def test_train_epoch_end_logs_resets_plots_and_steps_the_scheduler(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, no_cuda: None
) -> None:
    """Epoch 1 with ``plot_every_n_epochs=2`` after one train batch.

    Six metric logs, no ``batch_size`` (Lightning's default) and ``sync_dist=True``:
    MSE 2, RMSE sqrt 2, Pearson 15 / sqrt 228 in both collections (no inverse, so the
    two units agree). Both collections are reset, ``_plot_samples`` gets the train
    buffer once under "train_sample" and the buffer is replaced by an empty one, and
    the scheduler is stepped once with no argument (a list from ``lr_schedulers`` is
    stepped through its first element).
    """
    task = _make(cls, plot_every_n_epochs=2)
    _attach(task, tmp_path, epoch=1)
    log = _record(monkeypatch, task)
    task._shared_step(_coo(), 0, "train")
    buffered = task.train_samples
    plotted: list[tuple[Any, str]] = []
    monkeypatch.setattr(
        task, "_plot_samples", lambda s, stage: plotted.append((s, stage))
    )
    scheduler = _Scheduler()
    monkeypatch.setattr(task, "lr_schedulers", lambda: [scheduler])
    log.calls.clear()
    task.on_train_epoch_end()
    expected = {}
    for prefix in ("train/gene_interaction/", "train/transformed/gene_interaction/"):
        expected[prefix + "MSE"] = 2.0
        expected[prefix + "RMSE"] = math.sqrt(2.0)
        expected[prefix + "Pearson"] = PEARSON
    assert _metric_logs(log) == pytest.approx(expected, rel=1e-6)
    assert {
        k: kw for k, _, kw in log.calls if "gene_interaction" in k
    } == dict.fromkeys(expected, {"sync_dist": True})
    assert task.train_metrics["MSE"].update_count == 0
    assert task.train_transformed_metrics["MSE"].update_count == 0
    assert len(plotted) == 1 and plotted[0][0] is buffered
    assert plotted[0][1] == "train_sample"
    assert task.train_samples == {"true_values": [], "predictions": [], "latents": {}}
    assert scheduler.steps == [()]


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize(
    "config", [{"factor": 0.5}, {"type": "ReduceLROnPlateau", "factor": 0.5}]
)
def test_default_plateau_scheduler_crashes_the_first_training_epoch_end(
    cls: Any,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    no_cuda: None,
) -> None:
    """Finding: the default scheduler cannot be stepped by the manual-optimization path.

    ``configure_optimizers`` builds ``ReduceLROnPlateau`` for a config without
    ``type``, with an unknown ``type``, or with ``type: "ReduceLROnPlateau"`` (the
    case the 006 ``hetero_cell_bipartite_dango_gi_mmli.yaml`` and ``_test.yaml`` and
    the 004/005 ``hetero_cell_bipartite_dango_gi.yaml`` configs hit; both spellings
    are run here), and declares the monitor
    "val/gene_interaction/MSE"; ``on_train_epoch_end`` (lines 743 and 1595)
    calls ``step()`` with no metric, so a real ``fast_dev_run`` epoch fails with
    ``TypeError``. Lightning never steps a scheduler under manual optimization, so the
    monitor is never read either. The CGT trainer steps the plateau scheduler on its
    monitor at validation end (issue #534). Pinned until this task does the same.
    """
    monkeypatch.setattr("wandb.log", lambda *a, **k: None)
    task = _make(cls, lr_scheduler_config=config)
    batches: Any = [_coo()]
    loader: DataLoader[HeteroData] = DataLoader(batches, batch_size=None)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    with pytest.raises(
        TypeError,
        match=re.escape(
            "ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'"
        ),
    ):
        trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 1


@pytest.mark.parametrize("cls", TASKS)
def test_fast_dev_run_with_the_006_warmup_scheduler(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, no_cuda: None
) -> None:
    """The 006 scheduler (``CosineAnnealingWarmupRestarts``) through one real epoch.

    Construction leaves the rate at ``min_lr`` 1e-4 (``init_lr``), which is what
    ``learning_rate`` logs during the step; ``on_train_epoch_end`` steps it once, to
    step 1 of a 2-step warmup: 1e-4 + (1e-2 - 1e-4) * 1 / 2 = 5.05e-3. The train loss
    is the pre-step 2.0 and nothing reaches wandb (``plot_every_n_epochs`` 10).
    """
    calls: list[Any] = []
    monkeypatch.setattr("wandb.log", lambda *a, **k: calls.append(a))
    task = _make(
        cls,
        lr_scheduler_config={
            "type": "CosineAnnealingWarmupRestarts",
            "first_cycle_steps": 10,
            "max_lr": 1e-2,
            "min_lr": 1e-4,
            "warmup_steps": 2,
        },
    )
    batches: Any = [_coo()]
    loader: DataLoader[HeteroData] = DataLoader(batches, batch_size=None)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
    assert metrics["learning_rate"] == pytest.approx(1e-4)
    assert metrics["train/loss"] == pytest.approx(2.0)
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(5.05e-3)
    assert calls == []


@pytest.mark.parametrize("cls", TASKS)
def test_validation_epoch_end_plots_only_outside_the_sanity_check(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Epoch 0, ``plot_every_n_epochs=1``. During the sanity check the metrics are still
    logged and reset but the buffer is kept and nothing is plotted; afterwards the
    buffer is plotted once under "val_sample" and emptied.
    """
    task = _make(cls, plot_every_n_epochs=1)
    _attach(task, tmp_path)
    log = _record(monkeypatch, task)
    plotted: list[str] = []
    monkeypatch.setattr(task, "_plot_samples", lambda s, stage: plotted.append(stage))
    task._shared_step(_coo(), 0, "val")
    task.trainer.state.stage = RunningStage.SANITY_CHECKING
    log.calls.clear()
    task.on_validation_epoch_end()
    assert _metric_logs(log)["val/gene_interaction/MSE"] == 2.0
    assert task.val_metrics["MSE"].update_count == 0
    assert plotted == [] and len(task.val_samples["true_values"]) == 1
    task.trainer.state.stage = RunningStage.VALIDATING
    task.on_validation_epoch_end()
    assert plotted == ["val_sample"]
    assert task.val_samples == {"true_values": [], "predictions": [], "latents": {}}


@pytest.mark.parametrize("cls", TASKS)
def test_test_epoch_end_logs_and_plots_the_test_buffer(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Test metrics log (MSE 2.0 in both collections), the buffer is plotted once
    under "test_sample" and emptied; an empty buffer is not plotted.
    """
    task = _make(cls)
    _attach(task, tmp_path)
    log = _record(monkeypatch, task)
    plotted: list[str] = []
    monkeypatch.setattr(task, "_plot_samples", lambda s, stage: plotted.append(stage))
    task._shared_step(_coo(), 0, "test")
    log.calls.clear()
    task.on_test_epoch_end()
    logged = _metric_logs(log)
    assert logged["test/gene_interaction/MSE"] == 2.0
    assert logged["test/transformed/gene_interaction/MSE"] == 2.0
    assert plotted == ["test_sample"]
    assert task.test_samples == {"true_values": [], "predictions": [], "latents": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task.on_test_epoch_end()
    assert plotted == ["test_sample"]


@pytest.mark.parametrize("cls", TASKS)
def test_epoch_starts_clear_buffers_only_on_plot_epochs(
    cls: Any, tmp_path: Any, no_cuda: None
) -> None:
    """``plot_every_n_epochs=2``: at epoch 0 train and val buffers are kept, at epoch 1
    they are replaced by empty ones; the test buffer is always cleared.
    """
    task = _make(cls, plot_every_n_epochs=2)
    marker = torch.tensor([[9.0]])
    for name in ("train_samples", "val_samples", "test_samples"):
        getattr(task, name)["true_values"].append(marker)
    _attach(task, tmp_path, epoch=0)
    task.on_train_epoch_start()
    task.on_validation_epoch_start()
    task.on_test_epoch_start()
    assert task.train_samples["true_values"] == [marker]
    assert task.val_samples["true_values"] == [marker]
    assert task.test_samples["true_values"] == []
    _attach(task, tmp_path, epoch=1)
    task.on_train_epoch_start()
    task.on_validation_epoch_start()
    assert task.train_samples["true_values"] == []
    assert task.val_samples["true_values"] == []


@pytest.mark.parametrize("cls", TASKS)
def test_accumulation_schedule_takes_the_last_threshold_reached(
    cls: Any, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """{0: 1, 2: 2, 10: 4} at epochs 1, 2, 5, 10, 12 gives 1, 2, 2, 4, 4.

    A threshold applies from its own epoch on (``current_epoch >= threshold``): epoch
    2 is the first at 2 and epoch 10 the first at 4, so ``>`` would give 1 and 2 at
    those boundaries. Each epoch start prints the value it chose.
    """
    task = _make(cls, grad_accumulation_schedule={0: 1, 2: 2, 10: 4})
    chosen = []
    for epoch in (1, 2, 5, 10, 12):
        _attach(task, tmp_path, epoch=epoch)
        task.on_train_epoch_start()
        chosen.append(task.current_accumulation_steps)
    assert chosen == [1, 2, 2, 4, 4]
    assert capsys.readouterr().out == "".join(
        f"Epoch {e}: Using gradient accumulation steps = {k}\n"
        for e, k in zip((1, 2, 5, 10, 12), (1, 2, 2, 4, 4), strict=True)
    )


@pytest.mark.parametrize("cls", TASKS)
def test_string_keyed_accumulation_schedule_is_ordered_lexicographically(
    cls: Any, tmp_path: Any
) -> None:
    """Finding: string thresholds are sorted as strings, then compared as integers.

    ``on_train_epoch_start`` converts each key with ``int`` for the comparison but
    iterates ``sorted(keys)``, which orders "0", "10", "2". At epoch 12 every
    threshold is reached and the LAST one visited wins, so the "2" entry sets the
    steps to 2 instead of the "10" entry's 4. ``__init__`` also looks up the int 0, so
    {"0": 3, "2": 2, "10": 4} starts at the default 1 rather than its "0" entry 3
    until the first epoch start. Pinned until the keys are converted before sorting.
    """
    task = _make(cls, grad_accumulation_schedule={"0": 3, "2": 2, "10": 4})
    assert task.current_accumulation_steps == 1
    _attach(task, tmp_path, epoch=12)
    task.on_train_epoch_start()
    assert task.current_accumulation_steps == 2


UNSPACED_SCHEDULE_CONFIGS = {
    "hetero_cell_bipartite_dango_gi_cabbi_009.yaml": {"0:16": None},
    "hetero_cell_bipartite_dango_gi_cabbi_010.yaml": {"0:16": None},
    "hetero_cell_bipartite_dango_gi_cabbi_012.yaml": {"0:16": None},
    "hetero_cell_bipartite_dango_gi_mmli_011.yaml": {"0:8": None},
    "hetero_cell_bipartite_dango_gi_mmli_013.yaml": {"0:8": None},
}


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize("name", sorted(UNSPACED_SCHEDULE_CONFIGS))
def test_unspaced_flow_mapping_schedule_crashes_the_first_epoch_start(
    cls: Any, name: str, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: ``grad_accumulation_schedule: {0:16}`` (no space) is not ``{0: 16}``.

    YAML reads ``0:16`` inside a flow mapping as one plain scalar key with a null
    value, so OmegaConf loads ``{"0:16": None}`` (``{"0:8": None}`` for the mmli
    pair). Five committed 006 configs carry it: ``hetero_cell_bipartite_dango_gi_``
    ``cabbi_009``, ``cabbi_010``, ``cabbi_012`` (``{0:16}``) and ``mmli_011``,
    ``mmli_013`` (``{0:8}``). The task then starts with 1 accumulation step (the
    ``get(0, 1)`` default) and ``on_train_epoch_start`` raises ``ValueError`` on
    ``int("0:16")``. Hypothesis, not checked against W&B: a run launched from these
    files as committed stopped at its first epoch start. Pinned until the task
    validates the schedule's keys and values at construction.
    """
    config = OmegaConf.load(
        Path(__file__).parents[3] / "experiments/006-kuzmin-tmi/conf" / name
    )
    schedule = OmegaConf.to_container(config.regression_task.grad_accumulation_schedule)
    assert schedule == UNSPACED_SCHEDULE_CONFIGS[name]
    task = _make(cls, grad_accumulation_schedule=schedule)
    assert task.current_accumulation_steps == 1
    _attach(task, tmp_path)
    key = next(iter(UNSPACED_SCHEDULE_CONFIGS[name]))
    with pytest.raises(
        ValueError, match=re.escape(f"invalid literal for int() with base 10: '{key}'")
    ):
        task.on_train_epoch_start()
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------- configure_optimizers


@pytest.mark.parametrize("cls", TASKS)
def test_no_scheduler_returns_the_bare_optimizer_over_model_and_loss_parameters(
    cls: Any,
) -> None:
    """``lr_scheduler_config=None``: an ``AdamW`` with ``lr`` renamed from
    ``learning_rate`` (1e-2) and ``weight_decay`` 1e-3 passed through, one parameter
    group holding the task's parameters, which include a learnable loss parameter.
    """
    loss_func = _SquaredError()
    w = nn.Parameter(torch.ones(()))
    loss_func.register_parameter("w", w)
    model = _Fixed()
    task = _make(
        cls,
        model,
        loss_func=loss_func,
        optimizer_config={"type": "AdamW", "learning_rate": 1e-2, "weight_decay": 1e-3},
    )
    optimizer = task.configure_optimizers()
    assert type(optimizer) is torch.optim.AdamW
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (1e-2, 1e-3)
    assert [id(p) for p in group["params"]] == [id(model.scale), id(w)]


@pytest.mark.parametrize("cls", TASKS)
def test_warmup_restart_scheduler_branch(cls: Any) -> None:
    """``CosineAnnealingWarmupRestarts`` gets every non-type key, runs per epoch with
    frequency 1 and no monitor, and leaves the rate at ``min_lr`` (its ``init_lr``
    overrides the optimizer's 1e-2).
    """
    config = {
        "type": "CosineAnnealingWarmupRestarts",
        "first_cycle_steps": 10,
        "cycle_mult": 2.0,
        "max_lr": 1e-2,
        "min_lr": 1e-5,
        "warmup_steps": 3,
        "gamma": 0.5,
    }
    out = _make(cls, lr_scheduler_config=config).configure_optimizers()
    assert sorted(out) == ["lr_scheduler", "optimizer"]
    scheduler = out["lr_scheduler"]["scheduler"]
    assert out["lr_scheduler"] == {
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
    }
    assert type(scheduler) is CosineAnnealingWarmupRestarts
    assert scheduler.optimizer is out["optimizer"]
    assert (
        scheduler.first_cycle_steps,
        scheduler.cycle_mult,
        scheduler.base_max_lr,
        scheduler.min_lr,
        scheduler.warmup_steps,
        scheduler.gamma,
    ) == (10, 2.0, 1e-2, 1e-5, 3, 0.5)
    assert out["optimizer"].param_groups[0]["lr"] == 1e-5


@pytest.mark.parametrize("cls", TASKS)
def test_cosine_annealing_scheduler_branch(cls: Any) -> None:
    """``CosineAnnealingLR`` with ``T_max`` 7 and ``eta_min`` 1e-4, per epoch, no monitor."""
    out = _make(
        cls,
        lr_scheduler_config={"type": "CosineAnnealingLR", "T_max": 7, "eta_min": 1e-4},
    ).configure_optimizers()
    scheduler = out["lr_scheduler"]["scheduler"]
    assert type(scheduler) is torch.optim.lr_scheduler.CosineAnnealingLR
    assert (scheduler.T_max, scheduler.eta_min) == (7, 1e-4)
    assert out["lr_scheduler"] == {
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
    }
    assert out["optimizer"].param_groups[0]["lr"] == 1e-2


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize(
    "config",
    [
        {"factor": 0.5, "patience": 3},
        {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 3},
    ],
)
def test_plateau_is_the_default_branch_and_monitors_val_mse(
    cls: Any, config: dict[str, Any]
) -> None:
    """No ``type`` and ``type: ReduceLROnPlateau`` build the same scheduler (factor 0.5,
    patience 3, mode "min") monitoring "val/gene_interaction/MSE" per epoch.
    """
    out = _make(cls, lr_scheduler_config=config).configure_optimizers()
    scheduler = out["lr_scheduler"]["scheduler"]
    assert type(scheduler) is ReduceLROnPlateau
    assert (scheduler.factor, scheduler.patience, scheduler.mode) == (0.5, 3, "min")
    assert out["lr_scheduler"] == {
        "scheduler": scheduler,
        "monitor": "val/gene_interaction/MSE",
        "interval": "epoch",
        "frequency": 1,
    }


@pytest.mark.parametrize("cls", TASKS)
def test_unknown_scheduler_type_silently_becomes_plateau(cls: Any) -> None:
    """Finding: any unrecognized ``type`` falls into the ``ReduceLROnPlateau`` branch.

    A misspelled "CosineAnnealingWarmupRestart" with no other keys builds a plateau
    scheduler (the ``else`` at lines 888-899) instead of raising; with its usual keys
    it fails only as an unexpected keyword of ``ReduceLROnPlateau``. Pinned until an
    unknown type raises with the valid names.
    """
    out = _make(
        cls, lr_scheduler_config={"type": "CosineAnnealingWarmupRestart"}
    ).configure_optimizers()
    assert type(out["lr_scheduler"]["scheduler"]) is ReduceLROnPlateau
    with pytest.raises(
        TypeError, match="unexpected keyword argument 'first_cycle_steps'"
    ):
        _make(
            cls,
            lr_scheduler_config={
                "type": "CosineAnnealingWarmupRestart",
                "first_cycle_steps": 10,
            },
        ).configure_optimizers()


# ------------------------------------------------------------------ _plot_samples


def _record_plots(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    """Record every ``Visualization`` call and every ``wandb.log`` payload."""
    visual: list[Any] = []
    logged: list[Any] = []

    class _Vis:
        def __init__(self, base_dir: str, max_points: int) -> None:
            visual.append(("init", base_dir, max_points))

        def visualize_model_outputs(self, *args: Any, **kwargs: Any) -> None:
            visual.append((args, kwargs))

    module = "torchcell.trainers.int_hetero_cell"
    monkeypatch.setattr(f"{module}.Visualization", _Vis)
    monkeypatch.setattr(
        f"{module}.genetic_interaction_score.box_plot",
        lambda true, pred: ("fig", true.tolist(), pred.tolist()),
    )
    monkeypatch.setattr("wandb.Image", lambda fig: ("image", fig))
    monkeypatch.setattr("wandb.log", lambda payload: logged.append(payload))
    monkeypatch.setattr(f"{module}.plt.close", lambda fig: None)
    return visual, logged


@pytest.mark.parametrize("cls", TASKS)
def test_plot_samples_hands_concatenated_columns_latents_and_box_plot(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Two chunks (two rows and one row, the first chunk 1-D) concatenate to
    true [2, 5, -1] and predictions [1, 3, -2], lifted to columns.

    ``Visualization(default_root_dir, max_points=1000)`` gets ``(predictions, true,
    {"z_p": Z}, loss name, epoch 3, None, stage=...)``. The oversmoothing log is the
    centered Frobenius norm of Z: mean (3, 4), deviations (0, 0), (-3, -4), (3, 4),
    sqrt 50 (numpy). The box plot gets the first columns.
    """
    visual, logged = _record_plots(monkeypatch)
    task = _make(cls, loss_func=LogCoshLoss())
    _attach(task, tmp_path, epoch=3)
    samples = {
        "true_values": [torch.tensor([2.0, 5.0]), torch.tensor([-1.0])],
        "predictions": [torch.tensor([1.0, 3.0]), torch.tensor([-2.0])],
        "latents": {"z_p": [torch.tensor(Z_P[:2]), torch.tensor(Z_P[2:])], "h": []},
    }
    task._plot_samples(samples, "val_sample")
    assert visual[0] == ("init", str(tmp_path), 1000)
    args, kwargs = visual[1]
    predictions, true_values, latents, loss_name, epoch, stamp = args
    assert predictions.tolist() == _column(P) and true_values.tolist() == _column(Y)
    assert list(latents) == ["z_p"] and latents["z_p"].tolist() == Z_P
    assert (loss_name, epoch, stamp, kwargs) == (
        "LogCoshLoss",
        3,
        None,
        {"stage": "val_sample"},
    )
    centered = np.array(Z_P) - np.array(Z_P).mean(axis=0)
    assert logged == [
        {
            "val_sample/oversmoothing_z_p": pytest.approx(
                float(np.linalg.norm(centered))
            )
        },
        {"val_sample/gene_interaction_box_plot": ("image", ("fig", Y, P))},
    ]
    assert float(np.linalg.norm(centered)) == pytest.approx(math.sqrt(50))


@pytest.mark.parametrize("cls", TASKS)
def test_plot_samples_subsamples_over_the_ceiling_and_skips_empty_or_all_nan(
    cls: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ceiling 2 on three rows keeps ``randperm(3)[:2]`` (seed 3) of targets,
    predictions and z_p, aligned, and passes ``max_points=2``. An empty buffer calls
    nothing. All-NaN targets draw no box plot; no ``z_p`` logs no oversmoothing.
    """
    visual, logged = _record_plots(monkeypatch)
    task = _make(cls, plot_sample_ceiling=2)
    _attach(task, tmp_path)
    samples = {
        "true_values": [torch.tensor(_column(Y))],
        "predictions": [torch.tensor(_column(P))],
        "latents": {"z_p": [torch.tensor(Z_P)]},
    }
    with torch.random.fork_rng():
        torch.manual_seed(3)
        task._plot_samples(samples, "test_sample")
        torch.manual_seed(3)
        idx = torch.randperm(3)[:2].tolist()
    assert visual[0] == ("init", str(tmp_path), 2)
    args, _ = visual[1]
    assert args[0].tolist() == [[P[i]] for i in idx]
    assert args[1].tolist() == [[Y[i]] for i in idx]
    assert args[2]["z_p"].tolist() == [Z_P[i] for i in idx]
    assert logged[1] == {
        "test_sample/gene_interaction_box_plot": (
            "image",
            ("fig", [Y[i] for i in idx], [P[i] for i in idx]),
        )
    }

    visual.clear()
    logged.clear()
    task._plot_samples({"true_values": [], "predictions": [], "latents": {}}, "x")
    assert (visual, logged) == ([], [])
    nan = {
        "true_values": [torch.tensor([[float("nan")], [float("nan")]])],
        "predictions": [torch.tensor([[1.0], [2.0]])],
        "latents": {},
    }
    task._plot_samples(nan, "x")
    assert logged == [] and visual[1][0][2] == {}


@pytest.mark.parametrize("cls", TASKS)
def test_batch_size_and_device_hparams_are_stored_and_never_read(
    cls: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``batch_size`` ("Batch size used for logging") and ``device`` are saved
    as hyperparameters and read nowhere: with ``batch_size=7`` and ``device="cuda"``
    every step still logs ``batch_size=3`` on a CPU batch and the cell graph stays on
    the CPU. Pinned until the two arguments are used or removed.
    """
    task = _make(cls, batch_size=7, device="cuda")
    assert (task.hparams["batch_size"], task.hparams["device"]) == (7, "cuda")
    log = _record(monkeypatch, task)
    task._shared_step(_coo(), 0, "val")
    assert set(log.batch_sizes.values()) == {3}
    assert task._cell_graph_device == torch.device("cpu")
