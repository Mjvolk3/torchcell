# tests/torchcell/trainers/test_int_dango.py
# [[tests.torchcell.trainers.test_int_dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_dango.py
"""``int_dango.RegressionTask`` (the 005/006 Dango trainer) on a scripted stand-in.

Fixture: ``_Scripted`` returns ``w * PRED`` (``w`` a scalar parameter, 1) and the
outputs dict the real ``Dango`` returns: ``integrated_embeddings = w * EMB`` with
``EMB = [[3, 4], [0, 1], [6, 8]]`` (row norms 5, 1, 10, mean 16 / 3) and one
reconstruction per network. It also holds an ``unused`` parameter that no output
touches, and records every call. The cell graph has three genes with the two 006 edge
types: ``string12_0_neighborhood`` 0->1, 1->2 and ``string12_0_fusion`` 2->0.

``DangoLoss`` (lambda 0.1 for neighborhood, 1.0 for fusion) closed form used below:

* neighborhood reconstruction ``[[.5, 1, 0], [0, 0, 0], [0, 0, 1]]`` against the dense
  adjacency (row = source) ``[[0, 1, 0], [0, 0, 1], [0, 0, 0]]``: edge terms 0 + 1,
  non-edge terms 0.1 * (0.25 + 1), total 1.125 / 9 = 0.125;
* fusion reconstruction all zero against the single edge (2, 0): 1 / 9;
* reconstruction loss (0.125 + 1 / 9) / 2 = 0.1180556;
* interaction loss for predictions [1, 3] against targets [2, 5]:
  (log cosh 1 + log cosh 2) / 2 = 0.8793917;
* ``LinearUntilUniform(10)``: alpha = 1 - 0.5 * e / 10 below epoch 10, then 0.5.

``self.log`` is replaced by a recorder; no ``Trainer.fit`` runs. A bare
``lightning.Trainer`` is attached only where the code reads ``current_epoch``,
``sanity_checking`` or ``default_root_dir``.
"""

import math
import re
from collections.abc import Iterator
from typing import Any

import lightning as L
import numpy as np
import numpy.typing as npt
import pytest
import torch
from lightning.pytorch.trainer.states import RunningStage
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torchmetrics import Metric, MetricCollection

from torchcell.losses.dango import DangoLoss, LinearUntilUniform, PreThenPost
from torchcell.trainers.int_dango import RegressionTask

NEIGH = "string12_0_neighborhood"
FUSION = "string12_0_fusion"
PRED = [1.0, 3.0]
TARGET = [2.0, 5.0]
EMB = [[3.0, 4.0], [0.0, 1.0], [6.0, 8.0]]
RECON_N = [[0.5, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
RECON_LOSS = (1.125 / 9 + 1 / 9) / 2
LOGCOSH = (math.log(math.cosh(1.0)) + math.log(math.cosh(2.0))) / 2
OPT_006 = {"type": "AdamW", "lr": 1e-5, "weight_decay": 1e-6}
SCHED_006 = {
    "type": "ReduceLROnPlateau",
    "mode": "min",
    "factor": 0.2,
    "patience": 3,
    "threshold": 1e-4,
    "threshold_mode": "rel",
    "cooldown": 2,
    "min_lr": 1e-9,
    "eps": 1e-10,
}
METRIC_NAMES = ["MSE", "Pearson", "RMSE"]
MASK_MESSAGE = (
    "The shape of the mask [3, 1] at index 0 does not match the shape of the indexed "
    "tensor [2, 1] at index 0"
)
SIX = {"MSE": 2.5, "Pearson": 1.0, "RMSE": math.sqrt(2.5)}


def _six(stage: str) -> dict[str, Any]:
    """The six epoch-end values for predictions [1, 3] vs targets [2, 5]: MSE
    (1 + 4) / 2 = 2.5, RMSE sqrt(2.5), Pearson 1 (two points), in both spaces.
    """
    out: dict[str, Any] = {}
    for space in ("gene_interaction", "transformed/gene_interaction"):
        for m, v in SIX.items():
            out[f"{stage}/{space}/{m}"] = pytest.approx(v, rel=1e-6)
    return out


@pytest.fixture(autouse=True)
def _isolated_rng() -> Iterator[None]:
    """Every ``torch.manual_seed`` in this file runs inside a forked RNG state."""
    with torch.random.fork_rng():
        yield


class _Scripted(nn.Module):
    """Dango-shaped outputs scaled by one trainable scalar; records each call."""

    def __init__(
        self,
        pred: torch.Tensor,
        emb: torch.Tensor | None = None,
        with_recon: bool = True,
    ) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))
        self.unused = nn.Parameter(torch.full((2,), 3.0))
        self.pred = pred
        self.emb = emb
        self.with_recon = with_recon
        self.calls: list[tuple[HeteroData, HeteroData]] = []

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls.append((cell_graph, batch))
        out: dict[str, Any] = {}
        if self.emb is not None:
            out["integrated_embeddings"] = self.w * self.emb
        if self.with_recon:
            out["reconstructions"] = {
                NEIGH: self.w * torch.tensor(RECON_N),
                FUSION: self.w * torch.zeros(3, 3),
            }
        return self.w * self.pred, out


class _Recorder:
    """Stands in for ``LightningModule.log``: every call, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, dict[str, Any]]] = []

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.calls.append((name, float(value), kwargs))

    @property
    def values(self) -> dict[str, float]:
        return {name: value for name, value, _ in self.calls}

    @property
    def batch_sizes(self) -> dict[str, Any]:
        return {name: kw.get("batch_size") for name, _, kw in self.calls}


def _cell_graph() -> HeteroData:
    graph = HeteroData()
    graph["gene"].num_nodes = 3
    graph["gene", NEIGH, "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])
    graph["gene", FUSION, "gene"].edge_index = torch.tensor([[2], [0]])
    return graph


def _batch(values: list[float], original: list[float] | None = None) -> HeteroData:
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([0, 1, 2, 0, 1, 2])
    batch["gene"].phenotype_values = torch.tensor(values)
    if original is not None:
        batch["gene"].phenotype_values_original = torch.tensor(original)
    return batch


def _dango_loss() -> DangoLoss:
    return DangoLoss(
        edge_types=[NEIGH, FUSION],
        lambda_values={NEIGH: 0.1, FUSION: 1.0},
        scheduler=LinearUntilUniform(transition_epoch=10),
    )


def _task(
    monkeypatch: pytest.MonkeyPatch,
    model: _Scripted | None = None,
    loss_func: nn.Module | None = None,
    **overrides: Any,
) -> tuple[RegressionTask, _Recorder]:
    kwargs: dict[str, Any] = dict(
        optimizer_config=dict(OPT_006),
        lr_scheduler_config=dict(SCHED_006),
        batch_size=64,
        plot_every_n_epochs=2,
        plot_sample_ceiling=10000,
        device="cpu",
    )
    kwargs.update(overrides)
    scripted = model or _Scripted(torch.tensor(PRED), torch.tensor(EMB))
    task = RegressionTask(
        model=scripted,
        cell_graph=_cell_graph(),
        loss_func=_dango_loss() if loss_func is None else loss_func,
        **kwargs,
    )
    recorder = _Recorder()
    monkeypatch.setattr(task, "log", recorder)
    return task, recorder


def _scripted(task: RegressionTask) -> _Scripted:
    model = task.model
    assert isinstance(model, _Scripted)
    return model


def _attach(task: RegressionTask, tmp_path: Any, epoch: int = 0) -> L.Trainer:
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=100,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    task.trainer = trainer
    trainer.fit_loop.epoch_progress.current.completed = epoch
    assert task.current_epoch == epoch
    return trainer


def _computed(collection: MetricCollection) -> dict[str, float]:
    return {k: float(v) for k, v in collection.compute().items()}


# --- construction and forward ------------------------------------------------------ #
def test_init_registers_six_metric_collections_and_saves_hparams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train/val/test x (original, transformed) collections of MSE, Pearson, RMSE with
    the exact prefixes; manual optimization; hparams hold every init argument except the
    model and the loss.
    """
    task, _ = _task(monkeypatch)
    for stage in ("train", "val", "test"):
        assert list(task._metrics(f"{stage}_metrics").keys()) == [
            f"{stage}/gene_interaction/{m}" for m in METRIC_NAMES
        ]
        assert list(task._metrics(f"{stage}_transformed_metrics").keys()) == [
            f"{stage}/transformed/gene_interaction/{m}" for m in METRIC_NAMES
        ]
    assert task.automatic_optimization is False
    assert task.current_accumulation_steps == 1
    assert sorted(task._hp.keys()) == [
        "batch_size",
        "cell_graph",
        "clip_grad_norm",
        "clip_grad_norm_max_norm",
        "device",
        "execution_mode",
        "forward_transform",
        "grad_accumulation_schedule",
        "inverse_transform",
        "lr_scheduler_config",
        "optimizer_config",
        "plot_every_n_epochs",
        "plot_sample_ceiling",
    ]
    assert task.train_samples == {
        "true_values": [],
        "predictions": [],
        "latents": {"integrated_embeddings": []},
    }


def test_forward_calls_model_on_cell_graph_and_keeps_only_integrated_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HeteroData batch has no ``device`` attribute, so the device comes from
    ``perturbation_indices``; the representations dict carries only
    ``integrated_embeddings`` (None when the model omits it).
    """
    task, _ = _task(monkeypatch)
    batch = _batch(TARGET)
    predictions, reps = task(batch)
    assert predictions.tolist() == PRED
    assert list(reps) == ["integrated_embeddings"]
    assert reps["integrated_embeddings"].tolist() == EMB
    assert _scripted(task).calls[0] == (task.cell_graph, batch)
    assert task._cell_graph_device == torch.device("cpu")

    bare, _ = _task(monkeypatch, model=_Scripted(torch.tensor(PRED), None))
    _, reps = bare(batch)
    assert reps == {"integrated_embeddings": None}


# --- the shared step with DangoLoss ------------------------------------------------- #
@pytest.mark.parametrize(("epoch", "alpha"), [(0, 1.0), (4, 0.8), (10, 0.5), (12, 0.5)])
def test_shared_step_dango_loss_exact_value_logs_and_epoch_weighting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, epoch: int, alpha: float
) -> None:
    """Loss = alpha * 0.1180556 + (1 - alpha) * 0.8793917 with alpha from
    ``LinearUntilUniform(10)`` at ``current_epoch``; every component is logged under
    ``val/`` with batch_size 2 (the number of predictions), then ``val/loss`` and the
    mean row norm of the integrated embeddings, 16 / 3.
    """
    task, log = _task(monkeypatch)
    _attach(task, tmp_path, epoch)
    loss, predictions, targets = task._shared_step(_batch(TARGET), 0, "val")

    expected = alpha * RECON_LOSS + (1 - alpha) * LOGCOSH
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert predictions is not None and targets is not None
    assert predictions.tolist() == [[1.0], [3.0]]
    assert targets.tolist() == [[2.0], [5.0]]
    assert [name for name, _, _ in log.calls] == [
        "val/reconstruction_loss",
        "val/interaction_loss",
        "val/weighted_reconstruction_loss",
        "val/weighted_interaction_loss",
        "val/alpha",
        "val/loss",
        "val/integrated_embeddings_norm",
    ]
    values = log.values
    assert values["val/reconstruction_loss"] == pytest.approx(RECON_LOSS, rel=1e-6)
    assert values["val/interaction_loss"] == pytest.approx(LOGCOSH, rel=1e-6)
    assert values["val/weighted_reconstruction_loss"] == pytest.approx(
        alpha * RECON_LOSS, rel=1e-6
    )
    assert values["val/weighted_interaction_loss"] == pytest.approx(
        (1 - alpha) * LOGCOSH, rel=1e-6, abs=1e-12
    )
    assert values["val/alpha"] == pytest.approx(alpha)
    assert values["val/loss"] == pytest.approx(expected, rel=1e-6)
    assert values["val/integrated_embeddings_norm"] == pytest.approx(16 / 3, rel=1e-6)
    assert set(log.batch_sizes.values()) == {2}
    assert all(kw["sync_dist"] is True for _, _, kw in log.calls)


def test_shared_step_dango_loss_reconstruction_matches_numpy_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild the reconstruction term independently: dense adjacency with row = source
    of ``edge_index``, weighted MSE with lambda on the zero entries, divided by all
    N^2 entries, averaged over the two networks. Swapping row and column (target as
    row) would give (2.225 / 9 + 1 / 9) / 2 = 0.1791667 instead of 0.1180556 for this graph.
    """
    task, log = _task(monkeypatch)
    task._shared_step(_batch(TARGET), 0, "train")

    def wmse(
        recon: npt.NDArray[np.float64], adj: npt.NDArray[np.float64], lam: float
    ) -> float:
        sq = (recon - adj) ** 2
        return float((sq[adj != 0].sum() + lam * sq[adj == 0].sum()) / adj.size)

    adj_n = np.zeros((3, 3))
    adj_n[[0, 1], [1, 2]] = 1.0
    adj_f = np.zeros((3, 3))
    adj_f[2, 0] = 1.0
    oracle = (
        wmse(np.array(RECON_N), adj_n, 0.1) + wmse(np.zeros((3, 3)), adj_f, 1.0)
    ) / 2
    transposed = (
        wmse(np.array(RECON_N), adj_n.T, 0.1) + wmse(np.zeros((3, 3)), adj_f.T, 1.0)
    ) / 2
    assert oracle == pytest.approx(RECON_LOSS)
    assert transposed == pytest.approx(0.1791667, rel=1e-6)
    assert log.values["train/reconstruction_loss"] == pytest.approx(oracle, rel=1e-6)


def test_shared_step_with_dango_loss_runs_the_model_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: with ``DangoLoss`` the step calls the model once through ``self(batch)``
    (int_dango.py:214) and again for the reconstructions (int_dango.py:262), so every
    training/validation step runs the full pretrain GNN, meta-embedding and HyperSAGNN
    twice. Values are unaffected because ``Dango`` has no dropout.
    Pinned until the reconstructions are taken from the first forward.
    """
    task, _ = _task(monkeypatch)
    batch = _batch(TARGET)
    task._shared_step(batch, 0, "train")
    assert _scripted(task).calls == [(task.cell_graph, batch), (task.cell_graph, batch)]


# --- the shared step with other losses ---------------------------------------------- #
class _TupleLoss(nn.Module):
    """Sum of squared errors; returns (loss, dict) and records its arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.args: list[tuple[torch.Tensor, ...]] = []

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        self.args.append(args)
        pred, target = args[0], args[1]
        return ((pred - target) ** 2).sum(), {"aux": torch.tensor(0.5), "note": 3.0}


class _PlainLoss(nn.Module):
    """Mean squared error returned as a bare tensor; records how many arguments it got."""

    def __init__(self) -> None:
        super().__init__()
        self.arity: list[int] = []

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        self.arity.append(len(args))
        return ((args[0] - args[1]) ** 2).mean()


def test_shared_step_generic_tuple_loss_gets_embeddings_and_logs_tensor_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-Dango loss receives (predictions [2, 1], targets [2, 1], embeddings) and
    the model runs once; (1 - 2)^2 + (3 - 5)^2 = 5; only tensor entries of its dict are
    logged (``aux`` 0.5, not ``note``).
    """
    loss_func = _TupleLoss()
    task, log = _task(monkeypatch, loss_func=loss_func)
    loss, _, _ = task._shared_step(_batch(TARGET), 0, "test")
    assert loss.item() == pytest.approx(5.0)
    pred, target, emb = loss_func.args[0]
    assert pred.tolist() == [[1.0], [3.0]] and target.tolist() == [[2.0], [5.0]]
    assert emb.tolist() == EMB
    assert len(_scripted(task).calls) == 1
    assert [n for n, _, _ in log.calls] == [
        "test/aux",
        "test/loss",
        "test/integrated_embeddings_norm",
    ]
    assert log.values["test/aux"] == 0.5


def test_shared_step_plain_loss_without_embeddings_uses_two_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No integrated embeddings: the loss is called as (pred, target), MSE 2.5, and no
    embedding norm is logged.
    """
    loss_func = _PlainLoss()
    task, log = _task(
        monkeypatch,
        model=_Scripted(torch.tensor(PRED), None, with_recon=False),
        loss_func=loss_func,
    )
    loss, _, _ = task._shared_step(_batch(TARGET), 0, "train")
    assert loss.item() == pytest.approx(2.5)
    assert loss_func.arity == [2]
    assert log.values == {"train/loss": pytest.approx(2.5)}


def test_shared_step_without_loss_function_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, _ = _task(monkeypatch)
    task.loss_func = None
    with pytest.raises(ValueError, match=re.escape("No loss function provided")):
        task._shared_step(_batch(TARGET), 0, "train")


def test_shared_step_reshapes_scalar_and_keeps_first_target_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0-dim prediction and target become [1, 1] (batch_size 1); a [2, 2] target keeps
    only its first column.
    """
    task, log = _task(
        monkeypatch,
        model=_Scripted(torch.tensor(1.5), None, with_recon=False),
        loss_func=_PlainLoss(),
    )
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([0])
    batch["gene"].phenotype_values = torch.tensor(1.0)
    loss, predictions, targets = task._shared_step(batch, 0, "train")
    assert predictions is not None and targets is not None
    assert predictions.tolist() == [[1.5]] and targets.tolist() == [[1.0]]
    assert loss.item() == pytest.approx(0.25)
    assert log.batch_sizes["train/loss"] == 1

    task2, _ = _task(
        monkeypatch,
        model=_Scripted(torch.tensor(PRED), None, with_recon=False),
        loss_func=_PlainLoss(),
    )
    wide = _batch([0.0])
    wide["gene"].phenotype_values = torch.tensor([[2.0, 9.0], [5.0, 9.0]])
    loss2, _, targets2 = task2._shared_step(wide, 0, "train")
    assert targets2 is not None and targets2.tolist() == [[2.0], [5.0]]
    assert loss2.item() == pytest.approx(2.5)


def test_shared_step_nan_target_is_masked_for_metrics_but_not_for_the_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Finding: NaN targets are removed before both metric updates
    (int_dango.py:351-356, 381-386) but the loss sees them (int_dango.py:284-290), so the
    interaction loss is NaN for any NaN target. Whether the STEP loss is NaN depends on
    the schedule: under ``LinearUntilUniform`` (the 006 schedule, here epoch 12) the
    total is alpha * recon + (1 - alpha) * NaN = NaN at every epoch (at epoch 0 it is
    0 * NaN = NaN); under ``PreThenPost`` before its transition see the next test. The
    metrics see the two finite pairs [1, 3] vs [2, 5]: MSE 2.5. Not shown reachable in
    the 006 data. Pinned until the loss is masked as the metrics are.
    """
    task, log = _task(monkeypatch, model=_Scripted(torch.tensor([1.0, 0.0, 3.0]), None))
    _attach(task, tmp_path, epoch=12)
    loss, _, _ = task._shared_step(_batch([2.0, float("nan"), 5.0]), 0, "val")
    assert math.isnan(loss.item())
    assert math.isnan(log.values["val/interaction_loss"])
    assert _computed(task._metrics("val_metrics"))["val/gene_interaction/MSE"] == 2.5
    transformed = _computed(task._metrics("val_transformed_metrics"))
    assert transformed["val/transformed/gene_interaction/MSE"] == 2.5


def test_shared_step_nan_target_under_pre_then_post_keeps_a_finite_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding (schedule dependence of the NaN-loss finding): ``PreThenPost(10)`` at
    epoch 0 (the 005 default schedule) returns total = recon and never multiplies the
    NaN interaction loss, so the step loss is the finite reconstruction loss 0.1180556,
    ``train/interaction_loss`` is NaN and ``train/weighted_interaction_loss`` is
    zeros_like(NaN) = 0. After the transition the step loss would be NaN.
    Pinned until the loss is masked as the metrics are.
    """
    loss_func = DangoLoss(
        edge_types=[NEIGH, FUSION],
        lambda_values={NEIGH: 0.1, FUSION: 1.0},
        scheduler=PreThenPost(transition_epoch=10),
    )
    task, log = _task(
        monkeypatch,
        model=_Scripted(torch.tensor([1.0, 0.0, 3.0]), None),
        loss_func=loss_func,
    )
    loss, _, _ = task._shared_step(_batch([2.0, float("nan"), 5.0]), 0, "train")
    assert math.isfinite(loss.item())
    assert loss.item() == pytest.approx(RECON_LOSS, rel=1e-6)
    nan_logs = sorted(n for n, v, _ in log.calls if math.isnan(v))
    assert nan_logs == ["train/interaction_loss"]
    assert log.values["train/weighted_interaction_loss"] == 0.0


@pytest.mark.parametrize(
    ("epoch", "alpha", "expected"),
    [(0, 1.0, RECON_LOSS), (9, 1.0, RECON_LOSS), (10, 0.0, LOGCOSH)],
)
def test_shared_step_pre_then_post_switches_terms_at_the_transition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    epoch: int,
    alpha: float,
    expected: float,
) -> None:
    """``PreThenPost(10)`` (the 005 default) through the real step: epochs 0 and 9 give
    the reconstruction loss alone (0.1180556, weighted interaction 0), epoch 10 the
    interaction loss alone (0.8793917, weighted reconstruction 0); alpha logs 1 then 0.
    """
    loss_func = DangoLoss(
        edge_types=[NEIGH, FUSION],
        lambda_values={NEIGH: 0.1, FUSION: 1.0},
        scheduler=PreThenPost(transition_epoch=10),
    )
    task, log = _task(monkeypatch, loss_func=loss_func)
    _attach(task, tmp_path, epoch)
    loss, _, _ = task._shared_step(_batch(TARGET), 0, "train")
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    values = log.values
    assert values["train/alpha"] == alpha
    assert values["train/reconstruction_loss"] == pytest.approx(RECON_LOSS, rel=1e-6)
    assert values["train/interaction_loss"] == pytest.approx(LOGCOSH, rel=1e-6)
    assert values["train/weighted_reconstruction_loss"] == pytest.approx(
        alpha * RECON_LOSS, rel=1e-6
    )
    assert values["train/weighted_interaction_loss"] == pytest.approx(
        (1 - alpha) * LOGCOSH, rel=1e-6
    )
    assert values["train/loss"] == pytest.approx(expected, rel=1e-6)


class _ConstLoss(nn.Module):
    """Ignores the shapes: 0 * sum(predictions) + 1."""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return args[0].sum() * 0.0 + 1.0


def test_count_mismatch_from_an_empty_genotype_raises_in_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding (the model-side empty-genotype finding, seen from the trainer): two
    predictions for three targets (a trailing genotype with no perturbation indices)
    make ``DangoLoss`` raise at the log-cosh broadcast, before any metric update.
    Pinned until ``HyperSAGNN`` sizes its output by the number of graphs.
    """
    task, _ = _task(monkeypatch)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The size of tensor a (2) must match the size of tensor b (3) at "
            "non-singleton dimension 0"
        ),
    ):
        task._shared_step(_batch([2.0, 5.0, 4.0]), 0, "train")


def test_count_mismatch_with_a_shape_blind_loss_raises_at_the_metric_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a loss that ignores shapes, the same mismatch reaches
    ``predictions[mask]`` (int_dango.py:355), where the [3, 1] target mask cannot index
    the [2, 1] predictions.
    """
    task, _ = _task(monkeypatch, loss_func=_ConstLoss())
    with pytest.raises(IndexError, match=re.escape(MASK_MESSAGE)):
        task._shared_step(_batch([2.0, 5.0, 4.0]), 0, "train")


def test_learning_rate_log_batch_size_counts_targets_not_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``training_step`` logs ``learning_rate`` with len(phenotype_values) while the step
    logs use predictions.size(0). They can differ only when the counts disagree; with
    all-NaN targets both metric updates are skipped, so a shape-blind loss lets the step
    finish: ``train/loss`` carries batch_size 2 and ``learning_rate`` batch_size 3.
    """
    task, log = _task(monkeypatch, loss_func=_ConstLoss())
    _wire_optimizer(monkeypatch, task)
    task.training_step(_batch([float("nan")] * 3), 0)
    sizes = log.batch_sizes
    assert sizes["train/loss"] == 2
    assert sizes["train/integrated_embeddings_norm"] == 2
    assert sizes["learning_rate"] == 3


class _Affine(nn.Module):
    """Inverse transform x -> 2 x + 1 on ``gene_interaction``; records its input."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, data: HeteroData) -> HeteroData:
        values = data["gene"]["gene_interaction"]
        self.seen.append(values.clone())
        out = HeteroData()
        out["gene"].gene_interaction = 2 * values + 1
        return out


def test_shared_step_inverse_transform_puts_metrics_in_original_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inverse transform receives the squeezed predictions [1, 3] under the key
    ``gene_interaction`` and returns [3, 7]; the original-unit metrics compare [3, 7]
    with ``phenotype_values_original`` [4, 10]: MSE (1 + 9) / 2 = 5, Pearson 1; the
    transformed metrics compare raw [1, 3] with ``phenotype_values`` [2, 5]: MSE 2.5.
    The returned tuple pairs the RAW predictions with the ORIGINAL-unit targets.
    """
    inverse = _Affine()
    task, _ = _task(monkeypatch, loss_func=_PlainLoss(), inverse_transform=inverse)
    _, predictions, targets = task._shared_step(_batch(TARGET, [4.0, 10.0]), 0, "train")
    assert inverse.seen[0].tolist() == PRED
    original = _computed(task._metrics("train_metrics"))
    assert original["train/gene_interaction/MSE"] == pytest.approx(5.0)
    assert original["train/gene_interaction/RMSE"] == pytest.approx(math.sqrt(5.0))
    assert original["train/gene_interaction/Pearson"] == pytest.approx(1.0)
    transformed = _computed(task._metrics("train_transformed_metrics"))
    assert transformed["train/transformed/gene_interaction/MSE"] == pytest.approx(2.5)
    assert predictions is not None and targets is not None
    assert predictions.tolist() == [[1.0], [3.0]]
    assert targets.tolist() == [[4.0], [10.0]]


def test_shared_step_without_inverse_both_metric_spaces_see_the_same_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No transform (as the 005/006 scripts run): both collections get [1, 3] vs [2, 5]:
    MSE 2.5, RMSE sqrt(2.5), Pearson 1.
    """
    task, _ = _task(monkeypatch)
    task._shared_step(_batch(TARGET), 0, "train")
    expected = {"MSE": 2.5, "Pearson": 1.0, "RMSE": math.sqrt(2.5)}
    original = _computed(task._metrics("train_metrics"))
    transformed = _computed(task._metrics("train_transformed_metrics"))
    for m, v in expected.items():
        assert original[f"train/gene_interaction/{m}"] == pytest.approx(v)
        assert transformed[f"train/transformed/gene_interaction/{m}"] == pytest.approx(
            v
        )


# --- profiling mode and the dummy loss ---------------------------------------------- #
def test_dataloader_profiling_returns_zero_loss_touching_every_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``execution_mode="dataloader_profiling"`` never calls the model; the loss is 0
    and its backward gives every parameter a zero gradient; two logs with batch_size =
    number of targets (3).
    """
    task, log = _task(monkeypatch, execution_mode="dataloader_profiling")
    loss, predictions, targets = task._shared_step(_batch([1.0, 2.0, 3.0]), 0, "train")
    assert (predictions, targets) == (None, None)
    assert loss.item() == 0.0
    assert _scripted(task).calls == []
    loss.backward()
    for name, p in task.model.named_parameters():
        assert p.grad is not None and torch.equal(p.grad, torch.zeros_like(p)), name
    assert log.calls == [
        ("train/dataloader_profile_loss", 0.0, {"batch_size": 3, "sync_dist": True}),
        (
            "train/dataloader_profile_batch_size",
            3.0,
            {"batch_size": 3, "sync_dist": True},
        ),
    ]


def test_unused_params_loss_is_zero_and_touches_only_gradless_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh parameters: the dummy loss is a 0.0 tensor whose backward reaches every
    parameter (``unused`` included) with a zero gradient. When ``w`` already has a
    gradient only ``unused`` is touched; when all have one it is the int 0.
    """
    task, _ = _task(monkeypatch)
    dummy = task._ensure_no_unused_params_loss()
    assert isinstance(dummy, torch.Tensor) and dummy.item() == 0.0
    dummy.backward()
    model = _scripted(task)
    assert model.w.grad is not None and model.w.grad.item() == 0.0
    assert model.unused.grad is not None and model.unused.grad.tolist() == [0.0, 0.0]

    model.zero_grad(set_to_none=True)
    model.w.grad = torch.tensor(7.0)
    partial = task._ensure_no_unused_params_loss()
    assert isinstance(partial, torch.Tensor)
    partial.backward()
    assert model.w.grad.item() == 7.0
    assert model.unused.grad is not None

    done = task._ensure_no_unused_params_loss()
    assert done == 0 and isinstance(done, int)


def test_shared_step_loss_backward_reaches_the_unused_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real step adds the dummy term, so ``unused`` gets a zero (not None) grad and
    ``w`` gets the DangoLoss gradient at epoch 0: d/dw of the recon term
    alpha * recon(w) with recon quadratic in w.
    """
    task, _ = _task(monkeypatch)
    loss, _, _ = task._shared_step(_batch(TARGET), 0, "train")
    loss.backward()
    model = _scripted(task)
    assert model.unused.grad is not None and model.unused.grad.tolist() == [0.0, 0.0]
    # recon(w): neigh ((w/2)^2 * .1 + (w - 1)^2 + 1 + .1 w^2) / 9, fusion 1 / 9; mean of
    # the two; d/dw at w = 1: (0.1 * 0.5 + 0 + 0.2) / 9 / 2 = 0.25 / 18.
    assert model.w.grad is not None
    assert model.w.grad.item() == pytest.approx(0.25 / 18, rel=1e-5)


# --- training / validation / test steps --------------------------------------------- #
class _Opt:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.param_groups = [{"lr": 0.25}]

    def step(self) -> None:
        self.events.append("step")

    def zero_grad(self) -> None:
        self.events.append("zero_grad")


def _wire_optimizer(
    monkeypatch: pytest.MonkeyPatch, task: RegressionTask
) -> tuple[list[str], list[float], list[float]]:
    events: list[str] = []
    backward: list[float] = []
    clipped: list[float] = []
    opt = _Opt(events)
    monkeypatch.setattr(task, "optimizers", lambda: opt)
    monkeypatch.setattr(
        task, "manual_backward", lambda loss: backward.append(float(loss))
    )
    monkeypatch.setattr(
        "torchcell.trainers.int_dango.nn.utils.clip_grad_norm_",
        lambda params, max_norm: clipped.append(max_norm),
    )
    return events, backward, clipped


def test_training_step_backward_clip_step_zero_grad_and_lr_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No accumulation schedule, clipping on (max_norm 10.0 as in 006): backward on the
    step loss, clip, step, zero_grad, then ``learning_rate`` 0.25 logged with
    batch_size = len(phenotype_values) = 2.
    """
    task, log = _task(
        monkeypatch,
        loss_func=_PlainLoss(),
        clip_grad_norm=True,
        clip_grad_norm_max_norm=10.0,
    )
    events, backward, clipped = _wire_optimizer(monkeypatch, task)
    loss = task.training_step(_batch(TARGET), 0)
    assert loss.item() == pytest.approx(2.5)
    assert backward == [pytest.approx(2.5)]
    assert clipped == [10.0]
    assert events == ["step", "zero_grad"]
    assert log.calls[-1] == (
        "learning_rate",
        0.25,
        {"batch_size": 2, "sync_dist": True},
    )


def test_training_step_ignores_the_accumulation_schedule_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``grad_accumulation_schedule`` is only tested for None
    (int_dango.py:450, 455); nothing sets ``current_accumulation_steps`` from it, so
    {0: 4} still divides the loss by 1 and steps on every batch. Every 005/006 Dango
    config sets it to null, so no reported run is affected.
    Pinned until the schedule is applied per epoch (as ``int_hetero_cell`` does).
    """
    task, _ = _task(
        monkeypatch, loss_func=_PlainLoss(), grad_accumulation_schedule={0: 4}
    )
    events, backward, clipped = _wire_optimizer(monkeypatch, task)
    task.training_step(_batch(TARGET), 0)
    task.training_step(_batch(TARGET), 1)
    assert task.current_accumulation_steps == 1
    assert backward == [pytest.approx(2.5), pytest.approx(2.5)]
    assert events == ["step", "zero_grad", "step", "zero_grad"]
    assert clipped == []


def test_training_step_model_profiling_skips_the_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model_profiling`` runs the real step and returns before backward/step/log."""
    task, log = _task(
        monkeypatch, loss_func=_PlainLoss(), execution_mode="model_profiling"
    )
    events, backward, _ = _wire_optimizer(monkeypatch, task)
    loss = task.training_step(_batch(TARGET), 0)
    assert loss.item() == pytest.approx(2.5)
    assert (events, backward) == ([], [])
    assert [n for n, _, _ in log.calls] == [
        "train/loss",
        "train/integrated_embeddings_norm",
    ]
    assert log.values == {
        "train/loss": pytest.approx(2.5),
        "train/integrated_embeddings_norm": pytest.approx(16 / 3, rel=1e-6),
    }


@pytest.mark.parametrize("stage", ["val", "test"])
def test_validation_and_test_steps_return_the_stage_loss(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    task, log = _task(monkeypatch, loss_func=_PlainLoss())
    step = task.validation_step if stage == "val" else task.test_step
    loss = step(_batch(TARGET), 0)
    assert loss.item() == pytest.approx(2.5)
    assert log.values[f"{stage}/loss"] == pytest.approx(2.5)
    assert _computed(task._metrics(f"{stage}_metrics"))[
        f"{stage}/gene_interaction/MSE"
    ] == pytest.approx(2.5)


# --- metric computation and epoch hooks --------------------------------------------- #
class _Raises(Metric):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def update(self) -> None:
        return None

    def compute(self) -> torch.Tensor:
        raise ValueError(self.message)


class _Const(Metric):
    def update(self) -> None:
        return None

    def compute(self) -> torch.Tensor:
        return torch.tensor(1.25)


def test_compute_metrics_safely_skips_only_the_two_known_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metrics raising "Needs at least two samples" or "No samples to concatenate" are
    dropped from the result; any other ValueError propagates.
    """
    task, _ = _task(monkeypatch)
    collection = MetricCollection(
        {
            "a": _Raises("Needs at least two samples, got 1"),
            "b": _Raises("No samples to concatenate"),
            "c": _Const(),
        }
    )
    assert {
        k: float(v) for k, v in task._compute_metrics_safely(collection).items()
    } == {"c": 1.25}
    with pytest.raises(ValueError, match=re.escape("bad shape")):
        task._compute_metrics_safely(MetricCollection({"d": _Raises("bad shape")}))


def test_compute_metrics_safely_returns_nan_on_empty_and_single_sample_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: with the installed torchmetrics no stage metric raises the two guarded
    messages (int_dango.py:489-496 is dead code): an epoch with no valid target returns
    NaN for MSE, Pearson and RMSE, and a one-sample epoch returns MSE 1 and Pearson NaN.
    These NaNs are what ``on_*_epoch_end`` would log, and val MSE is the scheduler and
    checkpoint monitor. Pinned until the guard checks the update count instead.
    """
    task, _ = _task(monkeypatch)
    empty = task._compute_metrics_safely(task._metrics("val_metrics"))
    assert sorted(empty) == [f"val/gene_interaction/{m}" for m in METRIC_NAMES]
    assert all(math.isnan(float(v)) for v in empty.values())
    single = task._metrics("val_metrics")
    single.update(torch.tensor([1.0]), torch.tensor([2.0]))
    one = {k: float(v) for k, v in task._compute_metrics_safely(single).items()}
    assert one["val/gene_interaction/MSE"] == 1.0
    assert one["val/gene_interaction/RMSE"] == 1.0
    assert math.isnan(one["val/gene_interaction/Pearson"])


def test_on_train_epoch_end_logs_both_metric_spaces_and_resets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Epoch 0 with plot_every_n_epochs 2 (no plot): six metric logs without batch_size,
    in collection order, then both collections are reset to zero updates.
    """
    task, log = _task(monkeypatch, loss_func=_PlainLoss())
    _attach(task, tmp_path, epoch=0)
    task._shared_step(_batch(TARGET), 0, "train")
    log.calls.clear()
    plotted: list[str] = []
    monkeypatch.setattr(task, "_plot_samples", lambda s, stage: plotted.append(stage))
    task.on_train_epoch_end()
    assert [(n, kw) for n, _, kw in log.calls] == [
        (f"train/gene_interaction/{m}", {"sync_dist": True}) for m in METRIC_NAMES
    ] + [
        (f"train/transformed/gene_interaction/{m}", {"sync_dist": True})
        for m in METRIC_NAMES
    ]
    assert log.values == _six("train")
    assert plotted == []
    for name in ("train_metrics", "train_transformed_metrics"):
        for metric in task._metrics(name).values():
            assert metric.update_count == 0


def test_epoch_start_hooks_reset_samples_only_on_plot_epochs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """With plot_every_n_epochs 2, epoch 0 keeps the accumulators and epoch 1 replaces
    them with empty ones, for train and val alike.
    """
    task, _ = _task(monkeypatch)
    trainer = _attach(task, tmp_path, epoch=0)
    task.train_samples["true_values"].append(torch.ones(1, 1))
    task.val_samples["true_values"].append(torch.ones(1, 1))
    task.on_train_epoch_start()
    task.on_validation_epoch_start()
    assert len(task.train_samples["true_values"]) == 1
    assert len(task.val_samples["true_values"]) == 1
    trainer.fit_loop.epoch_progress.current.completed = 1
    task.on_train_epoch_start()
    task.on_validation_epoch_start()
    empty = {
        "true_values": [],
        "predictions": [],
        "latents": {"integrated_embeddings": []},
    }
    assert task.train_samples == empty and task.val_samples == empty


def test_on_validation_epoch_end_plots_only_outside_sanity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Epoch 1 (plot epoch): during the sanity check the samples are kept and nothing is
    plotted; afterwards ``_plot_samples(val_samples, "val_sample")`` runs once and the
    accumulators are emptied. Metrics are logged in both cases.
    """
    task, log = _task(monkeypatch, loss_func=_PlainLoss())
    trainer = _attach(task, tmp_path, epoch=1)
    task._shared_step(_batch(TARGET), 0, "val")
    log.calls.clear()
    plotted: list[tuple[int, str]] = []
    monkeypatch.setattr(
        task,
        "_plot_samples",
        lambda s, stage: plotted.append((len(s["true_values"]), stage)),
    )
    trainer.state.stage = RunningStage.SANITY_CHECKING
    task.on_validation_epoch_end()
    assert plotted == [] and len(task.val_samples["true_values"]) == 1
    assert log.values == _six("val")
    for name in ("val_metrics", "val_transformed_metrics"):
        for metric in task._metrics(name).values():
            assert metric.update_count == 0

    trainer.state.stage = RunningStage.VALIDATING
    task.on_validation_epoch_end()
    assert plotted == [(1, "val_sample")]
    assert task.val_samples["true_values"] == []


# --- sample collection and plotting ------------------------------------------------- #
def test_train_sample_collection_respects_the_ceiling_and_indexes_gene_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ceiling 3, batches of 2, plot epoch: batch 1 is kept whole; batch 2 keeps
    ``randperm(2)[:1]`` (seed 0); batch 3 is dropped. The latents of batch 2 are
    ``integrated_embeddings[idx]``, i.e. a GENE row picked by a SAMPLE index (the
    embeddings are per gene, [3, 2]); batch 1 appends the whole gene table.
    """
    task, _ = _task(monkeypatch, loss_func=_PlainLoss(), plot_sample_ceiling=3)
    _attach(task, tmp_path, epoch=1)
    task._shared_step(_batch(TARGET), 0, "train")
    torch.manual_seed(0)
    task._shared_step(_batch(TARGET), 1, "train")
    task._shared_step(_batch(TARGET), 2, "train")
    torch.manual_seed(0)
    idx = torch.randperm(2)[:1]
    samples = task.train_samples
    assert [t.tolist() for t in samples["true_values"]] == [
        [[2.0], [5.0]],
        [[TARGET[int(idx)]]],
    ]
    assert [t.tolist() for t in samples["predictions"]] == [
        [[1.0], [3.0]],
        [[PRED[int(idx)]]],
    ]
    assert [t.tolist() for t in samples["latents"]["integrated_embeddings"]] == [
        EMB,
        [EMB[int(idx)]],
    ]


def test_val_samples_collected_on_plot_epochs_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """plot_every_n_epochs 2: epoch 0 collects nothing, epoch 1 appends every batch."""
    task, _ = _task(monkeypatch, loss_func=_PlainLoss())
    trainer = _attach(task, tmp_path, epoch=0)
    task._shared_step(_batch(TARGET), 0, "val")
    assert task.val_samples["true_values"] == []
    trainer.fit_loop.epoch_progress.current.completed = 1
    task._shared_step(_batch(TARGET), 0, "val")
    task._shared_step(_batch(TARGET), 1, "val")
    assert [t.tolist() for t in task.val_samples["predictions"]] == [
        [[1.0], [3.0]],
        [[1.0], [3.0]],
    ]


def _record_plots(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    visual: list[Any] = []
    logged: list[Any] = []

    class _Vis:
        def __init__(self, base_dir: str, max_points: int) -> None:
            visual.append(("init", base_dir, max_points))

        def visualize_model_outputs(self, *args: Any, **kwargs: Any) -> None:
            visual.append((args, kwargs))

    monkeypatch.setattr("torchcell.trainers.int_dango.Visualization", _Vis)
    monkeypatch.setattr(
        "torchcell.trainers.int_dango.genetic_interaction_score.box_plot",
        lambda true, pred: ("fig", true.tolist(), pred.tolist()),
    )
    monkeypatch.setattr("wandb.Image", lambda fig: ("image", fig))
    monkeypatch.setattr("wandb.log", lambda payload: logged.append(payload))
    monkeypatch.setattr("torchcell.trainers.int_dango.plt.close", lambda fig: None)
    return visual, logged


def test_plot_epoch_hands_exact_arrays_and_a_duplicated_gene_table_smoothness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Two training batches on plot epoch 1, then ``on_train_epoch_end``:
    ``Visualization(base_dir=default_root_dir, max_points=10000)`` gets predictions
    [[1], [3], [1], [3]], targets [[2], [5], [2], [5]], empty latents, the loss class
    name, epoch 1, None, stage "train_sample"; the box plot gets the first columns.

    Finding: each training step appends the whole [num_genes, H] embedding table
    (int_dango.py:422-424), so the logged ``oversmoothing_integrated_embeddings`` is the
    Frobenius norm of the table stacked once per batch: sqrt(2) * ||EMB - mean|| here
    (EMB centered: rows (0, -1/3), (-3, -10/3), (3, 11/3); squared norm 0 + 1/9 + 9 +
    100/9 + 9 + 121/9 = 42.667, so sqrt(2 * 42.667) = 9.2376), growing as
    sqrt(number of collected batches). Pinned until the latents are per sample or the
    table is logged once.
    """
    visual, logged = _record_plots(monkeypatch)
    task, _ = _task(monkeypatch)
    _attach(task, tmp_path, epoch=1)
    task._shared_step(_batch(TARGET), 0, "train")
    task._shared_step(_batch(TARGET), 1, "train")
    task.on_train_epoch_end()

    assert visual[0] == ("init", str(tmp_path), 10000)
    args, kwargs = visual[1]
    assert args[0].tolist() == [[1.0], [3.0], [1.0], [3.0]]
    assert args[1].tolist() == [[2.0], [5.0], [2.0], [5.0]]
    assert args[2:] == ({}, "DangoLoss", 1, None)
    assert kwargs == {"stage": "train_sample"}
    centered = np.array(EMB) - np.array(EMB).mean(axis=0)
    single = float(np.linalg.norm(centered))
    assert single**2 == pytest.approx(128 / 3)
    assert logged[0] == {
        "train_sample/oversmoothing_integrated_embeddings": pytest.approx(
            math.sqrt(2) * single, rel=1e-6
        )
    }
    assert logged[1] == {
        "train_sample/gene_interaction_box_plot": (
            "image",
            ("fig", [2.0, 5.0, 2.0, 5.0], [1.0, 3.0, 1.0, 3.0]),
        )
    }
    assert len(logged) == 2
    assert task.train_samples == {
        "true_values": [],
        "predictions": [],
        "latents": {"integrated_embeddings": []},
    }


def test_plot_samples_noop_on_empty_and_skips_box_plot_for_all_nan_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Empty samples: nothing is constructed or logged. All-NaN targets and no latents:
    the scatter is drawn but neither wandb payload is logged.
    """
    visual, logged = _record_plots(monkeypatch)
    task, _ = _task(monkeypatch)
    _attach(task, tmp_path, epoch=1)
    task._plot_samples({"true_values": [], "predictions": []}, "val_sample")
    assert (visual, logged) == ([], [])
    task._plot_samples(
        {
            "true_values": [torch.tensor([float("nan"), float("nan")])],
            "predictions": [torch.tensor([1.0, 2.0])],
            "latents": {"integrated_embeddings": []},
        },
        "val_sample",
    )
    args, kwargs = visual[1]
    assert args[0].tolist() == [[1.0], [2.0]]
    assert kwargs == {"stage": "val_sample"}
    assert logged == []


# --- optimizer ---------------------------------------------------------------------- #
def test_configure_optimizers_matches_the_006_config_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AdamW(lr 1e-5, weight_decay 1e-6) over the model parameters and
    ReduceLROnPlateau with every 006 field, monitored on val/gene_interaction/MSE per
    epoch.
    """
    task, _ = _task(monkeypatch)
    config = task.configure_optimizers()
    optimizer = config["optimizer"]
    assert type(optimizer) is torch.optim.AdamW
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (1e-5, 1e-6)
    assert [id(p) for p in group["params"]] == [id(p) for p in task.parameters()]
    sched_cfg = config["lr_scheduler"]
    assert isinstance(sched_cfg, dict)
    assert {k: v for k, v in sched_cfg.items() if k != "scheduler"} == {
        "monitor": "val/gene_interaction/MSE",
        "interval": "epoch",
        "frequency": 1,
    }
    scheduler = sched_cfg["scheduler"]
    assert type(scheduler) is ReduceLROnPlateau
    assert (
        scheduler.mode,
        scheduler.factor,
        scheduler.patience,
        scheduler.threshold,
        scheduler.threshold_mode,
        scheduler.cooldown,
        scheduler.min_lrs,
        scheduler.eps,
    ) == ("min", 0.2, 3, 1e-4, "rel", 2, [1e-9], 1e-10)


def test_configure_optimizers_renames_learning_rate_and_ignores_scheduler_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``learning_rate`` becomes ``lr``. Finding: ``lr_scheduler_config["type"]`` is
    dropped (int_dango.py:660-663); a "CosineAnnealingLR" config still builds a
    ReduceLROnPlateau from the remaining keys. Every 005/006 Dango config says
    ReduceLROnPlateau, so no reported run is affected.
    Pinned until the type selects the scheduler or an unknown type is refused.
    """
    task, _ = _task(
        monkeypatch,
        optimizer_config={"type": "SGD", "learning_rate": 0.5},
        lr_scheduler_config={"type": "CosineAnnealingLR", "mode": "max"},
    )
    config = task.configure_optimizers()
    optimizer = config["optimizer"]
    assert type(optimizer) is torch.optim.SGD
    assert optimizer.param_groups[0]["lr"] == 0.5
    sched_cfg = config["lr_scheduler"]
    assert isinstance(sched_cfg, dict)
    scheduler = sched_cfg["scheduler"]
    assert type(scheduler) is ReduceLROnPlateau and scheduler.mode == "max"


def test_plot_samples_subsamples_to_the_ceiling_with_one_shared_permutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ceiling 2, three collected samples: one ``randperm(3)[:2]`` (seed 0) indexes the
    targets, the predictions and the latents alike, and ``max_points`` is the ceiling.
    """
    visual, logged = _record_plots(monkeypatch)
    task, _ = _task(monkeypatch, plot_sample_ceiling=2)
    _attach(task, tmp_path, epoch=1)
    latents = torch.tensor([[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    torch.manual_seed(0)
    task._plot_samples(
        {
            "true_values": [torch.tensor([[10.0], [20.0]]), torch.tensor([[30.0]])],
            "predictions": [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])],
            "latents": {"integrated_embeddings": [latents]},
        },
        "train_sample",
    )
    torch.manual_seed(0)
    idx = torch.randperm(3)[:2].tolist()
    assert visual[0] == ("init", str(tmp_path), 2)
    args, _ = visual[1]
    assert args[0][:, 0].tolist() == [[1.0, 2.0, 3.0][i] for i in idx]
    assert args[1][:, 0].tolist() == [[10.0, 20.0, 30.0][i] for i in idx]
    picked = latents[idx].double().numpy()
    expected = float(np.linalg.norm(picked - picked.mean(axis=0)))
    assert logged[0] == {
        "train_sample/oversmoothing_integrated_embeddings": pytest.approx(
            expected, rel=1e-6
        )
    }


def test_inverse_transform_on_a_single_genotype_returns_one_by_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch of one: the squeezed prediction is 0-dim, the inverse 2 x + 1 maps 1.5 to
    4.0 and the metrics see [[4.0]] against the original target 4.5 (MSE 0.25).
    """
    inverse = _Affine()
    task, _ = _task(
        monkeypatch,
        model=_Scripted(torch.tensor([1.5]), None, with_recon=False),
        loss_func=_PlainLoss(),
        inverse_transform=inverse,
    )
    task._shared_step(_batch([1.0], [4.5]), 0, "test")
    assert inverse.seen[0].dim() == 0 and inverse.seen[0].item() == 1.5
    metrics = task._metrics("test_metrics")
    mse = metrics["test/gene_interaction/MSE"]
    assert float(mse.compute()) == pytest.approx(0.25)
