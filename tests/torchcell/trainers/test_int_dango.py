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
from torch_geometric.data import Batch, HeteroData
from torchmetrics import Metric, MetricCollection

from torchcell.losses.dango import DangoLoss, LinearUntilUniform, PreThenPost
from torchcell.models.dango import Dango
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
SIX = {"MSE": 2.5, "Pearson": 1.0, "RMSE": math.sqrt(2.5)}
EMPTY_SAMPLES: dict[str, Any] = {
    "true_values": [],
    "predictions": [],
    "integrated_embeddings": None,
}


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
    assert task.train_samples == EMPTY_SAMPLES
    assert task.val_samples == EMPTY_SAMPLES


def test_init_refuses_an_accumulation_schedule_and_a_scheduler_it_does_not_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #616 item 6. ``grad_accumulation_schedule`` was only tested for None, so
    {0: 4} still stepped on every batch; and ``lr_scheduler_config["type"]`` was
    dropped, so "CosineAnnealingLR" built a ReduceLROnPlateau. Both are now refused at
    construction with the full message (a missing type is refused too). Every 005/006
    Dango config sets the schedule to null or omits it and names ReduceLROnPlateau, so
    no config changes behavior.
    """
    with pytest.raises(ValueError) as accumulation:
        _task(monkeypatch, grad_accumulation_schedule={0: 4})
    assert str(accumulation.value) == (
        "int_dango.RegressionTask does not implement gradient accumulation; "
        "grad_accumulation_schedule must be None, got {0: 4}"
    )
    with pytest.raises(ValueError) as cosine:
        _task(monkeypatch, lr_scheduler_config={"type": "CosineAnnealingLR"})
    assert str(cosine.value) == (
        "int_dango.RegressionTask builds only ReduceLROnPlateau; "
        "lr_scheduler_config type is 'CosineAnnealingLR'"
    )
    with pytest.raises(ValueError) as missing:
        _task(monkeypatch, lr_scheduler_config={"mode": "min"})
    assert str(missing.value) == (
        "int_dango.RegressionTask builds only ReduceLROnPlateau; "
        "lr_scheduler_config type is None"
    )


def test_forward_calls_model_once_and_returns_its_outputs_dict_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HeteroData batch has no ``device`` attribute, so the device comes from
    ``perturbation_indices``; the second element is the model's own outputs dict (the
    same object, with ``reconstructions``), so the step needs no second forward.
    """
    task, _ = _task(monkeypatch)
    batch = _batch(TARGET)
    returned: list[dict[str, Any]] = []
    model = _scripted(task)
    model.register_forward_hook(lambda m, i, o: returned.append(o[1]))
    predictions, reps = task(batch)
    assert predictions.tolist() == PRED
    assert reps is returned[0]
    assert sorted(reps) == ["integrated_embeddings", "reconstructions"]
    assert reps["integrated_embeddings"].tolist() == EMB
    assert reps["reconstructions"][NEIGH].tolist() == RECON_N
    assert model.calls == [(task.cell_graph, batch)]
    assert task._cell_graph_device == torch.device("cpu")

    bare, _ = _task(
        monkeypatch, model=_Scripted(torch.tensor(PRED), None, with_recon=False)
    )
    _, reps = bare(batch)
    assert reps == {}


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


def _real_cell_graph() -> HeteroData:
    """The four-gene graph of tests/torchcell/models/test_dango.py."""
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    graph["gene", NEIGH, "gene"].edge_index = torch.tensor(
        [[0, 1, 1, 2, 3], [1, 0, 2, 1, 2]]
    )
    graph["gene", FUSION, "gene"].edge_index = torch.tensor([[0, 3], [3, 0]])
    return graph


def _real_batch() -> Batch:
    """Triples [0, 1, 2], [1, 2, 3] and the pair [0, 3]; targets 0.1, -0.2, 0.05."""
    data = []
    for genes, value in (([0, 1, 2], 0.1), ([1, 2, 3], -0.2), ([0, 3], 0.05)):
        genotype = HeteroData()
        genotype["gene"].num_nodes = 4
        genotype["gene"].perturbation_indices = torch.tensor(genes)
        genotype["gene"].phenotype_values = torch.tensor([value])
        data.append(genotype)
    return Batch.from_data_list(data, follow_batch=["perturbation_indices"])


def _real_dango() -> Dango:
    torch.manual_seed(0)
    return Dango(gene_num=4, edge_types=[NEIGH, FUSION], hidden_channels=8, num_heads=2)


def _dense(graph: HeteroData, edge_type: str) -> torch.Tensor:
    adj = torch.zeros(4, 4)
    index = graph["gene", edge_type, "gene"].edge_index
    adj[index[0], index[1]] = 1.0
    return adj


def test_shared_step_with_dango_loss_runs_the_model_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Issue #616 item 1. The step used to call the model through ``self(batch)`` and
    again for the reconstructions, because ``forward`` returned only
    ``integrated_embeddings`` and the ``DangoLoss`` branch needed ``reconstructions``
    (both introduced together in 9f995c828, "dango works"). It now calls it once:
    the scripted model records one call, and a forward hook on the real ``Dango``
    with ``DangoLoss`` fires once per step.
    """
    task, _ = _task(monkeypatch)
    batch = _batch(TARGET)
    task._shared_step(batch, 0, "train")
    assert _scripted(task).calls == [(task.cell_graph, batch)]

    model = _real_dango()
    hooks: list[int] = []
    model.register_forward_hook(lambda m, i, o: hooks.append(1))
    real, _ = _task(monkeypatch, model=None, loss_func=_dango_loss())
    real.model = model
    real.cell_graph = _real_cell_graph()
    real._shared_step(_real_batch(), 0, "train")
    assert hooks == [1]


# Captured from the pre-fix code (main at 4a179a2e1, two forwards per step) by running
# this file's real-Dango step in a scratch script and printing float.hex of every value.
BEFORE_LUU_E4 = {
    "train/reconstruction_loss": 0.22111916542053223,
    "train/interaction_loss": 0.008982975035905838,
    "train/weighted_reconstruction_loss": 0.17689533531665802,
    "train/weighted_interaction_loss": 0.0017965950537472963,
    "train/alpha": 0.800000011920929,
    "train/loss": 0.17869192361831665,
    "train/integrated_embeddings_norm": 0.4157995879650116,
}
BEFORE_METRICS = {
    "MSE": 0.01808621548116207,
    "Pearson": 0.3456900715827942,
    "RMSE": 0.13448500633239746,
}


@pytest.mark.parametrize(
    ("scheduler", "epoch", "alpha"),
    [(LinearUntilUniform(10), 4, 0.8), (PreThenPost(10), 12, 0.0)],
)
def test_one_forward_step_equals_the_two_forward_protocol_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    scheduler: LinearUntilUniform | PreThenPost,
    epoch: int,
    alpha: float,
) -> None:
    """Issue #616 item 1: removing the second forward changes no value. The real seeded
    ``Dango`` (G 4, H 8, two heads) is stepped once with ``DangoLoss``; the old protocol
    is replayed beside it on an identically seeded model (predictions from call 1,
    reconstructions from call 2). The loss and every logged component are EQUAL
    (``==`` on float32), and so are all six epoch metrics. ``Dango`` has no dropout, so
    the second call's reconstructions were bit-identical to the first's.

    Pinned pre-fix values (``LinearUntilUniform(10)`` at epoch 4, captured from the
    two-forward code, ``BEFORE_LUU_E4``): loss 0.17869192361831665
    (``0x1.6df608p-3``) = 0.8 * recon 0.22111916542053223 + 0.2 * log-cosh
    0.008982975035905838; MSE 0.01808621548116207, Pearson 0.3456900715827942. Under
    ``PreThenPost(10)`` at epoch 12 the loss is the log-cosh term alone.

    Gradients are equal up to float32 summation order only: with both terms active the
    shared GNN now accumulates both upstream gradients before backpropagating once,
    instead of backpropagating each forward graph separately; the largest difference
    measured on this step was 9.3e-10 (LinearUntilUniform) and 0 (PreThenPost, one
    term). No reported value depends on it.
    """
    loss_func = DangoLoss(
        edge_types=[NEIGH, FUSION],
        lambda_values={NEIGH: 0.1, FUSION: 1.0},
        scheduler=scheduler,
    )
    graph = _real_cell_graph()
    batch = _real_batch()
    task, log = _task(monkeypatch, loss_func=loss_func)
    task.model = _real_dango()
    task.cell_graph = graph
    _attach(task, tmp_path, epoch)
    loss, _, _ = task._shared_step(batch, 0, "train")

    reference = _real_dango()
    first, _ = reference(graph, batch)
    _, second = reference(graph, batch)
    targets = batch["gene"].phenotype_values.view(-1, 1)
    ref_loss, ref_parts = loss_func(
        first.view(-1, 1),
        targets,
        second["reconstructions"],
        {e: _dense(graph, e) for e in (NEIGH, FUSION)},
        current_epoch=epoch,
    )
    norm = second["integrated_embeddings"].norm(p=2, dim=-1).mean()

    assert loss.item() == ref_loss.item()
    expected = {f"train/{k}": float(v) for k, v in ref_parts.items()}
    expected["train/loss"] = ref_loss.item()
    expected["train/integrated_embeddings_norm"] = norm.item()
    assert log.values == expected
    assert log.values["train/alpha"] == pytest.approx(alpha)

    ref_mse = float(((first - targets.view(-1)) ** 2).mean())
    for space in ("gene_interaction", "transformed/gene_interaction"):
        computed = {
            k.rsplit("/", 1)[1]: v
            for k, v in _computed(
                task._metrics(
                    "train_metrics"
                    if space == "gene_interaction"
                    else "train_transformed_metrics"
                )
            ).items()
        }
        assert computed["MSE"] == ref_mse
        assert computed == BEFORE_METRICS

    if epoch == 4:
        assert log.values == BEFORE_LUU_E4
    else:
        assert log.values["train/loss"] == BEFORE_LUU_E4["train/interaction_loss"]

    loss.backward()
    ref_loss.backward()
    for (name, p), (_, q) in zip(
        task.model.named_parameters(), reference.named_parameters(), strict=True
    ):
        # The step's zero-weight dummy term gives parameters outside the active loss
        # (the reconstruction heads under PreThenPost after the transition) a zero
        # gradient where the bare reference has None.
        assert p.grad is not None, name
        ref_grad = torch.zeros_like(p) if q.grad is None else q.grad
        torch.testing.assert_close(p.grad, ref_grad, atol=1e-8, rtol=0.0)


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


@pytest.mark.parametrize(
    ("scheduler", "epoch", "alpha"),
    [
        (LinearUntilUniform(10), 12, 0.5),
        (LinearUntilUniform(10), 0, 1.0),
        (PreThenPost(10), 0, 1.0),
        (PreThenPost(10), 10, 0.0),
    ],
)
def test_shared_step_masks_nan_targets_before_the_loss_under_both_schedules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    scheduler: LinearUntilUniform | PreThenPost,
    epoch: int,
    alpha: float,
) -> None:
    """Issue #616 item 3. Predictions [1, 0, 3] against targets [2, NaN, 5]: the NaN
    pair is dropped before the loss with the mask the transformed metrics use, so the
    interaction loss is the log-cosh of the two finite pairs, 0.8793917, under both
    schedules (it used to be NaN, making the ``LinearUntilUniform`` step loss NaN at
    every epoch and the ``PreThenPost`` loss NaN after its transition). Step loss =
    alpha * 0.1180556 + (1 - alpha) * 0.8793917. Both metric spaces see the same two
    pairs: MSE 2.5. Logs keep batch_size 3, the genotype count.
    """
    loss_func = DangoLoss(
        edge_types=[NEIGH, FUSION],
        lambda_values={NEIGH: 0.1, FUSION: 1.0},
        scheduler=scheduler,
    )
    task, log = _task(
        monkeypatch,
        model=_Scripted(torch.tensor([1.0, 0.0, 3.0]), None),
        loss_func=loss_func,
    )
    _attach(task, tmp_path, epoch)
    loss, _, _ = task._shared_step(_batch([2.0, float("nan"), 5.0]), 0, "val")
    expected = alpha * RECON_LOSS + (1 - alpha) * LOGCOSH
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    values = log.values
    assert values["val/interaction_loss"] == pytest.approx(LOGCOSH, rel=1e-6)
    assert values["val/weighted_interaction_loss"] == pytest.approx(
        (1 - alpha) * LOGCOSH, rel=1e-6, abs=1e-12
    )
    assert values["val/loss"] == pytest.approx(expected, rel=1e-6)
    assert values["val/alpha"] == pytest.approx(alpha)
    assert set(log.batch_sizes.values()) == {3}
    assert _computed(task._metrics("val_metrics"))["val/gene_interaction/MSE"] == 2.5
    transformed = _computed(task._metrics("val_transformed_metrics"))
    assert transformed["val/transformed/gene_interaction/MSE"] == 2.5


def test_shared_step_generic_loss_also_receives_only_finite_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mask applies to every loss: a plain MSE gets the [2, 1] finite pairs
    [[1], [3]] vs [[2], [5]] and returns 2.5 (it would be NaN unmasked).
    """
    loss_func = _TupleLoss()
    task, _ = _task(
        monkeypatch,
        model=_Scripted(torch.tensor([1.0, 0.0, 3.0]), torch.tensor(EMB)),
        loss_func=loss_func,
    )
    loss, _, _ = task._shared_step(_batch([2.0, float("nan"), 5.0]), 0, "train")
    assert loss.item() == 5.0
    pred, target, emb = loss_func.args[0]
    assert pred.tolist() == [[1.0], [3.0]] and target.tolist() == [[2.0], [5.0]]
    assert emb.tolist() == EMB


def test_shared_step_refuses_a_batch_whose_targets_are_all_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch with no finite target has no loss; it is refused by name instead of
    yielding a NaN (or, under ``PreThenPost``, a reconstruction-only) step loss.
    """
    task, log = _task(monkeypatch, model=_Scripted(torch.tensor([1.0, 0.0, 3.0]), None))
    with pytest.raises(ValueError) as excinfo:
        task._shared_step(_batch([float("nan")] * 3), 7, "train")
    assert str(excinfo.value) == (
        "train batch 7: all 3 targets are NaN, so the loss and metrics are undefined"
    )
    assert log.calls == []


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


@pytest.mark.parametrize("loss_kind", ["dango", "shape_blind"])
def test_shared_step_refuses_fewer_predictions_than_targets(
    monkeypatch: pytest.MonkeyPatch, loss_kind: str
) -> None:
    """Issue #616 item 2 at the trainer boundary: two predictions for three targets
    (what a trailing empty genotype used to produce) are refused by name before any
    loss or metric, whatever the loss. Previously ``DangoLoss`` raised a broadcast
    error, and a shape-blind loss either hit an IndexError at the metric mask or, with
    all-NaN targets, finished the step with mismatched batch sizes in the logs.
    """
    loss_func = _dango_loss() if loss_kind == "dango" else _ConstLoss()
    task, log = _task(monkeypatch, loss_func=loss_func)
    with pytest.raises(ValueError) as excinfo:
        task._shared_step(_batch([2.0, 5.0, 4.0]), 3, "val")
    assert str(excinfo.value) == (
        "val batch 3: the model returned 2 predictions for 3 targets"
    )
    assert log.calls == []
    assert task._metrics("val_metrics")["val/gene_interaction/MSE"].update_count == 0


def test_learning_rate_and_step_logs_share_the_genotype_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``training_step`` logs ``learning_rate`` with len(phenotype_values) and the step
    logs use predictions.size(0); the boundary check makes them equal: 3 genotypes,
    one with a NaN target, log batch_size 3 everywhere.
    """
    task, log = _task(
        monkeypatch,
        model=_Scripted(torch.tensor([1.0, 0.0, 3.0]), torch.tensor(EMB)),
        loss_func=_ConstLoss(),
    )
    _wire_optimizer(monkeypatch, task)
    task.training_step(_batch([2.0, float("nan"), 5.0]), 0)
    assert log.batch_sizes == {
        "train/loss": 3,
        "train/integrated_embeddings_norm": 3,
        "learning_rate": 3,
    }


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


def test_training_step_steps_the_optimizer_on_every_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No accumulation (a schedule is refused at construction): each batch backpropagates
    the undivided loss 2.5 and steps once; clipping off means no clip call.
    """
    task, _ = _task(monkeypatch, loss_func=_PlainLoss())
    events, backward, clipped = _wire_optimizer(monkeypatch, task)
    task.training_step(_batch(TARGET), 0)
    task.training_step(_batch(TARGET), 1)
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


def test_log_epoch_metrics_propagates_every_metric_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #616 item 4. ``_compute_metrics_safely`` swallowed two torchmetrics messages
    that torchmetrics 1.8.2 never raises; it is gone, and nothing is swallowed: a
    metric raising "Needs at least two samples, got 1" propagates (the collection
    orders metrics by name, so it is computed first and nothing is logged).
    """
    task, log = _task(monkeypatch)
    task.val_metrics = MetricCollection(
        {"c": _Const(), "a": _Raises("Needs at least two samples, got 1")}
    )
    task._metrics("val_metrics").update()
    with pytest.raises(
        ValueError, match=re.escape("Needs at least two samples, got 1")
    ):
        task._log_epoch_metrics("val_metrics")
    assert log.calls == []


def test_log_epoch_metrics_refuses_an_empty_epoch_and_logs_nan_pearson_for_one_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #616 item 4. An epoch with no finite-target sample used to log NaN for MSE,
    RMSE and Pearson, and validation MSE is the scheduler and checkpoint monitor; it is
    now refused by name, before anything is logged. A one-sample epoch logs MSE 1 and
    RMSE 1 for prediction 1 vs target 2, and Pearson NaN (undefined for one sample),
    then resets the collection.
    """
    task, log = _task(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        task._log_epoch_metrics("val_metrics")
    assert str(excinfo.value) == (
        "val/gene_interaction/MSE: the epoch ended with no sample with a finite "
        "target, so the metric is undefined"
    )
    assert log.calls == []

    task._metrics("val_metrics").update(torch.tensor([1.0]), torch.tensor([2.0]))
    task._log_epoch_metrics("val_metrics")
    assert [(n, kw) for n, _, kw in log.calls] == [
        (f"val/gene_interaction/{m}", {"sync_dist": True}) for m in METRIC_NAMES
    ]
    values = log.values
    assert values["val/gene_interaction/MSE"] == 1.0
    assert values["val/gene_interaction/RMSE"] == 1.0
    assert math.isnan(values["val/gene_interaction/Pearson"])
    for metric in task._metrics("val_metrics").values():
        assert metric.update_count == 0


def test_epoch_ends_log_no_metrics_under_dataloader_profiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``execution_mode="dataloader_profiling"`` (the 006 ``_dataloader_profile`` and
    ``_086_dataloader`` configs) never runs the model, so no metric is updated; the
    train and validation epoch ends log nothing instead of refusing the empty epoch.
    """
    task, log = _task(monkeypatch, execution_mode="dataloader_profiling")
    _attach(task, tmp_path, epoch=0)
    task._shared_step(_batch(TARGET), 0, "train")
    task._shared_step(_batch(TARGET), 0, "val")
    log.calls.clear()
    task.on_train_epoch_end()
    task.on_validation_epoch_end()
    assert log.calls == []


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
    assert task.train_samples == EMPTY_SAMPLES and task.val_samples == EMPTY_SAMPLES


def test_on_validation_epoch_end_plots_only_outside_sanity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Epoch 1 (plot epoch): during the sanity check the samples are kept and nothing is
    plotted; in the real validation epoch that follows (start hook, one step)
    ``_plot_samples(val_samples, "val_sample")`` runs once and the accumulators are
    emptied. Metrics are logged in both cases (each epoch has its own step, since an
    epoch with no sample is refused).
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
    task.on_validation_epoch_start()
    task._shared_step(_batch(TARGET), 0, "val")
    task.on_validation_epoch_end()
    assert plotted == [(1, "val_sample")]
    assert task.val_samples["true_values"] == []


# --- sample collection and plotting ------------------------------------------------- #
def test_train_sample_collection_respects_the_ceiling_and_keeps_the_last_gene_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ceiling 3, batches of 2, plot epoch: batch 1 is kept whole; batch 2 keeps
    ``randperm(2)[:1]`` (seed 0); batch 3's samples are dropped. Issue #616 item 5: the
    gene table is per gene, so it is neither indexed by a sample index (batch 2 used to
    store ``EMB[idx]``, a gene row) nor stacked; the buffer holds the LAST step's whole
    table, here 2 * EMB after ``w`` is set to 2 before batch 3.
    """
    task, _ = _task(monkeypatch, loss_func=_PlainLoss(), plot_sample_ceiling=3)
    _attach(task, tmp_path, epoch=1)
    task._shared_step(_batch(TARGET), 0, "train")
    torch.manual_seed(0)
    task._shared_step(_batch(TARGET), 1, "train")
    with torch.no_grad():
        _scripted(task).w.fill_(2.0)
    task._shared_step(_batch([4.0, 12.0]), 2, "train")
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
    assert samples["integrated_embeddings"].tolist() == [
        [2 * v for v in row] for row in EMB
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


def test_plot_epoch_hands_exact_arrays_and_the_single_gene_table_smoothness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Two training batches on plot epoch 1, then ``on_train_epoch_end``:
    ``Visualization(base_dir=default_root_dir, max_points=10000)`` gets predictions
    [[1], [3], [1], [3]], targets [[2], [5], [2], [5]], empty latents, the loss class
    name, epoch 1, None, stage "train_sample"; the box plot gets the first columns.

    Issue #616 item 5: the logged ``oversmoothing_integrated_embeddings`` is the
    Frobenius norm of ONE centered gene table, ||EMB - mean|| (EMB centered: rows
    (0, -1/3), (-3, -10/3), (3, 11/3); squared norm 0 + 1/9 + 9 + 100/9 + 9 + 121/9 =
    128/3, norm 6.5320), whatever the number of batches. It used to stack the table
    once per batch, logging sqrt(2) * 6.5320 = 9.2376 here.
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
    assert single == pytest.approx(6.5319726, rel=1e-6)
    assert logged[0] == {
        "train_sample/oversmoothing_integrated_embeddings": pytest.approx(
            single, rel=1e-6
        )
    }
    assert logged[1] == {
        "train_sample/gene_interaction_box_plot": (
            "image",
            ("fig", [2.0, 5.0, 2.0, 5.0], [1.0, 3.0, 1.0, 3.0]),
        )
    }
    assert len(logged) == 2
    assert task.train_samples == EMPTY_SAMPLES


def test_plot_samples_noop_on_empty_and_skips_box_plot_for_all_nan_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Empty samples: nothing is constructed or logged. All-NaN targets and no latents:
    the scatter is drawn but neither wandb payload is logged.
    """
    visual, logged = _record_plots(monkeypatch)
    task, _ = _task(monkeypatch)
    _attach(task, tmp_path, epoch=1)
    task._plot_samples(dict(EMPTY_SAMPLES), "val_sample")
    assert (visual, logged) == ([], [])
    task._plot_samples(
        {
            "true_values": [torch.tensor([float("nan"), float("nan")])],
            "predictions": [torch.tensor([1.0, 2.0])],
            "integrated_embeddings": None,
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


def test_configure_optimizers_renames_learning_rate_and_strips_the_scheduler_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``learning_rate`` becomes ``lr``; the (validated) scheduler ``type`` key is not
    passed to ReduceLROnPlateau and the remaining keys are (mode "max"). Any other
    type is refused at construction, see
    ``test_init_refuses_an_accumulation_schedule_and_a_scheduler_it_does_not_build``.
    """
    task, _ = _task(
        monkeypatch,
        optimizer_config={"type": "SGD", "learning_rate": 0.5},
        lr_scheduler_config={"type": "ReduceLROnPlateau", "mode": "max"},
    )
    config = task.configure_optimizers()
    optimizer = config["optimizer"]
    assert type(optimizer) is torch.optim.SGD
    assert optimizer.param_groups[0]["lr"] == 0.5
    sched_cfg = config["lr_scheduler"]
    assert isinstance(sched_cfg, dict)
    scheduler = sched_cfg["scheduler"]
    assert type(scheduler) is ReduceLROnPlateau and scheduler.mode == "max"


def test_plot_samples_subsamples_genotypes_to_the_ceiling_but_never_the_gene_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ceiling 2, three collected samples: one ``randperm(3)[:2]`` (seed 0) indexes the
    targets and the predictions alike, and ``max_points`` is the ceiling. The gene
    table is per gene, so the smoothness is that of the whole [3, 2] table (it used to
    be indexed by the genotype permutation, picking gene rows by sample index).
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
            "integrated_embeddings": latents,
        },
        "train_sample",
    )
    torch.manual_seed(0)
    idx = torch.randperm(3)[:2].tolist()
    assert visual[0] == ("init", str(tmp_path), 2)
    args, _ = visual[1]
    assert args[0][:, 0].tolist() == [[1.0, 2.0, 3.0][i] for i in idx]
    assert args[1][:, 0].tolist() == [[10.0, 20.0, 30.0][i] for i in idx]
    table = latents.double().numpy()
    expected = float(np.linalg.norm(table - table.mean(axis=0)))
    # rows centered on (5/3, 2): squared norm 25/9 + 4 + 4/9 + 0 + 49/9 + 4 = 50/3
    assert expected == pytest.approx(math.sqrt(50 / 3), rel=1e-12)
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


# --- 2026.10.07 (phase 24): the original-scale target reshapes and a 2-D inverse
# result (int_dango.py:262, 267, 407) ------------------------------------------------ #
def test_original_targets_are_reshaped_like_the_transformed_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``phenotype_values_original`` gets the same reshapes as the targets: a 0-dim 4.0
    becomes [[4.0]] (line 262) and a [2, 2] original [[4, 9], [10, 9]] keeps its first
    column [[4], [10]] (line 267). The returned targets are the original-scale ones.
    """
    task, _ = _task(
        monkeypatch,
        model=_Scripted(torch.tensor(1.5), None, with_recon=False),
        loss_func=_PlainLoss(),
    )
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([0])
    batch["gene"].phenotype_values = torch.tensor(1.0)
    batch["gene"].phenotype_values_original = torch.tensor(4.0)
    _, _, targets = task._shared_step(batch, 0, "train")
    assert targets is not None and targets.tolist() == [[4.0]]

    task2, _ = _task(
        monkeypatch,
        model=_Scripted(torch.tensor(PRED), None, with_recon=False),
        loss_func=_PlainLoss(),
    )
    wide = _batch(TARGET)
    wide["gene"].phenotype_values_original = torch.tensor([[4.0, 9.0], [10.0, 9.0]])
    _, _, targets2 = task2._shared_step(wide, 0, "train")
    assert targets2 is not None and targets2.tolist() == [[4.0], [10.0]]


class _Affine2D(_Affine):
    """``_Affine`` whose output is already [B, 1]: 2 x + 1 as a column."""

    def forward(self, data: HeteroData) -> HeteroData:
        out = super().forward(data)
        out["gene"].gene_interaction = out["gene"].gene_interaction.unsqueeze(1)
        return out


def test_a_two_dimensional_inverse_result_is_used_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A [2, 1] inverse result [[3], [7]] is kept unchanged (line 407): the
    original-unit metrics compare [3, 7] with [4, 10], MSE (1 + 9) / 2 = 5, the same
    numbers as the 1-D inverse test above.
    """
    task, _ = _task(monkeypatch, loss_func=_PlainLoss(), inverse_transform=_Affine2D())
    task._shared_step(_batch(TARGET, [4.0, 10.0]), 0, "train")
    original = _computed(task._metrics("train_metrics"))
    assert original["train/gene_interaction/MSE"] == pytest.approx(5.0)
    assert original["train/gene_interaction/RMSE"] == pytest.approx(math.sqrt(5.0))
