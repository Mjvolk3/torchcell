# tests/torchcell/trainers/test_int_hetero_cell_diffusion.py
# [[tests.torchcell.trainers.test_int_hetero_cell_diffusion]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_hetero_cell_diffusion.py
"""``DiffusionRegressionTask`` where it differs from ``RegressionTask`` (006 diffusion).

Behavior the two tasks share is pinned once, parametrized over both classes, in
``test_int_hetero_cell.py``; this file reuses its stand-ins. ``_Fixed`` returns the
predictions p = [1, 3, -2] times one trainable ``scale`` (1.0) with
``z_p = [[3, 4], [0, 0], [6, 8]]`` (row norms 5, 0, 10, mean 5); ``_coo`` is three
genotypes with targets y = [2, 5, -1], so the squared error is (1 + 4 + 1) / 3 = 2.0.

Since issue #614 ``DiffusionRegressionTask`` subclasses ``RegressionTask`` and both
run one ``_shared_step``. What differs is the stage loss and the train-stage scoring:
the train stage calls ``loss_func`` as ``RegressionTask`` does (the real
``DiffusionLoss`` gets ``(pred, target, z_p)`` and asks the model for
``compute_diffusion_loss(targets, z_p, t_mode="full")``, ignoring the predictions);
validation and test score ``F.mse_loss`` of the sampled predictions and never call
the loss; in training mode the real ``GeneInteractionDiff`` returns all-zero
placeholders, so the train stage updates no metric and keeps no plot sample; two
extra buffers feed ``train/avg_diffusion_loss`` and ``val/avg_inference_mse``.
"""

import logging
import re
import warnings
from types import SimpleNamespace
from typing import Any

import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict
from torch import nn
from torch_geometric.data import Batch, HeteroData

from tests.torchcell.trainers.test_int_hetero_cell import (
    TASKS,
    Z_P,
    P,
    Y,
    _attach,
    _column,
    _coo,
    _DiffusionSquaredError,
    _Fixed,
    _Log,
    _make,
    _record,
    _spy,
    _SquaredError,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.losses.diffusion_loss import DiffusionLoss
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.models.hetero_cell_bipartite_dango_diff_gi import GeneInteractionDiff
from torchcell.sequence import GeneSet
from torchcell.trainers.int_hetero_cell import DiffusionRegressionTask


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Epoch hooks free CUDA memory when it is available; keep them on CPU."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


class _Diffusing(_Fixed):
    """``_Fixed`` plus the two hooks ``DiffusionLoss`` requires.

    ``compute_diffusion_loss`` returns 0.3 * scale and records ``(targets, context,
    t_mode)``.
    """

    def __init__(self, **reps: Any) -> None:
        super().__init__(**reps)
        self.diffusion_decoder = SimpleNamespace(num_timesteps=10)
        self.diffusion_calls: list[tuple[Any, Any, str]] = []

    def compute_diffusion_loss(
        self, targets: torch.Tensor, context: torch.Tensor, *, t_mode: str
    ) -> torch.Tensor:
        self.diffusion_calls.append((targets.tolist(), context.tolist(), t_mode))
        return 0.3 * self.scale


def test_diffusion_train_step_logs_the_real_diffusion_loss_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DiffusionLoss(model, lambda_diffusion=2)`` on the train stage.

    The loss is called ``(pred, target, z_p)`` and asks the model for its diffusion
    loss on the [3, 1] targets with ``z_p`` as context and ``t_mode="full"``: 0.3,
    weighted to 0.6. The trainer logs ``train/diffusion_loss`` 0.3 and
    ``train/total_loss`` 0.6 before ``train/loss`` 0.6 (every log ``batch_size=3``),
    and keeps the detached 0.6 for the epoch average. The predictions themselves do
    not enter the loss.
    """
    model = _Diffusing()
    task = _make(
        DiffusionRegressionTask,
        model,
        loss_func=DiffusionLoss(model, lambda_diffusion=2.0),
    )
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, "train")
    assert loss.item() == pytest.approx(0.6)
    assert model.diffusion_calls == [(_column(Y), Z_P, "full")]
    assert log.names == [
        "train/diffusion_loss",
        "train/total_loss",
        "train/loss",
        "train/z_p_norm",
    ]
    assert log.values == pytest.approx(
        {
            "train/diffusion_loss": 0.3,
            "train/total_loss": 0.6,
            "train/loss": 0.6,
            "train/z_p_norm": 5.0,
        }
    )
    assert set(log.batch_sizes.values()) == {3}
    assert [t.item() for t in task.train_diffusion_loss] == pytest.approx([0.6])
    assert not task.train_diffusion_loss[0].requires_grad


@pytest.mark.parametrize("stage", ["val", "test"])
def test_diffusion_eval_stages_score_mse_and_never_call_the_loss(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation and test use ``F.mse_loss(pred, target)`` = (1 + 4 + 1) / 3 = 2.0 in
    model units, not ``loss_func`` (never called), logged as ``{stage}/inference_mse``
    (a float) and ``{stage}/loss``. Only validation keeps it for the epoch average.
    """
    loss_func = _DiffusionSquaredError()
    task = _make(DiffusionRegressionTask, loss_func=loss_func)
    log = _record(monkeypatch, task)
    loss, _, _ = task._shared_step(_coo(), 0, stage)
    assert loss.item() == 2.0
    assert loss_func.calls == []
    assert log.names == [f"{stage}/inference_mse", f"{stage}/loss", f"{stage}/z_p_norm"]
    assert type(log.calls[0][1]) is float and log.calls[0][1] == 2.0
    kept = [t.item() for t in task.val_mse_during_inference]
    assert kept == ([2.0] if stage == "val" else [])
    assert task.train_diffusion_loss == []


def test_diffusion_train_components_log_like_every_other_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diffusion task logs loss components through the shared logger: a
    one-element tensor as its value, a [1, 2] tensor element by element (``vec_0``,
    ``vec_1``), a plain number (``count`` 3) as is, an empty tensor not at all. It
    used to log the vector's mean 1.5 under one key and drop the number. The
    ``DiffusionLoss`` is called ``(pred, target, z_p)`` with no epoch.
    """
    components = {
        "one": torch.tensor(0.1),
        "vec": torch.tensor([[1.0, 2.0]]),
        "empty": torch.tensor([]),
        "count": 3,
    }
    loss_func = _DiffusionSquaredError("pair", components)
    task = _make(DiffusionRegressionTask, loss_func=loss_func)
    log = _record(monkeypatch, task)
    task._shared_step(_coo(), 0, "train")
    args, kwargs = loss_func.calls[0]
    assert ([a.tolist() for a in args], kwargs) == ([_column(P), _column(Y), Z_P], {})
    assert log.names == [
        "train/one",
        "train/vec_0",
        "train/vec_1",
        "train/count",
        "train/loss",
        "train/z_p_norm",
    ]
    assert log.values == pytest.approx(
        {
            "train/one": 0.1,
            "train/vec_0": 1.0,
            "train/vec_1": 2.0,
            "train/count": 3.0,
            "train/loss": 2.0,
            "train/z_p_norm": 5.0,
        }
    )


def test_diffusion_train_requires_a_loss_only_on_the_train_stage() -> None:
    """With ``loss_func=None`` the train stage raises ``ValueError("No loss function
    provided")`` as ``RegressionTask`` does (it was a bare ``assert``, an empty
    ``AssertionError`` and nothing under ``python -O``); the validation stage never
    needs the loss and returns the MSE 2.0.
    """
    task = _make(DiffusionRegressionTask, loss_func=None)
    task.log = _Log()
    with pytest.raises(ValueError, match=r"^No loss function provided$"):
        task._shared_step(_coo(), 0, "train")
    loss, _, _ = task._shared_step(_coo(), 0, "val")
    assert loss.item() == 2.0


@pytest.mark.parametrize(
    ("loss_func", "name"),
    [(LogCoshLoss(reduction="mean"), "LogCoshLoss"), (nn.MSELoss(), "MSELoss")],
)
def test_a_non_diffusion_loss_is_refused_at_construction(
    loss_func: nn.Module, name: str
) -> None:
    """The 006 diffusion script also offers ``loss: logcosh`` (and ``icloss``); with
    those the task would train on the model's all-zero training placeholders (on
    ``main`` ``LogCoshLoss`` raised ``TypeError`` at the first step). Any loss that is
    not a ``DiffusionLoss`` is refused by name when the task is built.
    """
    message = (
        f"DiffusionRegressionTask trains with a DiffusionLoss, got {name}: in "
        "training mode the diffusion model returns all-zero placeholder predictions, "
        "so any other loss would train on them"
    )
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        _make(DiffusionRegressionTask, loss_func=loss_func)


def test_diffusion_loss_on_a_model_without_z_p_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DiffusionLoss`` conditions on ``z_p``; a model that returns none is refused by
    the task, naming the loss, before the model's diffusion loss is computed.
    """
    model = _Diffusing(z_p=None)
    task = _make(DiffusionRegressionTask, model, loss_func=DiffusionLoss(model))
    _record(monkeypatch, task)
    message = (
        "DiffusionLoss needs representations['z_p'], which the model did not return"
    )
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        task._shared_step(_coo(), 0, "train")
    assert model.diffusion_calls == []


def test_diffusion_train_epoch_end_logs_no_train_metric_and_steps_the_scheduler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, no_cuda: None
) -> None:
    """Epoch 1 with ``plot_every_n_epochs=2`` after one train batch: the train
    collections were never updated and are neither computed nor logged (no NaN rows),
    nothing is plotted, ``train/avg_diffusion_loss`` logs the batch loss 2.0, and an
    epoch scheduler is stepped once with no argument.
    """
    task = _make(DiffusionRegressionTask, plot_every_n_epochs=2)
    _attach(task, tmp_path, epoch=1)
    log = _record(monkeypatch, task)
    task._shared_step(_coo(), 0, "train")
    plotted: list[str] = []
    monkeypatch.setattr(task, "_plot_samples", lambda s, stage: plotted.append(stage))
    steps: list[tuple[Any, ...]] = []
    scheduler = SimpleNamespace(step=lambda *a: steps.append(a))
    monkeypatch.setattr(task, "lr_schedulers", lambda: scheduler)
    log.calls.clear()
    task.on_train_epoch_end()
    assert [(n, float(v), kw) for n, v, kw in log.calls] == [
        ("train/avg_diffusion_loss", 2.0, {"sync_dist": True})
    ]
    assert task.train_metrics["MSE"].update_count == 0
    assert plotted == []
    assert steps == [()]


@pytest.mark.parametrize(
    ("hook", "buffer", "key"),
    [
        ("on_train_epoch_end", "train_diffusion_loss", "train/avg_diffusion_loss"),
        (
            "on_validation_epoch_end",
            "val_mse_during_inference",
            "val/avg_inference_mse",
        ),
    ],
)
def test_diffusion_epoch_ends_log_the_mean_of_their_buffer_and_clear_it(
    hook: str,
    buffer: str,
    key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    no_cuda: None,
) -> None:
    """Buffered 1.0, 2.0 and 4.5 log their mean 2.5 under the epoch key (no
    ``batch_size``, ``sync_dist=True``) and the buffer is emptied; an empty buffer
    logs nothing.
    """
    task = _make(DiffusionRegressionTask)
    _attach(task, tmp_path)
    monkeypatch.setattr(task, "lr_schedulers", lambda: None)
    log = _record(monkeypatch, task)
    setattr(task, buffer, [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(4.5)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # empty metric collections
        getattr(task, hook)()
        assert [(n, float(v), kw) for n, v, kw in log.calls if n == key] == [
            (key, 2.5, {"sync_dist": True})
        ]
        assert getattr(task, buffer) == []
        log.calls.clear()
        getattr(task, hook)()
    assert key not in log.names


# Real GeneInteractionDiff on a four-gene, two-graph fixture (as in
# tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py).


def _gene_multigraph() -> GeneMultiGraph:
    genes = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])
    graphs = {}
    for name in ["physical", "regulatory"]:
        graph = nx.Graph()
        graph.add_nodes_from(genes)
        graphs[name] = GeneGraph(name=name, graph=graph, max_gene_set=genes)
    return GeneMultiGraph(graphs=SortedDict(graphs))


EDGES = {"physical": [(0, 1), (1, 2), (2, 3)], "regulatory": [(3, 0), (0, 2)]}


def _graph_sample(pert: list[int] | None = None) -> HeteroData:
    keep = [g for g in range(4) if g not in (pert or [])]
    new_id = {g: i for i, g in enumerate(keep)}
    data = HeteroData()
    data["gene"].num_nodes = len(keep)
    if pert is not None:
        mask = torch.zeros(4, dtype=torch.bool)
        mask[pert] = True
        data["gene"].pert_mask = mask
        data["gene"].perturbation_indices = torch.tensor(pert, dtype=torch.long)
    for name, pairs in EDGES.items():
        kept = [(new_id[s], new_id[t]) for s, t in pairs if s in new_id and t in new_id]
        data["gene", name, "gene"].edge_index = (
            torch.tensor(kept, dtype=torch.long).t().contiguous()
        )
    return data


def test_real_diffusion_model_train_placeholders_are_not_scored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """``GeneInteractionDiff`` (diffusion decoder) returns ``zeros_like(targets)`` in
    training mode (hetero_cell_bipartite_dango_diff_gi.py, the ``self.training``
    branch of ``forward``): placeholders, not predictions. On a plot epoch, two train
    batches update neither train collection and keep no plot sample, and the skip is
    announced once (one INFO record). Before issue #614 the zeros were scored: MSE
    (0.25 + 0.0625 + 1) / 3 = 0.4375 and Pearson NaN for targets [0.5, -0.25, 1.0],
    whatever the model had learned. The loss is the model's own diffusion loss. In
    eval mode the model samples its decoder, and those predictions are scored: the
    val collection receives exactly the returned predictions, not all zero.
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = GeneInteractionDiff(
            gene_num=4,
            hidden_channels=8,
            num_layers=1,
            gene_multigraph=_gene_multigraph(),
            dropout=0.0,
            gene_encoder_config={
                "encoder_type": "gin",
                "graph_aggregation_method": "sum",
            },
            local_predictor_config={"num_heads": 2, "num_attention_layers": 1},
            diffusion_config={
                "num_layers": 1,
                "num_heads": 2,
                "num_timesteps": 10,
                "sampling_steps": 5,
            },
        )
        batch = Batch.from_data_list(
            [_graph_sample(p) for p in ([0], [1, 2], [3])],
            follow_batch=["perturbation_indices"],
        )
        batch["gene"].phenotype_values = torch.tensor([0.5, -0.25, 1.0])
        task = _make(
            DiffusionRegressionTask,
            model,
            optimizer_config={"type": "AdamW", "learning_rate": 1e-3},
            loss_func=DiffusionLoss(model),
            plot_every_n_epochs=1,
        )
        _attach(task, tmp_path)
        task.cell_graph = _graph_sample()
        log = _record(monkeypatch, task)
        train_seen = _spy(monkeypatch, task.train_metrics)
        transformed_seen = _spy(monkeypatch, task.train_transformed_metrics)
        model.train()
        with caplog.at_level(logging.INFO, logger="torchcell.trainers.int_hetero_cell"):
            loss, predictions, _ = task._shared_step(batch, 0, "train")
            task._shared_step(batch, 1, "train")
        model.eval()
        val_seen = _spy(monkeypatch, task.val_metrics)
        with torch.no_grad():
            _, sampled, _ = task._shared_step(batch, 0, "val")
    assert predictions is not None and sampled is not None
    assert predictions.tolist() == [[0.0], [0.0], [0.0]]
    assert (train_seen, transformed_seen) == ([], [])
    assert task.train_samples == {"true_values": [], "predictions": [], "latents": {}}
    trainer_records = [
        r.getMessage()
        for r in caplog.records
        if r.name == "torchcell.trainers.int_hetero_cell"
    ]
    assert trainer_records == [
        "DiffusionRegressionTask does not score train-stage predictions: "
        "train/gene_interaction/* and train/transformed/gene_interaction/* are not "
        "logged and no train samples are plotted"
    ]
    diffusion_logs = [float(v) for n, v, _ in log.calls if n == "train/diffusion_loss"]
    assert len(diffusion_logs) == 2
    assert loss.item() == pytest.approx(diffusion_logs[0])
    assert loss.item() > 0.0
    assert val_seen == [(sampled.view(-1).tolist(), [0.5, -0.25, 1.0])]
    assert any(v != 0.0 for v in sampled.view(-1).tolist())


@pytest.mark.parametrize("stage", ["train", "val"])
def test_both_tasks_share_one_step_except_the_stage_loss(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ``_shared_step`` serves both tasks; only the stage loss differs.

    Same model (``graph_reg_loss`` 0.25), same loss (``(loss, {"vec": [1, 2], "n": 3})``
    over the squared error 2.0), same batch:

    * train: both tasks log ``vec_0``, ``vec_1``, ``n``, add and log the graph term,
      and return 2.25 (the diffusion copy used to log the mean 1.5, drop ``n`` and
      ignore the graph term);
    * val: ``RegressionTask`` uses ``loss_func`` (2.25 with the graph term);
      ``DiffusionRegressionTask`` scores ``F.mse_loss`` of the sampled predictions
      (2.0, logged as ``val/inference_mse``) and adds the same graph term (2.25).
    """
    outputs = {}
    for cls in TASKS:
        components = {"vec": torch.tensor([1.0, 2.0]), "n": 3}
        task = _make(
            cls,
            _Fixed(graph_reg_loss=torch.tensor(0.25)),
            loss_func=(
                _DiffusionSquaredError("pair", components)
                if cls is DiffusionRegressionTask
                else _SquaredError("pair", components)
            ),
        )
        log = _record(monkeypatch, task)
        loss, _, _ = task._shared_step(_coo(), 0, stage)
        outputs[cls.__name__] = (loss.item(), log.values)
    regression, diffusion = (
        outputs["RegressionTask"],
        outputs["DiffusionRegressionTask"],
    )
    shared_tail = {
        f"{stage}/graph_reg_loss": 0.25,
        f"{stage}/loss": 2.25,
        f"{stage}/z_p_norm": 5.0,
    }
    components_logged = {
        f"{stage}/vec_0": 1.0,
        f"{stage}/vec_1": 2.0,
        f"{stage}/n": 3.0,
    }
    assert regression == (2.25, components_logged | shared_tail)
    if stage == "train":
        assert diffusion == regression
    else:
        assert diffusion == (2.25, {"val/inference_mse": 2.0} | shared_tail)
