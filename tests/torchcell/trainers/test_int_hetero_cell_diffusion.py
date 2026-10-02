# tests/torchcell/trainers/test_int_hetero_cell_diffusion.py
# [[tests.torchcell.trainers.test_int_hetero_cell_diffusion]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_hetero_cell_diffusion.py
"""``DiffusionRegressionTask`` where it differs from ``RegressionTask`` (006 diffusion).

Behavior the two tasks share is pinned once, parametrized over both classes, in
``test_int_hetero_cell.py``; this file reuses its stand-ins. ``_Fixed`` returns the
predictions p = [1, 3, -2] times one trainable ``scale`` (1.0) with
``z_p = [[3, 4], [0, 0], [6, 8]]`` (row norms 5, 0, 10, mean 5); ``_coo`` is three
genotypes with targets y = [2, 5, -1], so the squared error is (1 + 4 + 1) / 3 = 2.0.

What differs: the train stage calls ``loss_func(pred, target, z_p)`` (the real
``DiffusionLoss`` asks the model for ``compute_diffusion_loss(targets, z_p,
t_mode="full")`` and ignores the predictions); validation and test score
``F.mse_loss(pred, target)`` and never call the loss; the component logger averages
vectors and drops numbers; ``graph_reg_loss`` is ignored; two extra buffers feed
``train/avg_diffusion_loss`` and ``val/avg_inference_mse``. The trainer itself chooses
no sampling steps: validation predictions are whatever the model returns in eval mode
(``GeneInteractionDiff`` samples its decoder with its own ``sampling_steps``), and in
training mode the real model returns zeros, which the train metrics then score.
"""

import math
import warnings
from types import SimpleNamespace
from typing import Any

import networkx as nx
import pytest
import torch
from sortedcontainers import SortedDict
from torch_geometric.data import Batch, HeteroData

from tests.torchcell.trainers.test_int_hetero_cell import (
    TASKS,
    Z_P,
    Y,
    _attach,
    _column,
    _coo,
    _Fixed,
    _Log,
    _make,
    _record,
    _spy,
    _SquaredError,
)
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.losses.diffusion_loss import DiffusionLoss
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
    loss_func = _SquaredError()
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


def test_diffusion_train_components_mean_vectors_and_drop_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diffusion component logger (lines 1166-1173): a one-element tensor logs its
    value, a [1, 2] tensor logs its MEAN 1.5 under one key, an empty tensor logs NaN,
    and a plain number (``count`` 3) is not logged. ``z_p`` absent: the loss still
    receives a third positional argument, ``None``.
    """
    components = {
        "one": torch.tensor(0.1),
        "vec": torch.tensor([1.0, 2.0]),
        "empty": torch.tensor([]),
        "count": 3,
    }
    loss_func = _SquaredError("pair", components)
    task = _make(DiffusionRegressionTask, _Fixed(z_p=None), loss_func=loss_func)
    log = _record(monkeypatch, task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mean of an empty tensor
        task._shared_step(_coo(), 0, "train")
    args, kwargs = loss_func.calls[0]
    assert (len(args), args[2], kwargs) == (3, None, {})
    assert log.names == ["train/one", "train/vec", "train/empty", "train/loss"]
    assert log.values["train/one"] == pytest.approx(0.1)
    assert log.values["train/vec"] == 1.5
    assert math.isnan(log.values["train/empty"])


def test_diffusion_train_requires_a_loss_only_on_the_train_stage() -> None:
    """With ``loss_func=None`` the train stage fails a bare ``assert`` (an
    ``AssertionError`` with no message, and none at all under ``python -O``); the
    validation stage never needs the loss and returns the MSE 2.0.
    """
    task = _make(DiffusionRegressionTask, loss_func=None)
    task.log = _Log()
    with pytest.raises(AssertionError, match=r"^$"):
        task._shared_step(_coo(), 0, "train")
    loss, _, _ = task._shared_step(_coo(), 0, "val")
    assert loss.item() == 2.0


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


def test_real_diffusion_model_trains_metrics_on_zero_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: the train-stage metrics of the diffusion task score a constant zero.

    ``GeneInteractionDiff`` (diffusion decoder) returns ``zeros_like(targets)`` in
    training mode (hetero_cell_bipartite_dango_diff_gi.py lines 242-254), and the
    task feeds those to ``train/gene_interaction/*`` like real predictions: with
    targets [0.5, -0.25, 1.0] the train MSE is (0.25 + 0.0625 + 1) / 3 = 0.4375 and
    the Pearson is NaN (zero variance), whatever the model has learned. The loss is
    the model's own diffusion loss and is unaffected. Pinned until the train stage
    skips (or samples for) the original-unit metrics.
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
            diffusion_config={"num_layers": 1, "num_heads": 2, "num_timesteps": 10},
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
        )
        task.cell_graph = _graph_sample()
        log = _record(monkeypatch, task)
        seen = _spy(monkeypatch, task.train_metrics)
        model.train()
        loss, predictions, _ = task._shared_step(batch, 0, "train")
    assert predictions is not None
    assert predictions.tolist() == [[0.0], [0.0], [0.0]]
    assert seen == [([0.0, 0.0, 0.0], [0.5, -0.25, 1.0])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        computed = task._compute_metrics_safely(task.train_metrics)
    assert computed["train/gene_interaction/MSE"].item() == pytest.approx(0.4375)
    assert math.isnan(computed["train/gene_interaction/Pearson"].item())
    assert loss.item() == pytest.approx(log.values["train/diffusion_loss"])
    assert loss.item() > 0.0


@pytest.mark.parametrize("stage", ["train", "val"])
def test_two_shared_steps_drifted_apart(
    stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the two ``_shared_step`` bodies are copies that disagree on one batch.

    Same model (``graph_reg_loss`` 0.25), same loss (``(loss, {"vec": [1, 2], "n": 3})``
    over the squared error 2.0), same batch:

    * train: ``RegressionTask`` logs ``vec_0``, ``vec_1``, ``n`` and adds the graph term
      (loss 2.25); ``DiffusionRegressionTask`` logs ``vec`` as its mean 1.5, drops
      ``n``, ignores ``graph_reg_loss`` (loss 2.0) and logs no ``graph_reg_loss``;
    * val: ``RegressionTask`` still uses ``loss_func`` (2.25 with the graph term);
      ``DiffusionRegressionTask`` uses ``F.mse_loss`` (2.0) and logs
      ``val/inference_mse``.

    Everything after the loss (z_p norm, metrics, inverse, buffers) is identical (the
    parametrized tests above). Pinned until the shared part is one function.
    """
    outputs = {}
    for cls in TASKS:
        components = {"vec": torch.tensor([1.0, 2.0]), "n": 3}
        task = _make(
            cls,
            _Fixed(graph_reg_loss=torch.tensor(0.25)),
            loss_func=_SquaredError("pair", components),
        )
        log = _record(monkeypatch, task)
        loss, _, _ = task._shared_step(_coo(), 0, stage)
        outputs[cls.__name__] = (loss.item(), log.values)
    regression, diffusion = (
        outputs["RegressionTask"],
        outputs["DiffusionRegressionTask"],
    )
    if stage == "train":
        assert regression == (
            2.25,
            {
                "train/vec_0": 1.0,
                "train/vec_1": 2.0,
                "train/n": 3.0,
                "train/graph_reg_loss": 0.25,
                "train/loss": 2.25,
                "train/z_p_norm": 5.0,
            },
        )
        assert diffusion == (
            2.0,
            {"train/vec": 1.5, "train/loss": 2.0, "train/z_p_norm": 5.0},
        )
    else:
        assert regression == (
            2.25,
            {
                "val/vec_0": 1.0,
                "val/vec_1": 2.0,
                "val/n": 3.0,
                "val/graph_reg_loss": 0.25,
                "val/loss": 2.25,
                "val/z_p_norm": 5.0,
            },
        )
        assert diffusion == (
            2.0,
            {"val/inference_mse": 2.0, "val/loss": 2.0, "val/z_p_norm": 5.0},
        )
