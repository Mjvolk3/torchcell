# tests/torchcell/trainers/test_int_transformer_cell_methods.py
# [[tests.torchcell.trainers.test_int_transformer_cell_methods]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_transformer_cell_methods.py
"""``RegressionTask`` methods called directly, with exact logged payloads and metrics.

The model is a scripted stand-in (the boundary the trainer consumes): it returns fixed
predictions and a fixed representations dict, times one trainable scale that starts at 1,
so every quantity the trainer derives has a closed form. ``self.log`` is replaced by a
recorder, and the plotting classes plus ``wandb.log`` / ``wandb.Image`` by recorders
too, so what is asserted is the trainer's own arithmetic and routing.

The hand-made batch has two genotypes with predictions [1, 3] and targets [2, 5]:

* log-cosh loss: (log cosh 1 + log cosh 2) / 2 = (0.4337808 + 1.3250027) / 2 = 0.8793917;
* MSE (1 + 4) / 2 = 2.5, RMSE sqrt 2.5 = 1.5811388, Pearson of two points on a rising
  line = 1;
* ``h_CLS = [3, 4]`` has norm 5; ``H_genes_pert`` is all 2 over [2, 4, 2], so every token
  has norm 2 sqrt 2 = 2.8284271; ``H_genes`` is all 1 over [4, 2], so the residual ratio is
  ||1|| over 16 entries / ||1|| over 8 = 4 / sqrt 8 = sqrt 2.

The validation diagnostics use a 4-gene attention matrix A (rows below) against a graph
``g`` with edges 0 -> 1 and 1 -> 2. Row 0 puts its top weight on its true neighbor 1 and
row 1 on gene 0, a miss, so recall@degree is (1 + 0) / 2 = 0.5; every k in
[8, 32, 128, 320] is clipped to the 4 genes, one of which is a neighbor, so precision@k
is 1/4; the edge mass is (A[0, 1] + A[1, 2]) / sum(A) = (0.6 + 0.2) / 4 = 0.2. The degree
correlation is Spearman of degrees [1, 1, 0, 0] against column sums
[1.1, 1.3, 0.9, 0.7]: centered ranks [1, 1, -1, -1] and [0.5, 1.5, -0.5, -1.5] give
4 / sqrt(4 * 5) = 0.8944272.
"""

import math
from typing import Any

import lightning as L
import pytest
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torchmetrics import Metric, MetricCollection

from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchcell.trainers.int_transformer_cell import RegressionTask

LOGCOSH = (math.log(math.cosh(1.0)) + math.log(math.cosh(2.0))) / 2  # 0.8793917
ATTENTION = torch.tensor(
    [
        [0.1, 0.6, 0.2, 0.1],
        [0.5, 0.2, 0.2, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]
)
ADJACENCY = torch.tensor(
    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0] * 4, [0.0] * 4]
)


class _Scripted(nn.Module):
    """Returns fixed predictions (times a trainable scale) and a fixed reps dict."""

    # set per test; absent until then, which is what the trainer's hasattr checks see
    regularized_head_config: dict[str, Any] | None
    adjacency_matrices: dict[str, torch.Tensor]

    def __init__(self, predictions: torch.Tensor, reps: dict[str, Any]) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.predictions = predictions
        self.reps = reps
        self.calls: list[bool] = []

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData, return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.calls.append(return_attention)
        return self.predictions * self.scale, dict(self.reps)


class _Recorder:
    """Stands in for ``LightningModule.log``: keeps the last value per name as a float."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.values[name] = float(value)


def _reps(**overrides: Any) -> dict[str, Any]:
    reps: dict[str, Any] = {
        "h_CLS": torch.tensor([3.0, 4.0]),
        "H_genes": torch.ones(4, 2),
        "H_genes_pert": torch.full((2, 4, 2), 2.0),
        "graph_reg_loss": torch.tensor(0.25),
        "attention_weights": None,
        "residual_update_ratios": None,
    }
    reps.update(overrides)
    return reps


def _batch(targets: list[float], original: list[float] | None = None) -> HeteroData:
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor([0, 1])
    batch["gene"].perturbation_indices_batch = torch.tensor([0, 1])
    batch["gene"].phenotype_values = torch.tensor(targets)
    if original is not None:
        batch["gene"].phenotype_values_original = torch.tensor(original)
    return batch


def _task(
    monkeypatch: pytest.MonkeyPatch,
    reps: dict[str, Any] | None = None,
    predictions: torch.Tensor | None = None,
    **overrides: Any,
) -> tuple[RegressionTask, _Recorder]:
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    model = _Scripted(
        torch.tensor([1.0, 3.0]) if predictions is None else predictions,
        _reps() if reps is None else reps,
    )
    kwargs: dict[str, Any] = dict(
        optimizer_config={"type": "AdamW", "learning_rate": 1e-2},
        lr_scheduler_config=None,
        plot_every_n_epochs=0,
        plot_transformer_diagnostics_every_n_epochs=0,
        plot_edge_recovery_every_n_epochs=0,
        loss_func=LogCoshLoss(),
        device="cpu",
    )
    kwargs.update(overrides)
    graph_arg: Any = graph  # the trainer annotates a Tensor but clones a HeteroData
    task = RegressionTask(model=model, cell_graph=graph_arg, **kwargs)
    recorder = _Recorder()
    monkeypatch.setattr(task, "log", recorder)
    return task, recorder


def _scripted(task: RegressionTask) -> _Scripted:
    model = task.model
    assert isinstance(model, _Scripted)
    return model


def _attach_trainer(task: RegressionTask, tmp_path: Any, epoch: int = 0) -> None:
    task.trainer = L.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    task.trainer.fit_loop.epoch_progress.current.completed = epoch


# --- the shared step ---------------------------------------------------------------- #
def test_train_step_loss_payload_and_metrics_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1-D predictions and targets are lifted to [2, 1]; the loss is log-cosh plus the
    model's graph term 0.25 (the dummy unused-parameter term is 0); both metric spaces
    see predictions [1, 3] against targets [2, 5].
    """
    task, log = _task(monkeypatch)
    loss, predictions, targets = task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert loss.item() == pytest.approx(LOGCOSH + 0.25, rel=1e-6)
    assert isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor)
    assert torch.equal(predictions.detach(), torch.tensor([[1.0], [3.0]]))
    assert torch.equal(targets, torch.tensor([[2.0], [5.0]]))
    assert log.values == pytest.approx(
        {
            "train/cls_token_norm": 5.0,
            "train/graph_reg_loss": 0.25,
            "train/loss": LOGCOSH + 0.25,
            "train/z_p_norm": 2 * math.sqrt(2),
        }
    )
    assert _scripted(task).calls == [False]  # training never asks for attention
    for collection in (task.train_metrics, task.train_transformed_metrics):
        computed = task._compute_metrics_safely(collection)
        prefix = next(iter(computed)).rsplit("/", 1)[0]
        assert {k.rsplit("/", 1)[1]: v.item() for k, v in computed.items()} == (
            pytest.approx(
                {"MSE": 2.5, "RMSE": math.sqrt(2.5), "Pearson": 1.0}, rel=1e-6
            )
        ), prefix
    assert task.train_samples == {"true_values": [], "predictions": [], "latents": {}}


def test_original_scale_metrics_use_the_inverse_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transformed targets [2, 5], original [20, 50], inverse transform x -> 10 x.

    Transformed space compares [1, 3] with [2, 5]: MSE 2.5. Original space compares the
    inverted predictions [10, 30] with [20, 50]: (100 + 400) / 2 = 250.
    """

    class _TimesTen(nn.Module):
        def forward(self, data: HeteroData) -> HeteroData:
            assert data["gene"].phenotype_types == ["gene_interaction"]
            assert torch.equal(data["gene"].phenotype_sample_indices, torch.arange(2))
            data["gene"].phenotype_values = data["gene"].phenotype_values * 10
            return data

    task, _ = _task(monkeypatch, inverse_transform=_TimesTen())
    _, _, original = task._shared_step(_batch([2.0, 5.0], [20.0, 50.0]), 0, "val")
    assert isinstance(original, torch.Tensor)
    assert torch.equal(original, torch.tensor([[20.0], [50.0]]))
    orig = task._compute_metrics_safely(task.val_metrics)
    transformed = task._compute_metrics_safely(task.val_transformed_metrics)
    assert orig["val/gene_interaction/MSE"].item() == pytest.approx(250.0)
    assert transformed["val/transformed/gene_interaction/MSE"].item() == pytest.approx(
        2.5
    )


def test_nan_targets_are_masked_out_of_the_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Target [2, NaN] keeps only the first pair: MSE (1 - 2)^2 = 1. A one-sample
    Pearson computes to NaN in this torchmetrics version (no ValueError), so the safe
    compute returns it rather than skipping it.
    """
    task, _ = _task(monkeypatch)
    task._shared_step(_batch([2.0, float("nan")]), 0, "test")
    computed = task._compute_metrics_safely(task.test_metrics)
    assert computed["test/gene_interaction/MSE"].item() == 1.0
    assert math.isnan(computed["test/gene_interaction/Pearson"].item())


def test_compute_metrics_safely_skips_only_the_known_small_sample_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two small-sample messages are skipped; any other ValueError propagates.

    Skipped: "Needs at least two samples" and "No samples to concatenate".
    """

    class _Raises(Metric):
        def __init__(self, message: str) -> None:
            super().__init__()
            self.message = message

        def update(self) -> None:
            return None

        def compute(self) -> torch.Tensor:
            raise ValueError(self.message)

    task, _ = _task(monkeypatch)
    skipped = MetricCollection(
        {
            "a": _Raises("Needs at least two samples, got 1"),
            "b": _Raises("No samples to concatenate"),
        }
    )
    assert task._compute_metrics_safely(skipped) == {}
    with pytest.raises(ValueError, match="^boom$"):
        task._compute_metrics_safely(MetricCollection({"c": _Raises("boom")}))


def test_point_dist_graph_reg_owns_the_graph_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PointDistGraphReg adds the graph term itself (lambda 1), so the trainer does not
    add it again: loss = log-cosh + 0.25, not + 0.5. Its float components are logged
    under the stage prefix, and no separate trainer graph_reg log is written from the
    representations (the logged value is the loss's own component, also 0.25).
    """
    loss_func = PointDistGraphReg(distribution_loss={"lambda": 0.0})
    task, log = _task(monkeypatch, loss_func=loss_func)
    loss, _, _ = task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert loss.item() == pytest.approx(LOGCOSH + 0.25, rel=1e-6)
    assert log.values["train/point_loss"] == pytest.approx(LOGCOSH, rel=1e-6)
    assert log.values["train/graph_reg_loss"] == pytest.approx(0.25)
    assert log.values["train/dist_loss"] == 0.0
    assert log.values["train/total_loss"] == pytest.approx(LOGCOSH + 0.25, rel=1e-6)


@pytest.mark.parametrize("loss_class", [MleDistSupCR, MleWassSupCR])
def test_supcr_losses_receive_the_epoch_and_their_components_are_logged(
    loss_class: type[nn.Module], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The two SupCR composites are called as (pred, target, z_p, epoch=current_epoch).

    A recording subclass returns loss 0.5 and a component dict: a one-element tensor is
    logged as a scalar, a 2-element tensor as ``_0`` / ``_1``, an int as itself, and an
    empty tensor is skipped. The trainer adds the model's graph term: 0.5 + 0.25.
    """
    seen: dict[str, Any] = {}

    class _Recording(loss_class):  # type: ignore[valid-type,misc]
        def __init__(self) -> None:
            nn.Module.__init__(self)

        def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            z_p: torch.Tensor,
            epoch: int,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            seen.update(pred=pred, target=target, z_p=z_p, epoch=epoch)
            components = {
                "mse": torch.tensor(0.1),
                "vec": torch.tensor([1.0, 2.0]),
                "count": 3,
                "empty": torch.zeros(0),
            }
            return torch.tensor(0.5), components

    task, log = _task(monkeypatch, loss_func=_Recording())
    _attach_trainer(task, tmp_path, epoch=4)
    loss, _, _ = task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert seen["epoch"] == 4
    assert torch.equal(seen["z_p"], torch.full((2, 4, 2), 2.0))
    assert torch.equal(seen["target"], torch.tensor([[2.0], [5.0]]))
    assert loss.item() == pytest.approx(0.75)
    keys = ("train/mse", "train/vec_0", "train/vec_1", "train/count")
    assert {k: log.values[k] for k in keys} == pytest.approx(
        {"train/mse": 0.1, "train/vec_0": 1.0, "train/vec_1": 2.0, "train/count": 3.0}
    )
    assert [k for k in log.values if k.startswith("train/empty")] == []


def test_generic_losses_get_z_p_when_present_and_two_arguments_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain loss returning a tensor is the loss; with no ``z_p`` or ``H_genes_pert``
    in the representations it is called with (pred, target) only. Without a
    ``graph_reg_loss`` key nothing is added.
    """
    calls: list[int] = []

    class _Three(nn.Module):
        def forward(
            self, pred: torch.Tensor, target: torch.Tensor, z_p: torch.Tensor
        ) -> torch.Tensor:
            calls.append(3)
            return (pred - target).abs().mean()

    class _Two(nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            calls.append(2)
            return (pred - target).abs().sum()

    task, _ = _task(monkeypatch, loss_func=_Three())
    loss, _, _ = task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert loss.item() == pytest.approx(1.5 + 0.25)  # (1 + 2) / 2 plus the graph term
    reps = {"h_CLS": torch.tensor([3.0, 4.0])}
    task, log = _task(monkeypatch, reps=reps, loss_func=_Two())
    loss, _, _ = task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert loss.item() == 3.0
    assert calls == [3, 2]
    assert (
        "train/z_p_norm" not in log.values and "train/graph_reg_loss" not in log.values
    )


def test_gate_weights_are_averaged_over_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[[0.2, 0.8], [0.6, 0.4]] averages to global 0.4 and local 0.6; a one-column gate
    [[0.2], [0.6]] logs only the global weight, (0.2 + 0.6) / 2 = 0.4.
    """
    task, log = _task(
        monkeypatch, reps=_reps(gate_weights=torch.tensor([[0.2, 0.8], [0.6, 0.4]]))
    )
    task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert log.values["train/gate_weight_global"] == pytest.approx(0.4)
    assert log.values["train/gate_weight_local"] == pytest.approx(0.6)
    task, log = _task(
        monkeypatch, reps=_reps(gate_weights=torch.tensor([[0.2], [0.6]]))
    )
    task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert log.values["train/gate_weight_global"] == pytest.approx(0.4)
    assert "train/gate_weight_local" not in log.values


def test_dataloader_profiling_skips_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profiling mode never calls the model; the loss is an exact 0 connected to every
    parameter, and the batch size (2 genotypes) is logged as a float.
    """
    task, log = _task(monkeypatch, execution_mode="dataloader_profiling")
    loss, predictions, targets = task._shared_step(_batch([2.0, 5.0]), 0, "train")
    assert (loss.item(), predictions, targets) == (0.0, None, None)
    assert _scripted(task).calls == []
    assert log.values == {
        "train/dataloader_profile_loss": 0.0,
        "train/dataloader_profile_batch_size": 2.0,
    }
    loss.backward()
    grad = _scripted(task).scale.grad
    assert isinstance(grad, torch.Tensor) and grad.item() == 0.0


# --- validation diagnostics --------------------------------------------------------- #
def _entropy(ps: list[float]) -> float:
    return -sum(p * math.log(p) for p in ps)


def test_validation_diagnostics_accumulate_and_log_exact_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Every diagnostic scheduled (frequency 1 at epoch 0): the step asks the model for
    attention, accumulates attention stats, edge recovery and degree bias (numbers in the
    module docstring), and the epoch end logs recall and precision, hands the averages to
    the two plotting classes, resets the accumulators and plots the validation samples.
    """
    plots: dict[str, Any] = {}

    class _Diagnostics:
        def __init__(self, base_dir: str) -> None:
            plots["diagnostics_dir"] = base_dir

        def plot_attention_diagnostics(self, stats: Any, **kwargs: Any) -> None:
            plots["attention"] = (stats, kwargs)

    class _Recovery:
        def __init__(self, base_dir: str) -> None:
            plots["recovery_dir"] = base_dir

        def plot_edge_recovery_recall(
            self, recall: Any, *args: Any, **kwargs: Any
        ) -> None:
            plots["recall"] = recall

        def plot_edge_recovery_precision(
            self, precision: Any, ks: Any, *args: Any, **kwargs: Any
        ) -> None:
            plots["precision"] = (precision, ks)

        def plot_edge_mass_alignment(
            self, mass: Any, *args: Any, **kwargs: Any
        ) -> None:
            plots["mass"] = mass

        def plot_edge_recovery_per_graph(self, *args: Any, **kwargs: Any) -> None:
            plots["per_graph"] = True

    sampled: list[str] = []
    monkeypatch.setattr(
        "torchcell.viz.transformer_diagnostics.TransformerDiagnostics", _Diagnostics
    )
    monkeypatch.setattr(
        "torchcell.viz.graph_recovery.GraphRecoveryVisualization", _Recovery
    )
    reps = _reps(
        attention_weights=[ATTENTION.view(1, 1, 4, 4)], residual_update_ratios=[0.5]
    )
    task, log = _task(
        monkeypatch,
        reps=reps,
        plot_every_n_epochs=1,
        plot_transformer_diagnostics_every_n_epochs=1,
        plot_edge_recovery_every_n_epochs=1,
    )
    monkeypatch.setattr(
        task, "_plot_samples", lambda samples, stage: sampled.append(stage)
    )
    _scripted(task).regularized_head_config = {"g": {"layer": 0, "head": 0}}
    _scripted(task).adjacency_matrices = {"g": ADJACENCY}
    _attach_trainer(task, tmp_path)

    task.on_validation_epoch_start()
    task._shared_step(_batch([2.0, 5.0]), 0, "val")
    assert _scripted(task).calls == [True]
    assert log.values["val/residual_update_ratio"] == pytest.approx(math.sqrt(2))
    acc = task.edge_recovery_accumulators["g_L0_H0"]
    assert acc["sum_recall_deg"] == 0.5 * 2 and acc["count_nodes_deg"] == 2
    assert acc["sum_prec"] == pytest.approx({8: 0.5, 32: 0.5, 128: 0.5, 320: 0.5})
    assert acc["sum_edge_mass"] == pytest.approx(0.2)
    assert acc["degree_correlation_sum"] == pytest.approx(4 / math.sqrt(20))
    assert task.residual_update_accumulators == {0: {"sum_ratio": 0.5, "count": 1}}
    assert len(task.val_samples["true_values"]) == 1

    task.on_validation_epoch_end()
    edge_logs = {
        k: v for k, v in log.values.items() if k.startswith("val_edge_recovery")
    }
    assert edge_logs == pytest.approx(
        {
            "val_edge_recovery/g_L0_H0/recall_at_deg": 0.5,
            "val_edge_recovery/g_L0_H0/precision_k8": 0.25,
            "val_edge_recovery/g_L0_H0/precision_k32": 0.25,
            "val_edge_recovery/g_L0_H0/precision_k128": 0.25,
            "val_edge_recovery/g_L0_H0/precision_k320": 0.25,
        }
    )
    assert log.values["val/gene_interaction/MSE"] == 2.5
    stats, kwargs = plots["attention"]
    rows = [[0.1, 0.6, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1], [0.25] * 4, [0.25] * 4]
    entropy = sum(_entropy(r) for r in rows) / 4
    assert stats[0] == pytest.approx(
        {
            "entropy": entropy,
            "effective_rank": math.exp(entropy),
            "top5": 1.0,  # k = min(5, 4) keeps the whole row
            "top10": 1.0,
            "top50": 1.0,
            "max_row_weight": (0.6 + 0.5 + 0.25 + 0.25) / 4,
            "col_entropy": _entropy([1.1 / 4, 1.3 / 4, 0.9 / 4, 0.7 / 4]),
            "max_col_sum": 1.3,
        },
        rel=1e-5,
    )
    assert kwargs == {
        "residual_ratios": {0: 0.5},
        "gradient_norms": None,
        "num_epochs": 0,
        "stage": "val",
    }
    assert plots["recall"] == {"g_L0_H0": 0.5}
    assert plots["precision"] == (
        {"g_L0_H0": pytest.approx({8: 0.25, 32: 0.25, 128: 0.25, 320: 0.25})},
        [8, 32, 128, 320],
    )
    assert plots["mass"] == {"g_L0_H0": pytest.approx(0.2)}
    assert plots["per_graph"] is True
    assert plots["diagnostics_dir"] == plots["recovery_dir"] == str(tmp_path)
    assert sampled == ["val_sample"]
    assert (
        task.edge_recovery_accumulators == {}
        and task.attention_stats_accumulators == {}
    )
    assert task.val_samples == {"true_values": [], "predictions": [], "latents": {}}


def test_uniform_attention_diagnostics_are_log_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uniform 1/4 attention over 4 genes: row entropy log 4, effective rank 4, every
    top-k keeps the whole row (1.0), max row weight 1/4, column sums 1 so the column
    entropy is log 4 and the max column sum is 1. Two calls double the sums.
    """
    task, _ = _task(monkeypatch)
    uniform = torch.full((2, 3, 4, 4), 0.25)
    task._accumulate_attention_diagnostics([uniform], 0)
    task._accumulate_attention_diagnostics([uniform], 1)
    task._accumulate_attention_diagnostics([], 2)  # empty list is a no-op
    log4 = math.log(4)
    assert task.attention_stats_accumulators[0] == pytest.approx(
        {
            "entropy_sum": 2 * log4,
            "effective_rank_sum": 8.0,
            "top5_sum": 2.0,
            "top10_sum": 2.0,
            "top50_sum": 2.0,
            "max_row_weight_sum": 0.5,
            "col_entropy_sum": 2 * log4,
            "max_col_sum_sum": 2.0,
            "count": 2,
        },
        rel=1e-6,
    )


def test_edge_recovery_skips_models_without_graph_regularization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``regularized_head_config`` attribute, or one set to None, records nothing; a
    layer index past the attention list and an unknown graph name are skipped.
    """
    attention = [ATTENTION.view(1, 1, 4, 4)]
    task, _ = _task(monkeypatch)
    task._accumulate_edge_recovery_metrics(attention, 0)
    _scripted(task).regularized_head_config = None
    _scripted(task).adjacency_matrices = {"g": ADJACENCY}
    task._accumulate_edge_recovery_metrics(attention, 0)
    _scripted(task).regularized_head_config = {
        "g": {"layer": [3], "head": 0},
        "missing": {"layer": 0, "head": 0},
    }
    task._accumulate_edge_recovery_metrics(attention, 0)
    assert task.edge_recovery_accumulators == {}


def test_degree_bias_then_edge_recovery_share_one_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degree bias first creates the full record; edge recovery then fills the same key.
    A graph outside the regularized config is ignored.
    """
    task, _ = _task(monkeypatch)
    _scripted(task).regularized_head_config = {"g": {"layer": 0, "head": 0}}
    _scripted(task).adjacency_matrices = {"g": ADJACENCY}
    weights = ATTENTION.view(1, 1, 4, 4)
    task._accumulate_degree_bias(weights, "other", 0, 0)
    assert task.edge_recovery_accumulators == {}
    task._accumulate_degree_bias(weights, "g", 0, 0)
    task._accumulate_edge_recovery_metrics([weights], 0)
    acc = task.edge_recovery_accumulators["g_L0_H0"]
    assert (acc["degree_corr_count"], acc["count_batches"], acc["count_nodes_deg"]) == (
        1,
        1,
        2,
    )
    assert acc["degree_correlation_sum"] == pytest.approx(4 / math.sqrt(20))


def test_residual_update_ratio_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """x_out = 2 x_in gives ||x_out - x_in|| / ||x_in|| = 1 per call; two calls sum to 2."""
    task, _ = _task(monkeypatch)
    x = torch.ones(1, 3, 4)
    task._accumulate_residual_updates(x, 2 * x, 1)
    task._accumulate_residual_updates(x, 2 * x, 1)
    assert task.residual_update_accumulators == {1: {"sum_ratio": 2.0, "count": 2}}


# --- sample collection and plotting ------------------------------------------------- #
def test_train_samples_respect_the_ceiling_and_pool_the_latents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ceiling 1 on a 2-genotype batch keeps one random genotype with its own target and
    prediction: after ``torch.manual_seed(0)``, ``torch.randperm(2)[:1]`` is [0], so the
    kept pair is genotype 0, (target 2, prediction 1). H_pooled is the mean of h_CLS [3, 4] and the four [2, 2] gene rows:
    [(3 + 8) / 5, (4 + 8) / 5] = [2.2, 2.4].
    """
    torch.manual_seed(0)
    task, _ = _task(monkeypatch, plot_every_n_epochs=1, plot_sample_ceiling=1)
    task._shared_step(_batch([2.0, 5.0]), 0, "train")
    (kept,) = task.train_samples["true_values"]
    (pred,) = task.train_samples["predictions"]
    assert (kept.item(), pred.item()) == (2.0, 1.0)
    (pooled,) = task.train_samples["latents"]["H_pooled"]
    torch.testing.assert_close(pooled, torch.tensor([[2.2, 2.4]]))
    task._shared_step(_batch([2.0, 5.0]), 1, "train")  # ceiling reached: nothing added
    assert len(task.train_samples["true_values"]) == 1

    task, _ = _task(monkeypatch, plot_every_n_epochs=1)
    task._shared_step(_batch([2.0, 5.0]), 0, "train")
    torch.testing.assert_close(
        task.train_samples["latents"]["H_pooled"][0], torch.tensor([[2.2, 2.4]] * 2)
    )


def test_validation_ceiling_and_test_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation honors the same ceiling: seed 0 keeps genotype 0 (target 2), pooled
    [2.2, 2.4]; the test stage always keeps every sample, so its H_pooled is that row for
    both genotypes, [[2.2, 2.4], [2.2, 2.4]].
    """
    torch.manual_seed(0)
    task, _ = _task(monkeypatch, plot_every_n_epochs=1, plot_sample_ceiling=1)
    task._shared_step(_batch([2.0, 5.0]), 0, "val")
    (val_true,) = task.val_samples["true_values"]
    assert torch.equal(val_true, torch.tensor([[2.0]]))
    torch.testing.assert_close(
        task.val_samples["latents"]["H_pooled"][0], torch.tensor([[2.2, 2.4]])
    )
    task.on_test_epoch_start()
    task._shared_step(_batch([2.0, 5.0]), 0, "test")
    assert torch.equal(
        task.test_samples["true_values"][0], torch.tensor([[2.0], [5.0]])
    )
    assert torch.equal(
        task.test_samples["predictions"][0], torch.tensor([[1.0], [3.0]])
    )
    torch.testing.assert_close(
        task.test_samples["latents"]["H_pooled"][0], torch.tensor([[2.2, 2.4]] * 2)
    )


def test_plot_samples_routes_everything_to_the_plotters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Three samples over a ceiling of 2 are subsampled to 2 pairs that stay aligned; 1-D
    latents are lifted to rows; the ``z_p`` latent (row r is [r, r]) is subsampled with
    the same indices, and its oversmoothing score is the Frobenius norm of the two kept
    rows about their mean, 2 entries of +-(r2 - r1) / 2 per row, i.e. |r2 - r1|. The box
    plot is logged once. Empty samples return before touching anything.
    """
    calls: dict[str, Any] = {}

    class _Vis:
        def __init__(self, base_dir: str, max_points: int) -> None:
            calls["init"] = (base_dir, max_points)

        def visualize_model_outputs(
            self,
            predictions: torch.Tensor,
            true_values: torch.Tensor,
            latents: dict[str, torch.Tensor],
            loss_name: str,
            epoch: int,
            timestamp: Any,
            stage: str,
        ) -> None:
            calls["outputs"] = (
                predictions,
                true_values,
                latents,
                loss_name,
                epoch,
                stage,
            )

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr("torchcell.trainers.int_transformer_cell.Visualization", _Vis)
    monkeypatch.setattr("wandb.log", lambda payload, **kwargs: logged.append(payload))
    monkeypatch.setattr("wandb.Image", lambda figure: "image")
    task, _ = _task(monkeypatch, plot_sample_ceiling=2)
    _attach_trainer(task, tmp_path, epoch=3)

    task._plot_samples({"true_values": [], "predictions": []}, "val_sample")
    assert calls == {} and logged == []

    torch.manual_seed(0)
    truth = torch.tensor([10.0, 20.0, 30.0])
    samples = {
        "true_values": [truth],
        "predictions": [truth + 1],
        "latents": {
            "H_pooled": [
                torch.tensor([0.0, 0.0]),
                torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            ],
            "z_p": [torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])],
        },
    }
    task._plot_samples(samples, "test_sample")
    predictions, true_values, latents, loss_name, epoch, stage = calls["outputs"]
    assert calls["init"] == (str(tmp_path), 2)
    assert (loss_name, epoch, stage) == ("LogCoshLoss", 3, "test_sample")
    assert true_values.shape == (2, 1) and predictions.shape == (2, 1)
    torch.testing.assert_close(predictions, true_values + 1)
    assert list(latents) == ["H_pooled"] and latents["H_pooled"].shape == (2, 2)
    # the kept H_pooled rows are the rows of the kept samples (row r holds r)
    kept_rows = true_values[:, 0] / 10 - 1
    torch.testing.assert_close(latents["H_pooled"][:, 0], kept_rows)
    assert logged[0]["test_sample/oversmoothing_z_p"] == pytest.approx(
        abs(kept_rows[1] - kept_rows[0]).item()
    )
    assert [sorted(p) for p in logged] == [
        ["test_sample/oversmoothing_z_p"],
        ["test_sample/gene_interaction_box_plot"],
    ]
    assert logged[1]["test_sample/gene_interaction_box_plot"] == "image"


# --- epoch hooks, optimizers, the training step ------------------------------------- #
def test_train_epoch_end_logs_resets_and_steps_the_scheduler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Metrics are logged under both prefixes and reset; the manual scheduler steps once:
    cosine annealing from 1e-2 with T_max 2 lands on 1e-2 * (1 + cos(pi / 2)) / 2 = 5e-3.
    A ReduceLROnPlateau scheduler is refused by the assertion at line 1374.
    """
    task, log = _task(monkeypatch)
    _attach_trainer(task, tmp_path)
    task._shared_step(_batch([2.0, 5.0]), 0, "train")
    optimizer = torch.optim.SGD(task.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    task.trainer.strategy.lr_scheduler_configs = [LRSchedulerConfig(scheduler)]
    task.on_train_epoch_end()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)
    assert log.values["train/gene_interaction/MSE"] == 2.5
    assert log.values["train/transformed/gene_interaction/RMSE"] == pytest.approx(
        math.sqrt(2.5)
    )
    assert math.isnan(
        task._compute_metrics_safely(task.train_metrics)[
            "train/gene_interaction/MSE"
        ].item()
    )

    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    task.trainer.strategy.lr_scheduler_configs = [LRSchedulerConfig(plateau)]
    with pytest.raises(AssertionError):
        task.on_train_epoch_end()


def test_train_epoch_end_plots_scheduled_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """With plotting every epoch the collected train samples go to ``_plot_samples``
    under ``train_sample`` and the buffer is cleared.
    """
    task, _ = _task(monkeypatch, plot_every_n_epochs=1)
    _attach_trainer(task, tmp_path)
    stages: list[str] = []
    monkeypatch.setattr(
        task, "_plot_samples", lambda samples, stage: stages.append(stage)
    )
    task._shared_step(_batch([2.0, 5.0]), 0, "train")
    task.on_train_epoch_end()
    assert stages == ["train_sample"]
    assert task.train_samples == {"true_values": [], "predictions": [], "latents": {}}


def test_test_epoch_end_logs_and_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test metrics are logged under ``test/`` and the kept samples are plotted once."""
    task, log = _task(monkeypatch)
    stages: list[str] = []
    monkeypatch.setattr(
        task, "_plot_samples", lambda samples, stage: stages.append(stage)
    )
    task._shared_step(_batch([2.0, 5.0]), 0, "test")
    task.on_test_epoch_end()
    assert log.values["test/gene_interaction/MSE"] == 2.5
    assert log.values["test/transformed/gene_interaction/MSE"] == 2.5
    assert stages == ["test_sample"]
    assert task.test_samples["true_values"] == []


def test_accumulation_schedule_follows_the_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Integer keys: {0: 2, 3: 4} starts at 2 and switches to 4 from epoch 3.

    Finding: ``__init__`` reads the epoch-0 value with ``.get(0, 1)``
    (int_transformer_cell.py:74-76), so a schedule whose keys are strings (as YAML
    or a string-keyed dict delivers them) starts at 1 instead of the configured "0"
    value; ``on_train_epoch_start`` converts string keys and corrects it at epoch 0.
    """
    task, _ = _task(monkeypatch, grad_accumulation_schedule={0: 2, 3: 4})
    assert task.current_accumulation_steps == 2
    _attach_trainer(task, tmp_path, epoch=3)
    task.on_train_epoch_start()
    assert task.current_accumulation_steps == 4

    task, _ = _task(monkeypatch, grad_accumulation_schedule={"0": 3})
    assert task.current_accumulation_steps == 1
    _attach_trainer(task, tmp_path, epoch=0)
    task.on_train_epoch_start()
    assert task.current_accumulation_steps == 3


def test_cosine_warmup_restarts_scheduler_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The custom warm-restart scheduler is wrapped for epoch stepping at frequency 1 and
    carries the configured cycle length, learning-rate bounds and warmup.
    """
    task, _ = _task(
        monkeypatch,
        lr_scheduler_config={
            "type": "CosineAnnealingWarmupRestarts",
            "first_cycle_steps": 10,
            "max_lr": 1e-2,
            "min_lr": 1e-4,
            "warmup_steps": 2,
        },
    )
    config = task.configure_optimizers()
    scheduler = config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, CosineAnnealingWarmupRestarts)
    assert (
        scheduler.first_cycle_steps,
        scheduler.max_lr,
        scheduler.min_lr,
        scheduler.warmup_steps,
    ) == (10, 1e-2, 1e-4, 2)
    assert (
        config["lr_scheduler"]["interval"],
        config["lr_scheduler"]["frequency"],
    ) == ("epoch", 1)
    assert isinstance(config["optimizer"], torch.optim.AdamW)


def _loader() -> DataLoader[HeteroData]:
    batches = [_batch([2.0, 5.0]), _batch([2.0, 5.0])]
    dataset: Any = batches  # pre-collated batches handed through with batch_size=None
    return DataLoader(dataset, batch_size=None)


def _fit(task: RegressionTask, tmp_path: Any) -> L.Trainer:
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
    trainer.fit(task, train_dataloaders=_loader())
    return trainer


def _real_log_task(**overrides: Any) -> RegressionTask:
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    kwargs: dict[str, Any] = dict(
        optimizer_config={"type": "SGD", "lr": 1.0},
        lr_scheduler_config=None,
        plot_every_n_epochs=0,
        plot_transformer_diagnostics_every_n_epochs=0,
        plot_edge_recovery_every_n_epochs=0,
        loss_func=LogCoshLoss(),
        device="cpu",
    )
    kwargs.update(overrides)
    model = _Scripted(torch.tensor([1.0, 3.0]), _reps())
    graph_arg: Any = graph
    return RegressionTask(model=model, cell_graph=graph_arg, **kwargs)


def test_gradient_accumulation_defers_the_step_and_logs_the_effective_batch(
    tmp_path: Any,
) -> None:
    """Accumulating 2 batches, one fast_dev_run batch never reaches ``opt.step``: the
    scale stays 1 and ``global_step`` 0; the effective batch is 2 genotypes * 2 steps *
    1 process = 4.
    """
    task = _real_log_task(grad_accumulation_schedule={0: 2})
    trainer = _fit(task, tmp_path)
    assert trainer.global_step == 0
    assert _scripted(task).scale.item() == 1.0
    assert trainer.callback_metrics["effective_batch_size"].item() == 4.0


def test_gradient_clipping_bounds_the_sgd_step(tmp_path: Any) -> None:
    """SGD at lr 1 with the gradient clipped to norm 1e-3 moves the one parameter by
    exactly 1e-3. The raw gradient is d/ds mean(log cosh(p s - t)) at s = 1, i.e.
    (tanh(1 - 2) * 1 + tanh(3 - 5) * 3) / 2 = (-0.7615942 - 2.8920827) / 2 = -1.8268384,
    larger than 1e-3 in norm and negative, so the step raises the scale to 1.001.
    """
    task = _real_log_task(clip_grad_norm=True, clip_grad_norm_max_norm=1e-3)
    trainer = _fit(task, tmp_path)
    assert trainer.global_step == 1
    assert _scripted(task).scale.item() == pytest.approx(1.001, rel=1e-7)
    assert trainer.callback_metrics["learning_rate"].item() == 1.0


def test_model_profiling_skips_the_optimizer(tmp_path: Any) -> None:
    """``model_profiling`` returns the loss before any optimizer work: no step, no
    learning-rate log, the parameter untouched.
    """
    task = _real_log_task(execution_mode="model_profiling")
    trainer = _fit(task, tmp_path)
    assert trainer.global_step == 0
    assert _scripted(task).scale.item() == 1.0
    assert "learning_rate" not in trainer.callback_metrics
