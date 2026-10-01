# tests/torchcell/trainers/test_int_transformer_cell.py
# [[tests.torchcell.trainers.test_int_transformer_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_transformer_cell.py
"""``RegressionTask`` (the live CGT trainer) on CPU with a tiny model and two batches.

Lightning 2.5 ``fast_dev_run=True`` runs one training and one validation batch, disables
loggers and checkpoint callbacks, and skips the sanity check. Every plotting frequency is
0, which ``_is_scheduled`` reads as "never" (the ZeroDivisionError fix its docstring
records), so nothing reaches ``wandb.log``; a recorder on it proves that. The trainer
uses manual optimization, so the test also checks that the optimizer step moved the
weights and that the logged losses are finite.

2026.09.30 - exact values on a scripted stand-in (Phase 15). ``_Fixed`` returns the
predictions ``[1, 3]`` times one trainable ``scale`` (initially 1) and a fixed
representations dict (``h_CLS = [3, 4]``, ``H_genes`` all 1 over [4, 2],
``H_genes_pert`` all 2 over [2, 4, 2], ``graph_reg_loss = 0.25``); every batch has the
targets ``[2, 5]``. So, under ``fast_dev_run``:

* train loss = (log cosh 1 + log cosh 2) / 2 + 0.25 = 0.8793917 + 0.25, the train
  metrics see [1, 3] against [2, 5]: MSE (1 + 4) / 2 = 2.5;
* AdamW (lr 1e-2, weight decay 1e-2) takes one step: decay gives 1 - 1e-4, and the
  first Adam update is lr * g / (|g| + 1e-8), i.e. -1e-2 * sign(g), with
  g = (tanh(-1) * 1 + tanh(-2) * 3) / 2 < 0, so scale = 0.9999 + 0.01 = 1.0099;
* validation then sees predictions 1.0099 * [1, 3]: MSE ((1.0099 - 2)^2 +
  (3.0297 - 5)^2) / 2 and loss mean(log cosh(1.0099 p - t)) + 0.25;
* CosineAnnealingLR with T_max 2 is stepped once by ``on_train_epoch_end``:
  1e-2 * (1 + cos(pi / 2)) / 2 = 5e-3;
* the pooled latent is mean([h_CLS, 4 tokens of 2]) = ((3 + 8) / 5, (4 + 8) / 5)
  = (2.2, 2.4) for each genotype.
"""

import math
from typing import Any, cast

import lightning as L
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData

from torchcell.losses.logcosh import LogCoshLoss
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.trainers.int_transformer_cell import RegressionTask

N, D, HEADS = 8, 16, 4


def _labelled(batch: HeteroData, values: list[float]) -> HeteroData:
    out = batch.clone()
    out["gene"].phenotype_values = torch.tensor(values).unsqueeze(1)
    return out


def _task(cell_graph: HeteroData, **overrides: Any) -> RegressionTask:
    torch.manual_seed(0)
    model = CellGraphTransformer(
        gene_num=N,
        hidden_channels=D,
        num_transformer_layers=1,
        num_attention_heads=HEADS,
        cell_graph=cell_graph,
        dropout=0.0,
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
    return RegressionTask(model=model, cell_graph=cell_graph, **kwargs)


def _trainer(tmp_path: Any) -> L.Trainer:
    return L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )


@pytest.fixture
def loader(batch: HeteroData) -> DataLoader[HeteroData]:
    """Two pre-collated batches; batch_size=None hands each through uncollated."""
    batches = [_labelled(batch, [0.1, -0.2, 0.3]), _labelled(batch, [0.0, 0.5, -0.1])]
    return DataLoader(cast("Dataset[HeteroData]", batches), batch_size=None)


def test_fast_dev_run_trains_one_step_without_touching_wandb(
    cell_graph: HeteroData,
    loader: DataLoader[HeteroData],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One optimizer step moves every weight, logs finite train/val losses, calls wandb never."""
    calls: list[Any] = []
    monkeypatch.setattr("wandb.log", lambda *a, **k: calls.append((a, k)))
    task = _task(cell_graph)
    before = {n: p.detach().clone() for n, p in task.model.named_parameters()}

    _trainer(tmp_path).fit(task, train_dataloaders=loader, val_dataloaders=loader)

    assert task.trainer.global_step == 1
    moved = sorted(
        n for n, p in task.model.named_parameters() if not torch.equal(p, before[n])
    )
    assert moved == sorted(before)  # every parameter has a gradient and AdamW moves it
    metrics = task.trainer.callback_metrics
    assert torch.isfinite(metrics["train/loss"]) and torch.isfinite(metrics["val/loss"])
    assert metrics["learning_rate"].item() == pytest.approx(1e-2)
    assert calls == []


def test_forward_is_the_model_forward_on_the_cloned_cell_graph(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """task(batch) equals model(task.cell_graph, batch); the clone is made once and reused."""
    task = _task(cell_graph)
    labelled = _labelled(batch, [0.1, -0.2, 0.3])
    predictions, representations = task(labelled)
    expected, _ = task.model(task.cell_graph, labelled)
    torch.testing.assert_close(predictions, expected)
    assert predictions.shape == (3, 1)
    assert representations["attention_weights"] is None
    assert task.cell_graph is not cell_graph  # cloned in __init__
    graph_before = task.cell_graph
    task(labelled)
    assert task.cell_graph is graph_before
    assert task._cell_graph_device == torch.device("cpu")


def test_batch_size_is_the_number_of_genotypes(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Perturbation batches count distinct batch indices: 3, not the 6 perturbed genes."""
    task = _task(cell_graph)
    assert task._get_batch_size(_labelled(batch, [0.1, -0.2, 0.3])) == 3
    dense = HeteroData()
    dense["gene"].x = torch.zeros(5, 2)
    assert task._get_batch_size(dense) == 5


@pytest.mark.parametrize(
    ("freq", "epoch", "expected"),
    [
        (0, 0, False),
        (None, 0, False),
        (1, 0, True),
        (2, 0, False),
        (2, 1, True),
        (3, 5, True),
    ],
)
def test_is_scheduled_treats_zero_and_none_as_never(
    cell_graph: HeteroData, freq: int | None, epoch: int, expected: bool
) -> None:
    """(epoch + 1) % freq == 0, with freq 0 or None meaning never (the documented fix)."""
    task = _task(cell_graph)
    task.trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    task.trainer.fit_loop.epoch_progress.current.completed = epoch
    assert task.current_epoch == epoch
    assert task._is_scheduled(freq) is expected


def test_configure_optimizers_builds_the_named_optimizer_and_scheduler(
    cell_graph: HeteroData,
) -> None:
    """AdamW at lr 1e-2 alone; CosineAnnealingLR when asked; ReduceLROnPlateau by default."""
    alone = _task(cell_graph).configure_optimizers()
    assert isinstance(alone, torch.optim.AdamW)
    assert alone.param_groups[0]["lr"] == pytest.approx(1e-2)

    cosine = _task(
        cell_graph, lr_scheduler_config={"type": "CosineAnnealingLR", "T_max": 5}
    ).configure_optimizers()
    assert isinstance(
        cosine["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert cosine["lr_scheduler"]["interval"] == "epoch"

    plateau = _task(
        cell_graph, lr_scheduler_config={"factor": 0.5}
    ).configure_optimizers()
    assert isinstance(
        plateau["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert plateau["lr_scheduler"]["monitor"] == "val/gene_interaction/MSE"


def test_missing_loss_function_is_an_error_at_step_time(
    cell_graph: HeteroData,
    batch: HeteroData,
    loader: DataLoader[HeteroData],
    tmp_path: Any,
) -> None:
    """A task built without loss_func fails the first shared step, not silently."""
    task = _task(cell_graph, loss_func=None)
    with pytest.raises(ValueError, match="No loss function provided"):
        _trainer(tmp_path).fit(task, train_dataloaders=loader)


# ------------------------------------------------ Phase 15: a scripted stand-in model

LOGCOSH = (math.log(math.cosh(1.0)) + math.log(math.cosh(2.0))) / 2  # 0.8793917


class _Fixed(nn.Module):
    """Fixed predictions times one trainable scale, plus a fixed representations dict."""

    regularized_head_config: dict[str, Any] | None
    adjacency_matrices: dict[str, torch.Tensor]

    def __init__(self, predictions: torch.Tensor | None = None, **reps: Any) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.predictions = (
            torch.tensor([1.0, 3.0]) if predictions is None else predictions
        )
        self.reps: dict[str, Any] = {
            "h_CLS": torch.tensor([3.0, 4.0]),
            "H_genes": torch.ones(4, 2),
            "H_genes_pert": torch.full((2, 4, 2), 2.0),
            "graph_reg_loss": torch.tensor(0.25),
            "attention_weights": None,
            "residual_update_ratios": None,
        }
        self.reps.update(reps)
        self.return_attention: list[bool] = []

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData, return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.return_attention.append(return_attention)
        return self.predictions * self.scale, dict(self.reps)


def _fixed_batch(
    targets: list[float] | float | None = None,
    original: list[float] | float | None = None,
) -> HeteroData:
    batch = HeteroData()
    values = torch.tensor([2.0, 5.0] if targets is None else targets)
    n = 1 if values.dim() == 0 else values.numel()
    batch["gene"].perturbation_indices = torch.arange(n)
    batch["gene"].perturbation_indices_batch = torch.arange(n)
    batch["gene"].phenotype_values = values
    if original is not None:
        batch["gene"].phenotype_values_original = torch.tensor(original)
    return batch


def _fixed_task(model: _Fixed | None = None, **overrides: Any) -> RegressionTask:
    graph = HeteroData()
    graph["gene"].num_nodes = 4
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
    graph_arg: Any = graph  # annotated as a Tensor, cloned as a HeteroData
    return RegressionTask(model=model or _Fixed(), cell_graph=graph_arg, **kwargs)


def _fixed_loader() -> DataLoader[HeteroData]:
    dataset: Any = [_fixed_batch(), _fixed_batch()]
    return DataLoader(dataset, batch_size=None)


def _scale(task: RegressionTask) -> float:
    model = task.model
    assert isinstance(model, _Fixed)
    return float(model.scale.item())


class _LogRecorder:
    """Stands in for ``LightningModule.log``: the last value per name, as a float."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def __call__(self, name: str, value: Any, **kwargs: Any) -> None:
        self.values[name] = float(value)


def _recording(monkeypatch: pytest.MonkeyPatch, task: RegressionTask) -> _LogRecorder:
    recorder = _LogRecorder()
    monkeypatch.setattr(task, "log", recorder)
    return recorder


def _attach(task: RegressionTask, tmp_path: Any, epoch: int = 0) -> None:
    task.trainer = L.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    task.trainer.fit_loop.epoch_progress.current.completed = epoch


def test_fast_dev_run_exact_losses_metrics_step_and_cosine_schedule(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact train and val losses and metrics, the first AdamW step, one cosine step.

    The derivations are in the module docstring. The learning rate logged during the
    step is the pre-schedule 1e-2; the scheduler leaves 5e-3 on the optimizer.
    """
    calls: list[Any] = []
    monkeypatch.setattr("wandb.log", lambda *a, **k: calls.append((a, k)))
    task = _fixed_task(lr_scheduler_config={"type": "CosineAnnealingLR", "T_max": 2})
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        task, train_dataloaders=_fixed_loader(), val_dataloaders=_fixed_loader()
    )
    s = 0.9999 + 0.01
    assert _scale(task) == pytest.approx(s, abs=1e-6)
    p = [1.0 * s, 3.0 * s]
    val_loss = (math.log(math.cosh(p[0] - 2)) + math.log(math.cosh(p[1] - 5))) / 2
    metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
    assert metrics["train/loss"] == pytest.approx(LOGCOSH + 0.25, rel=1e-6)
    assert metrics["train/gene_interaction/MSE"] == pytest.approx(2.5)
    assert metrics["train/transformed/gene_interaction/Pearson"] == pytest.approx(1.0)
    assert metrics["val/loss"] == pytest.approx(val_loss + 0.25, rel=1e-5)
    assert metrics["val/gene_interaction/MSE"] == pytest.approx(
        ((p[0] - 2) ** 2 + (p[1] - 5) ** 2) / 2, rel=1e-5
    )
    assert metrics["val/residual_update_ratio"] == pytest.approx(math.sqrt(2))
    assert metrics["learning_rate"] == pytest.approx(1e-2)
    optimizer = trainer.optimizers[0]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    assert calls == []


def test_the_default_plateau_scheduler_steps_on_the_validation_mse(
    tmp_path: Any,
) -> None:
    """The default scheduler (no ``type``) is ReduceLROnPlateau on ``val/gene_interaction/MSE``.

    Contract (issue #534): under manual optimization Lightning steps no scheduler, so
    the plateau scheduler is stepped once per validation epoch on its monitor, which
    validation computes before ``on_train_epoch_end``; the training epoch end steps only
    epoch-interval schedulers. With a validation loop, one fast_dev_run epoch steps it
    once on the val MSE ``((s - 2)^2 + (3s - 5)^2) / 2`` with ``s = 0.9999 + 0.01`` (the
    scale after the first AdamW step), which becomes its ``best``; one step cannot
    reduce the rate, so it stays 1e-2. Without a validation loop the epoch completes
    and the scheduler is never stepped.
    """
    task = _fixed_task(lr_scheduler_config={"factor": 0.5})
    trainer = _fdr(tmp_path)
    trainer.fit(
        task, train_dataloaders=_fixed_loader(), val_dataloaders=_fixed_loader()
    )
    assert trainer.global_step == 1
    plateau = trainer.lr_scheduler_configs[0].scheduler
    assert isinstance(plateau, ReduceLROnPlateau)
    s = 0.9999 + 0.01
    val_mse = ((s - 2) ** 2 + (3 * s - 5) ** 2) / 2
    assert plateau.last_epoch == 1
    assert float(plateau.best) == pytest.approx(val_mse, rel=1e-5)
    assert trainer.callback_metrics["val/gene_interaction/MSE"].item() == (
        pytest.approx(val_mse, rel=1e-5)
    )
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-2)
    assert sorted(trainer.callback_metrics) == [
        "learning_rate",
        "train/cls_token_norm",
        "train/gene_interaction/MSE",
        "train/gene_interaction/Pearson",
        "train/gene_interaction/RMSE",
        "train/graph_reg_loss",
        "train/loss",
        "train/transformed/gene_interaction/MSE",
        "train/transformed/gene_interaction/Pearson",
        "train/transformed/gene_interaction/RMSE",
        "train/z_p_norm",
        "val/cls_token_norm",
        "val/gene_interaction/MSE",
        "val/gene_interaction/Pearson",
        "val/gene_interaction/RMSE",
        "val/graph_reg_loss",
        "val/loss",
        "val/residual_update_ratio",
        "val/transformed/gene_interaction/MSE",
        "val/transformed/gene_interaction/Pearson",
        "val/transformed/gene_interaction/RMSE",
        "val/z_p_norm",
    ]

    train_only = _fixed_task(lr_scheduler_config={"factor": 0.5})
    trainer_train_only = _fdr(tmp_path)
    trainer_train_only.fit(train_only, train_dataloaders=_fixed_loader())
    assert trainer_train_only.global_step == 1
    unstepped = trainer_train_only.lr_scheduler_configs[0].scheduler
    assert isinstance(unstepped, ReduceLROnPlateau)
    assert unstepped.last_epoch == 0
    assert sorted(trainer_train_only.callback_metrics) == [
        "learning_rate",
        "train/cls_token_norm",
        "train/gene_interaction/MSE",
        "train/gene_interaction/Pearson",
        "train/gene_interaction/RMSE",
        "train/graph_reg_loss",
        "train/loss",
        "train/transformed/gene_interaction/MSE",
        "train/transformed/gene_interaction/Pearson",
        "train/transformed/gene_interaction/RMSE",
        "train/z_p_norm",
    ]


def _fdr(tmp_path: Any) -> L.Trainer:
    return L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )


def _record_plots(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    """Record every ``Visualization`` call and every ``wandb.log`` payload."""
    visual: list[Any] = []
    logged: list[Any] = []

    class _Vis:
        def __init__(self, base_dir: str, max_points: int) -> None:
            visual.append(("init", base_dir, max_points))

        def visualize_model_outputs(self, *args: Any, **kwargs: Any) -> None:
            visual.append((args, kwargs))

    monkeypatch.setattr("torchcell.trainers.int_transformer_cell.Visualization", _Vis)
    monkeypatch.setattr(
        "torchcell.trainers.int_transformer_cell.genetic_interaction_score.box_plot",
        lambda true, pred: ("fig", true.tolist(), pred.tolist()),
    )
    monkeypatch.setattr("wandb.Image", lambda fig: ("image", fig))
    monkeypatch.setattr("wandb.log", lambda payload: logged.append(payload))
    monkeypatch.setattr(
        "torchcell.trainers.int_transformer_cell.plt.close", lambda fig: None
    )
    return visual, logged


def test_fast_dev_run_plots_val_then_train_samples_every_epoch(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``plot_every_n_epochs=1``: validation plots first (its epoch end runs inside the
    training epoch), then training. Train samples are the pre-step predictions [1, 3];
    validation sees 1.0099 * [1, 3]. Both carry the pooled latent (2.2, 2.4) per
    genotype and the loss name ``LogCoshLoss``.
    """
    visual, logged = _record_plots(monkeypatch)
    task = _fixed_task(plot_every_n_epochs=1)
    _fdr(tmp_path).fit(
        task, train_dataloaders=_fixed_loader(), val_dataloaders=_fixed_loader()
    )
    s = 0.9999 + 0.01
    assert logged == [
        {
            "val_sample/gene_interaction_box_plot": (
                "image",
                ("fig", [2.0, 5.0], pytest.approx([s, 3 * s], abs=1e-6)),
            )
        },
        {
            "train_sample/gene_interaction_box_plot": (
                "image",
                ("fig", [2.0, 5.0], [1.0, 3.0]),
            )
        },
    ]
    inits = [v for v in visual if v[0] == "init"]
    assert inits == [("init", str(tmp_path), 1000)] * 2
    calls = [v for v in visual if v[0] != "init"]
    stages = [kwargs["stage"] for _, kwargs in calls]
    assert stages == ["val_sample", "train_sample"]
    for args, _ in calls:
        predictions, true_values, latents, loss_name, epoch, stamp = args
        assert true_values.tolist() == [[2.0], [5.0]]
        assert torch.allclose(latents["H_pooled"], torch.tensor([[2.2, 2.4]] * 2))
        assert (loss_name, epoch, stamp) == ("LogCoshLoss", 0, None)
    assert calls[1][0][0].tolist() == [[1.0], [3.0]]


def test_trainer_test_logs_exact_test_metrics_and_plots_the_test_samples(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``trainer.test`` runs ``test_step`` once: loss log-cosh + 0.25, MSE 2.5, one
    ``test_sample`` plot, and the sample buffer is emptied afterwards.
    """
    visual, logged = _record_plots(monkeypatch)
    task = _fixed_task()
    results = _fdr(tmp_path).test(task, dataloaders=_fixed_loader(), verbose=False)
    assert results[0]["test/loss"] == pytest.approx(LOGCOSH + 0.25, rel=1e-6)
    assert results[0]["test/gene_interaction/MSE"] == pytest.approx(2.5)
    assert results[0]["test/gene_interaction/RMSE"] == pytest.approx(math.sqrt(2.5))
    assert list(logged[0]) == ["test_sample/gene_interaction_box_plot"]
    assert len(logged) == 1
    assert task.test_samples == {"true_values": [], "predictions": [], "latents": {}}
    calls = [v for v in visual if v[0] != "init"]
    assert [kwargs["stage"] for _, kwargs in calls] == ["test_sample"]


def test_scalar_prediction_and_target_are_lifted_to_one_by_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0-dim prediction, target and original all become [1, 1]: loss log cosh 1 + 0.25."""
    task = _fixed_task(_Fixed(predictions=torch.tensor(1.0)))
    log = _recording(monkeypatch, task)
    loss, predictions, original = task._shared_step(_fixed_batch(2.0, 2.0), 0, "train")
    assert predictions is not None and original is not None
    assert predictions.tolist() == [[1.0]]
    assert original.tolist() == [[2.0]]
    assert loss.item() == pytest.approx(math.log(math.cosh(1.0)) + 0.25, rel=1e-6)
    assert log.values["train/loss"] == pytest.approx(loss.item())


def test_batch_size_and_device_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch size falls back from ``x`` rows to perturbation count to value count to 1.

    Without ``perturbation_indices_batch`` the count is of perturbed GENES (3 here for
    what may be fewer genotypes), as the in-code comment says. ``forward`` and the
    profiling step read the device from whichever of those exists, else from the
    model's parameters, and both run on every shape.
    """
    task = _fixed_task()
    perts = HeteroData()
    perts["gene"].perturbation_indices = torch.tensor([4, 5, 6])
    values = HeteroData()
    values["gene"].phenotype_values = torch.tensor([2.0, 5.0])
    empty = HeteroData()
    empty["gene"].num_nodes = 0
    dense = HeteroData()
    dense["gene"].x = torch.zeros(5, 2)
    assert [task._get_batch_size(b) for b in (perts, values, empty)] == [3, 2, 1]
    for b in (dense, values, empty):
        predictions, _ = task(b)
        assert predictions.tolist() == [1.0, 3.0]
        assert task._cell_graph_device == torch.device("cpu")
    profiling = _fixed_task(execution_mode="dataloader_profiling")
    log = _recording(monkeypatch, profiling)
    sizes = []
    for b in (dense, values, empty):
        profiling._shared_step(b, 0, "val")
        sizes.append(log.values["val/dataloader_profile_batch_size"])
    assert sizes == [5.0, 2.0, 1.0]


def test_frozen_parameters_are_left_out_of_the_dummy_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only trainable parameters are tied into the unused-parameter and profiling losses.

    With ``scale`` frozen the dummy loss is the integer 0, and the profiling loss is the
    bare zero leaf: backward leaves the frozen parameter without a gradient.
    """
    model = _Fixed()
    model.scale.requires_grad_(False)
    task = _fixed_task(model, execution_mode="dataloader_profiling")
    _recording(monkeypatch, task)
    assert task._ensure_no_unused_params_loss() == 0
    loss, _, _ = task._shared_step(_fixed_batch(), 0, "train")
    assert loss.grad_fn is None
    assert loss.item() == 0.0
    trainable = _fixed_task(_Fixed())
    dummy = trainable._ensure_no_unused_params_loss()
    assert isinstance(dummy, torch.Tensor)
    assert dummy.item() == 0.0 and dummy.requires_grad


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


def test_validation_accumulates_ratios_and_degree_bias_with_edge_recovery_off(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostics every epoch, edge recovery never; two validation batches.

    The model's per-layer ratios [0.5, 0.25] sum to 1.0 and 0.5 over two batches. The
    degree bias runs on "g" layer 0 only: layer 5 is past the one attention map and
    "missing" has no adjacency. Its Spearman correlation of degrees [1, 1, 0, 0]
    against column sums [1.1, 1.3, 0.9, 0.7] is 4 / sqrt(20) = 0.8944272 per batch; no
    recall or precision is accumulated because that tier is off.
    """
    model = _Fixed(
        attention_weights=[ATTENTION.view(1, 1, 4, 4)],
        residual_update_ratios=[0.5, 0.25],
    )
    model.regularized_head_config = {
        "g": {"layer": [0, 5], "head": 0},
        "missing": {"layer": 0, "head": 0},
    }
    model.adjacency_matrices = {"g": ADJACENCY}
    task = _fixed_task(model, plot_transformer_diagnostics_every_n_epochs=1)
    _attach(task, tmp_path)
    _recording(monkeypatch, task)
    for batch_idx in (0, 1):
        task._shared_step(_fixed_batch(), batch_idx, "val")
    assert model.return_attention == [True, True]
    assert task.residual_update_accumulators == {
        0: {"sum_ratio": 1.0, "count": 2},
        1: {"sum_ratio": 0.5, "count": 2},
    }
    assert task.attention_stats_accumulators[0]["count"] == 2
    assert list(task.edge_recovery_accumulators) == ["g_L0_H0"]
    acc = task.edge_recovery_accumulators["g_L0_H0"]
    assert acc["degree_correlation_sum"] == pytest.approx(2 * 4 / math.sqrt(20))
    assert acc["degree_corr_count"] == 2
    assert (acc["count_nodes_deg"], acc["count_batches"]) == (0, 0)


def test_point_dist_graph_reg_and_generic_losses_log_components_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both loss branches share one component logger (issue #534).

    A one-element tensor and a number log as ``{stage}/{key}``; a multi-element tensor
    logs element by element as ``_0``, ``_1`` (the PointDistGraphReg branch used to drop
    it); a string is skipped; a non-dict second element logs nothing.
    """
    from torchcell.losses.point_dist_graph_reg import PointDistGraphReg

    class _Components(PointDistGraphReg):
        def __init__(self, components: Any) -> None:
            nn.Module.__init__(self)
            self.components = components

        def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, Any]:
            return torch.tensor(0.5), self.components

    components = {
        "one": torch.tensor(0.1),
        "vec": torch.tensor([1.0, 2.0]),
        "count": 3,
        "note": "text",
    }
    task = _fixed_task(loss_func=_Components(components))
    log = _recording(monkeypatch, task)
    loss, _, _ = task._shared_step(_fixed_batch(), 0, "train")
    assert loss.item() == 0.5  # the graph term is the loss's own, not added again
    component_keys = {"train/one", "train/count", "train/vec_0", "train/vec_1"}
    extra = {k: v for k, v in log.values.items() if k in component_keys}
    assert extra == pytest.approx(
        {"train/one": 0.1, "train/count": 3.0, "train/vec_0": 1.0, "train/vec_1": 2.0}
    )
    assert [k for k in log.values if "vec" in k or "note" in k] == [
        "train/vec_0",
        "train/vec_1",
    ]

    class _Tuple(nn.Module):
        def forward(self, *args: Any) -> tuple[torch.Tensor, Any]:
            return torch.tensor(0.5), components

    generic = _fixed_task(loss_func=_Tuple())
    generic_log = _recording(monkeypatch, generic)
    generic._shared_step(_fixed_batch(), 0, "train")
    assert {k: v for k, v in generic_log.values.items() if k in component_keys} == (
        extra
    )
    assert "train/note" not in generic_log.values

    bare = _fixed_task(loss_func=_Components(None))
    bare_log = _recording(monkeypatch, bare)
    bare._shared_step(_fixed_batch(), 0, "train")
    assert sorted(bare_log.values) == [
        "train/cls_token_norm",
        "train/loss",
        "train/z_p_norm",
    ]


class _Inverse(nn.Module):
    """An inverse transform that returns ``fn(values)`` as the phenotype values."""

    def __init__(self, fn: Any) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, data: HeteroData) -> HeteroData:
        data["gene"].phenotype_values = self.fn(data["gene"].phenotype_values)
        return data


def test_inverse_transform_output_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """0-dim and [B, 1] inverse outputs are both used: times ten gives MSE 250.

    Predictions [1, 3] invert to [10, 30] against originals [20, 50]: (100 + 400) / 2.
    A single genotype squeezes to 0-dim: 10 against 20 is MSE 100.

    An inverse whose values are not a tensor raises ``TypeError`` naming the transform
    and the returned type, and updates no original-scale metric (issue #534; it used to
    be ignored, scoring the transformed [1, 3] against [20, 50] as MSE 1285).
    """
    column = _fixed_task(inverse_transform=_Inverse(lambda v: (v * 10).unsqueeze(1)))
    _recording(monkeypatch, column)
    column._shared_step(_fixed_batch([2.0, 5.0], [20.0, 50.0]), 0, "val")
    mse = column._compute_metrics_safely(column.val_metrics)["val/gene_interaction/MSE"]
    assert mse.item() == pytest.approx(250.0)
    single = _fixed_task(
        _Fixed(predictions=torch.tensor([1.0])),
        inverse_transform=_Inverse(lambda v: v * 10),
    )
    _recording(monkeypatch, single)
    single._shared_step(_fixed_batch([2.0], [20.0]), 0, "val")
    mse = single._compute_metrics_safely(single.val_metrics)["val/gene_interaction/MSE"]
    assert mse.item() == pytest.approx(100.0)
    listed = _fixed_task(inverse_transform=_Inverse(lambda v: (v * 10).tolist()))
    _recording(monkeypatch, listed)
    with pytest.raises(
        TypeError,
        match=r"^inverse_transform _Inverse returned phenotype_values of type list; "
        r"expected torch\.Tensor$",
    ):
        listed._shared_step(_fixed_batch([2.0, 5.0], [20.0, 50.0]), 0, "val")
    assert listed.val_metrics["MSE"].update_count == 0


def test_all_nan_targets_update_no_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every target NaN: neither metric collection is updated, the loss is still logged."""
    task = _fixed_task()
    log = _recording(monkeypatch, task)
    task._shared_step(_fixed_batch([float("nan"), float("nan")]), 0, "train")
    assert task.train_metrics["MSE"].update_count == 0
    assert task.train_transformed_metrics["MSE"].update_count == 0
    assert math.isnan(log.values["train/loss"])


def test_sanity_check_never_plots_or_logs_diagnostics(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During the sanity check every plot is skipped even when scheduled; the
    accumulators are still reset and the validation samples are kept.
    """
    task = _fixed_task(
        plot_every_n_epochs=1,
        plot_transformer_diagnostics_every_n_epochs=1,
        plot_edge_recovery_every_n_epochs=1,
    )
    _attach(task, tmp_path)
    from lightning.pytorch.trainer.states import RunningStage

    task.trainer.state.stage = RunningStage.SANITY_CHECKING
    assert task.trainer.sanity_checking
    plotted: list[str] = []
    monkeypatch.setattr(task, "_plot_samples", lambda s, stage: plotted.append(stage))
    monkeypatch.setattr(
        task, "_plot_attention_diagnostics", lambda: plotted.append("attention")
    )
    monkeypatch.setattr(
        task, "_plot_edge_recovery_metrics", lambda: plotted.append("edges")
    )
    _recording(monkeypatch, task)
    task.val_samples["true_values"].append(torch.tensor([[2.0]]))
    task.attention_stats_accumulators = {0: {"count": 1}}
    task.edge_recovery_accumulators = {
        "g": {
            "count_nodes_deg": 0,
            "count_nodes_prec": dict.fromkeys(task.edge_recovery_ks, 0),
        }
    }
    task.on_validation_epoch_end()
    assert plotted == []
    kept = task.val_samples["true_values"]
    assert len(kept) == 1 and torch.equal(kept[0], torch.tensor([[2.0]]))
    assert (task.attention_stats_accumulators, task.edge_recovery_accumulators) == (
        {},
        {},
    )


def test_plot_samples_logs_oversmoothing_and_skips_an_all_nan_box_plot(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``z_p`` latent logs its centered Frobenius norm; all-NaN targets draw no box plot.

    z_p rows (0, 0) and (2, 0) have mean (1, 0), so the centered norm is sqrt 2. The
    empty ``H_pooled`` list is dropped, so the visualizer gets no latent.
    """
    visual, logged = _record_plots(monkeypatch)
    task = _fixed_task()
    _attach(task, tmp_path)
    samples = {
        "true_values": [torch.tensor([[float("nan")], [float("nan")]])],
        "predictions": [torch.tensor([[1.0], [2.0]])],
        "latents": {"z_p": [torch.tensor([[0.0, 0.0], [2.0, 0.0]])], "H_pooled": []},
    }
    task._plot_samples(samples, "x")
    assert logged == [{"x/oversmoothing_z_p": pytest.approx(math.sqrt(2))}]
    assert visual[1][0][2] == {}


def test_validation_step_frees_the_cuda_cache_every_fifty_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 1236 frees the cache when ``batch_idx > 0 and batch_idx % 50 == 0``, so the
    running count of calls after batches 0, 49, 50 and 100 is [0, 0, 1, 2]: batch 0 and
    batch 49 do not fire, batches 50 and 100 do.
    """
    task = _fixed_task()
    _recording(monkeypatch, task)
    emptied: list[int] = []
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: emptied.append(1))
    counts: list[int] = []
    for batch_idx in (0, 49, 50, 100):
        task.validation_step(_fixed_batch(), batch_idx)
        counts.append(len(emptied))
    assert counts == [0, 0, 1, 2]


def test_optimizer_config_renames_learning_rate_and_passes_the_rest() -> None:
    """``learning_rate`` becomes ``lr``; other keys reach the optimizer unchanged; an
    unknown optimizer name is an AttributeError from ``torch.optim``.
    """
    sgd = _fixed_task(
        optimizer_config={"type": "SGD", "learning_rate": 0.1, "momentum": 0.9}
    ).configure_optimizers()
    assert type(sgd) is torch.optim.SGD
    group = sgd.param_groups[0]
    assert (group["lr"], group["momentum"], group["nesterov"]) == (0.1, 0.9, False)
    with pytest.raises(
        AttributeError, match="^module 'torch.optim' has no attribute 'Nope'$"
    ):
        _fixed_task(optimizer_config={"type": "Nope"}).configure_optimizers()


def test_accumulation_schedule_keeps_the_last_threshold_reached(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """{0: 1, 5: 3} at epoch 2 stays at 1 and prints it; at epoch 5 it is 3."""
    task = _fixed_task(grad_accumulation_schedule={0: 1, 5: 3})
    _attach(task, tmp_path, epoch=2)
    task.on_train_epoch_start()
    assert task.current_accumulation_steps == 1
    _attach(task, tmp_path, epoch=5)
    task.on_train_epoch_start()
    assert task.current_accumulation_steps == 3
    assert capsys.readouterr().out == (
        "Epoch 2: Using gradient accumulation steps = 1\n"
        "Epoch 5: Using gradient accumulation steps = 3\n"
    )


def test_diagnostic_plots_skip_empty_accumulators(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Zero counts contribute nothing: no attention, recall, precision, mass or per-graph plot.

    A graph with no counted node at any k is left out of the precision metrics, so
    with every graph empty the precision plot is skipped with one logged INFO line
    rather than called with an empty inner dict (issue #534). Skipping, not raising:
    an empty accumulator is a legitimate state (no node reached degree k) and the
    recall and mass plots already skip it. A graph counted at some k plots only those
    k. With one counted attention layer and an empty residual accumulator, the
    attention plot gets ``residual_ratios=None`` and ``gradient_norms=None``.
    """
    calls: list[tuple[str, Any]] = []

    class _Recovery:
        def __init__(self, base_dir: str) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            return lambda *a, **k: calls.append((name, a))

    class _Diagnostics:
        def __init__(self, base_dir: str) -> None:
            pass

        def plot_attention_diagnostics(self, stats: Any, **kwargs: Any) -> None:
            calls.append(("attention", (stats, kwargs)))

    monkeypatch.setattr(
        "torchcell.viz.graph_recovery.GraphRecoveryVisualization", _Recovery
    )
    monkeypatch.setattr(
        "torchcell.viz.transformer_diagnostics.TransformerDiagnostics", _Diagnostics
    )
    task = _fixed_task()
    _attach(task, tmp_path)
    task.edge_recovery_accumulators = {
        "g_L0_H0": {
            "count_nodes_deg": 0,
            "count_nodes_prec": dict.fromkeys(task.edge_recovery_ks, 0),
            "count_batches": 0,
        }
    }
    with caplog.at_level("INFO", logger="torchcell.trainers.int_transformer_cell"):
        task._plot_edge_recovery_metrics()
    assert calls == []
    assert [
        r.getMessage()
        for r in caplog.records
        if r.name == "torchcell.trainers.int_transformer_cell"
    ] == [
        "Skipping the edge recovery precision plot at epoch 0: no graph has a "
        "counted node at any k."
    ]
    counts = dict.fromkeys(task.edge_recovery_ks, 0)
    counts[32] = 4
    task.edge_recovery_accumulators["g_L1_H0"] = {
        "count_nodes_deg": 0,
        "count_nodes_prec": counts,
        "sum_prec": dict.fromkeys(task.edge_recovery_ks, 1.0),
        "count_batches": 0,
    }
    task._plot_edge_recovery_metrics()
    assert calls == [
        (
            "plot_edge_recovery_precision",
            ({"g_L1_H0": {32: 0.25}}, [8, 32, 128, 320], 0, None),
        )
    ]
    calls.clear()
    zero = {"count": 0}
    task.attention_stats_accumulators = {0: dict(zero)}
    task._plot_attention_diagnostics()
    assert calls == []
    stats = {
        "entropy_sum": 2.0,
        "effective_rank_sum": 4.0,
        "top5_sum": 2.0,
        "top10_sum": 2.0,
        "top50_sum": 2.0,
        "max_row_weight_sum": 1.0,
        "col_entropy_sum": 3.0,
        "max_col_sum_sum": 5.0,
        "count": 2,
    }
    task.attention_stats_accumulators = {0: stats, 1: dict(zero)}
    task.residual_update_accumulators = {0: {"sum_ratio": 1.0, "count": 0}}
    task._plot_attention_diagnostics()
    assert calls == [
        (
            "attention",
            (
                {
                    0: {
                        "entropy": 1.0,
                        "effective_rank": 2.0,
                        "top5": 1.0,
                        "top10": 1.0,
                        "top50": 1.0,
                        "max_row_weight": 0.5,
                        "col_entropy": 1.5,
                        "max_col_sum": 2.5,
                    }
                },
                {
                    "residual_ratios": None,
                    "gradient_norms": None,
                    "num_epochs": 0,
                    "stage": "val",
                },
            ),
        )
    ]
