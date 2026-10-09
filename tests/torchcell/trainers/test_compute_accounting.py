# tests/torchcell/trainers/test_compute_accounting.py
"""ComputeAccounting logs N, D and C for a tiny run, and its state survives a resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from torch.utils.data import DataLoader, TensorDataset

from torchcell.trainers.compute_accounting import ComputeAccounting, count_parameters


class _Recorder(Logger):
    """A logger that keeps every logged metric so the test can read it back."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics: list[dict[str, float]] = []
        self.hparams: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "recorder"

    @property
    def version(self) -> str:
        return "0"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append(dict(metrics))

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        self.hparams.update(dict(params))


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gene_embedding = torch.nn.Embedding(50, 8)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        )

    @property
    def num_parameters(self) -> dict[str, int]:
        emb = sum(p.numel() for p in self.gene_embedding.parameters())
        body = sum(p.numel() for p in self.body.parameters())
        return {"gene_embedding": emb, "body": body, "total": emb + body}

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body(self.gene_embedding(idx)).squeeze(-1)
        return out


class _Task(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Net()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        idx, y = batch
        return torch.nn.functional.mse_loss(self.model(idx), y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader(n: int = 64, batch: int = 16) -> DataLoader[Any]:
    idx = torch.randint(0, 50, (n,))
    y = torch.randn(n)
    return DataLoader(TensorDataset(idx, y), batch_size=batch)


def _last(metrics: list[dict[str, float]], key: str) -> float:
    vals = [m[key] for m in metrics if key in m]
    assert vals, f"{key} never logged"
    return vals[-1]


def test_logs_params_flops_and_records(tmp_path: Path) -> None:
    rec = _Recorder()
    cb = ComputeAccounting(unit="records", embedding_keys=("gene_embedding",))
    trainer = Trainer(
        max_epochs=2,
        logger=rec,
        callbacks=[cb],
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    task = _Task()
    trainer.fit(task, train_dataloaders=_loader())
    params = count_parameters(task.model, ("gene_embedding",))
    assert params["embedding"] == 50 * 8
    assert params["non_embedding"] == 8 * 16 + 16 + 16 + 1
    assert _last(rec.metrics, "compute/params_non_embedding") == params["non_embedding"]
    assert _last(rec.metrics, "compute/flops_per_step") > 0
    # 4 steps per epoch of 16 records; FLOPs per epoch is per step times steps
    assert _last(rec.metrics, "compute/steps_per_epoch") == 4
    assert _last(rec.metrics, "compute/flops_per_epoch") == pytest.approx(
        _last(rec.metrics, "compute/flops_per_step") * 4
    )
    assert _last(rec.metrics, "compute/flops_cumulative") == pytest.approx(
        2 * _last(rec.metrics, "compute/flops_per_epoch")
    )
    assert _last(rec.metrics, "compute/records_per_epoch") == 64
    assert _last(rec.metrics, "compute/records_seen") == 128
    assert _last(rec.metrics, "compute/epochs_completed") == 2
    assert _last(rec.metrics, "compute/gpu_hours_cumulative") > 0
    assert rec.hparams["compute_unit"] == "records"


def test_state_survives_resume(tmp_path: Path) -> None:
    cb = ComputeAccounting()
    cb.flops_cumulative = 123.0
    cb.records_seen = 7
    cb.epochs_completed = 3
    state = cb.state_dict()
    cb2 = ComputeAccounting()
    cb2.load_state_dict(state)
    assert cb2.flops_cumulative == 123.0
    assert cb2.records_seen == 7
    assert cb2.epochs_completed == 3
    assert cb2._measured is False
