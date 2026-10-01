# tests/torchcell/trainers/test_int_hetero_cell.py
# [[tests.torchcell.trainers.test_int_hetero_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_int_hetero_cell.py
"""Batch sizing in the 006 hetero trainers (``RegressionTask``, ``DiffusionRegressionTask``).

Both tasks size a perturbation batch (no ``gene.x``) by the collated batch's
``num_graphs``. Reading ``max(perturbation_indices_batch) + 1`` instead drops a trailing
genotype with no perturbed gene (issue #567). The ladder below that is unchanged: no
``perturbation_indices_batch`` counts perturbed genes, then phenotype values, then 1.
"""

from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from torchcell.trainers.int_hetero_cell import DiffusionRegressionTask, RegressionTask

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
