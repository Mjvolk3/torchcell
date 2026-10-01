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

from typing import Any, Literal

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from torchcell.losses.dcell import DCellLoss
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
