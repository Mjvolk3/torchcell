"""COO label decode of RegressionTask: one value per row, conflicting rows masked."""

from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.trainers.int_transformer_cell import RegressionTask

GENE_NUM = 8


def _cell_graph() -> HeteroData:
    cg = HeteroData()
    cg["gene"].num_nodes = GENE_NUM
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    return cg


def _task(**kwargs: Any) -> RegressionTask:
    torch.manual_seed(0)
    model = CellGraphTransformer(
        gene_num=GENE_NUM,
        hidden_channels=16,
        num_transformer_layers=1,
        num_attention_heads=4,
        cell_graph=_cell_graph(),
    )
    return RegressionTask(
        model=model,
        cell_graph=_cell_graph(),
        optimizer_config={"type": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
        lr_scheduler_config=None,
        device="cpu",
        **kwargs,
    )


def _batch(rows: list[int], types: list[int], values: list[float]) -> HeteroData:
    b = HeteroData()
    b["gene"].phenotype_types = [["fitness", "gene_interaction"]]
    b["gene"].phenotype_type_indices = torch.tensor(types)
    b["gene"].phenotype_values_batch = torch.tensor(rows)
    b["gene"].phenotype_values = torch.tensor(values)
    b["gene"].phenotype_values_original = torch.tensor(values) * 10
    return b


def test_one_value_per_row_and_nan_where_absent() -> None:
    task = _task()
    # rows 0 and 2 carry fitness, row 1 carries only gene_interaction
    batch = _batch(rows=[0, 1, 2], types=[0, 1, 0], values=[0.9, -0.1, 1.1])
    out = task._coo_label(batch, "fitness", 3, original=False)
    assert out.shape == (3, 1)
    assert out[0, 0].item() == pytest.approx(0.9) and out[2, 0].item() == pytest.approx(
        1.1
    )
    assert torch.isnan(out[1, 0])
    orig = task._coo_label(batch, "fitness", 3, original=True)
    assert orig[0, 0].item() == pytest.approx(9.0)
    assert task.coo_conflict_rows == {}


def test_a_row_with_two_values_of_one_label_is_masked_and_counted() -> None:
    task = _task()
    # row 1 carries fitness twice (a measured 1.0 beside a SynthLethDB 0.0), row 0 once
    batch = _batch(rows=[0, 1, 1, 2], types=[0, 0, 0, 1], values=[0.9, 1.0, 0.0, -0.2])
    out = task._coo_label(batch, "fitness", 3, original=False)
    assert out[0, 0].item() == pytest.approx(0.9)
    assert torch.isnan(out[1, 0]), "the conflicting row is masked, never averaged"
    assert torch.isnan(out[2, 0])
    assert task.coo_conflict_rows == {"fitness": 1}
    # the other label on the same batch is untouched
    gi = task._coo_label(batch, "gene_interaction", 3, original=False)
    assert gi[2, 0].item() == pytest.approx(-0.2) and torch.isnan(gi[1, 0])
    # counted cumulatively across calls
    task._coo_label(batch, "fitness", 3, original=True)
    assert task.coo_conflict_rows == {"fitness": 2}
