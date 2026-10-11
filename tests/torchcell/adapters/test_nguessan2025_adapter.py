# tests/torchcell/adapters/test_nguessan2025_adapter.py
"""The N'Guessan 2025 adapter's conf, and a gapped stage partition on the graph."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import torchcell.adapters.nguessan2025_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    PHENOTYPE_METHOD,
    AdapterCase,
    Shape,
    assert_graph_integrity,
    assert_missing_conf,
    dataset_genotype_is_segregant,
    dataset_phenotype_class,
    expected_conf,
    expected_methods,
    install_recorder,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.nguessan2025_adapter import Nguessan2025Adapter
from torchcell.datamodels.schema import SingleCellExpressionPhenotype
from torchcell.datasets.scerevisiae.nguessan2025 import (
    CELL_CYCLE_GAP,
    Nguessan2025Dataset,
)


def test_a_gapped_stage_partition_projects_as_json_null() -> None:
    """A pooled segregant with no stage labels: the partition property is ``null``."""
    phenotype = SingleCellExpressionPhenotype(
        umi_count={"YAL001C": 2},
        total_umis=2,
        n_features=6240,
        feature_universe_sha256="c" * 64,
        n_cells=3,
        cell_cycle_stage_counts=None,
        library_cell_counts={"1": 3},
        provenance_gaps=[CELL_CYCLE_GAP],
    )
    props = CellAdapter._single_cell_expression_properties(phenotype)
    assert props["cell_cycle_stage_counts"] == "null"
    assert props["library_cell_counts"] == '{"1": 3}'
    assert (props["n_cells"], props["total_umis"]) == (3, 2)


_INIT_CASES = [
    AdapterCase(
        Nguessan2025Adapter,
        "nguessan2025_adapter.yaml",
        Shape("single cell expression phenotype", perturbation=False),
        Nguessan2025Dataset,
    )
]


def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Nguessan2025Adapter`` loads ``conf/nguessan2025_adapter.yaml``: segregant
    genotype nodes, NO perturbation and NO environment-perturbation nodes, the ``single
    cell expression phenotype`` pair, memory_reduction_factor 1.0 on every chunked method.

    The shared harness pairs an adapter with its dataset through the dataset-to-adapter
    map, which this dataset is not in until #1014; the pairing is named here instead.
    """
    install_recorder(monkeypatch)
    dataset = SimpleNamespace(name="FakeDataset")
    adapter = Nguessan2025Adapter(
        dataset=dataset,  # type: ignore[arg-type]
        process_workers=3,
        io_workers=2,
        chunk_size=500,
        loader_batch_size=50,
    )
    shape = _INIT_CASES[0].shape
    assert dataset_genotype_is_segregant(Nguessan2025Dataset)
    assert OmegaConf.to_container(adapter.config) == expected_conf(shape, True)
    nodes, edges = expected_methods(shape, True)
    assert [m for m in adapter.supported_node_methods if m in set(nodes)] == nodes
    assert [m for m in adapter.supported_edge_methods if m in set(edges)] == edges
    assert_graph_integrity(nodes, edges, shape.phenotype)
    assert PHENOTYPE_METHOD[dataset_phenotype_class(Nguessan2025Dataset)] == (
        shape.phenotype
    )
    assert (adapter.dataset, adapter.chunk_size) == (dataset, 500)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
