# tests/torchcell/adapters/test_boocock2025_adapter.py
"""The single-cell expression node method and the Boocock 2025 adapter's conf."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import pytest

import torchcell.adapters.boocock2025_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.boocock2025_adapter import Boocock2025Adapter
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import SingleCellExpressionPhenotype
from torchcell.datasets.scerevisiae.boocock2025 import Boocock2025Dataset


def _phenotype() -> SingleCellExpressionPhenotype:
    return SingleCellExpressionPhenotype(
        umi_count={"YAL002W": 3, "YAL001C": 1},
        total_umis=6,
        n_features=5435,
        feature_universe_sha256="b" * 64,
        n_cells=2,
        cell_cycle_stage_counts={"G1": 1, "S": 1},
        library_cell_counts={"04_ByxRM_51-": 2},
    )


def test_single_cell_expression_properties_project_every_field() -> None:
    """Dicts become JSON strings in sorted key order; totals stay typed."""
    props = CellAdapter._single_cell_expression_properties(_phenotype())
    assert props == {
        "graph_level": "node",
        "label_name": "umi_count",
        "label_statistic_name": None,
        "umi_count": '{"YAL001C": 1, "YAL002W": 3}',
        "total_umis": 6,
        "n_features": 5435,
        "feature_universe_sha256": "b" * 64,
        "n_cells": 2,
        "cell_cycle_stage_counts": '{"G1": 1, "S": 1}',
        "library_cell_counts": '{"04_ByxRM_51-": 2}',
        "measurement_type": "scrnaseq_umi_count",
    }


def test_phenotype_node_is_content_addressed() -> None:
    phenotype = _phenotype()

    class _Experiment:
        pass

    experiment = _Experiment()
    experiment.phenotype = phenotype  # type: ignore[attr-defined]
    adapter = CellAdapter.__new__(CellAdapter)
    undecorated = cast(
        Any, CellAdapter._single_cell_expression_phenotype_node
    ).__wrapped__
    node = undecorated(
        adapter,
        {"experiment": experiment},
        "single cell expression phenotype (chunked)",
    )
    expected = hashlib.sha256(
        json.dumps(phenotype.model_dump()).encode("utf-8")
    ).hexdigest()
    assert node.get_id() == expected
    assert node.get_label() == "single cell expression phenotype"
    assert node.get_properties()["total_umis"] == 6


def test_reference_nodes_are_deduplicated_by_content() -> None:
    phenotype = _phenotype()

    class _Reference:
        phenotype_reference = phenotype

    class _Entry:
        reference = _Reference()

    adapter = CellAdapter.__new__(CellAdapter)
    adapter.dataset = type(
        "D", (), {"experiment_reference_index": [_Entry(), _Entry()]}
    )()
    nodes = adapter._get_single_cell_expression_phenotype_reference_nodes()
    assert [n.get_label() for n in nodes] == ["single cell expression phenotype"]
    assert nodes[0].get_properties()["n_cells"] == 2


_INIT_CASES = [
    AdapterCase(
        Boocock2025Adapter,
        "boocock2025_adapter.yaml",
        Shape("single cell expression phenotype", perturbation=False),
        Boocock2025Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Boocock2025Adapter`` loads ``conf/boocock2025_adapter.yaml``: segregant genotype
    nodes, NO perturbation and NO environment-perturbation nodes (no culture carries
    one), the ``single cell expression phenotype`` pair, memory_reduction_factor 1.0 on
    every chunked method.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
