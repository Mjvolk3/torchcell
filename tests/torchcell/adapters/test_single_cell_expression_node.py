# tests/torchcell/adapters/test_single_cell_expression_node.py
# [[tests.torchcell.adapters.test_single_cell_expression_node]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_single_cell_expression_node.py
"""The ``single cell expression phenotype`` node in its Jackson 2020 shape (#1016).

Jackson 2020's records project through the one single-cell UMI-count node class they
share with Boocock 2025 (whose shape is pinned in ``test_boocock2025_adapter.py``): the
count dicts and the per-cell lists are JSON strings, a gapped field is the JSON ``null``,
the totals, the cell count and the universe pin stay typed, the node id is the sha256 of
the phenotype dump, the reference collector emits one node per distinct reference
phenotype, and the emitted property set is exactly the schema config's declared set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import yaml

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import SingleCellExpressionPhenotype
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load(
    (REPO_ROOT / "biocypher" / "config" / "torchcell_schema_config.yaml").read_text(
        encoding="utf-8"
    )
)
NODE = "single cell expression phenotype"


def _phenotype(**overrides: Any) -> SingleCellExpressionPhenotype:
    fields: dict[str, Any] = {
        "umi_count": {"YAL002W": 5, "YAL001C": 2},
        "marker_umi_count": {"NATMX": 1, "KANMX": 0},
        "total_umis": 8,
        "n_features": 3,
        "feature_universe_sha256": "f" * 64,
        "n_cells": 2,
        "cell_cycle_stage_counts": None,
        "library_cell_counts": {"YPD": 2},
        "cell_barcodes": ["GGGT", "AAAC"],
        "cell_libraries": ["YPD", "YPD"],
        "cell_total_umis": [6, 2],
        "measurement_type": "scrnaseq_umi_count",
        "provenance_gaps": [
            ProvenanceGap(
                field="cell_cycle_stage_counts",
                reason=ProvenanceGapReason.not_reported_by_primary,
            )
        ],
    }
    fields.update(overrides)
    return SingleCellExpressionPhenotype(**fields)


def _sha(model: Any) -> str:
    return hashlib.sha256(json.dumps(model.model_dump()).encode("utf-8")).hexdigest()


def test_the_node_projects_dicts_and_lists_as_json_and_keeps_scalars_typed() -> None:
    phenotype = _phenotype()
    method = cast(Any, CellAdapter._single_cell_expression_phenotype_node).__wrapped__
    node = method(
        CellAdapter.__new__(CellAdapter),
        {"experiment": SimpleNamespace(phenotype=phenotype)},
        f"{NODE} (chunked)",
    )
    pid = _sha(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        NODE,
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
        "graph_level": "node",
        "label_name": "umi_count",
        "label_statistic_name": None,
        "umi_count": '{"YAL001C": 2, "YAL002W": 5}',
        "marker_umi_count": '{"KANMX": 0, "NATMX": 1}',
        "total_umis": 8,
        "n_features": 3,
        "feature_universe_sha256": "f" * 64,
        "n_cells": 2,
        "cell_cycle_stage_counts": "null",
        "library_cell_counts": '{"YPD": 2}',
        "cell_barcodes": '["GGGT", "AAAC"]',
        "cell_libraries": '["YPD", "YPD"]',
        "cell_total_umis": "[6, 2]",
        "measurement_type": "scrnaseq_umi_count",
    }
    declared = set(SCHEMA[NODE]["properties"])
    assert set(node.get_properties()) - {"id", "preferred_id"} == declared
    assert declared == set(SingleCellExpressionPhenotype.model_fields) - {
        "provenance_gaps"
    }
    assert NODE in SCHEMA["phenotype member of"]["source"]
    assert "pseudobulk umi count phenotype" not in SCHEMA


def test_the_reference_collector_emits_each_distinct_phenotype_once() -> None:
    a = _phenotype()
    b = _phenotype(cell_barcodes=["AAAC", "GGGT"], cell_total_umis=[2, 6])
    adapter = CellAdapter.__new__(CellAdapter)
    adapter.dataset = cast(
        Any,
        SimpleNamespace(
            experiment_reference_index=[
                SimpleNamespace(reference=SimpleNamespace(phenotype_reference=p))
                for p in (a, b, a)
            ]
        ),
    )
    nodes = adapter._get_single_cell_expression_phenotype_reference_nodes()
    assert [n.get_id() for n in nodes] == [_sha(a), _sha(b)]
    assert {n.get_preferred_id() for n in nodes} == {NODE}
    assert nodes[1].get_properties()["cell_barcodes"] == '["AAAC", "GGGT"]'
