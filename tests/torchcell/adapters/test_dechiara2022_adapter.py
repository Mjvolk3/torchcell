# tests/torchcell/adapters/test_dechiara2022_adapter.py
# [[tests.torchcell.adapters.test_dechiara2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_dechiara2022_adapter.py
"""``torchcell.adapters.dechiara2022_adapter`` constructor tests on name-only datasets.

The three adapters add no node or edge builder of their own; what each adds is the conf
it loads (the knowledge-graph enable-list) and the ``CellAdapter`` wiring. The checks
are ``_adapter_init_harness``'s: the exact conf rebuilt from the dataset's graph shape,
the wiring, and the refusal of a missing conf.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import torchcell.adapters.dechiara2022_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.dechiara2022_adapter import (
    CellYieldDeChiara2022Adapter,
    DoublingTimeDeChiara2022Adapter,
    LifeCycleDeChiara2022Adapter,
)
from torchcell.datamodels.schema import LifeCycleAssay, LifeCyclePhenotype, SampleUnit
from torchcell.datasets.scerevisiae.dechiara2022 import (
    CellYieldDeChiara2022Dataset,
    DoublingTimeDeChiara2022Dataset,
    LifeCycleDeChiara2022Dataset,
)

_INIT_CASES = [
    AdapterCase(
        LifeCycleDeChiara2022Adapter,
        "life_cycle_dechiara2022_adapter.yaml",
        Shape("life cycle phenotype", perturbation=False),
        LifeCycleDeChiara2022Dataset,
    ),
    AdapterCase(
        DoublingTimeDeChiara2022Adapter,
        "doubling_time_dechiara2022_adapter.yaml",
        Shape(
            "environment response phenotype", perturbation=False, env_perturbation=True
        ),
        DoublingTimeDeChiara2022Dataset,
    ),
    AdapterCase(
        CellYieldDeChiara2022Adapter,
        "cell_yield_dechiara2022_adapter.yaml",
        Shape(
            "environment response phenotype", perturbation=False, env_perturbation=True
        ),
        CellYieldDeChiara2022Dataset,
    ),
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf each constructor loads is exactly the one its dataset's shape needs.

    * ``LifeCycleDeChiara2022Adapter``: ``life cycle phenotype``; gene-keyed genotype
      with NO perturbation nodes (each isolate carries ~5,000); no environment
      perturbation (the five protocols carry none).
    * ``DoublingTimeDeChiara2022Adapter`` and ``CellYieldDeChiara2022Adapter``:
      ``environment response phenotype``; no perturbation nodes; the carbon or nitrogen
      source served as ``environment perturbation`` nodes.

    Each adapter is checked against the dataset ``dataset_adapter_map`` pairs it with,
    the conf order is the registration order, no edge dangles, every chunked entity
    node is linked and the phenotype method matches the dataset's ``experiment_class``.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)


# --------------------------------------------------------------------------- #
# The CellAdapter's life-cycle phenotype node and reference collector
# --------------------------------------------------------------------------- #
def _life_cycle(**kw: Any) -> LifeCyclePhenotype:
    fields: dict[str, Any] = {
        "life_cycle": {"sporulation_efficiency@24h": 22.0, "dyad_fraction@24h": 0.0},
        "assay": LifeCycleAssay.microscopy_ascus_count,
        "cells_per_sample": 100,
        "cells_per_sample_is_minimum": True,
    }
    fields.update(kw)
    return LifeCyclePhenotype(**fields)


def _id(phenotype: LifeCyclePhenotype) -> str:
    return hashlib.sha256(
        json.dumps(phenotype.model_dump()).encode("utf-8")
    ).hexdigest()


def test_the_life_cycle_node_projects_the_time_course_as_json() -> None:
    """The keyed time course is one JSON string; assay and counts stay typed."""
    phenotype = _life_cycle()
    handler = cast(Any, CellAdapter._life_cycle_phenotype_node).__wrapped__
    node = handler(
        CellAdapter.__new__(CellAdapter),
        {"experiment": SimpleNamespace(phenotype=phenotype)},
        "life cycle phenotype (chunked)",
    )
    pid = _id(phenotype)
    assert (node.get_id(), node.get_label(), node.get_preferred_id()) == (
        pid,
        "life cycle phenotype",
        f"phenotype_{pid}",
    )
    assert node.get_properties() == {
        "graph_level": "global",
        "label_name": "life_cycle",
        "label_statistic_name": None,
        "life_cycle": '{"dyad_fraction@24h": 0.0, "sporulation_efficiency@24h": 22.0}',
        "assay": "microscopy_ascus_count",
        "n_samples": None,
        "sample_unit": None,
        "cells_per_sample": 100,
        "cells_per_sample_is_minimum": True,
        "id": pid,
        "preferred_id": f"phenotype_{pid}",
    }


def test_life_cycle_reference_nodes_are_deduplicated() -> None:
    """Three references, two equal: two nodes, with a stated sample unit typed."""
    first = _life_cycle(
        life_cycle={"stationary_phase_viability@7d": 63.7},
        assay=LifeCycleAssay.flow_cytometry_viability_stain,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        cells_per_sample=10_000,
        cells_per_sample_is_minimum=False,
    )
    second = _life_cycle()
    adapter = CellAdapter.__new__(CellAdapter)
    adapter.dataset = cast(
        Any,
        SimpleNamespace(
            experiment_reference_index=[
                SimpleNamespace(reference=SimpleNamespace(phenotype_reference=p))
                for p in (first, first, second)
            ]
        ),
    )
    nodes = adapter._get_life_cycle_phenotype_reference_nodes()
    assert [(n.get_id(), n.get_label()) for n in nodes] == [
        (_id(first), "life cycle phenotype"),
        (_id(second), "life cycle phenotype"),
    ]
    props = nodes[0].get_properties()
    assert (props["sample_unit"], props["n_samples"], props["preferred_id"]) == (
        "biological_replicate",
        1,
        "life cycle phenotype",
    )
