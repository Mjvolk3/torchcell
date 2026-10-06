# tests/torchcell/adapters/test_mormino2022_adapter.py
"""The Mormino 2022 adapter conf enables only methods the CellAdapter actually has, and
this dataset's three environment edits and categorical readout reach the graph.
"""

from __future__ import annotations

import os.path as osp
from typing import Any, cast

import pytest

import torchcell.adapters.mormino2022_adapter as _init_module
import torchcell.adapters.mormino2022_adapter as adapter_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters._crispr_adapter_conf import (
    adapter_method_names,
    conf_methods,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.mormino2022_adapter import Mormino2022Adapter
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    ResponseCategory,
    SampleUnit,
)
from torchcell.datasets.scerevisiae import mormino2022 as m
from torchcell.datasets.scerevisiae.mormino2022 import CrispriMormino2022Dataset

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "crispri_mormino2022_adapter.yaml",
)


def test_every_configured_method_exists_in_the_adapter_tables() -> None:
    node_names, edge_names = adapter_method_names()
    conf_nodes, conf_edges = conf_methods(CONF)
    assert set(conf_nodes) <= node_names, set(conf_nodes) - node_names
    assert set(conf_edges) <= edge_names, set(conf_edges) - edge_names


def test_the_conf_serves_the_crispr_construct_and_environment_perturbation_pairs() -> (
    None
):
    conf_nodes, conf_edges = conf_methods(CONF)
    assert "crispr construct (chunked)" in conf_nodes
    assert "crispr construct to perturbation (chunked)" in conf_edges
    assert "environment perturbation (chunked)" in conf_nodes
    assert "environment perturbation to environment (chunked)" in conf_edges


def test_all_three_environment_edits_become_distinct_nodes() -> None:
    dataset = m.CrispriMormino2022Dataset.__new__(m.CrispriMormino2022Dataset)
    environment = m.CrispriMormino2022Dataset._environment(dataset)
    nodes = [
        CellAdapter._environment_perturbation_node_from(perturbation)
        for perturbation in environment.perturbations
    ]
    assert len({node.get_id() for node in nodes}) == 3
    types = {node.get_properties()["perturbation_type"] for node in nodes}
    assert types == {"small_molecule", "environment_physical"}


def test_the_categorical_readout_projects_its_typed_call_and_source_symbol() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        assay_type=AssayType.biosensor_readout,
        category=ResponseCategory.enhanced,
        category_label="+",
        n_samples=2,
        sample_unit=SampleUnit.biological_replicate,
        units=m.UNITS,
    )
    props = cast(Any, CellAdapter)._environment_response_properties(phenotype)
    assert props["category"] == "enhanced"
    assert props["category_label"] == "+"
    assert props["assay_type"] == "biosensor_readout"
    assert props["environment_response"] is None
    # the full record (units included) travels in the Experiment blob, not here
    assert "serialized_data" not in props
    assert phenotype.model_dump()["units"] == m.UNITS


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        Mormino2022Adapter,
        "crispri_mormino2022_adapter.yaml",
        Shape("environment response phenotype", crispr=True, env_perturbation=True),
        CrispriMormino2022Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``Mormino2022Adapter`` loads ``conf/crispri_mormino2022_adapter.yaml``: 18 node and 16 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; CRISPR construct nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

    The adapter is checked against the dataset ``dataset_adapter_map`` pairs it with
    (asserted to be the class above). The conf lists its methods in the order the
    adapter runs them, no edge dangles, every chunked entity node is linked, and the
    phenotype method matches that dataset's ``experiment_class``. The adapter keeps the dataset and the worker / chunk sizes
    it was given (3, 2, 500, 50), calls ``wandb.init`` once and logs the method table
    (event number, name, node/edge, factor or NaN for a non-chunked method) then the
    dataset name and the pinned start time; nothing is printed.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, _init_module, monkeypatch)
