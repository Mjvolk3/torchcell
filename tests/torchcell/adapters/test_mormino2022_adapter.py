# tests/torchcell/adapters/test_mormino2022_adapter.py
"""The Mormino 2022 adapter conf enables only methods the CellAdapter actually has, and
this dataset's three environment edits and categorical readout reach the graph.
"""

from __future__ import annotations

import json
import os.path as osp
from typing import Any, cast

import torchcell.adapters.mormino2022_adapter as adapter_module
from tests.torchcell.adapters._crispr_adapter_conf import (
    adapter_method_names,
    conf_methods,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    ResponseCategory,
    SampleUnit,
)
from torchcell.datasets.scerevisiae import mormino2022 as m

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
    assert json.loads(props["serialized_data"])["units"] == m.UNITS
