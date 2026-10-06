# tests/torchcell/adapters/test_smith2006_adapter.py
"""The Smith 2006 adapter conf enables only methods the CellAdapter actually has, and
this dataset's ordinal records project onto the typed node properties.

A method name in the conf that is not in ``CellAdapter``'s tables fails SILENTLY at KG
build time (the method is simply never called), so the enable-list is checked against
the tables rather than eyeballed.
"""

from __future__ import annotations

import inspect
import os.path as osp
import re
from typing import Any, cast

import pytest
import yaml

import torchcell.adapters.smith2006_adapter as _init_module
import torchcell.adapters.smith2006_adapter as adapter_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.smith2006_adapter import Smith2006Adapter
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPBO
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    MeasurementType,
    PhysicalFactor,
    ResponseCategory,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.scerevisiae.smith2006 import FattyAcidSmith2006Dataset

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "env_chemgen_smith2006_adapter.yaml",
)


def adapter_method_names() -> tuple[set[str], set[str]]:
    """Every node and edge method name ``CellAdapter.__init__`` registers."""
    source = inspect.getsource(CellAdapter.__init__)
    node_block, edge_block = source.split("self.edge_methods = [", 1)
    node_block = node_block.split("self.node_methods = [", 1)[1]
    pattern = r'"([^"]+)",\s*\n?\s*self\._'
    return set(re.findall(pattern, node_block)), set(re.findall(pattern, edge_block))


def conf_methods(path: str) -> tuple[list[str], list[str]]:
    """The node and edge method names an adapter conf enables."""
    with open(path) as handle:
        conf = yaml.safe_load(handle)["cell_adapter"]
    return (
        [method["method_name"] for method in conf["node_methods"]],
        [method["method_name"] for method in conf["edge_methods"]],
    )


def test_every_configured_method_exists_in_the_adapter_tables() -> None:
    node_names, edge_names = adapter_method_names()
    assert len(node_names) > 20 and len(edge_names) > 10
    conf_nodes, conf_edges = conf_methods(CONF)
    assert set(conf_nodes) <= node_names, set(conf_nodes) - node_names
    assert set(conf_edges) <= edge_names, set(conf_edges) - edge_names


def test_the_conf_serves_the_environment_perturbation_and_response_phenotype() -> None:
    conf_nodes, conf_edges = conf_methods(CONF)
    assert "environment perturbation (chunked)" in conf_nodes
    assert "environment response phenotype (chunked)" in conf_nodes
    assert "environment response phenotype reference" in conf_nodes
    assert "environment perturbation to environment (chunked)" in conf_edges


def test_the_conf_has_no_crispr_construct_pair() -> None:
    """This screen's genotypes are KanMX deletions; a construct node would be empty."""
    conf_nodes, conf_edges = conf_methods(CONF)
    assert "crispr construct (chunked)" not in conf_nodes
    assert "crispr construct to perturbation (chunked)" not in conf_edges


def _environment() -> Environment:
    return Environment(
        media=YPBO,
        temperature=Temperature(value=30.0),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=Concentration(value=0.1, unit=ConcentrationUnit.percent_w_v),
                agent=resolved_compound("oleic acid"),
            )
        ],
        aerobicity="aerobic",
        duration_hours=72.0,
    )


def test_the_carbon_source_factor_becomes_an_environment_perturbation_node() -> None:
    (perturbation,) = _environment().perturbations
    node = CellAdapter._environment_perturbation_node_from(perturbation)
    assert node.get_label() == "environment perturbation"
    props = node.get_properties()
    assert props["perturbation_type"] == "environment_physical"
    assert "serialized_data" not in props


def test_the_ordinal_phenotype_projects_its_typed_call() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.ordinal,
        assay_type=AssayType.halo_zone,
        environment_response=1.0,
        category=ResponseCategory.severely_reduced,
        category_label="defective",
        n_samples=3,
        sample_unit=SampleUnit.biological_replicate,
        units="clear-zone size",
    )
    props = cast(Any, CellAdapter)._environment_response_properties(phenotype)
    assert props["category"] == ResponseCategory.severely_reduced.value
    assert props["category_label"] == "defective"
    assert props["environment_response"] == 1.0
    assert props["assay_type"] == AssayType.halo_zone.value


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        Smith2006Adapter,
        "env_chemgen_smith2006_adapter.yaml",
        Shape("environment response phenotype", env_perturbation=True),
        FattyAcidSmith2006Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``Smith2006Adapter`` loads ``conf/env_chemgen_smith2006_adapter.yaml``: 17 node and 15 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
