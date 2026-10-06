# tests/torchcell/adapters/test_mota2024_adapter.py
"""Mota 2024 adapter: the conf enables only methods CellAdapter has, and
the nodes this dataset depends on carry the properties it depends on.
"""

from __future__ import annotations

import inspect
import json
import os.path as osp
import re
from typing import Any

import pytest
import yaml

import torchcell.adapters.mota2024_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.mota2024_adapter import EnvChemgenMota2024Adapter
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    ResponseCategory,
)
from torchcell.datasets.scerevisiae import mota2024 as m
from torchcell.datasets.scerevisiae.mota2024 import EnvChemgenMota2024Dataset

_CONF_DIR = osp.join(osp.dirname(inspect.getfile(CellAdapter)), "conf")
_METHOD_NAME_RE = re.compile(r'\(\s*"([^"]+)",\s*self\._', re.MULTILINE)


def _available_method_names() -> set[str]:
    """Every method name ``CellAdapter.__init__`` registers in its two tables."""
    return set(_METHOD_NAME_RE.findall(inspect.getsource(CellAdapter.__init__)))


def _conf(slug: str) -> dict[str, Any]:
    with open(osp.join(_CONF_DIR, f"{slug}_adapter.yaml")) as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)["cell_adapter"]
    return loaded


def _check_conf(slug: str) -> None:
    conf = _conf(slug)
    nodes = [entry["method_name"] for entry in conf["node_methods"]]
    edges = [entry["method_name"] for entry in conf["edge_methods"]]
    missing = (set(nodes) | set(edges)) - _available_method_names()
    assert not missing, missing
    # gene-keyed genotype, not the segregant node Bloom 2019 uses
    assert "genotype (chunked)" in nodes and "perturbation (chunked)" in nodes
    assert "segregant genotype (chunked)" not in nodes
    assert "perturbation to genotype (chunked)" in edges
    for name in (
        "environment (chunked)",
        "media (chunked)",
        "temperature (chunked)",
        "environment perturbation (chunked)",
        "environment response phenotype (chunked)",
        "environment response phenotype reference",
    ):
        assert name in nodes, name
    for name in (
        "media to environment (chunked)",
        "temperature to environment (chunked)",
        "environment perturbation to environment (chunked)",
        "phenotype to experiment (chunked)",
    ):
        assert name in edges, name


def test_conf_enables_only_methods_the_cell_adapter_has() -> None:
    _check_conf("mota2024")


def test_adapter_class_reads_its_own_conf() -> None:
    assert osp.exists(osp.join(_CONF_DIR, "mota2024_adapter.yaml"))
    assert EnvChemgenMota2024Adapter.__mro__[1] is CellAdapter


def test_both_perturbations_become_distinct_environment_perturbation_nodes() -> None:
    spec = next(s for s in m._ACID_SPECS if s["acid"] == "acetic")
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    acid, ph = dataset._environment(spec).perturbations
    acid_node = CellAdapter._environment_perturbation_node_from(acid)
    ph_node = CellAdapter._environment_perturbation_node_from(ph)
    assert acid_node.get_id() != ph_node.get_id()
    assert acid_node.get_properties()["compound_name"] == "acetic acid"
    # a physical factor projects its factor, its magnitude and the agent that set
    # it onto the same columns a small molecule uses, so pH 4.5 is queryable
    ph_props = ph_node.get_properties()
    assert ph_props["perturbation_type"] == "environment_physical"
    assert ph_props["factor"] == "pH"
    assert ph_props["compound_name"] == "hydrochloric acid"
    assert ph_props["concentration_value"] == 4.5
    assert acid_node.get_properties()["factor"] is None
    # the full record travels in the Experiment blob (inside its environment)
    assert "serialized_data" not in ph_props
    serialized = json.loads(json.dumps(ph.model_dump()))
    assert serialized["factor"] == "pH"
    assert serialized["magnitude"]["value"] == 4.5
    assert serialized["agent"]["name"] == "hydrochloric acid"


def test_the_ordinal_rank_and_its_label_are_both_projected() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.ordinal,
        assay_type=AssayType.spot_dilution,
        environment_response=2.0,
        category=ResponseCategory.severely_reduced,
        category_label="++",
        units=m.MEASUREMENT_UNITS,
    )
    props = CellAdapter._environment_response_properties(phenotype)
    assert props["environment_response"] == 2.0
    assert props["category"] == "severely_reduced"
    assert props["category_label"] == "++"
    assert props["measurement_type"] == "ordinal"


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        EnvChemgenMota2024Adapter,
        "mota2024_adapter.yaml",
        Shape("environment response phenotype", env_perturbation=True),
        EnvChemgenMota2024Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``EnvChemgenMota2024Adapter`` loads ``conf/mota2024_adapter.yaml``: 17 node and 15 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
