# tests/torchcell/adapters/test_costanzo2021_adapter.py
"""Costanzo 2021 adapter: the conf enables only methods CellAdapter has, and
the nodes this dataset depends on carry the properties it depends on.
"""

from __future__ import annotations

import inspect
import os.path as osp
import re
from typing import Any

import pytest
import yaml

import torchcell.adapters.costanzo2021_adapter as _init_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.costanzo2021_adapter import EnvChemgenCostanzo2021Adapter
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import Media
from torchcell.datasets.scerevisiae import costanzo2021 as c
from torchcell.datasets.scerevisiae.costanzo2021 import EnvChemgenCostanzo2021Dataset

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


def _media_node_id(media: Media) -> str:
    """The adapter's media node id: identity by COMPOSITION (cell_adapter)."""
    return CellAdapter._media_node_id(media)


def test_conf_enables_only_methods_the_cell_adapter_has() -> None:
    _check_conf("costanzo2021")


def test_adapter_class_reads_its_own_conf() -> None:
    assert osp.exists(osp.join(_CONF_DIR, "costanzo2021_adapter.yaml"))
    assert EnvChemgenCostanzo2021Adapter.__mro__[1] is CellAdapter


def test_media_nodes_join_the_served_sga_family() -> None:
    """The 13 dosed conditions and served Costanzo 2016 must hash to ONE media node."""
    spec = next(s for s in c._CONDITIONS if s["col"] == "Benomyl")
    dataset = c.EnvChemgenCostanzo2021Dataset.__new__(c.EnvChemgenCostanzo2021Dataset)
    node_id = _media_node_id(dataset._environment(spec).media)
    assert node_id == _media_node_id(SGA_DM_SELECTION)
    galactose = next(s for s in c._CONDITIONS if s["col"] == "Galactose")
    assert _media_node_id(dataset._environment(galactose).media) != node_id


def test_the_dosed_compound_is_projected_on_the_perturbation_node() -> None:
    spec = next(s for s in c._CONDITIONS if s["col"] == "Benomyl")
    dataset = c.EnvChemgenCostanzo2021Dataset.__new__(c.EnvChemgenCostanzo2021Dataset)
    (perturbation,) = dataset._environment(spec).perturbations
    props = CellAdapter._environment_perturbation_node_from(
        perturbation
    ).get_properties()
    assert props["compound_name"] == "benomyl"
    assert props["inchikey"] == "RIOXQFHNBCKOKP-UHFFFAOYSA-N"


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        EnvChemgenCostanzo2021Adapter,
        "costanzo2021_adapter.yaml",
        Shape("environment response phenotype", env_perturbation=True),
        EnvChemgenCostanzo2021Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``EnvChemgenCostanzo2021Adapter`` loads ``conf/costanzo2021_adapter.yaml``: 17 node and 15 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
