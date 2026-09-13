# tests/torchcell/adapters/test_auesukaree2009_adapter.py
"""Auesukaree 2009 adapter: the conf enables only methods CellAdapter has, and
the nodes this dataset depends on carry the properties it depends on.
"""

from __future__ import annotations

import inspect
import os.path as osp
import re
from typing import Any

import yaml

from torchcell.adapters.auesukaree2009_adapter import EnvChemgenAuesukaree2009Adapter
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    Media,
    ResponseCategory,
)
from torchcell.datasets.scerevisiae import auesukaree2009 as a

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
    _check_conf("auesukaree2009")


def test_adapter_class_reads_its_own_conf() -> None:
    assert osp.exists(osp.join(_CONF_DIR, "auesukaree2009_adapter.yaml"))
    assert EnvChemgenAuesukaree2009Adapter.__mro__[1] is CellAdapter


def test_media_node_is_the_shared_solid_ypd_node() -> None:
    spec = next(s for s in a._STRESS_SPECS if s["stress"] == "ethanol")
    dataset = a.EnvChemgenAuesukaree2009Dataset.__new__(
        a.EnvChemgenAuesukaree2009Dataset
    )
    assert _media_node_id(dataset._environment(spec).media) == _media_node_id(YPD_AGAR)


def test_the_typed_category_is_projected_on_the_phenotype_node() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        assay_type=AssayType.spot_dilution,
        category=ResponseCategory.sensitive,
        category_label="sensitive",
        units=a.MEASUREMENT_UNITS,
    )
    props = CellAdapter._environment_response_properties(phenotype)
    assert props["category"] == "sensitive"
    assert props["category_label"] == "sensitive"
    assert props["assay_type"] == "spot_dilution"


def test_the_heat_environment_emits_no_perturbation_node() -> None:
    spec = next(s for s in a._STRESS_SPECS if s["stress"] == "heat")
    dataset = a.EnvChemgenAuesukaree2009Dataset.__new__(
        a.EnvChemgenAuesukaree2009Dataset
    )
    assert dataset._environment(spec).perturbations == []
