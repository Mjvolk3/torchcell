# tests/torchcell/adapters/test_smith2016_adapter.py
"""The Smith 2016 adapter conf enables only methods the CellAdapter actually has, and the
per-guide construct becomes its own queryable node.

A method name in the conf that is not in ``CellAdapter``'s tables fails SILENTLY at KG
build time, so the enable-list is checked against the tables rather than eyeballed.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp

import pytest

import torchcell.adapters.smith2016_adapter as _init_module
import torchcell.adapters.smith2016_adapter as adapter_module
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
from torchcell.adapters.smith2016_adapter import Smith2016Adapter
from torchcell.datamodels.schema import CrisprConstruct
from torchcell.datasets.scerevisiae.smith2016 import CrispriChemgenSmith2016Dataset

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "crispri_chemgen_smith2016_adapter.yaml",
)


def test_every_configured_method_exists_in_the_adapter_tables() -> None:
    node_names, edge_names = adapter_method_names()
    conf_nodes, conf_edges = conf_methods(CONF)
    assert set(conf_nodes) <= node_names, set(conf_nodes) - node_names
    assert set(conf_edges) <= edge_names, set(conf_edges) - edge_names


def test_the_conf_serves_the_crispr_construct_pair() -> None:
    conf_nodes, conf_edges = conf_methods(CONF)
    assert "crispr construct (chunked)" in conf_nodes
    assert "crispr construct to perturbation (chunked)" in conf_edges
    assert "environment response phenotype (chunked)" in conf_nodes
    assert "environment perturbation (chunked)" in conf_nodes


def test_the_library_pool_is_a_queryable_construct_property() -> None:
    """The SAME spacer in two pools is two strains, so the pool must reach the graph."""
    broad = CrisprConstruct(
        effector="dCas9-Mxi1",
        guide_sequence="GTCAGGTACTCCGAATTCGA",
        n_guides=1,
        library_pool="broad_tiling",
    )
    tiling = broad.model_copy(update={"library_pool": "gene_tiling_20bp"})
    broad_node = CellAdapter._crispr_construct_node_from(broad)
    tiling_node = CellAdapter._crispr_construct_node_from(tiling)
    assert broad_node.get_label() == "crispr construct"
    assert broad_node.get_properties()["library_pool"] == "broad_tiling"
    assert tiling_node.get_properties()["library_pool"] == "gene_tiling_20bp"
    assert broad_node.get_id() != tiling_node.get_id()
    assert (
        broad_node.get_id()
        == hashlib.sha256(json.dumps(broad.model_dump()).encode("utf-8")).hexdigest()
    )
    assert "serialized_data" not in broad_node.get_properties()


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        Smith2016Adapter,
        "crispri_chemgen_smith2016_adapter.yaml",
        Shape("environment response phenotype", crispr=True, env_perturbation=True),
        CrispriChemgenSmith2016Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``Smith2016Adapter`` loads ``conf/crispri_chemgen_smith2016_adapter.yaml``: 18 node and 16 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; CRISPR construct nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
