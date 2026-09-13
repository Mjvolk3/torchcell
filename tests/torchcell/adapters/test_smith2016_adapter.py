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

import torchcell.adapters.smith2016_adapter as adapter_module
from tests.torchcell.adapters._crispr_adapter_conf import (
    adapter_method_names,
    conf_methods,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.schema import CrisprConstruct

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
    assert (
        json.loads(broad_node.get_properties()["serialized_data"]) == broad.model_dump()
    )
