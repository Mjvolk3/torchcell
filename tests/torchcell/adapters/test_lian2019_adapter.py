# tests/torchcell/adapters/test_lian2019_adapter.py
"""The Lian 2019 adapter conf enables only methods the CellAdapter actually has, and each
modality's construct (including the CRISPRd donor) reaches the graph as its own node.
"""

from __future__ import annotations

import json
import os.path as osp

import torchcell.adapters.lian2019_adapter as adapter_module
from tests.torchcell.adapters._crispr_adapter_conf import (
    adapter_method_names,
    conf_methods,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.lian2019 import (
    crispr_perturbation,
    split_deletion_cassette,
)

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "crispr_magic_lian2019_adapter.yaml",
)

ACS1_CASSETTE = (
    "TGGGATGAACACCTTATCGAATGGCTTAGACCAGTTTAAAAATTGGGTAGTTCAATAGACTCCTTGTGCAAGC"
    "GCTGATAGTCCTGCAACCCGTCCAAGTCTTTAGAACCGAAGAACTTAG"
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


def test_the_three_effectors_give_three_distinct_construct_nodes() -> None:
    spacer, _donor = split_deletion_cassette(ACS1_CASSETTE)
    constructs = [
        crispr_perturbation("YAL054C", "ACS1", "a", "A" * 23).crispr,
        crispr_perturbation("YAL054C", "ACS1", "i", "C" * 20).crispr,
        crispr_perturbation("YAL054C", "ACS1", "d", spacer, _donor).crispr,
    ]
    nodes = [CellAdapter._crispr_construct_node_from(c) for c in constructs]
    assert len({node.get_id() for node in nodes}) == 3
    assert [node.get_properties()["effector"] for node in nodes] == [
        "dLbCas12a-VP",
        "dSpCas9-RD1152",
        "SaCas9",
    ]


def test_the_deletion_donor_rides_on_the_perturbation_not_the_construct() -> None:
    """The donor rides the perturbation, the spacer rides the construct.

    ``donor_sequence`` is a ``CrisprDeletionPerturbation`` field, so it reaches the graph
    through the perturbation node's serialized_data rather than the construct's.
    """
    spacer, donor = split_deletion_cassette(ACS1_CASSETTE)
    perturbation = crispr_perturbation("YAL054C", "ACS1", "d", spacer, donor)
    node = CellAdapter._crispr_construct_node_from(perturbation.crispr)
    assert node.get_properties()["guide_sequence"] == spacer
    assert "donor_sequence" not in node.get_properties()
    assert perturbation.model_dump()["donor_sequence"] == donor
    assert (
        json.loads(node.get_properties()["serialized_data"])["guide_sequence"] == spacer
    )
