# tests/torchcell/adapters/test_lian2019_adapter.py
"""The Lian 2019 adapter conf enables only methods the CellAdapter actually has, and each
modality's construct (including the CRISPRd donor) reaches the graph as its own node.
"""

from __future__ import annotations

import os.path as osp

import pytest

import torchcell.adapters.lian2019_adapter as _init_module
import torchcell.adapters.lian2019_adapter as adapter_module
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
from torchcell.adapters.lian2019_adapter import Lian2019Adapter
from torchcell.datasets.scerevisiae.lian2019 import (
    CrisprMagicLian2019Dataset,
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
    through the perturbation (in the Experiment blob's genotype) rather than the
    construct.
    """
    spacer, donor = split_deletion_cassette(ACS1_CASSETTE)
    perturbation = crispr_perturbation("YAL054C", "ACS1", "d", spacer, donor)
    node = CellAdapter._crispr_construct_node_from(perturbation.crispr)
    assert node.get_properties()["guide_sequence"] == spacer
    assert "donor_sequence" not in node.get_properties()
    assert perturbation.model_dump()["donor_sequence"] == donor
    assert "serialized_data" not in node.get_properties()


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        Lian2019Adapter,
        "crispr_magic_lian2019_adapter.yaml",
        Shape("environment response phenotype", crispr=True, env_perturbation=True),
        CrisprMagicLian2019Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``Lian2019Adapter`` loads ``conf/crispr_magic_lian2019_adapter.yaml``: 18 node and 16 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; CRISPR construct nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
