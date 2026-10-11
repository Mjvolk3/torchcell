# tests/torchcell/adapters/test_parsons2006_adapter.py
"""The Parsons 2006 adapter conf enables only methods the CellAdapter has, and a real
scratch build of the loader projects onto the strain-resolved nodes.

The compound node is what lets a Parsons condition join the other chemogenomic
datasets, so its CAS-resolved InChIKey has to reach the graph as a property.
"""

from __future__ import annotations

import gc
import json
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import torchcell.adapters.cell_adapter as cell_adapter_module
import torchcell.adapters.parsons2006_adapter as adapter_module
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters.test_vanacloig2022_adapter import (
    adapter_method_names,
    conf_methods,
)
from tests.torchcell.datasets.scerevisiae.test_parsons2006 import _build
from torchcell.adapters.parsons2006_adapter import EnvChemgenParsons2006Adapter
from torchcell.datasets.scerevisiae.parsons2006 import EnvChemgenParsons2006Dataset

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "env_chemgen_parsons2006_adapter.yaml",
)


def test_every_configured_method_exists_in_the_adapter_tables() -> None:
    node_names, edge_names = adapter_method_names()
    conf_nodes, conf_edges = conf_methods(CONF)
    assert set(conf_nodes) <= node_names, set(conf_nodes) - node_names
    assert set(conf_edges) <= edge_names, set(conf_edges) - edge_names
    assert "environment response phenotype (chunked)" in conf_nodes
    assert "environment perturbation (chunked)" in conf_nodes
    assert "genotype (chunked)" in conf_nodes and "perturbation (chunked)" in conf_nodes
    assert "perturbation to genotype (chunked)" in conf_edges


class _Table:
    def __init__(self, columns: list[Any], data: list[Any]) -> None:
        self.columns = columns
        self.data = data


class _Recorder:
    """Stands in for ``wandb`` (the adapter logs a method table)."""

    Table = _Table

    def init(self) -> None:
        return None

    def log(self, payload: dict[str, Any]) -> None:
        return None


def test_adapter_emits_the_strain_resolved_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 11 synthetic records give 11 experiment nodes, three references (one per
    medium) on ONE BY4741 genome node, and a barcoded kanMX deletion per genotype. The
    compound nodes carry the pinned table's InChIKeys (sulfometuron-methyl, MMS, the
    thialysine hydrochloride Table S1's CAS number names) at the Table S1 doses.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(cell_adapter_module, "wandb", _Recorder())
    dataset = _build(tmp_path, monkeypatch)
    assert len(dataset) == 11
    adapter = EnvChemgenParsons2006Adapter(
        dataset=dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=11,
        loader_batch_size=11,
    )
    try:
        nodes = list(adapter.get_nodes())
        edges = list(adapter.get_edges())
    finally:
        gc.unfreeze()
        dataset.close_lmdb()
    by_label: dict[str, list[Any]] = {}
    for node in nodes:
        by_label.setdefault(node.get_label(), []).append(node)
    assert len(by_label["experiment"]) == 11
    assert len(by_label["experiment reference"]) == 3
    references = [
        json.loads(n.get_properties()["serialized_data"])
        for n in by_label["experiment reference"]
    ]
    assert {r["experiment_reference_type"] for r in references} == {
        "strain_environment_response"
    }
    assert {r["phenotype_reference"]["environment_response"] for r in references} == {
        0.0
    }
    assert sorted(
        r["environment_reference"]["media"]["base_medium"] for r in references
    ) == ["SC", "SC", "YPD"]
    assert len(by_label["genome"]) == 1
    genome = json.loads(by_label["genome"][0].get_properties()["serialized_data"])
    assert (genome["strain"], genome["background"]["mating_type"]) == ("BY4741", "a")
    assert {
        n.get_properties()["perturbation_type"] for n in by_label["perturbation"]
    } == {"barcoded_kanmx_deletion"}
    compounds = {
        (
            n.get_properties()["compound_name"],
            n.get_properties()["inchikey"],
            n.get_properties()["concentration_value"],
            n.get_properties()["concentration_unit"],
        )
        for n in by_label["environment perturbation"]
    }
    assert compounds == {
        ("sulfometuron-methyl", "ZDXMLEQEMNLCQG-UHFFFAOYSA-N", 0.2, "nM"),
        ("methyl methanesulfonate", "MBABOKRGFJTBAE-UHFFFAOYSA-N", 0.004, "percent"),
        (
            "l-cysteine, s-(2-aminoethyl)-, hydrochloride",
            "CVHKULVNPGAEQM-WCCKRBBISA-N",
            3.0,
            "uM",
        ),
    }
    phenotypes = by_label["environment response phenotype"]
    # one node per record plus ONE reference phenotype: the three references' phenotypes
    # are identical (log2 ratio 0, the same units), and phenotype ids are content
    # hashes, so two records with the same (ratio, p-value) also share an id
    assert len(phenotypes) == 11 + 1
    assert len({ph.get_id() for ph in phenotypes}) == 9 + 1
    assert {p.get_properties()["measurement_type"] for p in phenotypes} == {
        "log2_ratio"
    }
    perturbation_edges = [e for e in edges if e.get_label() == "perturbation member of"]
    assert len(perturbation_edges) == 11


_INIT_CASES = [
    AdapterCase(
        EnvChemgenParsons2006Adapter,
        "env_chemgen_parsons2006_adapter.yaml",
        Shape("environment response phenotype", env_perturbation=True),
        EnvChemgenParsons2006Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs:
    17 node and 15 edge methods, the dataset ``dataset_adapter_map`` pairs it with, the
    worker / chunk sizes it was given, one ``wandb.init`` and its method table.
    """
    assert_construction(case, monkeypatch, capsys)


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_refuses_a_missing_conf_before_wandb(
    case: AdapterCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the conf absent the error names ``<adapters dir>/conf/<conf name>``."""
    assert_missing_conf(case, adapter_module, monkeypatch)
