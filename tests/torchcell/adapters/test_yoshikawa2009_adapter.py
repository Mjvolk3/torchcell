# tests/torchcell/adapters/test_yoshikawa2009_adapter.py
"""The Yoshikawa 2009 adapter conf enables only methods the CellAdapter has, and a real
scratch build of the loader projects onto the strain-resolved nodes.

The two stresses are compound nodes, which is what lets them join the other datasets that
add ethanol or sodium chloride, so their InChIKeys and doses have to reach the graph.
"""

from __future__ import annotations

import gc
import json
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import torchcell.adapters.cell_adapter as cell_adapter_module
import torchcell.adapters.yoshikawa2009_adapter as adapter_module
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
from tests.torchcell.datasets.scerevisiae.test_yoshikawa2009 import _build
from torchcell.adapters.yoshikawa2009_adapter import EnvChemgenYoshikawa2009Adapter
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datasets.scerevisiae.yoshikawa2009 import EnvChemgenYoshikawa2009Dataset

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "env_chemgen_yoshikawa2009_adapter.yaml",
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
    """The 56 synthetic records give 56 experiment nodes, four references (one per
    condition, each a standard-strain Table 1 rate; the 5% ethanol one reads the
    nonstress row and says so in its units) on ONE BY4742 genome node, a barcoded kanMX
    deletion per genotype, and ethanol at both doses and sodium chloride.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(cell_adapter_module, "wandb", _Recorder())
    dataset = _build(tmp_path, monkeypatch)
    assert len(dataset) == 56
    adapter = EnvChemgenYoshikawa2009Adapter(
        dataset=dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=56,
        loader_batch_size=56,
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
    assert len(by_label["experiment"]) == 56
    assert len(by_label["experiment reference"]) == 4
    references = [
        json.loads(n.get_properties()["serialized_data"])
        for n in by_label["experiment reference"]
    ]
    assert {r["experiment_reference_type"] for r in references} == {
        "strain_environment_response"
    }
    assert sorted(
        r["phenotype_reference"]["environment_response"] for r in references
    ) == [0.178, 0.203, 0.452, 0.452]
    assert len(by_label["genome"]) == 1
    genome = json.loads(by_label["genome"][0].get_properties()["serialized_data"])
    assert (genome["strain"], genome["background"]["mating_type"]) == (
        "BY4742",
        "alpha",
    )
    assert {
        n.get_properties()["perturbation_type"] for n in by_label["perturbation"]
    } == {"barcoded_kanmx_deletion"}
    compounds = {
        (
            n.get_properties()["inchikey"],
            n.get_properties()["concentration_value"],
            n.get_properties()["concentration_unit"],
        )
        for n in by_label["environment perturbation"]
    }
    assert compounds == {
        (resolved_compound("ethanol").inchikey, 5.0, "percent"),
        (resolved_compound("ethanol").inchikey, 8.0, "percent_v/v"),
        (resolved_compound("sodium chloride").inchikey, 1.0, "M"),
    }
    phenotypes = by_label["environment response phenotype"]
    assert {p.get_properties()["measurement_type"] for p in phenotypes} == {
        "growth_rate"
    }
    assert len({ph.get_id() for ph in phenotypes}) == 56 + 4
    perturbation_edges = [e for e in edges if e.get_label() == "perturbation member of"]
    assert len(perturbation_edges) == 56


_INIT_CASES = [
    AdapterCase(
        EnvChemgenYoshikawa2009Adapter,
        "env_chemgen_yoshikawa2009_adapter.yaml",
        Shape("environment response phenotype", env_perturbation=True),
        EnvChemgenYoshikawa2009Dataset,
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
