# tests/torchcell/adapters/test_cooper2010_adapter.py
"""Unit tests for the Cooper 2010 amino-acid adapter conf and its phenotype projection.

The conf is the whole surface of this adapter (the class only picks which enable-list to
load), so what is worth pinning is that every method it enables exists in the
``CellAdapter`` tables, that it is the same enable-list as the Mulleder 2016 metabolite
conf it was cloned from (no environment-perturbation methods, since the environment
carries none), that the adapter points at its own conf and dataset, and that the
metabolite phenotype node id is the sha256 of the phenotype's serialized form, so the
ragged key set and the linear-ratio measurement type are what the node is keyed on.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from typing import Any, cast

import yaml

import torchcell.adapters as adapters_package
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.cooper2010_adapter import AminoAcidCooper2010Adapter
from torchcell.datamodels.schema import MetabolitePhenotype
from torchcell.datasets.scerevisiae.cooper2010 import (
    MEASUREMENT_TYPE,
    AminoAcidCooper2010Dataset,
)

CONF = "amino_acid_cooper2010_adapter.yaml"
MULLEDER_CONF = "amino_acid_mulleder2016_adapter.yaml"


def _load(name: str = CONF) -> dict[str, Any]:
    path = osp.join(osp.dirname(adapters_package.__file__), "conf", name)
    with open(path) as handle:
        conf: dict[str, Any] = yaml.safe_load(handle)
    return conf


def test_conf_exists_and_parses() -> None:
    conf = _load()["cell_adapter"]
    assert conf["node_methods"] and conf["edge_methods"]


def test_every_enabled_method_exists_in_the_cell_adapter_tables() -> None:
    """A conf naming a method the adapter does not implement is a silent no-op."""
    import torchcell.adapters.cell_adapter as module

    with open(module.__file__) as handle:
        source = handle.read()
    conf = _load()["cell_adapter"]
    for entry in conf["node_methods"] + conf["edge_methods"]:
        assert f'"{entry["method_name"]}"' in source, entry["method_name"]


def test_conf_is_the_mulleder_metabolite_enable_list() -> None:
    assert _load() == _load(MULLEDER_CONF)
    nodes = {entry["method_name"] for entry in _load()["cell_adapter"]["node_methods"]}
    assert "metabolite phenotype (chunked)" in nodes
    assert "metabolite phenotype reference" in nodes
    assert "environment perturbation (chunked)" not in nodes


def test_adapter_points_at_its_own_conf_and_dataset() -> None:
    import inspect

    source = inspect.getsource(AminoAcidCooper2010Adapter)
    assert CONF in source
    assert AminoAcidCooper2010Dataset.__name__ in source


def test_metabolite_phenotype_node_is_keyed_on_the_serialized_phenotype() -> None:
    phenotype = MetabolitePhenotype(
        metabolite_level={"arginine": 3.543331384, "glutamine+valine": 1.154314489},
        metabolite_level_se=None,
        n_replicates={"arginine": 1, "glutamine+valine": 1},
        measurement_type=MEASUREMENT_TYPE,
    )

    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.phenotype = phenotype  # type: ignore[attr-defined]
    adapter = CellAdapter.__new__(CellAdapter)
    undecorated = cast(Any, CellAdapter._metabolite_phenotype_node).__wrapped__
    node = undecorated(
        adapter, {"experiment": experiment}, "metabolite phenotype (chunked)"
    )
    expected = hashlib.sha256(
        json.dumps(phenotype.model_dump()).encode("utf-8")
    ).hexdigest()
    assert node.get_id() == expected
    props = node.get_properties()
    assert props["measurement_type"] == MEASUREMENT_TYPE
    assert json.loads(props["metabolite_level"])["glutamine+valine"] == 1.154314489
    assert json.loads(props["serialized_data"]) == phenotype.model_dump()
