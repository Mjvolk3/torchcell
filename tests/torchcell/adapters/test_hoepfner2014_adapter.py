# tests/torchcell/adapters/test_hoepfner2014_adapter.py
"""Unit tests for the Hoepfner 2014 HIP-HOP adapter conf and its node projections.

The conf is the whole surface of this adapter (the class only picks which enable-list to
load), so what is worth pinning is that every method it enables exists, that the
gene-keyed genotype/perturbation pair and the environment-perturbation pair ARE enabled
(the first separates this conf from the Bloom 2019 segregant one, the second is what makes
the dosed compound a node carrying its InChIKey), and that the phenotype projection
carries the screen id the records are keyed on.
"""

from __future__ import annotations

import json
import os.path as osp
from typing import Any, cast

import yaml

import torchcell.adapters as adapters_package
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.hoepfner2014_adapter import EnvChemgenHoepfner2014Adapter
from torchcell.datamodels.identity import (
    environment_perturbation_identity,
    identity_sha256,
)
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EnvironmentResponsePhenotype,
    MeasurementType,
    SampleUnit,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae.hoepfner2014 import EnvChemgenHoepfner2014Dataset

CONF = "env_chemgen_hoepfner2014_adapter.yaml"


def _load() -> dict[str, Any]:
    path = osp.join(osp.dirname(adapters_package.__file__), "conf", CONF)
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


def test_gene_keyed_genotype_and_environment_perturbation_are_enabled() -> None:
    conf = _load()["cell_adapter"]
    nodes = {entry["method_name"] for entry in conf["node_methods"]}
    edges = {entry["method_name"] for entry in conf["edge_methods"]}
    assert "genotype (chunked)" in nodes and "perturbation (chunked)" in nodes
    assert "perturbation to genotype (chunked)" in edges
    assert "environment perturbation (chunked)" in nodes
    assert "environment perturbation to environment (chunked)" in edges
    assert "environment response phenotype (chunked)" in nodes
    assert "environment response phenotype reference" in nodes
    # A segregant genotype is Bloom's shape, never this dataset's.
    assert "segregant genotype (chunked)" not in nodes


def test_adapter_points_at_its_own_conf_and_dataset() -> None:
    import inspect

    source = inspect.getsource(EnvChemgenHoepfner2014Adapter)
    assert CONF in source
    assert EnvChemgenHoepfner2014Dataset.__name__ in source


def test_environment_perturbation_node_projects_the_compound_identity() -> None:
    """The node id hashes the perturbation, so the cleaned compound is what joins."""
    perturbation = SmallMoleculePerturbation(
        compound=EnvChemgenHoepfner2014Dataset._vehicle(),
        concentration=Concentration(
            value=2.0, unit=ConcentrationUnit.percent_v_v, basis=DoseBasis.fixed
        ),
    )

    class FakeEnvironment:
        perturbations = [perturbation]

    class FakeExperiment:
        environment = FakeEnvironment()

    adapter = CellAdapter.__new__(CellAdapter)
    undecorated = cast(Any, CellAdapter._environment_perturbation_node).__wrapped__
    nodes = undecorated(
        adapter, {"experiment": FakeExperiment()}, "environment perturbation (chunked)"
    )
    assert len(nodes) == 1
    expected = identity_sha256(environment_perturbation_identity(perturbation))
    assert nodes[0].get_id() == expected
    props = nodes[0].get_properties()
    assert props["inchikey"] == "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
    assert json.loads(props["serialized_data"]) == perturbation.model_dump()


def test_environment_response_phenotype_node_carries_the_screen_id() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.sensitivity_score,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=-3.25,
        n_samples=2,
        sample_unit=SampleUnit.technical_replicate,
        units="adjusted MADL sensitivity score",
        screen_id="0077",
    )

    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.phenotype = phenotype  # type: ignore[attr-defined]
    adapter = CellAdapter.__new__(CellAdapter)
    undecorated = cast(
        Any, CellAdapter._environment_response_phenotype_node
    ).__wrapped__
    node = undecorated(
        adapter, {"experiment": experiment}, "environment response phenotype (chunked)"
    )
    props = node.get_properties()
    assert props["screen_id"] == "0077"
    assert props["environment_response"] == -3.25
    assert json.loads(props["serialized_data"]) == phenotype.model_dump()
