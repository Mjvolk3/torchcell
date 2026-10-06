# tests/torchcell/adapters/test_wildenhain2015_adapter.py
"""The Wildenhain 2015 adapter conf enables only methods the CellAdapter actually has,
and this dataset's records project onto the typed node properties.

The compound node is the point of the dataset: its PubChem CID and InChIKey have to
reach the graph as properties, or a CGM cell joins nothing.
"""

from __future__ import annotations

import gc
import json
import os.path as osp
from typing import Any

import pytest

import torchcell.adapters.cell_adapter as cell_adapter_module
import torchcell.adapters.wildenhain2015_adapter as adapter_module
from tests.torchcell.adapters.test_vanacloig2022_adapter import (
    adapter_method_names,
    conf_methods,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponsePhenotype,
    MeasurementType,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
    UncertaintyType,
)

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "env_chemgen_wildenhain2015_adapter.yaml",
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


def test_compound_node_carries_the_cid_and_the_inchikey() -> None:
    environment = Environment(
        media=SC,
        temperature=Temperature(value=30.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("CID 1183", pubchem_cid=1183),
                concentration=Concentration(
                    value=20.0, unit=ConcentrationUnit.micromolar
                ),
                solvent=Solvent(
                    name="DMSO", compound=resolved_compound("dimethyl sulfoxide")
                ),
            )
        ],
        aerobicity="aerobic",
        duration_hours=18.0,
    )
    node = CellAdapter._environment_perturbation_node_from(environment.perturbations[0])
    props = node.get_properties()
    assert (
        props["compound_name"] == "vanillin"
    )  # the canonical PubChem title, not 'CID 1183'
    assert props["inchikey"] == "MWOOGOJBHIARFG-UHFFFAOYSA-N"
    assert props["concentration_value"] == 20.0
    assert props["concentration_unit"] == "uM"
    # the full record travels in the Experiment blob (inside its environment)
    assert "serialized_data" not in props
    payload = environment.perturbations[0].model_dump()
    assert payload["compound"]["pubchem_cid"] == 1183
    assert payload["solvent"]["name"] == "DMSO"


def test_environment_response_properties_project_the_z_score_axes() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.liquid_od_growth,
        environment_response=-5.0,
        category=ResponseCategory.sensitive,
        category_label="Active / sensitive",
        environment_response_uncertainty=1.4142135623730951,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=2,
        sample_unit=SampleUnit.screen,
        units="z-score of growth inhibition",
    )
    props = CellAdapter._environment_response_properties(phenotype)
    assert props["environment_response"] == -5.0
    assert props["measurement_type"] == "z_score"
    assert props["assay_type"] == "liquid_od_growth"
    assert props["category"] == "sensitive"
    assert props["category_label"] == "Active / sensitive"
    assert props["environment_response_se"] == 1.0
    assert "serialized_data" not in props


# ---- #504: the strain-resolved records through the real adapter ------------------ #
def _panel_store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real scratch build: CDC28 (conditional allele), NNK1 on YKL171W, the wild type,
    plus the held TSCII / YGL11 / wtn01 labels (4 served records).
    """
    from tests.torchcell.datasets.scerevisiae.test_wildenhain2015 import (
        _build,
        _panel_row,
        _panel_rows,
        _PanelGenome,
        _write_raw,
    )

    rows = [*_panel_rows(), _panel_row("NULL", "wild type", "8", "1183", "-2.0")]
    _write_raw(tmp_path, rows)
    return _build(tmp_path, monkeypatch, _PanelGenome())


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
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (#504 / #507): the experiment, reference, genome and phenotype nodes come
    out for the new family with no adapter change. One reference and one genome node
    (the BY4741 background, MATa, four alleles) serve all four records; the
    perturbation nodes carry the conditional-allele and kanMX-deletion leaves; the
    wild-type record has a genotype node with no perturbation edge.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(cell_adapter_module, "wandb", _Recorder())
    dataset = _panel_store(tmp_path, monkeypatch)
    assert len(dataset) == 4
    adapter = adapter_module.EnvChemgenWildenhain2015Adapter(
        dataset=dataset,
        process_workers=1,
        io_workers=1,
        chunk_size=4,
        loader_batch_size=4,
    )
    try:
        nodes = list(adapter.get_nodes())
        edges = list(adapter.get_edges())
    finally:
        gc.unfreeze()
    by_label: dict[str, list[Any]] = {}
    for node in nodes:
        by_label.setdefault(node.get_label(), []).append(node)
    assert len(by_label["experiment"]) == 4
    assert len(by_label["experiment reference"]) == 1
    reference = json.loads(
        by_label["experiment reference"][0].get_properties()["serialized_data"]
    )
    assert reference["experiment_reference_type"] == "strain_environment_response"
    assert reference["environment_reference"]["perturbations"] == []
    assert reference["phenotype_reference"]["environment_response"] == 0.0
    assert len(by_label["genome"]) == 1
    genome = json.loads(by_label["genome"][0].get_properties()["serialized_data"])
    assert genome["strain"] == "BY4741"
    assert genome["background"]["mating_type"] == "a"
    assert [a["allele_name"] for a in genome["background"]["alleles"]] == [
        "his3Δ1",
        "leu2Δ0",
        "met15Δ0",
        "ura3Δ0",
    ]
    experiments = [
        json.loads(n.get_properties()["serialized_data"])
        for n in by_label["experiment"]
    ]
    assert {e["experiment_type"] for e in experiments} == {
        "strain_environment_response"
    }
    assert {
        n.get_properties()["perturbation_type"] for n in by_label["perturbation"]
    } == {"conditional_allele", "barcoded_kanmx_deletion"}
    assert len(by_label["environment response phenotype"]) == 5  # 4 records + reference
    assert len(by_label["genotype"]) == 4
    perturbation_edges = [e for e in edges if e.get_label() == "perturbation member of"]
    assert len(perturbation_edges) == 3  # the wild type has no perturbation
