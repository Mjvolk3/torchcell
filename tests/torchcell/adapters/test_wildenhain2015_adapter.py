# tests/torchcell/adapters/test_wildenhain2015_adapter.py
"""The Wildenhain 2015 adapter conf enables only methods the CellAdapter actually has,
and this dataset's records project onto the typed node properties.

The compound node is the point of the dataset: its PubChem CID and InChIKey have to
reach the graph as properties, or a CGM cell joins nothing.
"""

from __future__ import annotations

import json
import os.path as osp

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
    payload = json.loads(props["serialized_data"])
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
    assert json.loads(props["serialized_data"]) == phenotype.model_dump()
