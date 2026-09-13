# tests/torchcell/adapters/test_vanacloig2022_adapter.py
"""The Vanacloig 2022 adapter conf enables only methods the CellAdapter actually has,
and this dataset's records project onto the typed node properties.

A method name in the conf that is not in ``CellAdapter``'s tables fails SILENTLY at KG
build time (the method is simply never called), so the enable-list is checked against
the tables rather than eyeballed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os.path as osp
import re
from typing import Any, cast

import yaml

import torchcell.adapters.vanacloig2022_adapter as adapter_module
import torchcell.datasets.scerevisiae.vanacloig2022 as loader
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datamodels.media import SYNBASE
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    DoseBasis,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    PhysicalFactor,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)

CONF = osp.join(
    osp.dirname(osp.abspath(adapter_module.__file__)),
    "conf",
    "env_chemgen_vanacloig2022_adapter.yaml",
)


def adapter_method_names() -> tuple[set[str], set[str]]:
    """Every node and edge method name ``CellAdapter.__init__`` registers.

    Read from the source because building the tables means constructing an adapter,
    which starts a wandb run.
    """
    source = inspect.getsource(CellAdapter.__init__)
    node_block, edge_block = source.split("self.edge_methods = [", 1)
    node_block = node_block.split("self.node_methods = [", 1)[1]
    pattern = r'"([^"]+)",\s*\n?\s*self\._'
    return set(re.findall(pattern, node_block)), set(re.findall(pattern, edge_block))


def conf_methods(path: str) -> tuple[list[str], list[str]]:
    """The node and edge method names an adapter conf enables."""
    with open(path) as handle:
        conf = yaml.safe_load(handle)["cell_adapter"]
    return (
        [method["method_name"] for method in conf["node_methods"]],
        [method["method_name"] for method in conf["edge_methods"]],
    )


def test_every_configured_method_exists_in_the_adapter_tables() -> None:
    node_names, edge_names = adapter_method_names()
    assert len(node_names) > 20 and len(edge_names) > 10
    conf_nodes, conf_edges = conf_methods(CONF)
    assert set(conf_nodes) <= node_names, set(conf_nodes) - node_names
    assert set(conf_edges) <= edge_names, set(conf_edges) - edge_names


def test_conf_serves_the_readout_and_the_gene_keyed_genotype() -> None:
    conf_nodes, conf_edges = conf_methods(CONF)
    assert "environment response phenotype (chunked)" in conf_nodes
    assert "environment response phenotype reference" in conf_nodes
    assert "environment perturbation (chunked)" in conf_nodes
    assert "genotype (chunked)" in conf_nodes and "perturbation (chunked)" in conf_nodes
    assert "perturbation to genotype (chunked)" in conf_edges
    # a gene-keyed genotype is NOT a segregant mosaic
    assert "segregant genotype (chunked)" not in conf_nodes


def _environment() -> Environment:
    """The loader's own environment for one IC30 compound (not a hand-built copy)."""
    dataset = loader.EnvChemgenVanacloig2022Dataset.__new__(
        loader.EnvChemgenVanacloig2022Dataset
    )
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    environment = dataset._environment("Furfural")
    assert environment.media == SYNBASE
    assert isinstance(environment.perturbations[0], SmallMoleculePerturbation)
    assert isinstance(environment.perturbations[1], EnvironmentPhysicalPerturbation)
    assert environment.perturbations[1].factor is PhysicalFactor.ph
    return environment


def test_environment_perturbation_nodes_carry_the_compound_and_the_ph() -> None:
    environment = _environment()
    compound_node = CellAdapter._environment_perturbation_node_from(
        environment.perturbations[0]
    )
    props = compound_node.get_properties()
    assert props["compound_name"] == "furfural"
    assert props["inchikey"] == "HYBBIBNJHNGZAN-UHFFFAOYSA-N"
    assert props["concentration_value"] is None  # IC30 with no released molar value
    ph_node = CellAdapter._environment_perturbation_node_from(
        environment.perturbations[1]
    )
    ph_props = ph_node.get_properties()
    assert ph_props["perturbation_type"] == "environment_physical"
    # a physical factor projects onto the SAME columns: factor + magnitude + the agent
    # that realizes it, so pH 5.0 set with HCl is as queryable as a dosed compound
    assert ph_props["factor"] == "pH"
    assert ph_props["concentration_value"] == 5.0
    assert ph_props["concentration_unit"] == "pH"
    assert ph_props["compound_name"] == "hydrochloric acid"
    assert ph_props["inchikey"] == "VEXZGXHMUGYJMC-UHFFFAOYSA-N"
    assert ph_props["factor"] != compound_node.get_properties()["factor"]
    assert ph_node.get_id() != compound_node.get_id()


def test_dosed_perturbation_carries_its_typed_solvent_gap() -> None:
    perturbation = _environment().perturbations[0]
    assert isinstance(perturbation, SmallMoleculePerturbation)
    gaps = perturbation.provenance_gaps
    assert [gap.field for gap in gaps] == ["solvent"]
    assert gaps[0].reason.value == "deferred_pending_source_review"
    assert gaps[0].resolve_with is not None
    # the gapped field is None and the dose is NOT gapped: an IC30 basis is known
    assert perturbation.solvent is None
    assert perturbation.concentration.basis is DoseBasis.IC30
    node = CellAdapter._environment_perturbation_node_from(perturbation)
    assert json.loads(node.get_properties()["serialized_data"])["provenance_gaps"]


def test_barcoded_deletion_projects_as_a_perturbation_node() -> None:
    perturbation = BarcodedKanMxDeletionPerturbation(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="TFC3",
        barcode="ACGTACGTACGTACGTACGT",
        collection="3DeltaAlpha drug-sensitive yeast deletion collection",
    )

    class FakeExperiment:
        pass

    experiment = FakeExperiment()
    experiment.genotype = Genotype(perturbations=[perturbation])  # type: ignore[attr-defined]
    undecorated = cast(Any, CellAdapter._perturbation_node).__wrapped__
    nodes = undecorated(
        CellAdapter.__new__(CellAdapter),
        {"experiment": experiment},
        "perturbation (chunked)",
    )
    assert len(nodes) == 1
    props = nodes[0].get_properties()
    assert props["systematic_gene_name"] == "YAL001C"
    assert props["perturbation_type"] == "barcoded_kanmx_deletion"
    assert json.loads(props["serialized_data"])["barcode"] == "ACGTACGTACGTACGTACGT"
    assert (
        nodes[0].get_id()
        == hashlib.sha256(
            json.dumps(perturbation.model_dump()).encode("utf-8")
        ).hexdigest()
    )


def test_environment_response_properties_project_the_typed_axes() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=-1.25,
        environment_response_uncertainty=0.3,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=3,
        sample_unit=SampleUnit.biological_replicate,
        units="log2(inhibitor/control)",
    )
    props = CellAdapter._environment_response_properties(phenotype)
    assert props["environment_response"] == -1.25
    assert props["measurement_type"] == "log2_ratio"
    assert props["assay_type"] == "pooled_competitive_growth_barcode"
    assert props["environment_response_se"] is not None
    assert json.loads(props["serialized_data"]) == phenotype.model_dump()
