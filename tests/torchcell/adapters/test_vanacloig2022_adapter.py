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

import pytest
import yaml

import torchcell.adapters.vanacloig2022_adapter as _init_module
import torchcell.adapters.vanacloig2022_adapter as adapter_module
import torchcell.datasets.scerevisiae.vanacloig2022 as loader
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    assert_construction,
    assert_missing_conf,
)
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.vanacloig2022_adapter import EnvChemgenVanacloig2022Adapter
from torchcell.datamodels.media import SYNBASE
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    CultureEnvironment,
    DoseBasis,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Genotype,
    MatingType,
    MeasurementType,
    PhysicalFactor,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae.vanacloig2022 import EnvChemgenVanacloig2022Dataset

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
    # the gap travels in the Experiment blob (inside its environment), not the node
    assert perturbation.model_dump()["provenance_gaps"]
    node = CellAdapter._environment_perturbation_node_from(perturbation)
    assert "serialized_data" not in node.get_properties()


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
    # the barcode travels in the Experiment blob (inside its genotype), not the node
    assert "serialized_data" not in props
    assert perturbation.model_dump()["barcode"] == "ACGTACGTACGTACGTACGT"
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
    assert "serialized_data" not in props


# --------------------------------------------------------------------------- #
# #500 / #501: the strain-resolved family through the node builders, end to end
# --------------------------------------------------------------------------- #
class _Gene:
    """The slice of a genome resolution the loader's gene-name policy reads."""

    def __init__(self, systematic: str) -> None:
        self.systematic_name = systematic
        self.is_current_gene = True


class _TinyGenome:
    gene_set = {"YAL001C", "YAL002W"}
    feature_index = {"standard_to_ids": {"TFC3": ["YAL001C"], "VPS8": ["YAL002W"]}}

    def resolve_gene_name(self, name: str) -> _Gene:
        return _Gene({"TFC3": "YAL001C", "VPS8": "YAL002W"}.get(name, name))


def _tiny_matrix() -> Any:
    import pandas as pd

    data: dict[str, Any] = {
        "gene": ["YAL001C_AAAACCCC", "YAL002W_CCCCGGGG"],
        "std_name": ["TFC3", "VPS8"],
        "Control1_CG003": [100, 200],
        "Control2_CG003": [120, 220],
    }
    for token in ("Furfural", "DMSO"):
        for rep, counts in enumerate(([110, 190], [90, 230], [130, 210]), start=1):
            data[f"{token}_CG003_rep{rep}"] = counts
    return pd.DataFrame(data)


@pytest.fixture()
def tiny(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Two genes x (Furfural, DMSO), built through the real ``process``."""
    import gzip
    import os

    monkeypatch.setattr(loader, "default_genome", lambda: _TinyGenome())
    monkeypatch.setattr(
        loader.EnvChemgenVanacloig2022Dataset, "download", lambda self: None
    )
    monkeypatch.setattr(
        loader.EnvChemgenVanacloig2022Dataset,
        "_load_matrix",
        lambda self: _tiny_matrix(),
    )
    root = str(tmp_path / "env_chemgen_vanacloig2022")
    os.makedirs(osp.join(root, "raw"), exist_ok=True)
    with gzip.open(osp.join(root, "raw", loader.DATA_FILENAME), "wt") as handle:
        handle.write("placeholder\n")
    return loader.EnvChemgenVanacloig2022Dataset(root=root)


def _adapter(dataset: Any) -> CellAdapter:
    adapter = CellAdapter.__new__(CellAdapter)
    adapter.dataset = dataset
    return adapter


def test_strain_resolved_records_emit_the_expected_nodes(tiny: Any) -> None:
    assert len(tiny) == 4  # 2 genes x (DMSO, Furfural)
    adapter = _adapter(tiny)
    data = tiny.transform_item(tiny[0])
    experiment = data["experiment"]
    assert isinstance(experiment, StrainEnvironmentResponseExperiment)
    assert isinstance(data["reference"], StrainEnvironmentResponseExperimentReference)
    assert isinstance(experiment.environment, CultureEnvironment)

    (experiment_node, *constants) = cast(Any, CellAdapter._experiment_node).__wrapped__(
        adapter, data, "experiment (chunked)"
    )
    assert experiment_node.get_label() == "experiment"
    blob = json.loads(experiment_node.get_properties()["serialized_data"])
    assert blob["experiment_type"] == "strain_environment_response"
    assert len(blob["genotype"]["perturbations"]) == 1
    assert {c.get_label() for c in constants} <= {"interned constant"}

    (genome,) = adapter._get_genome_nodes()
    assert genome.get_label() == "genome"
    props = genome.get_properties()
    assert props["strain"] == loader.LIBRARY_STRAIN
    back = StrainReferenceGenome.model_validate_json(props["serialized_data"])
    assert back.background == loader.library_background()
    assert back.background.mating_type is MatingType.a

    (reference_node,) = adapter._get_experiment_reference_nodes()
    assert reference_node.get_label() == "experiment reference"
    reference = StrainEnvironmentResponseExperimentReference.model_validate_json(
        reference_node.get_properties()["serialized_data"]
    )
    assert reference.experiment_reference_type == "strain_environment_response"
    assert reference.environment_reference.culture_format is not None

    phenotype = cast(Any, CellAdapter._environment_response_phenotype_node).__wrapped__(
        adapter, data, "environment response phenotype (chunked)"
    )
    assert phenotype.get_label() == "environment response phenotype"
    assert phenotype.get_properties()["environment_response"] == pytest.approx(
        experiment.phenotype.environment_response
    )
    assert phenotype.get_properties()["measurement_type"] == "log2_ratio"

    dmso = cast(Any, CellAdapter._environment_perturbation_node).__wrapped__(
        adapter, data, "environment perturbation (chunked)"
    )
    names = {node.get_properties()["compound_name"] for node in dmso}
    assert names == {"dimethyl sulfoxide", "hydrochloric acid"}


# 2026.10.06, Phase 21: the constructor (exact conf content, wiring, refusal); the
# checks are in tests/torchcell/adapters/_adapter_init_harness.py.
_INIT_CASES = [
    AdapterCase(
        EnvChemgenVanacloig2022Adapter,
        "env_chemgen_vanacloig2022_adapter.yaml",
        Shape("environment response phenotype", env_perturbation=True),
        EnvChemgenVanacloig2022Dataset,
    )
]


@pytest.mark.parametrize("case", _INIT_CASES, ids=lambda c: c.adapter_cls.__name__)
def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    case: AdapterCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conf the constructor loads is exactly the one its dataset's shape needs.

    * ``EnvChemgenVanacloig2022Adapter`` loads ``conf/env_chemgen_vanacloig2022_adapter.yaml``: 17 node and 15 edge methods (``environment response phenotype``; gene-keyed genotype; perturbation nodes; environment-perturbation nodes; memory_reduction_factor 1.0 on every chunked method).

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
