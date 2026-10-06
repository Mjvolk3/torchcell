# tests/torchcell/adapters/test_hillenmeyer2008_adapter.py
"""Unit tests for the Hillenmeyer 2008 HIP/HOP adapter confs and node projection.

The confs are the whole surface of these adapters (the classes only pick which enable-list
to load), so what is worth pinning is that every method they enable exists, that the
gene-keyed genotype pair IS enabled (it is what separates this conf from the Bloom 2019
segregant one), and that the phenotype projection carries the control-set id the records
are keyed on.
"""

from __future__ import annotations

import json
import os.path as osp
from typing import Any

import yaml

import torchcell.adapters as adapters_package
from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.adapters.hillenmeyer2008_adapter import (
    HetHillenmeyer2008Adapter,
    HomHillenmeyer2008Adapter,
)
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    SampleUnit,
    UncertaintyType,
)

CONFS = ("het_hillenmeyer2008_adapter.yaml", "hom_hillenmeyer2008_adapter.yaml")


def _load(name: str) -> dict[str, Any]:
    path = osp.join(osp.dirname(adapters_package.__file__), "conf", name)
    with open(path) as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)
    return loaded


def test_both_confs_exist_and_parse() -> None:
    for name in CONFS:
        conf = _load(name)
        assert conf["cell_adapter"]["node_methods"]
        assert conf["cell_adapter"]["edge_methods"]


def test_every_enabled_method_exists_in_the_cell_adapter_tables() -> None:
    """A conf naming a method the adapter does not implement is a silent no-op."""
    import torchcell.adapters.cell_adapter as module

    with open(module.__file__) as handle:
        source = handle.read()
    for name in CONFS:
        conf = _load(name)["cell_adapter"]
        for entry in conf["node_methods"] + conf["edge_methods"]:
            assert f'"{entry["method_name"]}"' in source, entry["method_name"]


def test_gene_keyed_genotype_methods_are_enabled_unlike_the_segregant_conf() -> None:
    for name in CONFS:
        conf = _load(name)["cell_adapter"]
        nodes = {entry["method_name"] for entry in conf["node_methods"]}
        edges = {entry["method_name"] for entry in conf["edge_methods"]}
        assert "genotype (chunked)" in nodes
        assert "perturbation (chunked)" in nodes
        assert "segregant genotype (chunked)" not in nodes
        assert "perturbation to genotype (chunked)" in edges
        assert "environment response phenotype (chunked)" in nodes
        assert "environment response phenotype reference" in nodes
        assert "environment perturbation (chunked)" in nodes
        assert "temperature (chunked)" in nodes


def test_the_two_adapters_are_distinct_classes_over_one_enable_list() -> None:
    assert HetHillenmeyer2008Adapter.__name__ != HomHillenmeyer2008Adapter.__name__
    assert issubclass(HetHillenmeyer2008Adapter, CellAdapter)
    assert issubclass(HomHillenmeyer2008Adapter, CellAdapter)
    assert _load(CONFS[0])["cell_adapter"] == _load(CONFS[1])["cell_adapter"]


def test_environment_response_properties_carry_the_screen_and_the_derived_se() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=0.42,
        n_samples=4,
        sample_unit=SampleUnit.biological_replicate,
        environment_response_uncertainty=0.4,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        screen_id="het_04_01_2::old scanner::20::tag3::YPD::dmso::0",
        units="HIP fitness-defect log-ratio",
    )
    props = CellAdapter._environment_response_properties(phenotype)
    assert props["environment_response"] == 0.42
    assert props["environment_response_se"] == 0.2  # 0.4 / sqrt(4)
    assert props["measurement_type"] == "log2_ratio"
    assert props["assay_type"] == "pooled_competitive_growth_barcode"
    assert props["screen_id"] == phenotype.screen_id
    assert "serialized_data" not in props


# --------------------------------------------------------------------------- #
# #505: the strain-resolved records go through the existing node builders
# --------------------------------------------------------------------------- #
_CS = "het_06_03::new scanner::-5::tag3::YPD::dmso::0"


def _hillenmeyer_record() -> tuple[Any, Any]:
    """One het record and its reference, built with the loader's own builders."""
    from torchcell.datasets.scerevisiae import hillenmeyer2008 as h

    spec = h.MATRICES["het"]
    header = ["Orf", "a1:benomyl:6.9:um::::-5gen:het_06_03:new scanner"]
    columns = h.parse_columns(header, {"a1": _CS}, {"a1": "benomyl"})
    groups, _ = h.group_columns(columns)
    background = h.hillenmeyer_background()
    rows = h.MatrixRows(
        rows=[
            h.StrainRow(
                row_id="YBR115C:chr2_3",
                source_orf="YBR115C",
                orf="YBR115C",
                batch="chr2_3",
                values=[None, 0.7],
            )
        ],
        dropped_strains=[],
        dropped_genes={},
        constructed={},
    )
    ((experiment, control_set),) = list(
        h.iter_records("HetHillenmeyer2008Dataset", spec, rows, groups, background)
    )
    reference = h.build_reference(
        "HetHillenmeyer2008Dataset", spec, control_set, 4, background
    )
    return experiment, reference


def test_strain_resolved_record_emits_genome_reference_experiment_and_phenotype_nodes() -> (
    None
):
    from types import SimpleNamespace

    from torchcell.data.data import ExperimentReferenceIndex
    from torchcell.datamodels.schema import (
        StrainEnvironmentResponseExperiment,
        StrainEnvironmentResponseExperimentReference,
        StrainReferenceGenome,
    )

    experiment, reference = _hillenmeyer_record()
    adapter = CellAdapter.__new__(CellAdapter)
    adapter.dataset = SimpleNamespace(
        experiment_reference_index=[
            ExperimentReferenceIndex(reference=reference, member_indices=[0])
        ]
    )
    (genome,) = adapter._get_genome_nodes()
    assert genome.get_label() == "genome"
    props = genome.get_properties()
    assert props["strain"] == "BY4743"
    back = StrainReferenceGenome.model_validate_json(props["serialized_data"])
    assert back.background.functional_copies("YOR202W") == 0
    (ref_node,) = adapter._get_experiment_reference_nodes()
    assert ref_node.get_label() == "experiment reference"
    assert (
        StrainEnvironmentResponseExperimentReference.model_validate_json(
            ref_node.get_properties()["serialized_data"]
        )
        == reference
    )
    data = {"experiment": experiment}
    experiment_nodes = CellAdapter._experiment_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, data, "experiment (chunked)"
    )
    assert experiment_nodes[0].get_label() == "experiment"
    (perturbation,) = CellAdapter._perturbation_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, data, "perturbation (chunked)"
    )
    assert perturbation.get_label() == "perturbation"
    assert perturbation.get_properties()["perturbation_type"] == "heterozygous_deletion"
    assert perturbation.get_properties()["systematic_gene_name"] == "YBR115C"
    environment = CellAdapter._environment_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, data, "environment (chunked)"
    )
    assert environment.get_label() == "environment"
    assert (
        json.loads(environment.get_properties()["serialized_data"])["pre_culture"][
            "source"
        ]
        == "frozen_stock"
    )
    phenotype = CellAdapter._environment_response_phenotype_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, data, "environment response phenotype (chunked)"
    )
    assert phenotype.get_label() == "environment response phenotype"
    assert phenotype.get_properties()["screen_id"] == _CS
    assert phenotype.get_properties()["environment_response"] == 0.7
    assert isinstance(experiment, StrainEnvironmentResponseExperiment)
    assert experiment.experiment_type == "strain_environment_response"
