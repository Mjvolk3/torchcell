# tests/torchcell/adapters/test_strain_background_genome_node.py
# [[tests.torchcell.adapters.test_strain_background_genome_node]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_strain_background_genome_node.py
"""The strain background reaches the graph through the existing adapter path (#507).

No adapter method changed for the background: it is a field of
``StrainReferenceGenome``, which the strain-resolved reference class declares, so the ``genome`` node's ``serialized_data`` and the ``experiment reference`` node's
blob carry it, once per distinct reference. These tests pin that the round trip
through those two node builders reproduces the typed background exactly, and that a
new chemogenomic leaf passes through the perturbation node builder unchanged.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.data.data import ExperimentReferenceIndex
from torchcell.datamodels import schema as s
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.strain_background import BRACHMANN_1998, standard_background

SPECIES = "Saccharomyces cerevisiae"


def _reference() -> s.StrainEnvironmentResponseExperimentReference:
    return s.StrainEnvironmentResponseExperimentReference(
        dataset_name="Toy",
        genome_reference=s.StrainReferenceGenome(
            species=SPECIES,
            strain="BY4743",
            ploidy="diploid",
            background=standard_background("BY4743", resolve_with=BRACHMANN_1998),
        ),
        environment_reference=s.CultureEnvironment(
            media=MEDIA_LIBRARY["YPD"], temperature=s.Temperature(value=30.0)
        ),
        phenotype_reference=s.EnvironmentResponsePhenotype(
            measurement_type=s.MeasurementType.log2_ratio, environment_response=0.0
        ),
    )


def _adapter(reference: s.StrainEnvironmentResponseExperimentReference) -> CellAdapter:
    adapter = CellAdapter.__new__(CellAdapter)
    adapter.dataset = SimpleNamespace(
        experiment_reference_index=[
            ExperimentReferenceIndex(reference=reference, member_indices=[0, 1])
        ]
    )
    return adapter


def test_genome_node_carries_the_background_and_round_trips() -> None:
    reference = _reference()
    nodes = _adapter(reference)._get_genome_nodes()
    assert len(nodes) == 1
    node = nodes[0]
    payload = json.dumps(reference.genome_reference.model_dump())
    assert node.get_id() == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    props = node.get_properties()
    assert props["strain"] == "BY4743"
    back = s.StrainReferenceGenome.model_validate_json(props["serialized_data"])
    assert back == reference.genome_reference
    assert back.background.functional_copies("YOR202W") == 0


def test_experiment_reference_blob_round_trips_the_background() -> None:
    reference = _reference()
    nodes = _adapter(reference)._get_experiment_reference_nodes()
    assert len(nodes) == 1
    back = s.StrainEnvironmentResponseExperimentReference.model_validate_json(
        nodes[0].get_properties()["serialized_data"]
    )
    assert back == reference


def test_new_leaves_pass_through_the_perturbation_node_builder() -> None:
    het = s.HeterozygousDeletionPerturbation(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", cassette="kanMX4"
    )
    conditional = s.ConditionalAllelePerturbation(
        systematic_gene_name="YBR160W",
        perturbed_gene_name="CDC28",
        allele_class=s.ConditionalAlleleClass.temperature_sensitive,
        allele_name="cdc28-4",
    )
    experiment = SimpleNamespace(genotype=s.Genotype(perturbations=[het, conditional]))
    adapter = CellAdapter.__new__(CellAdapter)
    nodes = CellAdapter._perturbation_node.__wrapped__(  # type: ignore[attr-defined]
        adapter, {"experiment": experiment}, "perturbation (chunked)"
    )
    assert [n.get_properties()["perturbation_type"] for n in nodes] == [
        "heterozygous_deletion",
        "conditional_allele",
    ]
    for node, pert in zip(nodes, (het, conditional), strict=True):
        assert (
            node.get_id()
            == hashlib.sha256(json.dumps(pert.model_dump()).encode("utf-8")).hexdigest()
        )
        assert node.get_properties()["strain_id"] is None
