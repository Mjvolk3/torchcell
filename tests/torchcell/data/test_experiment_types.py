# tests/torchcell/data/test_experiment_types.py
"""A stored pair maps back to its classes; a retired tag fails by name, never by guess."""

from __future__ import annotations

import pytest

from torchcell.data.experiment_types import (
    RetiredExperimentTypeError,
    experiment_class,
    experiment_pair,
    experiment_reference_class,
)
from torchcell.datamodels import schema as s


def test_every_live_tag_maps_to_its_class() -> None:
    for tag, cls in s.EXPERIMENT_TYPE_MAP.items():
        assert experiment_class(tag) is cls
    for tag, cls in s.EXPERIMENT_REFERENCE_TYPE_MAP.items():
        assert experiment_reference_class(tag) is cls


@pytest.mark.parametrize("tag", sorted(s.RETIRED_EXPERIMENT_TYPES))
def test_a_retired_tag_names_itself_and_its_successor(tag: str) -> None:
    """The 14 retired tags are absent from both maps, and the reader says which."""
    assert tag not in s.EXPERIMENT_TYPE_MAP
    assert tag not in s.EXPERIMENT_REFERENCE_TYPE_MAP
    successor = s.RETIRED_EXPERIMENT_TYPES[tag]
    assert successor in s.EXPERIMENT_TYPE_MAP
    with pytest.raises(RetiredExperimentTypeError, match=f"{tag!r}.*{successor!r}"):
        experiment_class(tag)
    with pytest.raises(
        RetiredExperimentTypeError, match="unreadable until it is rebuilt"
    ):
        experiment_reference_class(tag)


def test_an_unknown_tag_is_a_plain_key_error() -> None:
    with pytest.raises(KeyError, match="unknown experiment_type 'no_such_tag'"):
        experiment_class("no_such_tag")
    assert not isinstance(
        pytest.raises(KeyError, match="no_such_tag"), RetiredExperimentTypeError
    )


def test_the_retired_table_is_exactly_the_fourteen_pairs() -> None:
    assert len(s.RETIRED_EXPERIMENT_TYPES) == 14
    assert sum(k.startswith("bacterial_") for k in s.RETIRED_EXPERIMENT_TYPES) == 13
    assert s.RETIRED_EXPERIMENT_TYPES["yeast_protein_turnover"] == "protein_turnover"
    assert s.RETIRED_EXPERIMENT_TYPES["bacterial_fitness"] == "fitness"


def test_experiment_pair_rebuilds_a_stored_record() -> None:
    reference = s.ReferenceGenome(
        species="Escherichia coli",
        strain="MG1655",
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        assembly_accession="GCA_000005845.2",
    )
    env = s.Environment(
        media=s.Media(name="LB", state="liquid", is_synthetic=False),
        temperature=s.Temperature(value=37.0),
    )
    experiment = s.FitnessExperiment(
        dataset_name="t",
        genotype=s.Genotype(
            perturbations=[
                s.DeletionPerturbation(
                    systematic_gene_name="b0001",
                    perturbed_gene_name="thrL",
                    gene_namespace="ecoli_k12_mg1655_bnumber",
                )
            ]
        ),
        environment=env,
        phenotype=s.FitnessPhenotype(fitness=0.5),
    )
    ref = s.FitnessExperimentReference(
        dataset_name="t",
        genome_reference=reference,
        environment_reference=env,
        phenotype_reference=s.FitnessPhenotype(fitness=1.0),
    )
    stored = {
        "experiment": experiment.model_dump(),
        "experiment_reference": ref.model_dump(),
    }
    rebuilt = experiment_pair(stored)
    assert rebuilt["experiment"] == experiment
    assert rebuilt["experiment_reference"] == ref
    assert rebuilt["experiment_reference"].genome_reference.assembly_set == (
        "ecoli_K12_MG1655_ASM584v2"
    )
