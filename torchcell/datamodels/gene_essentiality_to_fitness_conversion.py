# torchcell/datamodels/gene_essentiality_to_fitness_conversion
# [[torchcell.datamodels.gene_essentiality_to_fitness_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/gene_essentiality_to_fitness_conversion
# Test file: tests/torchcell/datamodels/test_gene_essentiality_to_fitness_conversion.py
"""Convert gene-essentiality experiments into equivalent fitness experiments."""

from typing import TYPE_CHECKING

from torchcell.datamodels.conversion import ConversionEntry, ConversionMap, Converter
from torchcell.datamodels.schema import (
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
)

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw


def gene_essentiality_to_fitness_experiment(
    experiment: GeneEssentialityExperiment,
) -> FitnessExperiment | None:
    """Map an essential gene to a zero-fitness experiment, else drop it.

    Returns ``None`` when the gene is not essential so the experiment is
    excluded from the converted dataset.
    """
    if experiment.phenotype.is_essential:
        fitness_phenotype = FitnessPhenotype(fitness=0.0, fitness_std=None)
        return FitnessExperiment(
            experiment_type="fitness",
            dataset_name=experiment.dataset_name,
            genotype=experiment.genotype,
            environment=experiment.environment,
            phenotype=fitness_phenotype,
        )
    return None  # Drop the experiment if the gene is not essential


def gene_essentiality_to_fitness_reference(
    reference: GeneEssentialityExperimentReference,
) -> FitnessExperimentReference:
    """Build the wild-type (fitness 1.0) reference for the fitness dataset."""
    fitness_phenotype_reference = FitnessPhenotype(fitness=1.0, fitness_std=None)
    return FitnessExperimentReference(
        experiment_reference_type="fitness",
        dataset_name=reference.dataset_name,
        genome_reference=reference.genome_reference,
        environment_reference=reference.environment_reference,
        phenotype_reference=fitness_phenotype_reference,
    )


class GeneEssentialityToFitnessConverter(Converter):
    """Converter turning gene-essentiality records into fitness records."""

    def __init__(self, root: str, query: "Neo4jQueryRaw"):
        """Initialize the converter over ``root`` and a raw Neo4j query."""
        super().__init__(root, query)

    @property
    def conversion_map(self) -> ConversionMap:
        """Return the experiment/reference conversion mapping for fitness."""
        entry = ConversionEntry(
            experiment_input_type=GeneEssentialityExperiment,
            experiment_conversion_function=gene_essentiality_to_fitness_experiment,
            experiment_output_type=FitnessExperiment,
            experiment_reference_input_type=GeneEssentialityExperimentReference,
            experiment_reference_conversion_function=gene_essentiality_to_fitness_reference,
            experiment_reference_output_type=FitnessExperimentReference,
        )

        return ConversionMap(entries=[entry])


if __name__ == "__main__":
    pass
