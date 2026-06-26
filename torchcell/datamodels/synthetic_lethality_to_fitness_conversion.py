# torchcell/datamodels/synthetic_lethality_to_fitness_conversion
# [[torchcell.datamodels.synthetic_lethality_to_fitness_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/synthetic_lethality_to_fitness_conversion
# Test file: tests/torchcell/datamodels/test_synthetic_lethality_to_fitness_conversion.py
"""Convert synthetic-lethality experiments into equivalent fitness experiments."""

from typing import TYPE_CHECKING

from torchcell.datamodels.conversion import ConversionEntry, ConversionMap, Converter
from torchcell.datamodels.schema import (
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
)

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw


def synthetic_lethality_to_fitness_experiment(
    experiment: SyntheticLethalityExperiment,
) -> FitnessExperiment | None:
    """Map a synthetic-lethal experiment to a fitness-0 experiment, else drop it."""
    if experiment.phenotype.is_synthetic_lethal:
        fitness_phenotype = FitnessPhenotype(fitness=0.0, fitness_std=None)
        return FitnessExperiment(
            experiment_type="fitness",
            dataset_name=experiment.dataset_name,
            genotype=experiment.genotype,
            environment=experiment.environment,
            phenotype=fitness_phenotype,
        )
    return None  # Drop the experiment if it's not synthetically lethal


def synthetic_lethality_to_fitness_reference(
    reference: SyntheticLethalityExperimentReference,
) -> FitnessExperimentReference:
    """Map a synthetic-lethality reference to a fitness-1 reference experiment."""
    fitness_phenotype_reference = FitnessPhenotype(fitness=1.0, fitness_std=None)
    return FitnessExperimentReference(
        experiment_reference_type="fitness",
        dataset_name=reference.dataset_name,
        genome_reference=reference.genome_reference,
        environment_reference=reference.environment_reference,
        phenotype_reference=fitness_phenotype_reference,
    )


class SyntheticLethalityToFitnessConverter(Converter):
    """Converter turning synthetic-lethality records into fitness records."""

    def __init__(self, root: str, query: "Neo4jQueryRaw"):
        """Initialize the converter with its cache root and source Neo4j query."""
        super().__init__(root, query)

    @property
    def conversion_map(self) -> ConversionMap:
        """Return the experiment/reference conversion mapping for this converter."""
        entry = ConversionEntry(
            experiment_input_type=SyntheticLethalityExperiment,
            experiment_conversion_function=synthetic_lethality_to_fitness_experiment,
            experiment_output_type=FitnessExperiment,
            experiment_reference_input_type=SyntheticLethalityExperimentReference,
            experiment_reference_conversion_function=synthetic_lethality_to_fitness_reference,
            experiment_reference_output_type=FitnessExperimentReference,
        )

        return ConversionMap(entries=[entry])


if __name__ == "__main__":
    pass
