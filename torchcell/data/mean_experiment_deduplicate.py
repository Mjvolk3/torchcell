import hashlib
import numpy as np
from typing import Any
import logging
from torchcell.datamodels import (
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    MeanDeletionPerturbation,
    GeneInteractionPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
)
from scipy.stats import t
from torchcell.data import Deduplicator

log = logging.getLogger(__name__)


class MeanExperimentDeduplicator(Deduplicator):
    def duplicate_check(self, data: list[dict[str, Any]]) -> dict[str, list[int]]:
        duplicate_check = {}
        for idx, item in enumerate(data):
            perturbations = item["experiment"].genotype.perturbations
            sorted_gene_names = sorted(
                [pert.systematic_gene_name for pert in perturbations]
            )
            hash_key = hashlib.sha256(str(sorted_gene_names).encode()).hexdigest()

            if hash_key not in duplicate_check:
                duplicate_check[hash_key] = []
            duplicate_check[hash_key].append(idx)
        return duplicate_check

    def create_deduplicate_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        experiment_type = duplicate_experiments[0]["experiment"].experiment_type
        if experiment_type == "fitness":
            return self._create_mean_fitness_entry(duplicate_experiments)
        elif experiment_type == "gene interaction":
            return self._create_mean_gene_interaction_entry(duplicate_experiments)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    def _create_mean_fitness_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        fitness_values = [
            exp["experiment"].phenotype.fitness
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness is not None
        ]
        fitness_stds = [
            exp["experiment"].phenotype.std
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.std is not None
        ]

        mean_fitness = np.mean(fitness_values) if fitness_values else None
        mean_fitness_std = (
            np.sqrt(np.mean(np.array(fitness_stds) ** 2)) if fitness_stds else None
        )

        mean_phenotype = FitnessPhenotype(fitness=mean_fitness, std=mean_fitness_std)

        mean_genotype = self._create_mean_genotype(duplicate_experiments)

        dataset_name = ("+").join(
            [i["experiment"].dataset_name for i in duplicate_experiments]
        )

        mean_experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        mean_reference = self._create_mean_fitness_reference(duplicate_experiments)

        return {"experiment": mean_experiment, "experiment_reference": mean_reference}

    def _create_mean_gene_interaction_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        interaction_values = [
            exp["experiment"].phenotype.gene_interaction
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.gene_interaction is not None
        ]
        p_values = [
            exp["experiment"].phenotype.p_value
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.p_value is not None
        ]

        mean_interaction = np.mean(interaction_values) if interaction_values else None
        aggregated_p_value = self._compute_p_value_for_mean(
            interaction_values, p_values
        )

        mean_phenotype = GeneInteractionPhenotype(
            gene_interaction=mean_interaction, p_value=aggregated_p_value
        )

        mean_genotype = self._create_mean_genotype(duplicate_experiments)

        dataset_name = ("+").join(
            [i["experiment"].dataset_name for i in duplicate_experiments]
        )

        mean_experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        mean_reference = self._create_mean_gene_interaction_reference(
            duplicate_experiments
        )

        return {"experiment": mean_experiment, "experiment_reference": mean_reference}

    def _create_mean_genotype(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> Genotype:
        mean_perturbations = []
        for pert in duplicate_experiments[0]["experiment"].genotype.perturbations:
            mean_pert = MeanDeletionPerturbation(
                systematic_gene_name=pert.systematic_gene_name,
                perturbed_gene_name=pert.perturbed_gene_name,
                num_duplicates=len(duplicate_experiments),
            )
            mean_perturbations.append(mean_pert)
        return Genotype(perturbations=mean_perturbations)

    def _create_mean_fitness_reference(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> FitnessExperimentReference:
        fitness_ref_values = [
            exp["experiment_reference"].phenotype_reference.fitness
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.fitness is not None
        ]
        fitness_ref_stds = [
            exp["experiment_reference"].phenotype_reference.std
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.std is not None
        ]

        mean_fitness_ref = np.mean(fitness_ref_values) if fitness_ref_values else None
        mean_fitness_ref_std = (
            np.sqrt(np.mean(np.array(fitness_ref_stds) ** 2))
            if fitness_ref_stds
            else None
        )

        mean_phenotype_reference = FitnessPhenotype(
            fitness=mean_fitness_ref, std=mean_fitness_ref_std
        )

        dataset_name = ("+").join(
            [i["experiment_reference"].dataset_name for i in duplicate_experiments]
        )

        return FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=duplicate_experiments[0][
                "experiment_reference"
            ].genome_reference,
            environment_reference=duplicate_experiments[0][
                "experiment_reference"
            ].environment_reference,
            phenotype_reference=mean_phenotype_reference,
        )

    def _create_mean_gene_interaction_reference(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> GeneInteractionExperimentReference:
        interaction_ref_values = [
            exp["experiment_reference"].phenotype_reference.gene_interaction
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.gene_interaction
            is not None
        ]

        mean_interaction_ref = (
            np.mean(interaction_ref_values) if interaction_ref_values else None
        )

        mean_phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=mean_interaction_ref, p_value=None
        )

        dataset_name = ("+").join(
            [i["experiment_reference"].dataset_name for i in duplicate_experiments]
        )

        return GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=duplicate_experiments[0][
                "experiment_reference"
            ].genome_reference,
            environment_reference=duplicate_experiments[0][
                "experiment_reference"
            ].environment_reference,
            phenotype_reference=mean_phenotype_reference,
        )

    def _compute_p_value_for_mean(self, x: list[float], p_values: list[float]) -> float:
        if len(x) != len(p_values):
            raise ValueError("x and p_values must have the same length.")

        n = len(x)

        if n < 2:
            raise ValueError("At least two data points are required.")

        mean_x = np.mean(x)
        sample_std_dev = np.std(x, ddof=1)
        sem = sample_std_dev / np.sqrt(n)
        t_stat = mean_x / sem
        p_value_for_mean = t.sf(np.abs(t_stat), df=n - 1) * 2

        return p_value_for_mean


if __name__ == "__main__":
    pass
