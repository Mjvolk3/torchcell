# torchcell/data/deduplicate
# [[torchcell.data.deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/deduplicate
# Test file: tests/torchcell/data/test_deduplicate.py

import hashlib
import numpy as np
from typing import Any
from torchcell.datamodels import (
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Experiment,
    ExperimentReference,
    MeanDeletionPerturbation,
    GeneInteractionPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
)
from abc import ABC, abstractmethod
from scipy.stats import t
import os
import lmdb
import pickle
from abc import ABC, abstractmethod
from typing import Any
from tqdm import tqdm


class Deduplicator(ABC):
    def __init__(self, root: str):
        self.root = root

    @abstractmethod
    def duplicate_check(self, data: Any) -> dict[str, list[int]]:
        pass

    @abstractmethod
    def create_mean_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        pass

    def process(self, input_path: str, output_path: str) -> None:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Open input and output LMDB environments
        env_input = lmdb.open(input_path, readonly=True)
        env_output = lmdb.open(output_path, map_size=int(1e12))

        # Read all data from input LMDB
        data = []
        with env_input.begin() as txn_input:
            cursor = txn_input.cursor()
            for _, value in cursor:
                data.append(pickle.loads(value))

        # Perform deduplication
        duplicate_check = self.duplicate_check(data)
        deduplicated_data = []

        with env_output.begin(write=True) as txn_output:
            for idx, (hash_key, indices) in enumerate(
                tqdm(duplicate_check.items(), desc="Deduplicating and writing to LMDB")
            ):
                if len(indices) > 1:
                    # Compute mean entry for duplicate experiments
                    duplicate_experiments = [data[i] for i in indices]
                    mean_entry = self.create_mean_entry(duplicate_experiments)
                    deduplicated_data.append(mean_entry)
                else:
                    # Keep non-duplicate experiments as is
                    deduplicated_data.append(data[indices[0]])

                # Serialize and write the deduplicated data to LMDB
                txn_output.put(f"{idx}".encode(), pickle.dumps(deduplicated_data[-1]))

        # Close LMDB environments
        env_input.close()
        env_output.close()

        print(f"Deduplication complete. LMDB database written to {output_path}")


# class Deduplicator(ABC):
#     @abstractmethod
#     def duplicate_check(self, data: Any) -> dict[str, list[int]]: ...

#     @abstractmethod
#     def create_mean_entry(
#         self, duplicate_experiments: list[dict[str, Any]]
#     ) -> dict[str, Experiment | ExperimentReference]: ...


# used for computing p-values when we only have label and p-value.
def compute_p_value_for_mean(x: list[float], p_values: list[float]) -> float:
    if len(x) != len(p_values):
        raise ValueError("x and p_values must have the same length.")

    n = len(x)

    if n < 2:
        raise ValueError("At least two data points are required.")

    # Calculate the mean of the x values
    mean_x = np.mean(x)

    # Calculate the sample standard deviation (Bessel's correction applied)
    sample_std_dev = np.std(x, ddof=1)

    # Calculate the standard error of the mean (SEM)
    sem = sample_std_dev / np.sqrt(n)

    # Compute the t-statistic for the mean
    t_stat = mean_x / sem

    # Compute the p-value (two-tailed test)
    p_value_for_mean = t.sf(np.abs(t_stat), df=n - 1) * 2

    return p_value_for_mean


class ExperimentDeduplicator(Deduplicator):
    def duplicate_check(self, data) -> dict[str, list[int]]:
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

    def create_mean_entry(
        self, duplicate_experiments
    ) -> dict[str, Experiment | ExperimentReference]:
        # Check if all phenotypes have the same graph_level and label
        graph_levels = set(
            exp["experiment"].phenotype.graph_level for exp in duplicate_experiments
        )
        labels = set(
            exp["experiment"].phenotype.label_name for exp in duplicate_experiments
        )

        if len(graph_levels) > 1 or len(labels) > 1:
            raise ValueError(
                "Duplicate experiments have different phenotype graph_level or label values."
            )

        interaction_values = [
            exp["experiment"].phenotype.interaction
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.interaction is not None
        ]

        interaction_p_values = [
            exp["experiment"].phenotype.p_value
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.p_value is not None
        ]

        # Calculate the mean fitness and mean standard deviation, handling empty lists
        mean_interaction = np.mean(interaction_values) if interaction_values else None
        aggregated_p_value = compute_p_value_for_mean(
            interaction_values, interaction_p_values
        )

        # Create a new GeneInteractionPhenotype with the mean values
        mean_phenotype = GeneInteractionPhenotype(
            graph_level=duplicate_experiments[0]["experiment"].phenotype.graph_level,
            label=duplicate_experiments[0]["experiment"].phenotype.label_name,
            label_statistic=duplicate_experiments[0][
                "experiment"
            ].phenotype.label_statistic,
            interaction=mean_interaction,
            p_value=aggregated_p_value,
        )

        mean_perturbations = []
        for pert in duplicate_experiments[0]["experiment"].genotype.perturbations:
            mean_pert = MeanDeletionPerturbation(
                systematic_gene_name=pert.systematic_gene_name,
                perturbed_gene_name=pert.perturbed_gene_name,
                num_duplicates=len(duplicate_experiments),
            )
            mean_perturbations.append(mean_pert)

        mean_genotype = Genotype(perturbations=mean_perturbations)

        mean_experiment = GeneInteractionExperiment(
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        # Create a new FitnessExperimentReference with the mean values
        interaction_ref_values = [
            exp["experiment_reference"].phenotype_reference.interaction
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.interaction is not None
        ]

        # Calculate the mean reference fitness and mean reference standard deviation, handling empty lists
        mean_fitness_ref = (
            np.mean(interaction_ref_values) if interaction_ref_values else None
        )

        mean_phenotype_reference = GeneInteractionPhenotype(
            graph_level=duplicate_experiments[0][
                "experiment_reference"
            ].phenotype_reference.graph_level,
            label=duplicate_experiments[0][
                "experiment_reference"
            ].phenotype_reference.label,
            label_statistic=duplicate_experiments[0][
                "experiment_reference"
            ].phenotype_reference.label_statistic,
            interaction=mean_fitness_ref,
            p_value=None,
        )

        # For now we don't deal with reference harmonization - just take first reference
        mean_reference = GeneInteractionExperimentReference(
            genome_reference=duplicate_experiments[0][
                "experiment_reference"
            ].genome_reference,
            environment_reference=duplicate_experiments[0][
                "experiment_reference"
            ].environment_reference,
            phenotype_reference=mean_phenotype_reference,
        )

        return {"experiment": mean_experiment, "experiment_reference": mean_reference}


class FitnessExperimentDeduplicator(Deduplicator):
    def duplicate_check(self, data) -> dict[str, list[int]]:
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

    def create_mean_entry(
        self, duplicate_experiments
    ) -> dict[str, Experiment | ExperimentReference]:
        # Check if all phenotypes have the same graph_level and label
        graph_levels = set(
            exp["experiment"].phenotype.graph_level for exp in duplicate_experiments
        )
        labels = set(exp["experiment"].phenotype.label for exp in duplicate_experiments)

        if len(graph_levels) > 1 or len(labels) > 1:
            raise ValueError(
                "Duplicate experiments have different phenotype graph_level or label values."
            )

        # Extract fitness values and standard deviations, excluding None values
        fitness_values = [
            exp["experiment"].phenotype.fitness
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness is not None
        ]
        fitness_stds = [
            exp["experiment"].phenotype.fitness_std
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness_std is not None
        ]

        # Calculate the mean fitness and mean standard deviation, handling empty lists
        mean_fitness = np.mean(fitness_values) if fitness_values else None
        mean_fitness_std = np.mean(fitness_stds) if fitness_stds else None

        # Create a new FitnessPhenotype with the mean values
        mean_phenotype = FitnessPhenotype(
            graph_level=duplicate_experiments[0]["experiment"].phenotype.graph_level,
            label=duplicate_experiments[0]["experiment"].phenotype.label,
            label_statistic=duplicate_experiments[0][
                "experiment"
            ].phenotype.label_statistic,
            fitness=mean_fitness,
            fitness_std=mean_fitness_std,
        )

        mean_perturbations = []
        for pert in duplicate_experiments[0]["experiment"].genotype.perturbations:
            mean_pert = MeanDeletionPerturbation(
                systematic_gene_name=pert.systematic_gene_name,
                perturbed_gene_name=pert.perturbed_gene_name,
                num_duplicates=len(duplicate_experiments),
            )
            mean_perturbations.append(mean_pert)

        mean_genotype = Genotype(perturbations=mean_perturbations)

        mean_experiment = FitnessExperiment(
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        # Create a new FitnessExperimentReference with the mean values
        fitness_ref_values = [
            exp["reference"].phenotype_reference.fitness
            for exp in duplicate_experiments
            if exp["reference"].phenotype_reference.fitness is not None
        ]
        fitness_ref_stds = [
            exp["reference"].phenotype_reference.fitness_std
            for exp in duplicate_experiments
            if exp["reference"].phenotype_reference.fitness_std is not None
        ]

        # Calculate the mean reference fitness and mean reference standard deviation, handling empty lists
        mean_fitness_ref = np.mean(fitness_ref_values) if fitness_ref_values else None
        mean_fitness_ref_std = np.mean(fitness_ref_stds) if fitness_ref_stds else None

        mean_phenotype_reference = FitnessPhenotype(
            graph_level=duplicate_experiments[0][
                "reference"
            ].phenotype_reference.graph_level,
            label=duplicate_experiments[0]["reference"].phenotype_reference.label,
            label_statistic=duplicate_experiments[0][
                "reference"
            ].phenotype_reference.label_statistic,
            fitness=mean_fitness_ref,
            fitness_std=mean_fitness_ref_std,
        )

        # For now we don't deal with reference harmonization - just take first reference
        mean_reference = FitnessExperimentReference(
            genome_reference=duplicate_experiments[0]["reference"].genome_reference,
            environment_reference=duplicate_experiments[0][
                "reference"
            ].environment_reference,
            phenotype_reference=mean_phenotype_reference,
        )

        return {"experiment": mean_experiment, "reference": mean_reference}


if __name__ == "__main__":
    pass
