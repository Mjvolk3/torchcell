# torchcell/data/deduplicate
# [[torchcell.data.deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/deduplicate
# Test file: tests/torchcell/data/test_deduplicate.py

import hashlib
import numpy as np
from typing import Any
import logging
import json
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
from abc import ABC, abstractmethod
from scipy.stats import t
import os
import lmdb
import pickle
from abc import ABC, abstractmethod
from typing import Any
from tqdm import tqdm
import json
import os
import os.path as osp
import lmdb
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any, Union
from torchcell.datamodels.schema import (
    EXPERIMENT_TYPE_MAP,
    EXPERIMENT_REFERENCE_TYPE_MAP,
)

log = logging.getLogger(__name__)


class Deduplicator(ABC):
    def __init__(self, root: str):
        self.root = root
        self.lmdb_dir = os.path.join(self.root, "deduplication", "lmdb")
        self.env = None

    @abstractmethod
    def duplicate_check(self, data: Any) -> dict[str, list[int]]:
        pass

    @abstractmethod
    def create_deduplicate_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        pass

    def _init_lmdb(self, readonly=True):
        """Initialize the LMDB environment."""
        if self.env is not None:
            self.close_lmdb()
        os.makedirs(os.path.dirname(self.lmdb_dir), exist_ok=True)
        if not readonly or osp.exists(self.lmdb_dir):
            self.env = lmdb.open(
                self.lmdb_dir,
                map_size=int(1e12),
                readonly=readonly,
                create=not readonly,
                lock=not readonly,
                readahead=False,
                meminit=False,
            )
        else:
            self.env = None

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def process(self, input_path: str, output_path: str) -> None:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._init_lmdb(readonly=False)  # Initialize LMDB for writing

        env_input = lmdb.open(input_path, readonly=True)

        # Read all data from input LMDB
        data = []
        with env_input.begin() as txn_input:
            cursor = txn_input.cursor()
            for _, value in cursor:
                json_data = json.loads(value.decode("utf-8"))
                # Reconstruct Pydantic objects
                experiment_class = EXPERIMENT_TYPE_MAP[
                    json_data["experiment"]["experiment_type"]
                ]
                experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                    json_data["experiment_reference"]["experiment_reference_type"]
                ]
                reconstructed_data = {
                    "experiment": experiment_class(**json_data["experiment"]),
                    "experiment_reference": experiment_reference_class(
                        **json_data["experiment_reference"]
                    ),
                }
                data.append(reconstructed_data)

        # Perform deduplication
        duplicate_check = self.duplicate_check(data)
        deduplicated_data = []
        deduplicated_count = 0

        with self.env.begin(write=True) as txn_output:
            for idx, (hash_key, indices) in enumerate(
                tqdm(duplicate_check.items(), desc="Deduplicating and writing to LMDB")
            ):
                if len(indices) > 1:
                    # Compute mean entry for duplicate experiments
                    duplicate_experiments = [data[i] for i in indices]
                    mean_entry = self.create_deduplicate_entry(duplicate_experiments)
                    deduplicated_data.append(mean_entry)
                    deduplicated_count += len(indices) - 1
                else:
                    # Keep non-duplicate experiments as is
                    deduplicated_data.append(data[indices[0]])

                # Serialize to JSON and write the deduplicated data to LMDB
                json_data = {
                    "experiment": deduplicated_data[-1]["experiment"].model_dump(),
                    "experiment_reference": deduplicated_data[-1][
                        "experiment_reference"
                    ].model_dump(),
                }
                txn_output.put(f"{idx}".encode(), json.dumps(json_data).encode())

        env_input.close()
        self.close_lmdb()

        log.info(f"Deduplication complete. LMDB database written to {output_path}")
        log.info(f"Number of instances deduplicated: {deduplicated_count}")
        log.info(
            f"Total number of instances after deduplication: {len(deduplicated_data)}"
        )

    def __getitem__(self, index: Union[int, slice, list]):
        self._init_lmdb(readonly=True)  # Initialize LMDB for reading
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(self, index: int):
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        data_key = f"{index}".encode()
        with self.env.begin() as txn:
            value = txn.get(data_key)
            if value is None:
                raise IndexError(f"No item found at index {index}")
            json_data = json.loads(value.decode("utf-8"))
            experiment_class = EXPERIMENT_TYPE_MAP[
                json_data["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                json_data["experiment_reference"]["experiment_reference_type"]
            ]
            return {
                "experiment": experiment_class(**json_data["experiment"]),
                "experiment_reference": experiment_reference_class(
                    **json_data["experiment_reference"]
                ),
            }

    def _get_records_by_slice(self, slice_obj: slice):
        if self.env is None:
            raise ValueError("LMDB environment is not initialized.")
        start, stop, step = slice_obj.indices(len(self))
        data_keys = [f"{i}".encode() for i in range(start, stop, step)]
        with self.env.begin() as txn:
            results = []
            for key in data_keys:
                value = txn.get(key)
                if value is not None:
                    json_data = json.loads(value.decode())
                    experiment_class = EXPERIMENT_TYPE_MAP[
                        json_data["experiment"]["experiment_type"]
                    ]
                    experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                        json_data["experiment_reference"]["experiment_reference_type"]
                    ]
                    reconstructed_data = {
                        "experiment": experiment_class(**json_data["experiment"]),
                        "experiment_reference": experiment_reference_class(
                            **json_data["experiment_reference"]
                        ),
                    }
                    results.append(reconstructed_data)
            return results

    def __len__(self):
        self._init_lmdb(readonly=True)
        if self.env is None:
            return 0  # Return 0 if the LMDB doesn't exist yet
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def __bool__(self):
        return os.path.exists(self.lmdb_dir)

    def __repr__(self):
        return f"Deduplicator(root={self.root})"


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

        mean_experiment = FitnessExperiment(
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

        mean_experiment = GeneInteractionExperiment(
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

        return FitnessExperimentReference(
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

        return GeneInteractionExperimentReference(
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
