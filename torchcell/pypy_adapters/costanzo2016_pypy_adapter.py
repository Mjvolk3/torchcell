# torchcell/pypy_adapters/costanzo2016_pypy_adapter
# [[torchcell.pypy_adapters.costanzo2016_pypy_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/pypy_adapters/costanzo2016_pypy_adapter
# Test file: tests/torchcell/pypy_adapters/test_costanzo2016_pypy_adapter.py


from tqdm import tqdm
import hashlib
import json
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import logger
from typing import Generator, Set
from torchcell.datamodels import BaseGenotype, InterferenceGenotype, DeletionGenotype

from torchcell.dataset_readers import LmdbDatasetReader

logger.debug(f"Loading module {__name__}.")


class SmfCostanzo2016Adapter:
    def __init__(self, dataset: LmdbDatasetReader):
        self.dataset = dataset

    def get_nodes(self) -> None:
        logger.info("Getting nodes.")
        logger.info("Get experiment reference nodes.")
        yield from self._get_experiment_reference_nodes()
        logger.info("Get genome nodes.")
        yield from self._get_genome_nodes()
        logger.info("Get experiment nodes.")
        yield from self._get_experiment_nodes()
        logger.info("Get dataset nodes.")
        yield from self._get_dataset_nodes()
        logger.info("Get genotype nodes.")
        logger.info("--- perturbation nodes.")
        yield from self._get_genotype_nodes()
        logger.info("Get environment nodes.")
        yield from self._get_environment_nodes()
        logger.info("Get media nodes.")
        yield from self._get_media_nodes()
        logger.info("Get temperature nodes.")
        yield from self._get_temperature_nodes()
        logger.info("Get phenotype nodes.")
        yield from self._get_phenotype_nodes()

    def _get_experiment_reference_nodes(self) -> None:
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"SmfCostanzo2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )

    def _get_genome_nodes(self) -> None:
        seen_node_ids: Set[str] = set()

        for i, data in enumerate(self.dataset.experiment_reference_index):
            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                yield BioCypherNode(
                    node_id=genome_id,
                    preferred_id=f"reference_genome_{i}",
                    node_label="genome",
                    properties={
                        "species": data.reference.reference_genome.species,
                        "strain": data.reference.reference_genome.strain,
                        "serialized_data": json.dumps(
                            data.reference.reference_genome.model_dump()
                        ),
                    },
                )

    def _get_experiment_nodes(self) -> None:
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"SmfCostanzo2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )

    def _get_genotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            genotype_id = hashlib.md5(
                json.dumps(data["experiment"].genotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if genotype_id not in seen_node_ids:
                seen_node_ids.add(genotype_id)
                systematic_gene_name = data[
                    "experiment"
                ].genotype.perturbation.systematic_gene_name
                perturbed_gene_name = data[
                    "experiment"
                ].genotype.perturbation.perturbed_gene_name
                description = data["experiment"].genotype.perturbation.description
                perturbation_type = data[
                    "experiment"
                ].genotype.perturbation.perturbation_type

                self._get_perturbation(data["experiment"].genotype)

                yield BioCypherNode(
                    node_id=genotype_id,
                    preferred_id=f"genotype_{i}",
                    node_label="genotype",
                    properties={
                        "systematic_gene_names": [systematic_gene_name],
                        "perturbed_gene_names": [perturbed_gene_name],
                        "is_deletion_genotype": isinstance(
                            data["experiment"].genotype, DeletionGenotype
                        ),
                        "is_interference_genotype": isinstance(
                            data["experiment"].genotype, InterferenceGenotype
                        ),
                        "description": description,
                        "perturbation_types": [perturbation_type],
                        "serialized_data": json.dumps(
                            data["experiment"].genotype.model_dump()
                        ),
                    },
                )

    @staticmethod
    def _get_perturbation(
        genotype: BaseGenotype,
    ) -> Generator[BioCypherNode, None, None]:
        perturbation_id = hashlib.md5(
            json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
        ).hexdigest()

        yield BioCypherNode(
            node_id=perturbation_id,
            preferred_id=genotype.perturbation.perturbation_type,
            node_label="perturbation",
            properties={
                "systematic_gene_name": [genotype.perturbation.systematic_gene_name],
                "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                "description": genotype.perturbation.description,
                "perturbation_type": genotype.perturbation.perturbation_type,
                "strain_id": genotype.perturbation.strain_id,
                "serialized_data": json.dumps(genotype.perturbation.model_dump()),
            },
        )

    def _get_environment_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(data["experiment"].environment.media.model_dump())

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data["experiment"].environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.model_dump()
                        ),
                    },
                )
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["reference"].reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                )

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.model_dump()
                        ),
                    },
                )

    def _get_media_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["experiment"].environment.media.name
                state = data["experiment"].environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.media.model_dump()
                        ),
                    },
                )
        for i, data in enumerate(self.dataset):
            media_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["reference"].reference_environment.media.name
                state = data["reference"].reference_environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.media.model_dump()
                        ),
                    },
                )

    def _get_temperature_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data["experiment"].environment.temperature.value,
                        "unit": data["experiment"].environment.temperature.unit,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.temperature.model_dump()
                        ),
                    },
                )

        for i, data in enumerate(self.dataset):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "description": data[
                            "reference"
                        ].reference_environment.temperature.description,
                        "serialized_data": json.dumps(
                            data[
                                "reference"
                            ].reference_environment.temperature.model_dump()
                        ),
                    },
                )

    def _get_phenotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["experiment"].phenotype.model_dump()
                        ),
                    },
                )

        # References
        for i, data in enumerate(self.dataset):
            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["reference"].reference_phenotype.graph_level
                label = data["reference"].reference_phenotype.label
                label_error = data["reference"].reference_phenotype.label_error
                fitness = data["reference"].reference_phenotype.fitness
                fitness_std = data["reference"].reference_phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["reference"].reference_phenotype.model_dump()
                        ),
                    },
                )

    def _get_dataset_nodes(self) -> None:
        yield BioCypherNode(
            node_id="SmfCostanzo2016",
            preferred_id="SmfCostanzo2016",
            node_label="dataset",
        )

    def get_edges(self) -> None:
        logger.info("Generating edges.")
        logger.info("Get dataset experiment reference edges.")
        yield from self._get_dataset_experiment_ref_edges()
        logger.info("Get experiment dataset edges.")
        yield from self._get_experiment_dataset_edges()
        logger.info("Get experiment reference experiment edges.")
        yield from self._get_experiment_ref_experiment_edges()
        logger.info("Get genotype experiment edges.")
        logger.info("--- perturbation genotype edges.")
        yield from self._get_genotype_experiment_edges()
        logger.info("Get environment experiment edges.")
        yield from self._get_environment_experiment_edges()
        logger.info("Get environment experiment reference edges.")
        yield from self._get_environment_experiment_ref_edges()
        logger.info("Get phenotype experiment edges.")
        yield from self._get_phenotype_experiment_edges()
        logger.info("Get phenotype experiment reference edges.")
        yield from self._get_phenotype_experiment_ref_edges()
        logger.info("Get media environment edges.")
        yield from self._get_media_environment_edges()
        logger.info("Get temperature environment edges.")
        yield from self._get_temperature_environment_edges()
        logger.info("Get genome experiment reference edges.")
        yield from self._get_genome_edges()

    def _get_dataset_experiment_ref_edges(self):
        # concept level
        for data in self.dataset:
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="SmfCostanzo2016",
                relationship_label="experiment reference member of",
            )

    def _get_experiment_dataset_edges(self):
        # concept level
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_id,
                target_id="SmfCostanzo2016",
                relationship_label="experiment member of",
            )

    def _get_experiment_ref_experiment_edges(self):
        # instance level
        for data in self.dataset.experiment_reference_index:
            dataset_subset = self.dataset[data.index]
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.md5(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                yield BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )

    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        # CHECK if needed - don't think needed since exp ref index
        # seen_genotype_experiment_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            genotype_id = hashlib.md5(
                json.dumps(data["experiment"].genotype.model_dump()).encode("utf-8")
            ).hexdigest()

            self._get_perturbation_genotype_edges(
                genotype=data["experiment"].genotype, genotype_id=genotype_id
            )

            # CHECK if needed - don't think needed since exp ref index
            # genotype_experiment_pair = (genotype_id, experiment_id)
            # if genotype_experiment_pair not in seen_genotype_experiment_pairs:
            #     seen_genotype_experiment_pairs.add(genotype_experiment_pair)

            yield BioCypherEdge(
                source_id=genotype_id,
                target_id=experiment_id,
                relationship_label="genotype member of",
            )

    @staticmethod
    def _get_perturbation_genotype_edges(
        genotype: BaseGenotype, genotype_id: str
    ) -> Generator[BioCypherEdge, None, None]:
        perturbation_id = hashlib.md5(
            json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
        ).hexdigest()

        yield BioCypherEdge(
            source_id=perturbation_id,
            target_id=genotype_id,
            relationship_label="perturbation member of",
        )

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_pairs: Set[tuple] = set()

        # Linking environments to experiments
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_ref_pairs: Set[tuple] = set()

        # Linking environments to experiment references
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.md5(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_pairs: Set[tuple] = set()

        # Linking phenotypes to experiments
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()

        # Linking phenotypes to experiment references
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_media_environment_pairs: Set[tuple] = set()

        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)

                yield BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )

    def _get_temperature_environment_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_temperature_environment_pairs: Set[tuple] = set()

        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)

                yield BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )

    def _get_genome_edges(self) -> None:
        seen_genome_experiment_ref_pairs: Set[tuple] = set()

        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )


class DmfCostanzo2016Adapter:
    def __init__(self, dataset: LmdbDatasetReader):
        self.dataset = dataset

    def get_nodes(self) -> None:
        logger.info("Getting nodes.")
        logger.info("Get experiment reference nodes.")
        yield from self._get_experiment_reference_nodes()
        logger.info("Get genome nodes.")
        yield from self._get_genome_nodes()
        logger.info("Get experiment nodes.")
        yield from self._get_experiment_nodes()
        logger.info("Get dataset nodes.")
        yield from self._get_dataset_nodes()
        logger.info("Get genotype nodes.")
        logger.info("--- perturbation nodes.")
        yield from self._get_genotype_nodes()
        logger.info("Get environment nodes.")
        yield from self._get_environment_nodes()
        logger.info("Get media nodes.")
        yield from self._get_media_nodes()
        logger.info("Get temperature nodes.")
        yield from self._get_temperature_nodes()
        logger.info("Get phenotype nodes.")
        yield from self._get_phenotype_nodes()

    def _get_experiment_reference_nodes(self) -> None:
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"DmfCostanzo2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )

    def _get_genome_nodes(self) -> None:
        seen_node_ids: Set[str] = set()

        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                yield BioCypherNode(
                    node_id=genome_id,
                    preferred_id=f"reference_genome_{i}",
                    node_label="genome",
                    properties={
                        "species": data.reference.reference_genome.species,
                        "strain": data.reference.reference_genome.strain,
                        "serialized_data": json.dumps(
                            data.reference.reference_genome.model_dump()
                        ),
                    },
                )

    def _get_experiment_nodes(self) -> None:
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"DmfCostanzo2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )

    def _get_genotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.md5(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                if genotype_id not in seen_node_ids:
                    seen_node_ids.add(genotype_id)
                    systematic_gene_name = genotype.perturbation.systematic_gene_name
                    perturbed_gene_name = genotype.perturbation.perturbed_gene_name
                    description = genotype.perturbation.description
                    perturbation_type = genotype.perturbation.perturbation_type
                    self._get_perturbation(genotype)

                    yield BioCypherNode(
                        node_id=genotype_id,
                        preferred_id=f"genotype_{i}",
                        node_label="genotype",
                        properties={
                            "systematic_gene_names": [systematic_gene_name],
                            "perturbed_gene_names": [perturbed_gene_name],
                            "is_deletion_genotype": isinstance(
                                data["experiment"].genotype, DeletionGenotype
                            ),
                            "is_interference_genotype": isinstance(
                                data["experiment"].genotype, InterferenceGenotype
                            ),
                            "description": description,
                            "perturbation_types": [perturbation_type],
                            "serialized_data": json.dumps(genotype.model_dump()),
                        },
                    )

    @staticmethod
    def _get_perturbation(
        genotype: BaseGenotype,
    ) -> Generator[BioCypherNode, None, None]:
        perturbation_id = hashlib.md5(
            json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
        ).hexdigest()

        yield BioCypherNode(
            node_id=perturbation_id,
            preferred_id=genotype.perturbation.perturbation_type,
            node_label="perturbation",
            properties={
                "systematic_gene_name": [genotype.perturbation.systematic_gene_name],
                "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                "description": genotype.perturbation.description,
                "perturbation_type": genotype.perturbation.perturbation_type,
                "strain_id": genotype.perturbation.strain_id,
                "serialized_data": json.dumps(genotype.perturbation.model_dump()),
            },
        )

    def _get_environment_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(data["experiment"].environment.media.model_dump())

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data["experiment"].environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.model_dump()
                        ),
                    },
                )
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["reference"].reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                )

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.model_dump()
                        ),
                    },
                )

    def _get_media_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["experiment"].environment.media.name
                state = data["experiment"].environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.media.model_dump()
                        ),
                    },
                )
        for i, data in tqdm(enumerate(self.dataset)):
            media_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["reference"].reference_environment.media.name
                state = data["reference"].reference_environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.media.model_dump()
                        ),
                    },
                )

    def _get_temperature_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data["experiment"].environment.temperature.value,
                        "unit": data["experiment"].environment.temperature.unit,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.temperature.model_dump()
                        ),
                    },
                )

        for i, data in tqdm(enumerate(self.dataset)):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "description": data[
                            "reference"
                        ].reference_environment.temperature.description,
                        "serialized_data": json.dumps(
                            data[
                                "reference"
                            ].reference_environment.temperature.model_dump()
                        ),
                    },
                )

    def _get_phenotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["experiment"].phenotype.model_dump()
                        ),
                    },
                )

        # References
        for i, data in tqdm(enumerate(self.dataset)):
            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["reference"].reference_phenotype.graph_level
                label = data["reference"].reference_phenotype.label
                label_error = data["reference"].reference_phenotype.label_error
                fitness = data["reference"].reference_phenotype.fitness
                fitness_std = data["reference"].reference_phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["reference"].reference_phenotype.model_dump()
                        ),
                    },
                )

    def _get_dataset_nodes(self) -> None:
        yield BioCypherNode(
            node_id="DmfCostanzo2016",
            preferred_id="DmfCostanzo2016",
            node_label="dataset",
        )

    def get_edges(self) -> None:
        logger.info("Generating edges.")
        logger.info("Get dataset experiment reference edges.")
        yield from self._get_dataset_experiment_ref_edges()
        logger.info("Get experiment dataset edges.")
        yield from self._get_experiment_dataset_edges()
        logger.info("Get experiment reference experiment edges.")
        yield from self._get_experiment_ref_experiment_edges()
        logger.info("Get genotype experiment edges.")
        logger.info("--- perturbation genotype edges.")
        yield from self._get_genotype_experiment_edges()
        logger.info("Get environment experiment edges.")
        yield from self._get_environment_experiment_edges()
        logger.info("Get environment experiment reference edges.")
        yield from self._get_environment_experiment_ref_edges()
        logger.info("Get phenotype experiment edges.")
        yield from self._get_phenotype_experiment_edges()
        logger.info("Get phenotype experiment reference edges.")
        yield from self._get_phenotype_experiment_ref_edges()
        logger.info("Get media environment edges.")
        yield from self._get_media_environment_edges()
        logger.info("Get temperature environment edges.")
        yield from self._get_temperature_environment_edges()
        logger.info("Get genome experiment reference edges.")
        yield from self._get_genome_edges()

    def _get_dataset_experiment_ref_edges(self):
        # concept level
        for data in tqdm(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="DmfCostanzo2016",
                relationship_label="experiment reference member of",
            )

    def _get_experiment_dataset_edges(self):
        # concept level
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_id,
                target_id="DmfCostanzo2016",
                relationship_label="experiment member of",
            )

    def _get_experiment_ref_experiment_edges(self):
        # instance level
        print()
        for data in tqdm(self.dataset.experiment_reference_index):
            dataset_subset = self.dataset[data.index]
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.md5(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                yield BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )

    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        # CHECK if needed - don't think needed since exp ref index
        # seen_genotype_experiment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.md5(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                self._get_perturbation_genotype_edges(
                    genotype=genotype, genotype_id=genotype_id
                )

                # CHECK if needed - don't think needed since exp ref index
                # genotype_experiment_pair = (genotype_id, experiment_id)
                # if genotype_experiment_pair not in seen_genotype_experiment_pairs:
                #     seen_genotype_experiment_pairs.add(genotype_experiment_pair)

                yield BioCypherEdge(
                    source_id=genotype_id,
                    target_id=experiment_id,
                    relationship_label="genotype member of",
                )

    @staticmethod
    def _get_perturbation_genotype_edges(
        genotype: BaseGenotype, genotype_id: str
    ) -> Generator[BioCypherEdge, None, None]:
        if genotype.perturbation:
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherEdge(
                source_id=perturbation_id,
                target_id=genotype_id,
                relationship_label="perturbation member of",
            )

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_pairs: Set[tuple] = set()

        # Linking environments to experiments
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_ref_pairs: Set[tuple] = set()

        # Linking environments to experiment references
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.md5(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_pairs: Set[tuple] = set()

        # Linking phenotypes to experiments
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()

        # Linking phenotypes to experiment references
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_media_environment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)

                yield BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )

    def _get_temperature_environment_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_temperature_environment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)

                yield BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )

    def _get_genome_edges(self) -> None:
        seen_genome_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )


if __name__ == "__main__":
    from biocypher import BioCypher

    # # # Simple Testing
    # dataset = LmdbDatasetReader(root="data/torchcell/smf_costanzo2016")
    # adapter = SmfCostanzo2016Adapter(dataset=dataset)
    # [i for i in adapter.get_nodes()]
    # [i for i in adapter.get_edges()]

    ## Advanced Testing
    # bc = BioCypher()
    # dataset = LmdbDatasetReader("data/torchcell/smf_costanzo2016")
    # adapter = SmfCostanzo2016Adapter(dataset=dataset)
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())

    # # # Write admin import statement and schema information (for biochatter)
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)

    # # # Print summary
    # bc.summary()
    # print()~

    ## Dmf
    # Simple Testing
    # dataset = LmdbDatasetReader(root="data/torchcell/dmf_costanzo2016")
    # dataset = LmdbDatasetReader(root="dmf_costanzo2016_subset_n_10000")
    # adapter = DmfCostanzo2016Adapter(dataset=dataset)
    # [i for i in adapter.get_nodes()]
    # [i for i in adapter.get_edges()]

    # Advanced Testing
    bc = BioCypher()
    # dataset = LmdbDatasetReader("data/torchcell/dmf_costanzo2016")
    dataset = LmdbDatasetReader("data/torchcell/dmf_costanzo2016_subset_n_10000")
    adapter = DmfCostanzo2016Adapter(dataset=dataset)
    bc.show_ontology_structure()
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()
