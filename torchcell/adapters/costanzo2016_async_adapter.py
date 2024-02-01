# torchcell/adapters/costanzo2016_adapter.py
# [[torchcell.adapters.costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/costanzo2016_adapter.py
# Test file: tests/torchcell/adapters/test_costanzo2016_adapter.py

from tqdm import tqdm
import hashlib
import random
import string
from enum import Enum, auto
from functools import lru_cache
from itertools import chain
from typing import Optional
import json
import pandas as pd
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import logger
from typing import Generator, Set
import torch
from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
)
from torchcell.datamodels import BaseGenotype, InterferenceGenotype, DeletionGenotype
from sortedcontainers import SortedList
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logger.debug(f"Loading module {__name__}.")

dataset = SmfCostanzo2016Dataset()


class SmfCostanzo2016Adapter:
    def __init__(self, dataset: SmfCostanzo2016Dataset, num_workers: int = 1):
        self.dataset = dataset
        self.num_workers = num_workers

    def get_nodes(self):
        methods = [
            self._get_experiment_reference_nodes,
            self._get_genome_nodes,
            self._get_experiment_nodes,
            self._get_dataset_nodes,
            self._get_genotype_nodes,
            self._get_environment_nodes,
            self._get_media_nodes,
            self._get_temperature_nodes,
            self._get_phenotype_nodes,
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(method) for method in methods]

            for future in as_completed(futures):
                try:
                    node_generator = future.result()
                    for node in node_generator:
                        yield node
                except Exception as exc:
                    logger.error(
                        f"Node generation method generated an exception: {exc}"
                    )

    def _get_experiment_reference_nodes(self) -> None:
        for i, data in enumerate(self.dataset.experiment_reference_index):
            nodes = []
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"SmfCostanzo2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )
            nodes.append(node)

        return nodes

    def _get_genome_nodes(self) -> None:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset.experiment_reference_index):
            genome_id = hashlib.sha256(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                node = BioCypherNode(
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
                nodes.append(node)
        return nodes

    def _get_experiment_nodes(self) -> None:
        for i, data in enumerate(self.dataset):
            nodes = []
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            node = BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"SmfCostanzo2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_genotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            genotype_id = hashlib.sha256(
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

                node = BioCypherNode(
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
                nodes.append(node)
            return

    @staticmethod
    def _get_perturbation(
        genotype: BaseGenotype,
    ) -> Generator[BioCypherNode, None, None]:
        if genotype.perturbation:
            i = 1
            perturbation_id = hashlib.sha256(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            node = BioCypherNode(
                node_id=perturbation_id,
                preferred_id=f"perturbation_{i}",
                node_label="perturbation",
                properties={
                    "systematic_gene_name": [
                        genotype.perturbation.systematic_gene_name
                    ],
                    "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                    "description": genotype.perturbation.description,
                    "perturbation_type": genotype.perturbation.perturbation_type,
                    "strain_id": genotype.perturbation.strain_id,
                    "serialized_data": json.dumps(genotype.perturbation.model_dump()),
                },
            )
        return node

    def _get_environment_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(data["experiment"].environment.media.model_dump())
                node = BioCypherNode(
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
                nodes.append(node)

        for i, data in enumerate(self.dataset):
            environment_id = hashlib.sha256(
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

                node = BioCypherNode(
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
                nodes.append(node)
        return nodes

    def _get_media_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            media_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["experiment"].environment.media.name
                state = data["experiment"].environment.media.state

                node = BioCypherNode(
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
                nodes.append(node)
        for i, data in enumerate(self.dataset):
            media_id = hashlib.sha256(
                json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["reference"].reference_environment.media.name
                state = data["reference"].reference_environment.media.state

                node = BioCypherNode(
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
                nodes.append(node)
            return nodes

    def _get_temperature_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            temperature_id = hashlib.sha256(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                node = BioCypherNode(
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
                nodes.append(node)

        for i, data in enumerate(self.dataset):
            temperature_id = hashlib.sha256(
                json.dumps(
                    data["reference"].reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                node = BioCypherNode(
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
                nodes.append(node)
            return nodes

    def _get_phenotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            phenotype_id = hashlib.sha256(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                node = BioCypherNode(
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
                nodes.append(node)
            return nodes

        for i, data in enumerate(self.dataset):
            phenotype_id = hashlib.sha256(
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

                node = BioCypherNode(
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
                nodes.append(node)
            return nodes

    def _get_dataset_nodes(self) -> None:
        nodes = [
            BioCypherNode(
                node_id="SmfCostanzo2016",
                preferred_id="SmfCostanzo2016",
                node_label="dataset",
            )
        ]
        return nodes

    def get_edges(self):
        methods = [
            self._get_dataset_experiment_ref_edges,
            self._get_experiment_dataset_edges,
            self._get_experiment_ref_experiment_edges,
            self._get_genotype_experiment_edges,
            self._get_environment_experiment_edges,
            self._get_environment_experiment_ref_edges,
            self._get_phenotype_experiment_edges,
            self._get_phenotype_experiment_ref_edges,
            self._get_media_environment_edges,
            self._get_temperature_environment_edges,
            self._get_genome_edges,
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(method) for method in methods]
            for future in as_completed(futures):
                try:
                    edge_generator = future.result()
                    for edge in edge_generator:
                        yield edge
                except Exception as exc:
                    logger.error(
                        f"Edge generation method generated an exception: {exc}"
                    )

    def _get_dataset_experiment_ref_edges(self):
        edges = []
        for data in self.dataset.experiment_reference_index:
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="SmfCostanzo2016",
                relationship_label="experiment reference member of",
            )
            edges.append(edge)
        return edges

    def _get_experiment_dataset_edges(self):
        edges = []
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=experiment_id,
                target_id="SmfCostanzo2016",
                relationship_label="experiment member of",
            )
            edges.append(edge)
        return edges

    def _get_experiment_ref_experiment_edges(self):
        edges = []
        for data in self.dataset.experiment_reference_index:
            dataset_subset = self.dataset[torch.tensor(data.index)]
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.sha256(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                edge = BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )
                edges.append(edge)

    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            genotype_id = hashlib.sha256(
                json.dumps(data["experiment"].genotype.model_dump()).encode("utf-8")
            ).hexdigest()

            self._get_perturbation_genotype_edges(
                genotype=data["experiment"].genotype, genotype_id=genotype_id
            )

            edge = BioCypherEdge(
                source_id=genotype_id,
                target_id=experiment_id,
                relationship_label="genotype member of",
            )
            edges.append(edge)

    @staticmethod
    def _get_perturbation_genotype_edges(
        genotype: BaseGenotype, genotype_id: str
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        if genotype.perturbation:
            perturbation_id = hashlib.sha256(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            edge = BioCypherEdge(
                source_id=perturbation_id,
                target_id=genotype_id,
                relationship_label="perturbation member of",
            )
            edges.append(edge)
        return edges

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_environment_experiment_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)
                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
        return edges

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_environment_experiment_ref_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.sha256(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
            return edges

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_phenotype_experiment_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.sha256(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)

                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
            return edges

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
        return edges

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_media_environment_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            media_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)

                edge = BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )
                edges.append(edge)
            return edges

    def _get_temperature_environment_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_temperature_environment_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            temperature_id = hashlib.sha256(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)

                edge = BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )
                edges.append(edge)
        return edges

    def _get_genome_edges(self) -> None:
        edges = []
        seen_genome_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.sha256(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )
                edges.append(edge)
        return edges


class DmfCostanzo2016Adapter:
    def __init__(self, dataset: DmfCostanzo2016Dataset, num_workers: int = 1):
        self.dataset = dataset
        self.num_workers = num_workers

    def get_nodes(self):
        # for node in self._get_experiment_nodes():
        #     yield node
        # for node in self._get_genotype_nodes():
        #     yield node
        # for node in self._get_perturbation_nodes():
        #     yield node
        # for node in self._get_phenotype_nodes():
        #     yield node

        methods = [
            self._get_experiment_nodes, # chunked
            self._get_genotype_nodes, # chunked
            self._get_perturbation_nodes, # chunked
            self._get_phenotype_nodes, # chunked
            self._get_experiment_reference_nodes,
            self._get_genome_nodes,
            self._get_dataset_nodes,
            self._get_environment_nodes,
            self._get_media_nodes,
            self._get_temperature_nodes,
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(method) for method in methods]
            for future in as_completed(futures):
                try:
                    node_generator = future.result()
                    for node in node_generator:
                        yield node
                except Exception as exc:
                    logger.error(
                        f"Node generation method generated an exception: {exc}"
                    )

    def _get_experiment_reference_nodes(self) -> None:
        nodes = []
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"DmfCostanzo2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_genome_nodes(self) -> None:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            genome_id = hashlib.sha256(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                node = BioCypherNode(
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
                nodes.append(node)
        return nodes

    @staticmethod
    def _chunk_experiment_nodes(data_chunk):
        nodes = []
        for i, data in enumerate(data_chunk):
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"DmfCostanzo2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_experiment_nodes(self):
        chunk_size = 1000  # Adjust as needed
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]

        nodes = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._chunk_experiment_nodes, chunk)
                for chunk in data_chunks
            ]
            for future in futures:
                nodes.extend(future.result())
        return nodes

    def _get_genotype_nodes(self):
        seen_node_ids = set()
        nodes = []

        for i, data in tqdm(enumerate(self.dataset)):
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.sha256(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                if genotype_id not in seen_node_ids:
                    seen_node_ids.add(genotype_id)

                    systematic_gene_name = genotype.perturbation.systematic_gene_name
                    perturbed_gene_name = genotype.perturbation.perturbed_gene_name
                    description = genotype.perturbation.description
                    perturbation_type = genotype.perturbation.perturbation_type

                    node = BioCypherNode(
                        node_id=genotype_id,
                        preferred_id=f"genotype_{i}",
                        node_label="genotype",
                        properties={
                            "systematic_gene_name": systematic_gene_name,
                            "perturbed_gene_name": perturbed_gene_name,
                            "description": description,
                            "perturbation_type": perturbation_type,
                            "serialized_data": json.dumps(genotype.model_dump()),
                        },
                    )
                    nodes.append(node)

        return nodes

    def _get_perturbation_nodes(self):
        nodes = []
        seen_node_ids = set()

        for data in tqdm(self.dataset):
            for genotype in data["experiment"].genotype:
                perturbation_id = hashlib.sha256(
                    json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
                ).hexdigest()

                if perturbation_id not in seen_node_ids:
                    seen_node_ids.add(perturbation_id)
                    node = BioCypherNode(
                        node_id=perturbation_id,
                        preferred_id=genotype.perturbation.perturbation_type,
                        node_label="perturbation",
                        properties={
                            "systematic_gene_name": genotype.perturbation.systematic_gene_name,
                            "perturbed_gene_name": genotype.perturbation.perturbed_gene_name,
                            "description": genotype.perturbation.description,
                            "perturbation_type": genotype.perturbation.perturbation_type,
                            "strain_id": genotype.perturbation.strain_id,
                            "serialized_data": json.dumps(
                                genotype.perturbation.model_dump()
                            ),
                        },
                    )
                    nodes.append(node)
        return nodes

    def _get_environment_nodes(self):
        nodes = []
        seen_node_ids = set()

        for data in tqdm(self.dataset.experiment_reference_index):
            environment_id = hashlib.sha256(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if environment_id not in seen_node_ids:
                seen_node_ids.add(environment_id)
                media = json.dumps(
                    data.reference.reference_environment.media.model_dump()
                )

                node = BioCypherNode(
                    node_id=environment_id,
                    preferred_id=f"environment_ref_{environment_id}",
                    node_label="environment",
                    properties={
                        "temperature": data.reference.reference_environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data.reference.reference_environment.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_media_nodes(self):
        seen_node_ids = set()
        nodes = []
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            media_id = hashlib.sha256(
                json.dumps(
                    data.reference.reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data.reference.reference_environment.media.name
                state = data.reference.reference_environment.media.state

                node = BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data.reference.reference_environment.media.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    # Function for processing experiment reference temperatures
    def _get_temperature_nodes(self):
        seen_node_ids = set()
        nodes = []
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            temperature_id = hashlib.sha256(
                json.dumps(
                    data.reference.reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                node = BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data.reference.reference_environment.temperature.value,
                        "unit": data.reference.reference_environment.temperature.unit,
                        "serialized_data": json.dumps(
                            data.reference.reference_environment.temperature.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_phenotype_nodes(self):
        seen_node_ids = set()
        nodes = []

        for i, data in tqdm(enumerate(self.dataset)):
            phenotype_id = hashlib.sha256(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)

                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                node = BioCypherNode(
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
                nodes.append(node)

        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)

                graph_level = data.reference.reference_phenotype.graph_level
                label = data.reference.reference_phenotype.label
                label_error = data.reference.reference_phenotype.label_error
                fitness = data.reference.reference_phenotype.fitness
                fitness_std = data.reference.reference_phenotype.fitness_std

                node = BioCypherNode(
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
                            data.reference.reference_phenotype.model_dump()
                        ),
                    },
                )
                nodes.append(node)

        return nodes

    def _get_dataset_nodes(self) -> None:
        nodes = [
            BioCypherNode(
                node_id="DmfCostanzo2016",
                preferred_id="DmfCostanzo2016",
                node_label="dataset",
            )
        ]
        return nodes

    # def get_edges(self):
    #     for edge in self._get_experiment_dataset_edges():
    #         yield edge
    #     for edge in self._get_dataset_experiment_ref_edges():
    #         yield edge
    #     for edge in self._get_experiment_ref_experiment_edges():
    #         yield edge
    #     for edge in self._get_genotype_experiment_edges():
    #         yield edge
    #     for edge in self._get_environment_experiment_edges():
    #         yield edge
    #     for edge in self._get_environment_experiment_ref_edges():
    #         yield edge
    #     for edge in self._get_phenotype_experiment_edges():
    #         yield edge
    #     for edge in self._get_phenotype_experiment_ref_edges():
    #         yield edge
    #     # First, generate all perturbation edges
    #     for edge in self._get_perturbation_genotype_edges():
    #         yield edge
    #     for edge in self._get_media_environment_edges():
    #         yield edge
    #     for edge in self._get_temperature_environment_edges():
    #         yield edge
    #     for edge in self._get_genome_edges():
    #         yield edge

    def get_edges(self):
        # Create a list of all the generator methods
        generator_methods = [
            self._get_experiment_dataset_edges,
            self._get_dataset_experiment_ref_edges,
            self._get_experiment_ref_experiment_edges,
            self._get_genotype_experiment_edges,
            self._get_environment_experiment_edges,
            self._get_environment_experiment_ref_edges,
            self._get_phenotype_experiment_edges,
            self._get_phenotype_experiment_ref_edges,
            self._get_perturbation_genotype_edges,
            self._get_media_environment_edges,
            self._get_temperature_environment_edges,
            self._get_genome_edges,
        ]

        # Use ProcessPoolExecutor to execute the generators in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for method in generator_methods:
                # Submit the generator method to the executor and get a future
                future = executor.submit(method)
                # Yield edges as they are produced by the generator
                try:
                    for edge in future.result():
                        yield edge
                except Exception as exc:
                    logger.error(f"Exception occurred: {exc}")

    @staticmethod
    def _chunk_experiment_dataset_edges(data_chunk):
        # Process a chunk of data
        edges = []
        for data in data_chunk:
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=experiment_id,
                target_id="DmfCostanzo2016",
                relationship_label="experiment member of",
            )
            edges.append(edge)
        return edges

    def _get_experiment_dataset_edges(self):
        chunk_size = 1000  # Adjust as needed
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]

        edges = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._chunk_experiment_dataset_edges, chunk)
                for chunk in data_chunks
            ]
            for future in futures:
                edges.extend(future.result())
        return edges

    def _get_dataset_experiment_ref_edges(self):
        edges = []
        for data in tqdm(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="DmfCostanzo2016",
                relationship_label="experiment reference member of",
            )
            edges.append(edge)
        return edges

    def _get_experiment_ref_experiment_edges(self):
        edges = []
        for data in tqdm(self.dataset.experiment_reference_index):
            dataset_subset = self.dataset[torch.tensor(data.index)]
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.sha256(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                edge = BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )
                edges.append(edge)
        return edges

    @staticmethod
    def _chunk_genotype_experiment_edges(data_chunk):
        edges = []
        for data in data_chunk:
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.sha256(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                edge = BioCypherEdge(
                    source_id=genotype_id,
                    target_id=experiment_id,
                    relationship_label="genotype member of",
                )
                edges.append(edge)
        return edges

    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        chunk_size = 1000  # Adjust as needed
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]

        edges = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._chunk_genotype_experiment_edges, chunk)
                for chunk in data_chunks
            ]
            for future in futures:
                edges.extend(future.result())
        return edges

    def _get_perturbation_genotype_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_edges = set()  # Set to track seen edges

        for data in tqdm(self.dataset):
            for genotype in data["experiment"].genotype:
                if genotype.perturbation:
                    genotype_id = hashlib.sha256(
                        json.dumps(genotype.model_dump()).encode("utf-8")
                    ).hexdigest()
                    perturbation_id = hashlib.sha256(
                        json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
                    ).hexdigest()

                    # Create a tuple representing the edge
                    edge_tuple = (
                        perturbation_id,
                        genotype_id,
                        "perturbation member of",
                    )

                    # Check if this edge is a duplicate
                    if edge_tuple not in seen_edges:
                        seen_edges.add(edge_tuple)
                        edge = BioCypherEdge(
                            source_id=perturbation_id,
                            target_id=genotype_id,
                            relationship_label="perturbation member of",
                        )
                        edges.append(edge)

        return edges

    @staticmethod
    def _chunk_environment_experiment_edges(data_chunk):
        edges = []
        seen_environment_experiment_pairs = set()
        for data in data_chunk:
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.sha256(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)
                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
        return edges

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        chunk_size = 1000  # Adjust as needed
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]

        edges = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._chunk_environment_experiment_edges, chunk)
                for chunk in data_chunks
            ]
            for future in futures:
                edges.extend(future.result())
        return edges

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_environment_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.sha256(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
        return edges

    @staticmethod
    def _chunk_phenotype_experiment_edges(data_chunk):
        edges = []
        seen_phenotype_experiment_pairs = set()
        for data in data_chunk:
            experiment_id = hashlib.sha256(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.sha256(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)
                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
        return edges

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        chunk_size = 1000  # Adjust as needed
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]

        edges = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._chunk_phenotype_experiment_edges, chunk)
                for chunk in data_chunks
            ]
            for future in futures:
                edges.extend(future.result())
        return edges

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
        return edges

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        # Optimized by using reference
        # We know reference contains all media and envs
        edges = []
        seen_media_environment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            environment_id = hashlib.sha256(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()
            media_id = hashlib.sha256(
                json.dumps(
                    data.reference.reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()
            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)
                edge = BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )
                edges.append(edge)
        return edges

    def _get_temperature_environment_edges(
        self,
        # Optimized by using reference
        # We know reference contain all envs and temps
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_temperature_environment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            environment_id = hashlib.sha256(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()
            temperature_id = hashlib.sha256(
                json.dumps(
                    data.reference.reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()
            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)
                edge = BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )
                edges.append(edge)
        return edges

    def _get_genome_edges(self) -> None:
        edges = []
        seen_genome_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.sha256(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )
                edges.append(edge)
        return edges


if __name__ == "__main__":
    from biocypher import BioCypher

    # # # Simple Testing
    # dataset = SmfCostanzo2016Dataset()
    # adapter = SmfCostanzo2016Adapter(dataset=dataset)
    # [i for i in adapter.get_nodes()]
    # [i for i in adapter.get_edges()]

    # Advanced Testing
    # bc = BioCypher()
    # dataset = SmfCostanzo2016Dataset()
    # adapter = SmfCostanzo2016Adapter(dataset=dataset, num_workers=10)
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())

    # # # Write admin import statement and schema information (for biochatter)
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)

    # # # # Print summary
    # bc.summary()
    # print()

    ## Dmf
    # Simple Testing
    # dataset = DmfCostanzo2016Dataset()
    # adapter = DmfCostanzo2016Adapter(dataset=dataset)
    # [i for i in adapter.get_nodes()]
    # [i for i in adapter.get_edges()]

    # Advanced Testing
    bc = BioCypher()
    dataset = DmfCostanzo2016Dataset()
    # dataset = DmfCostanzo2016Dataset(
    #     root="data/torchcell/dmf_costanzo2016_subset_n_1000",
    #     subset_n=1000,
    #     preprocess=None,
    # )
    # dataset = DmfCostanzo2016Dataset(
    #     root="data/torchcell/dmf_costanzo2016_subset_n_100000",
    #     subset_n=100000,
    #     preprocess=None,
    # )
    # dataset = DmfCostanzo2016Dataset(
    #     root="data/torchcell/dmf_costanzo2016_subset_n_1e6",
    #     subset_n=int(1e6),
    #     preprocess=None,
    # )
    # dataset = DmfCostanzo2016Dataset(
    #     root="data/torchcell/dmf_costanzo2016_subset_n_1e7",
    #     subset_n=int(1e7),
    #     preprocess=None,
    # )
    adapter = DmfCostanzo2016Adapter(dataset=dataset, num_workers=10)
    bc.show_ontology_structure()
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()
