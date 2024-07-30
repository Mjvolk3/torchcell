# torchcell/adapters/cell_adapter
# [[torchcell.adapters.cell_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/cell_adapter
# Test file: tests/torchcell/adapters/test_cell_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher._create import BioCypherEdge, BioCypherNode
import torch
from omegaconf import OmegaConf, DictConfig
from torchcell.loader import CpuExperimentLoaderMultiprocessing
from concurrent.futures import ProcessPoolExecutor
from torch_geometric.data import Dataset
from typing import List, Tuple, Callable, Generator, Set, Optional
from functools import wraps
import logging
import wandb
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CellAdapter:
    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        if loader_batch_size > chunk_size:
            raise ValueError(
                "chunk_size must be greater than or equal to loader_batch_size."
                "Our recommendation are chunk_size 2-3 order of magnitude in size."
            )
        self.config = config
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
        self.event = 0
        wandb.init()

        # Supported methods
        self.node_methods = [
            ("experiment_reference", self._get_experiment_reference_nodes),
            ("genome", self._get_genome_nodes),
            ("experiment", self._experiment_node),
            ("genotype", self._genotype_node),
            ("perturbation", self._perturbation_node),
            ("environment", self._environment_node),
            ("environment reference", self._get_environment_reference_nodes),
            ("media", self._media_node),
            ("media reference", self._get_media_reference_nodes),
            ("temperature", self._temperature_node),
            ("temperature reference", self._get_temperature_reference_nodes),
            ("fitness phenotype", self._fitness_phenotype_node),
            ("gene interaction phenotype", self._gene_interaction_phenotype_node),
            (
                "fitness phenotype reference",
                self._get_fitness_phenotype_reference_nodes,
            ),
            (
                "gene interaction phenotype reference",
                self._get_gene_interaction_phenotype_reference_nodes,
            ),
            ("dataset", self._get_dataset_nodes),
        ]
        self.edge_methods = [
            (
                "experiment reference to dataset",
                self._get_experiment_reference_to_dataset_edges,
            ),
            ("experiment to dataset", self._experiment_to_dataset_edge),
            (
                "experiment reference to experiment",
                self._experiment_reference_to_experiment_edge,
            ),
            ("genotype to experiment", self._genotype_to_experiment_edge),
            ("perturbation to genotype", self._perturbation_to_genotype_edges),
            ("environment to experiment", self._environment_to_experiment_edge),
            (
                "environment to experiment reference",
                self._get_environment_to_experiment_reference_edges,
            ),
            ("phenotype to experiment", self._phenotype_to_experiment_edge),
            ("media to environment", self._media_to_environment_edge),
            ("temperature to environment", self._temperature_to_environment_edge),
            (
                "genome to experiment reference",
                self._get_genome_to_experiment_reference_edges,
            ),
        ]

    def get_data_by_type(
        self, chunk_processing_func: Callable, chunk_size: Optional[int] = None
    ):
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
            futures = [
                executor.submit(chunk_processing_func, chunk) for chunk in data_chunks
            ]
            for future in futures:
                for data in future.result():
                    yield data

    def data_chunker(data_creation_logic):
        @wraps(data_creation_logic)
        def decorator(self, data_chunk: dict):
            # data_loader = CpuExperimentLoader(
            data_loader = CpuExperimentLoaderMultiprocessing(
                data_chunk,
                batch_size=self.loader_batch_size,
                num_workers=self.io_workers,
            )
            datas = []
            for batch in tqdm(data_loader):
                for data in batch:
                    transformed_data = data_chunk.transform_item(data)
                    data = data_creation_logic(self, transformed_data)
                    if isinstance(data, list):
                        # list[BiocypherNode] (e.g. perturbations)
                        datas.extend(data)
                    else:
                        # BiocypherNode
                        datas.append(data)
            return datas

        return decorator

    @property
    def supported_node_methods(self) -> List[str]:
        return [method_name for method_name, _ in self.node_methods]

    @property
    def supported_edge_methods(self) -> List[str]:
        return [method_name for method_name, _ in self.edge_methods]

    def get_nodes(self):
        for method_name, method in self.node_methods:
            if method_name in self.config.cell_adapter.node_methods:
                log.info(f"Running: {method_name}")
                if method.__name__.startswith("_get_"):
                    yield from method()
                else:
                    yield from self.get_data_by_type(method)
                self.event += 1
                wandb.log({"event": self.event})

    def get_edges(self):
        for method_name, method in self.edge_methods:
            if method_name in self.config.cell_adapter.edge_methods:
                log.info(f"Running: {method_name}")
                if method.__name__.startswith("_get_"):
                    yield from method()
                else:
                    yield from self.get_data_by_type(method)
                self.event += 1
                wandb.log({"event": self.event})

    # nodes
    def _get_experiment_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id="experiment reference",
                node_label="experiment reference",
                properties={
                    "dataset_name": self.dataset.name,
                    "pubmed_id": data.reference.pubmed_id,
                    "pubmed_url": data.reference.pubmed_url,
                    "doi": data.reference.doi,
                    "doi_url": data.reference.doi_url,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_genome_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            genome_id = hashlib.sha256(
                json.dumps(data.reference.genome_reference.model_dump()).encode("utf-8")
            ).hexdigest()
            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                node = BioCypherNode(
                    node_id=genome_id,
                    preferred_id="genome",
                    node_label="genome",
                    properties={
                        "species": data.reference.genome_reference.species,
                        "strain": data.reference.genome_reference.strain,
                        "serialized_data": json.dumps(
                            data.reference.genome_reference.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    @data_chunker
    def _experiment_node(self, data: dict) -> BioCypherNode:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=experiment_id,
            preferred_id="experiment",
            node_label="experiment",
            properties={
                "dataset_name": self.dataset.name,
                "pubmed_id": data["experiment"].pubmed_id,
                "pubmed_url": data["experiment"].pubmed_url,
                "doi": data["experiment"].doi,
                "doi_url": data["experiment"].doi_url,
                "serialized_data": json.dumps(data["experiment"].model_dump()),
            },
        )

    @data_chunker
    def _genotype_node(self, data: dict) -> BioCypherNode:
        genotype = data["experiment"].genotype
        genotype_id = hashlib.sha256(
            json.dumps(genotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=genotype_id,
            preferred_id="genotype",
            node_label="genotype",
            properties={
                "systematic_gene_names": genotype.systematic_gene_names,
                "perturbed_gene_names": genotype.perturbed_gene_names,
                "perturbation_types": genotype.perturbation_types,
                "serialized_data": json.dumps(genotype.model_dump()),
            },
        )

    @data_chunker
    def _perturbation_node(self, data: dict) -> list[BioCypherNode]:
        perturbations = data["experiment"].genotype.perturbations
        nodes = []
        for perturbation in perturbations:
            perturbation_id = hashlib.sha256(
                json.dumps(perturbation.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=perturbation_id,
                preferred_id=perturbation.perturbation_type,
                node_label="perturbation",
                properties={
                    "systematic_gene_name": perturbation.systematic_gene_name,
                    "perturbed_gene_name": perturbation.perturbed_gene_name,
                    "perturbation_type": perturbation.perturbation_type,
                    "description": perturbation.description,
                    "strain_id": perturbation.strain_id,
                    "serialized_data": json.dumps(perturbation.model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    @data_chunker
    def _environment_node(self, data: dict) -> BioCypherNode:
        environment_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
        ).hexdigest()
        media = json.dumps(data["experiment"].environment.media.model_dump())
        return BioCypherNode(
            node_id=environment_id,
            preferred_id="environment",
            node_label="environment",
            properties={
                "temperature": data["experiment"].environment.temperature.value,
                "media": media,
                "serialized_data": json.dumps(
                    data["experiment"].environment.model_dump()
                ),
            },
        )

    def _get_environment_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            environment_id = hashlib.sha256(
                json.dumps(data.reference.environment_reference.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()
            if environment_id not in seen_node_ids:
                seen_node_ids.add(environment_id)
                media = json.dumps(
                    data.reference.environment_reference.media.model_dump()
                )
                node = BioCypherNode(
                    node_id=environment_id,
                    preferred_id="environment",
                    node_label="environment",
                    properties={
                        "temperature": data.reference.environment_reference.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data.reference.environment_reference.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    @data_chunker
    def _media_node(self, data: dict) -> BioCypherNode:
        media_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.media.model_dump()).encode(
                "utf-8"
            )
        ).hexdigest()
        name = data["experiment"].environment.media.name
        state = data["experiment"].environment.media.state
        return BioCypherNode(
            node_id=media_id,
            preferred_id="media",
            node_label="media",
            properties={
                "name": name,
                "state": state,
                "serialized_data": json.dumps(
                    data["experiment"].environment.media.model_dump()
                ),
            },
        )

    def _get_media_reference_nodes(self) -> list[BioCypherNode]:
        seen_node_ids = set()
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            media_id = hashlib.sha256(
                json.dumps(
                    data.reference.environment_reference.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()
            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data.reference.environment_reference.media.name
                state = data.reference.environment_reference.media.state
                node = BioCypherNode(
                    node_id=media_id,
                    preferred_id="media",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data.reference.environment_reference.media.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    @data_chunker
    def _temperature_node(self, data: dict) -> BioCypherNode:
        temperature_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.temperature.model_dump()).encode(
                "utf-8"
            )
        ).hexdigest()
        return BioCypherNode(
            node_id=temperature_id,
            preferred_id="temperature",
            node_label="temperature",
            properties={
                "value": data["experiment"].environment.temperature.value,
                "unit": data["experiment"].environment.temperature.unit,
                "serialized_data": json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ),
            },
        )

    def _get_temperature_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            temperature_id = hashlib.sha256(
                json.dumps(
                    data.reference.environment_reference.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()
            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)
                node = BioCypherNode(
                    node_id=temperature_id,
                    preferred_id="temperature",
                    node_label="temperature",
                    properties={
                        "value": data.reference.environment_reference.temperature.value,
                        "unit": data.reference.environment_reference.temperature.unit,
                        "serialized_data": json.dumps(
                            data.reference.environment_reference.temperature.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    @data_chunker
    def _fitness_phenotype_node(self, data: dict) -> BioCypherNode:
        phenotype_id = hashlib.sha256(
            json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = data["experiment"].phenotype.graph_level
        label = data["experiment"].phenotype.label
        label_statistic = data["experiment"].phenotype.label_statistic
        fitness = data["experiment"].phenotype.fitness
        fitness_std = data["experiment"].phenotype.fitness_std

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="fitness phenotype",
            properties={
                "graph_level": graph_level,
                "label": label,
                "label_statistic": label_statistic,
                "fitness": fitness,
                "fitness_std": fitness_std,
                "serialized_data": json.dumps(
                    data["experiment"].phenotype.model_dump()
                ),
            },
        )

    @data_chunker
    def _gene_interaction_phenotype_node(self, data: dict) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label = phenotype.label
        label_statistic = phenotype.label_statistic
        interaction = phenotype.interaction
        p_value = phenotype.p_value

        properties = {
            "graph_level": graph_level,
            "label": label,
            "label_statistic": label_statistic,
            "interaction": interaction,
            "p_value": p_value,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="gene interaction phenotype",
            properties=properties,
        )

    def _get_fitness_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.phenotype_reference.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            graph_level = data.reference.phenotype_reference.graph_level
            label = data.reference.phenotype_reference.label
            label_statistic = data.reference.phenotype_reference.label_statistic
            fitness = data.reference.phenotype_reference.fitness
            fitness_std = data.reference.phenotype_reference.fitness_std

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="fitness phenotype",
                node_label="fitness phenotype",
                properties={
                    "graph_level": graph_level,
                    "label": label,
                    "label_statistic": label_statistic,
                    "fitness": fitness,
                    "fitness_std": fitness_std,
                    "serialized_data": json.dumps(
                        data.reference.phenotype_reference.model_dump()
                    ),
                },
            )
            nodes.append(node)
        return nodes

    def _get_gene_interaction_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.phenotype_reference.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            graph_level = data.reference.phenotype_reference.graph_level
            label = data.reference.phenotype_reference.label
            label_statistic = data.reference.phenotype_reference.label_statistic
            interaction = data.reference.phenotype_reference.interaction
            p_value = data.reference.phenotype_reference.p_value

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="gene interaction phenotype",
                node_label="gene interaction phenotype",
                properties={
                    "graph_level": graph_level,
                    "label": label,
                    "label_statistic": label_statistic,
                    "interaction": interaction,
                    "p_value": p_value,
                    "serialized_data": json.dumps(
                        data.reference.phenotype_reference.model_dump()
                    ),
                },
            )
            nodes.append(node)
        return nodes

    def _get_dataset_nodes(self) -> list[BioCypherNode]:
        nodes = [
            BioCypherNode(
                node_id=self.dataset.__class__.__name__,
                preferred_id=self.dataset.__class__.__name__,
                node_label="dataset",
            )
        ]
        return nodes

    # edges
    def _get_experiment_reference_to_dataset_edges(self) -> list[BioCypherEdge]:
        edges = []
        for data in self.dataset.experiment_reference_index:
            reference_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=reference_id,
                target_id=self.dataset.__class__.__name__,
                relationship_label="experiment reference member of",
            )
            edges.append(edge)
        return edges

    @data_chunker
    def _experiment_to_dataset_edge(self, data: dict) -> list[BioCypherEdge]:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=experiment_id,
            target_id=self.dataset.__class__.__name__,
            relationship_label="experiment member of",
        )
        return edge

    @data_chunker
    def _experiment_reference_to_experiment_edge(self, data: dict) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        experiment_ref_id = hashlib.sha256(
            json.dumps(data["reference"].model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=experiment_ref_id,
            target_id=experiment_id,
            relationship_label="experiment reference of",
        )
        return edge

    @data_chunker
    def _genotype_to_experiment_edge(self, data: dict) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        genotype = data["experiment"].genotype
        genotype_id = hashlib.sha256(
            json.dumps(genotype.model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=genotype_id,
            target_id=experiment_id,
            relationship_label="genotype member of",
        )
        return edge

    @data_chunker
    def _perturbation_to_genotype_edges(self, data: dict) -> list[BioCypherEdge]:
        edges = []
        genotype = data["experiment"].genotype
        for perturbation in genotype.perturbations:
            genotype_id = hashlib.sha256(
                json.dumps(genotype.model_dump()).encode("utf-8")
            ).hexdigest()
            perturbation_id = hashlib.sha256(
                json.dumps(perturbation.model_dump()).encode("utf-8")
            ).hexdigest()
            edges.append(
                BioCypherEdge(
                    source_id=perturbation_id,
                    target_id=genotype_id,
                    relationship_label="perturbation member of",
                )
            )
        return edges

    @data_chunker
    def _environment_to_experiment_edge(self, data: dict) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        environment_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=environment_id,
            target_id=experiment_id,
            relationship_label="environment member of",
        )
        return edge

    def _get_environment_to_experiment_reference_edges(self) -> list[BioCypherEdge]:
        edges = []
        seen_environment_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.sha256(
                json.dumps(data.reference.environment_reference.model_dump()).encode(
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

    @data_chunker
    def _phenotype_to_experiment_edge(self, data: dict) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        phenotype_id = hashlib.sha256(
            json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=phenotype_id,
            target_id=experiment_id,
            relationship_label="phenotype member of",
        )
        return edge

    @data_chunker
    def _media_to_environment_edge(self, data: dict) -> BioCypherEdge:
        environment_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
        ).hexdigest()
        media_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.media.model_dump()).encode(
                "utf-8"
            )
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=media_id,
            target_id=environment_id,
            relationship_label="media member of",
        )
        return edge

    @data_chunker
    def _temperature_to_environment_edge(self, data: dict) -> BioCypherEdge:
        environment_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
        ).hexdigest()
        temperature_id = hashlib.sha256(
            json.dumps(data["experiment"].environment.temperature.model_dump()).encode(
                "utf-8"
            )
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=temperature_id,
            target_id=environment_id,
            relationship_label="temperature member of",
        )
        return edge

    def _get_genome_to_experiment_reference_edges(self) -> list[BioCypherEdge]:
        edges = []
        seen_genome_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            genome_id = hashlib.sha256(
                json.dumps(data.reference.genome_reference.model_dump()).encode("utf-8")
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
    pass
