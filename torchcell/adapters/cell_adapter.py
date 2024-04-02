# torchcell/adapters/cell_adapter
# [[torchcell.adapters.cell_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/cell_adapter
# Test file: tests/torchcell/adapters/test_cell_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher._create import BioCypherEdge, BioCypherNode
from typing import Set, Optional
import torch
from torchcell.loader import CpuExperimentLoaderMultiprocessing
from concurrent.futures import ProcessPoolExecutor
from torch_geometric.data import Dataset
from typing import Callable
from functools import wraps
import logging
import wandb
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CellAdapter:
    def __init__(
        self,
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
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
        self.event = 0

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

    # @abstractmethod
    # def get_nodes(self) -> Generator[BioCypherNode, None, None]:
    #     pass

    # @abstractmethod
    # def get_edges(self) -> Generator[BioCypherEdge, None, None]:
    #     pass

    # Nodes put here for now since we are trying to generalize as much as possible to reduce code here.

    def get_nodes(self):
        print("Running: self._get_experiment_reference_nodes()")
        yield from self._get_experiment_reference_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_genome_nodes()")
        yield from self._get_genome_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._experiment_node)")
        yield from self.get_data_by_type(self._experiment_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._genotype_node)")
        yield from self.get_data_by_type(self._genotype_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._perturbation_node)")
        yield from self.get_data_by_type(self._perturbation_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._environment_node)")
        yield from self.get_data_by_type(self._environment_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_reference_environment_nodes()")
        yield from self._get_reference_environment_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._media_node)")
        yield from self.get_data_by_type(self._media_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_reference_media_nodes()")
        yield from self._get_reference_media_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._temperature_node)")
        yield from self.get_data_by_type(self._temperature_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_reference_temperature_nodes()")
        yield from self._get_reference_temperature_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._phenotype_node)")
        yield from self.get_data_by_type(self._phenotype_node)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_reference_phenotype_nodes()")
        yield from self._get_reference_phenotype_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_dataset_nodes()")
        yield from self.get_dataset_nodes()
        self.event += 1
        wandb.log({"event": self.event})
        print("Finished: get_nodes")

    def get_edges(self):
        print("Running: self.get_reference_dataset_edges()")
        yield from self.get_reference_dataset_edges()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._experiment_dataset_edge)")
        yield from self.get_data_by_type(self._experiment_dataset_edge)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._get_reference_experiment_edges)")
        yield from self.get_data_by_type(self._reference_experiment_edge)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._genotype_experiment_edge)")
        yield from self.get_data_by_type(self._genotype_experiment_edge)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._perturbation_genotype_edges)")
        yield from self.get_data_by_type(self._perturbation_genotype_edges)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._environment_experiment_edges)")
        yield from self.get_data_by_type(self._environment_experiment_edges)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_environment_experiment_ref_edges()")
        yield from self._get_environment_experiment_ref_edges()
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._phenotype_experiment_edges)")
        yield from self.get_data_by_type(self._phenotype_experiment_edges)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._media_environment_edge)")
        yield from self.get_data_by_type(self._media_environment_edge)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self.get_data_by_type(self._temperature_environment_edge)")
        yield from self.get_data_by_type(self._temperature_environment_edge)
        self.event += 1
        wandb.log({"event": self.event})
        print("Running: self._get_genome_edges()")
        yield from self._get_genome_edges()
        self.event += 1
        wandb.log({"event": self.event})
        print("Finished: get_edges")

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
                properties={"serialized_data": json.dumps(data.reference.model_dump())},
            )
            nodes.append(node)
        return nodes

    def _get_genome_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            genome_id = hashlib.sha256(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()
            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                node = BioCypherNode(
                    node_id=genome_id,
                    preferred_id="genome",
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

    @data_chunker
    def _experiment_node(self, data: dict) -> BioCypherNode:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=experiment_id,
            preferred_id="experiment",
            node_label="experiment",
            properties={"serialized_data": json.dumps(data["experiment"].model_dump())},
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

    def _get_reference_environment_nodes(self) -> list[BioCypherNode]:
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
                    preferred_id="environment",
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

    def _get_reference_media_nodes(self) -> list[BioCypherNode]:
        seen_node_ids = set()
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
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
                    preferred_id="media",
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

    def _get_reference_temperature_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            temperature_id = hashlib.sha256(
                json.dumps(
                    data.reference.reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()
            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)
                node = BioCypherNode(
                    node_id=temperature_id,
                    preferred_id="temperature",
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

    @data_chunker
    def _phenotype_node(self, data: dict) -> BioCypherNode:
        phenotype_id = hashlib.sha256(
            json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = data["experiment"].phenotype.graph_level
        label = data["experiment"].phenotype.label
        label_error = data["experiment"].phenotype.label_error
        fitness = data["experiment"].phenotype.fitness
        fitness_std = data["experiment"].phenotype.fitness_std

        return BioCypherNode(
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

    def _get_reference_phenotype_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            graph_level = data.reference.reference_phenotype.graph_level
            label = data.reference.reference_phenotype.label
            label_error = data.reference.reference_phenotype.label_error
            fitness = data.reference.reference_phenotype.fitness
            fitness_std = data.reference.reference_phenotype.fitness_std

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id=f"phenotype",
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

    def get_dataset_nodes(self) -> list[BioCypherNode]:
        nodes = [
            BioCypherNode(
                node_id=self.dataset.__class__.__name__,
                preferred_id=self.dataset.__class__.__name__,
                node_label="dataset",
            )
        ]
        return nodes

    # edges
    def get_reference_dataset_edges(self) -> list[BioCypherEdge]:
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
    def _experiment_dataset_edge(self, data: dict) -> list[BioCypherEdge]:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=experiment_id,
            target_id=self.dataset.__class__.__name__,
            relationship_label="experiment member of",
        )
        return edge

    # BUG This doesn't use multiprocessing and is therefore extremely slow.
    # def _get_reference_experiment_edges(self) -> list[BioCypherEdge]:
    #     edges = []
    #     for data in tqdm(self.dataset.experiment_reference_index):
    #         dataset_subset = self.dataset[torch.tensor(data.index)]
    #         experiment_ref_id = hashlib.sha256(
    #             json.dumps(data.reference.model_dump()).encode("utf-8")
    #         ).hexdigest()
    #         for i, data in enumerate(dataset_subset):
    #             data = self.dataset.transform_item(data)
    #             experiment_id = hashlib.sha256(
    #                 json.dumps(data["experiment"].model_dump()).encode("utf-8")
    #             ).hexdigest()
    #             edge = BioCypherEdge(
    #                 source_id=experiment_ref_id,
    #                 target_id=experiment_id,
    #                 relationship_label="experiment reference of",
    #             )
    #             edges.append(edge)
    #     return edges

    @data_chunker
    def _reference_experiment_edge(self, data: dict) -> BioCypherEdge:
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
    def _genotype_experiment_edge(self, data: dict) -> BioCypherEdge:
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
    def _perturbation_genotype_edges(self, data: dict) -> list[BioCypherEdge]:
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
    def _environment_experiment_edges(self, data: dict) -> BioCypherEdge:
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

    def _get_environment_experiment_ref_edges(self) -> list[BioCypherEdge]:
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

    @data_chunker
    def _phenotype_experiment_edges(self, data: dict) -> BioCypherEdge:
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
    def _media_environment_edge(self, data: dict) -> BioCypherEdge:
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
    def _temperature_environment_edge(self, data: dict) -> BioCypherEdge:
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

    def _get_genome_edges(self) -> list[BioCypherEdge]:
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
    pass
