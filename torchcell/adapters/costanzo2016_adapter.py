# torchcell/adapters/costanzo2016_adapter.py
# [[torchcell.adapters.costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/costanzo2016_adapter.py
# Test file: tests/torchcell/adapters/test_costanzo2016_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Generator, Set
import torch
from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
)
from torchcell.datamodels import Genotype
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.loader import DataLoader
from torchcell.dataset import Dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from typing import Callable
from functools import wraps
from abc import abstractmethod
from omegaconf import OmegaConf
from typing import Any
from torchcell.adapters.cell_adapter import CellAdapter

# logging
# Get the biocypher logger
logger = get_logger("biocypher")
logger.setLevel(logging.ERROR)


class SmfCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: SmfCostanzo2016Dataset,
        compute_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, compute_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmfCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmfCostanzo2016Dataset,
        compute_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, compute_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


# class DmfCostanzo2016Adapter:
#     def __init__(
#         self,
#         dataset: DmfCostanzo2016Dataset,
#         compute_workers: int = 1,
#         chunk_size: int = 1000,
#     ):
#         self.dataset = dataset
#         self.compute_workers = compute_workers
#         self.chunk_size = chunk_size

#     def get_nodes(self) -> Generator[BioCypherNode, None, None]:
#         for node in self._get_experiment_nodes():
#             yield node
#         for node in self._get__genotype_nodes():
#             yield node
#         for node in self._get__perturbation_nodes():
#             yield node
#         for node in self._get_phenotype_nodes():
#             yield node

#         # methods = [
#         #     self._get_experiment_reference_nodes,
#         #     self._get_genome_nodes,
#         #     self._get_dataset_nodes,
#         #     self._get_environment_nodes,
#         #     self._get_media_nodes,
#         #     self._get_temperature_nodes,
#         # ]

#         # with ProcessPoolExecutor(max_workers=self.compute_workers) as executor:
#         #     futures = [executor.submit(method) for method in methods]
#         #     for future in as_completed(futures):
#         #         try:
#         #             node_generator = future.result()
#         #             for node in node_generator:
#         #                 yield node
#         #         except Exception as exc:
#         #             logger.error(
#         #                 f"Node generation method generated an exception: {exc}"
#         #             )

#     def _get_experiment_reference_nodes(self) -> list[BioCypherNode]:
#         nodes = []
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             experiment_ref_id = hashlib.sha256(
#                 json.dumps(data.reference.model_dump()).encode("utf-8")
#             ).hexdigest()
#             node = BioCypherNode(
#                 node_id=experiment_ref_id,
#                 preferred_id=f"DmfCostanzo2016_reference_{i}",
#                 node_label="experiment reference",
#                 properties={
#                     "dataset_index": i,
#                     "serialized_data": json.dumps(data.reference.model_dump()),
#                 },
#             )
#             nodes.append(node)
#         return nodes

#     def _get_genome_nodes(self) -> list[BioCypherNode]:
#         nodes = []
#         seen_node_ids: Set[str] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             genome_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
#             ).hexdigest()
#             if genome_id not in seen_node_ids:
#                 seen_node_ids.add(genome_id)
#                 node = BioCypherNode(
#                     node_id=genome_id,
#                     preferred_id=f"reference_genome_{i}",
#                     node_label="genome",
#                     properties={
#                         "species": data.reference.reference_genome.species,
#                         "strain": data.reference.reference_genome.strain,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_genome.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     @staticmethod
#     def _chunk_experiment_nodes(data_chunk: dict) -> list[BioCypherNode]:
#         nodes = []
#         for i, data in enumerate(data_chunk):
#             experiment_id = hashlib.sha256(
#                 json.dumps(data["experiment"].model_dump()).encode("utf-8")
#             ).hexdigest()
#             node = BioCypherNode(
#                 node_id=experiment_id,
#                 preferred_id=f"DmfCostanzo2016_{i}",
#                 node_label="experiment",
#                 properties={
#                     "dataset_index": i,
#                     "serialized_data": json.dumps(data["experiment"].model_dump()),
#                 },
#             )
#             nodes.append(node)
#         return nodes

#     def _get_experiment_nodes(self) -> list[BioCypherNode]:
#         data_chunks = [
#             self.dataset[i : i + self.chunk_size]
#             for i in range(0, len(self.dataset), self.chunk_size)
#         ]
#         nodes = []
#         with ProcessPoolExecutor(max_workers=self.compute_workers) as executor:
#             futures = [
#                 executor.submit(self._chunk_experiment_nodes, chunk)
#                 for chunk in data_chunks
#             ]
#             for future in futures:
#                 nodes.extend(future.result())
#         return nodes

#     def _get__genotype_nodes(self) -> list[BioCypherNode]:
#         nodes = []
#         seen_node_ids = set()
#         for i, data in tqdm(enumerate(self.dataset)):
#             genotype = data["experiment"].genotype
#             genotype_id = hashlib.sha256(
#                 json.dumps(genotype.model_dump()).encode("utf-8")
#             ).hexdigest()
#             if genotype_id not in seen_node_ids:
#                 seen_node_ids.add(genotype_id)
#                 node = BioCypherNode(
#                     node_id=genotype_id,
#                     preferred_id=f"genotype_{i}",
#                     node_label="genotype",
#                     properties={
#                         "systematic_gene_names": genotype.systematic_gene_names,
#                         "perturbed_gene_names": genotype.perturbed_gene_names,
#                         "perturbation_types": genotype.perturbation_types,
#                         "serialized_data": json.dumps(
#                             data["experiment"].genotype.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get__perturbation_nodes(self) -> list[BioCypherNode]:
#         nodes = []
#         seen_node_ids = set()
#         for data in tqdm(self.dataset):
#             perturbations = data["experiment"].genotype.perturbations
#             for perturbation in perturbations:
#                 perturbation_id = hashlib.sha256(
#                     json.dumps(perturbation.model_dump()).encode("utf-8")
#                 ).hexdigest()
#                 if perturbation_id not in seen_node_ids:
#                     seen_node_ids.add(perturbation_id)
#                     node = BioCypherNode(
#                         node_id=perturbation_id,
#                         preferred_id=perturbation.perturbation_type,
#                         node_label="perturbation",
#                         properties={
#                             "systematic_gene_name": perturbation.systematic_gene_name,
#                             "perturbed_gene_name": perturbation.perturbed_gene_name,
#                             "perturbation_type": perturbation.perturbation_type,
#                             "description": perturbation.description,
#                             "strain_id": perturbation.strain_id,
#                             "serialized_data": json.dumps(perturbation.model_dump()),
#                         },
#                     )
#                     nodes.append(node)
#         return nodes

#     def _get_environment_nodes(self) -> list[BioCypherNode]:
#         # HACK - we know we can loop ref for this node type
#         nodes = []
#         seen_node_ids = set()
#         for data in tqdm(self.dataset.experiment_reference_index):
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             if environment_id not in seen_node_ids:
#                 seen_node_ids.add(environment_id)
#                 media = json.dumps(
#                     data.reference.reference_environment.media.model_dump()
#                 )
#                 node = BioCypherNode(
#                     node_id=environment_id,
#                     preferred_id=f"environment_ref_{environment_id}",
#                     node_label="environment",
#                     properties={
#                         "temperature": data.reference.reference_environment.temperature.value,
#                         "media": media,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_environment.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get_media_nodes(self) -> list[BioCypherNode]:
#         # HACK - we know we can loop ref for this node type
#         seen_node_ids = set()
#         nodes = []
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             media_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.media.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             if media_id not in seen_node_ids:
#                 seen_node_ids.add(media_id)
#                 name = data.reference.reference_environment.media.name
#                 state = data.reference.reference_environment.media.state
#                 node = BioCypherNode(
#                     node_id=media_id,
#                     preferred_id=f"media_{media_id}",
#                     node_label="media",
#                     properties={
#                         "name": name,
#                         "state": state,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_environment.media.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get_temperature_nodes(self) -> list[BioCypherNode]:
#         # HACK - we know we can loop ref for this node type
#         seen_node_ids = set()
#         nodes = []
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             temperature_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.temperature.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             if temperature_id not in seen_node_ids:
#                 seen_node_ids.add(temperature_id)
#                 node = BioCypherNode(
#                     node_id=temperature_id,
#                     preferred_id=f"temperature_{temperature_id}",
#                     node_label="temperature",
#                     properties={
#                         "value": data.reference.reference_environment.temperature.value,
#                         "unit": data.reference.reference_environment.temperature.unit,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_environment.temperature.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get_phenotype_nodes(self) -> list[BioCypherNode]:
#         nodes = []
#         seen_node_ids = set()
#         for i, data in tqdm(enumerate(self.dataset)):
#             phenotype_id = hashlib.sha256(
#                 json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
#             ).hexdigest()
#             if phenotype_id not in seen_node_ids:
#                 seen_node_ids.add(phenotype_id)
#                 graph_level = data["experiment"].phenotype.graph_level
#                 label = data["experiment"].phenotype.label
#                 label_error = data["experiment"].phenotype.label_error
#                 fitness = data["experiment"].phenotype.fitness
#                 fitness_std = data["experiment"].phenotype.fitness_std

#                 node = BioCypherNode(
#                     node_id=phenotype_id,
#                     preferred_id=f"phenotype_{phenotype_id}",
#                     node_label="phenotype",
#                     properties={
#                         "graph_level": graph_level,
#                         "label": label,
#                         "label_error": label_error,
#                         "fitness": fitness,
#                         "fitness_std": fitness_std,
#                         "serialized_data": json.dumps(
#                             data["experiment"].phenotype.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)

#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             phenotype_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_phenotype.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             if phenotype_id not in seen_node_ids:
#                 seen_node_ids.add(phenotype_id)
#                 graph_level = data.reference.reference_phenotype.graph_level
#                 label = data.reference.reference_phenotype.label
#                 label_error = data.reference.reference_phenotype.label_error
#                 fitness = data.reference.reference_phenotype.fitness
#                 fitness_std = data.reference.reference_phenotype.fitness_std

#                 node = BioCypherNode(
#                     node_id=phenotype_id,
#                     preferred_id=f"phenotype_{phenotype_id}",
#                     node_label="phenotype",
#                     properties={
#                         "graph_level": graph_level,
#                         "label": label,
#                         "label_error": label_error,
#                         "fitness": fitness,
#                         "fitness_std": fitness_std,
#                         "serialized_data": json.dumps(
#                             data.reference.reference_phenotype.model_dump()
#                         ),
#                     },
#                 )
#                 nodes.append(node)
#         return nodes

#     def _get_dataset_nodes(self) -> list[BioCypherNode]:
#         nodes = [
#             BioCypherNode(
#                 node_id="DmfCostanzo2016",
#                 preferred_id="DmfCostanzo2016",
#                 node_label="dataset",
#             )
#         ]
#         return nodes

#     def get_edges(self) -> Generator[BioCypherEdge, None, None]:
#         pass
#         # for edge in self._get_experiment_dataset_edges():
#         #     yield edge
#         # for edge in self._get_dataset_experiment_ref_edges():
#         #     yield edge
#         # for edge in self._get_experiment_ref_experiment_edges():
#         #     yield edge
#         # for edge in self._get_genotype_experiment_edges():
#         #     yield edge
#         # for edge in self._get_environment_experiment_edges():
#         #     yield edge
#         # for edge in self._get_environment_experiment_ref_edges():
#         #     yield edge
#         # for edge in self._get_phenotype_experiment_edges():
#         #     yield edge
#         # for edge in self._get_phenotype_experiment_ref_edges():
#         #     yield edge
#         # for edge in self._get_perturbation_genotype_edges():
#         #     yield edge
#         # for edge in self._get_media_environment_edges():
#         #     yield edge
#         # for edge in self._get_temperature_environment_edges():
#         #     yield edge
#         # for edge in self._get_genome_edges():
#         #     yield edge

#     # FLAG
#     @staticmethod
#     def _chunk_experiment_dataset_edges(data_chunk) -> list[BioCypherEdge]:
#         # Process a chunk of data
#         edges = []
#         for data in data_chunk:
#             experiment_id = hashlib.sha256(
#                 json.dumps(data["experiment"].model_dump()).encode("utf-8")
#             ).hexdigest()
#             edge = BioCypherEdge(
#                 source_id=experiment_id,
#                 target_id="DmfCostanzo2016",
#                 relationship_label="experiment member of",
#             )
#             edges.append(edge)
#         return edges

#     # FLAG
#     def _get_experiment_dataset_edges(self) -> list[BioCypherEdge]:
#         data_chunks = [
#             self.dataset[i : i + self.chunk_size]
#             for i in range(0, len(self.dataset), self.chunk_size)
#         ]
#         edges = []
#         with ProcessPoolExecutor(max_workers=self.compute_workers) as executor:
#             futures = [
#                 executor.submit(self._chunk_experiment_dataset_edges, chunk)
#                 for chunk in data_chunks
#             ]
#             for future in futures:
#                 edges.extend(future.result())
#         return edges

#     # FLAG
#     def _get_dataset_experiment_ref_edges(self) -> list[BioCypherEdge]:
#         edges = []
#         for data in tqdm(self.dataset.experiment_reference_index):
#             experiment_ref_id = hashlib.sha256(
#                 json.dumps(data.reference.model_dump()).encode("utf-8")
#             ).hexdigest()
#             edge = BioCypherEdge(
#                 source_id=experiment_ref_id,
#                 target_id="DmfCostanzo2016",
#                 relationship_label="experiment reference member of",
#             )
#             edges.append(edge)
#         return edges

#     # FLAG
#     def _get_experiment_ref_experiment_edges(self) -> list[BioCypherEdge]:
#         edges = []
#         for data in tqdm(self.dataset.experiment_reference_index):
#             dataset_subset = self.dataset[torch.tensor(data.index)]
#             experiment_ref_id = hashlib.sha256(
#                 json.dumps(data.reference.model_dump()).encode("utf-8")
#             ).hexdigest()
#             for i, data in enumerate(dataset_subset):
#                 experiment_id = hashlib.sha256(
#                     json.dumps(data["experiment"].model_dump()).encode("utf-8")
#                 ).hexdigest()
#                 edge = BioCypherEdge(
#                     source_id=experiment_ref_id,
#                     target_id=experiment_id,
#                     relationship_label="experiment reference of",
#                 )
#                 edges.append(edge)
#         return edges

#     # FLAG
#     @staticmethod
#     def _chunk_genotype_experiment_edges(data_chunk: dict) -> list[BioCypherEdge]:
#         edges = []
#         for data in data_chunk:
#             experiment_id = hashlib.sha256(
#                 json.dumps(data["experiment"].model_dump()).encode("utf-8")
#             ).hexdigest()
#             genotype = data["experiment"].genotype
#             genotype_id = hashlib.sha256(
#                 json.dumps(genotype.model_dump()).encode("utf-8")
#             ).hexdigest()
#             edge = BioCypherEdge(
#                 source_id=genotype_id,
#                 target_id=experiment_id,
#                 relationship_label="genotype member of",
#             )
#             edges.append(edge)
#         return edges

#     # FLAG
#     def _get_genotype_experiment_edges(self) -> list[BioCypherEdge]:
#         data_chunks = [
#             self.dataset[i : i + self.chunk_size]
#             for i in range(0, len(self.dataset), self.chunk_size)
#         ]
#         edges = []
#         with ProcessPoolExecutor(max_workers=self.compute_workers) as executor:
#             futures = [
#                 executor.submit(self._chunk_genotype_experiment_edges, chunk)
#                 for chunk in data_chunks
#             ]
#             for future in futures:
#                 edges.extend(future.result())
#         return edges

#     # FLAG
#     def _get_perturbation_genotype_edges(self) -> list[BioCypherEdge]:
#         edges = []
#         seen_edges = set()
#         for data in tqdm(self.dataset):
#             genotype = data["experiment"].genotype
#             for perturbation in genotype.perturbations:
#                 genotype_id = hashlib.sha256(
#                     json.dumps(genotype.model_dump()).encode("utf-8")
#                 ).hexdigest()
#                 perturbation_id = hashlib.sha256(
#                     json.dumps(perturbation.model_dump()).encode("utf-8")
#                 ).hexdigest()
#                 edge_tuple = (perturbation_id, genotype_id, "perturbation member of")
#                 if edge_tuple not in seen_edges:
#                     seen_edges.add(edge_tuple)
#                     edge = BioCypherEdge(
#                         source_id=perturbation_id,
#                         target_id=genotype_id,
#                         relationship_label="perturbation member of",
#                     )
#                     edges.append(edge)

#         return edges

#     # FLAG
#     @staticmethod
#     def _chunk_environment_experiment_edges(data_chunk) -> list[BioCypherEdge]:
#         edges = []
#         seen_environment_experiment_pairs = set()
#         for data in data_chunk:
#             experiment_id = hashlib.sha256(
#                 json.dumps(data["experiment"].model_dump()).encode("utf-8")
#             ).hexdigest()
#             environment_id = hashlib.sha256(
#                 json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
#             ).hexdigest()
#             env_experiment_pair = (environment_id, experiment_id)
#             if env_experiment_pair not in seen_environment_experiment_pairs:
#                 seen_environment_experiment_pairs.add(env_experiment_pair)
#                 edge = BioCypherEdge(
#                     source_id=environment_id,
#                     target_id=experiment_id,
#                     relationship_label="environment member of",
#                 )
#                 edges.append(edge)
#         return edges

#     # FLAG
#     def _get_environment_experiment_edges(self) -> list[BioCypherEdge]:
#         data_chunks = [
#             self.dataset[i : i + self.chunk_size]
#             for i in range(0, len(self.dataset), self.chunk_size)
#         ]
#         edges = []
#         with ProcessPoolExecutor(max_workers=self.compute_workers) as executor:
#             futures = [
#                 executor.submit(self._chunk_environment_experiment_edges, chunk)
#                 for chunk in data_chunks
#             ]
#             for future in futures:
#                 edges.extend(future.result())
#         return edges

#     # FLAG
#     def _get_environment_experiment_ref_edges(self) -> list[BioCypherEdge]:
#         edges = []
#         seen_environment_experiment_ref_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             experiment_ref_id = hashlib.sha256(
#                 json.dumps(data.reference.model_dump()).encode("utf-8")
#             ).hexdigest()
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             env_experiment_ref_pair = (environment_id, experiment_ref_id)
#             if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
#                 seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

#                 edge = BioCypherEdge(
#                     source_id=environment_id,
#                     target_id=experiment_ref_id,
#                     relationship_label="environment member of",
#                 )
#                 edges.append(edge)
#         return edges

#     # FLAG
#     @staticmethod
#     def _chunk_phenotype_experiment_edges(data_chunk) -> list[BioCypherEdge]:
#         edges = []
#         seen_phenotype_experiment_pairs = set()
#         for data in data_chunk:
#             experiment_id = hashlib.sha256(
#                 json.dumps(data["experiment"].model_dump()).encode("utf-8")
#             ).hexdigest()
#             phenotype_id = hashlib.sha256(
#                 json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
#             ).hexdigest()
#             phenotype_experiment_pair = (phenotype_id, experiment_id)
#             if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
#                 seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)
#                 edge = BioCypherEdge(
#                     source_id=phenotype_id,
#                     target_id=experiment_id,
#                     relationship_label="phenotype member of",
#                 )
#                 edges.append(edge)
#         return edges

#     # FLAG
#     def _get_phenotype_experiment_edges(self) -> list[BioCypherEdge]:
#         data_chunks = [
#             self.dataset[i : i + self.chunk_size]
#             for i in range(0, len(self.dataset), self.chunk_size)
#         ]
#         edges = []
#         with ProcessPoolExecutor(max_workers=self.compute_workers) as executor:
#             futures = [
#                 executor.submit(self._chunk_phenotype_experiment_edges, chunk)
#                 for chunk in data_chunks
#             ]
#             for future in futures:
#                 edges.extend(future.result())
#         return edges

#     # FLAG
#     def _get_phenotype_experiment_ref_edges(self) -> list[BioCypherEdge]:
#         edges = []
#         seen_phenotype_experiment_ref_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             experiment_ref_id = hashlib.sha256(
#                 json.dumps(data.reference.model_dump()).encode("utf-8")
#             ).hexdigest()
#             phenotype_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_phenotype.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
#             if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
#                 seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

#                 edge = BioCypherEdge(
#                     source_id=phenotype_id,
#                     target_id=experiment_ref_id,
#                     relationship_label="phenotype member of",
#                 )
#                 edges.append(edge)
#         return edges

#     def _get_media_environment_edges(self) -> list[BioCypherEdge]:
#         # HACK Optimized by using reference
#         # We know reference contains all media and envs
#         edges = []
#         seen_media_environment_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             media_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.media.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             media_environment_pair = (media_id, environment_id)
#             if media_environment_pair not in seen_media_environment_pairs:
#                 seen_media_environment_pairs.add(media_environment_pair)
#                 edge = BioCypherEdge(
#                     source_id=media_id,
#                     target_id=environment_id,
#                     relationship_label="media member of",
#                 )
#                 edges.append(edge)
#         return edges

#     def _get_temperature_environment_edges(self) -> list[BioCypherEdge]:
#         # HACK Optimized by using reference
#         # We know reference contain all envs and temps
#         edges = []
#         seen_temperature_environment_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             environment_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_environment.model_dump()).encode(
#                     "utf-8"
#                 )
#             ).hexdigest()
#             temperature_id = hashlib.sha256(
#                 json.dumps(
#                     data.reference.reference_environment.temperature.model_dump()
#                 ).encode("utf-8")
#             ).hexdigest()
#             temperature_environment_pair = (temperature_id, environment_id)
#             if temperature_environment_pair not in seen_temperature_environment_pairs:
#                 seen_temperature_environment_pairs.add(temperature_environment_pair)
#                 edge = BioCypherEdge(
#                     source_id=temperature_id,
#                     target_id=environment_id,
#                     relationship_label="temperature member of",
#                 )
#                 edges.append(edge)
#         return edges

#     def _get_genome_edges(self) -> list[BioCypherEdge]:
#         edges = []
#         seen_genome_experiment_ref_pairs: Set[tuple] = set()
#         for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
#             experiment_ref_id = hashlib.sha256(
#                 json.dumps(data.reference.model_dump()).encode("utf-8")
#             ).hexdigest()
#             genome_id = hashlib.sha256(
#                 json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
#             ).hexdigest()
#             genome_experiment_ref_pair = (genome_id, experiment_ref_id)
#             if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
#                 seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

#                 edge = BioCypherEdge(
#                     source_id=genome_id,
#                     target_id=experiment_ref_id,
#                     relationship_label="genome member of",
#                 )
#                 edges.append(edge)
#         return edges


if __name__ == "__main__":
    import os.path as osp
    from dotenv import load_dotenv
    import math
    from datetime import datetime
    import os

    ##
    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")

    ## SMF
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = SmfCostanzo2016Dataset(
    #     osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    # )
    # num_workers = mp.cpu_count()
    # io_workers = math.ceil(0.2 * num_workers)
    # compute_workers = num_workers - io_workers
    # adapter = SmfCostanzo2016Adapter(
    #     dataset=dataset,
    #     compute_workers=6,
    #     io_workers=4,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    ## DMF
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    # dataset = DmfCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016")
    # )
    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_subset_n_1e3"),
        subset_n=int(1e3),
    )
    adapter = DmfCostanzo2016Adapter(
        dataset=dataset,
        compute_workers=1,
        io_workers=1,
        chunk_size=int(2 * 1e2),
        loader_batch_size=int(1e2),
    )
    bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()
