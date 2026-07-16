"""BioCypher adapter that turns a cell dataset into graph nodes and edges."""

# torchcell/adapters/cell_adapter
# [[torchcell.adapters.cell_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/cell_adapter
# Test file: tests/torchcell/adapters/test_cell_adapter.py

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import wraps
from typing import Any, cast

import wandb
from biocypher._create import BioCypherEdge, BioCypherNode
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from tqdm import tqdm

from torchcell.loader import CpuExperimentLoaderMultiprocessing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CellAdapter:
    """Convert a cell experiment dataset into BioCypher nodes and edges for import."""

    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Store config, dataset, and worker/chunk sizes for graph generation.

        Args:
            config: Hydra config holding the adapter's node and edge methods.
            dataset: Dataset whose items are converted to nodes and edges.
            process_workers: Number of processes for parallel chunk processing.
            io_workers: Number of workers used by the experiment data loader.
            chunk_size: Number of dataset items processed per chunk.
            loader_batch_size: Batch size used within each chunk; must not
                exceed ``chunk_size``.
        """
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
        self.log_method_table()
        wandb.log(
            {
                "current_adapter_dataset_name": self.dataset.name,
                "current_adapter_dataset_start_time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )

        # Supported methods
        self.node_methods = [
            ("experiment reference", self._get_experiment_reference_nodes),
            ("genome", self._get_genome_nodes),
            ("experiment (chunked)", self._experiment_node),
            ("genotype (chunked)", self._genotype_node),
            ("perturbation (chunked)", self._perturbation_node),
            ("environment (chunked)", self._environment_node),
            ("environment reference", self._get_environment_reference_nodes),
            ("media (chunked)", self._media_node),
            ("media reference", self._get_media_reference_nodes),
            ("temperature (chunked)", self._temperature_node),
            ("temperature reference", self._get_temperature_reference_nodes),
            ("fitness phenotype (chunked)", self._fitness_phenotype_node),
            (
                "gene interaction phenotype (chunked)",
                self._gene_interaction_phenotype_node,
            ),
            (
                "gene essentiality phenotype (chunked)",
                self._gene_essentiality_phenotype_node,
            ),
            (
                "synthetic lethality phenotype (chunked)",
                self._synthetic_lethality_phenotype_node,
            ),
            (
                "synthetic rescue phenotype (chunked)",
                self._synthetic_rescue_phenotype_node,
            ),
            ("calmorph phenotype (chunked)", self._calmorph_phenotype_node),
            (
                "microarray expression phenotype (chunked)",
                self._microarray_expression_phenotype_node,
            ),
            (
                "rnaseq expression phenotype (chunked)",
                self._rnaseq_expression_phenotype_node,
            ),
            ("visual score phenotype (chunked)", self._visual_score_phenotype_node),
            ("metabolite phenotype (chunked)", self._metabolite_phenotype_node),
            (
                "protein abundance phenotype (chunked)",
                self._protein_abundance_phenotype_node,
            ),
            (
                "fitness phenotype reference",
                self._get_fitness_phenotype_reference_nodes,
            ),
            (
                "gene interaction phenotype reference",
                self._get_gene_interaction_phenotype_reference_nodes,
            ),
            (
                "gene essentiality phenotype reference",
                self._get_gene_essentiality_phenotype_reference_nodes,
            ),
            (
                "synthetic lethality phenotype reference",
                self._get_synthetic_lethality_phenotype_reference_nodes,
            ),
            (
                "synthetic rescue phenotype reference",
                self._get_synthetic_rescue_phenotype_reference_nodes,
            ),
            (
                "calmorph phenotype reference",
                self._get_calmorph_phenotype_reference_nodes,
            ),
            (
                "microarray expression phenotype reference",
                self._get_microarray_expression_phenotype_reference_nodes,
            ),
            (
                "rnaseq expression phenotype reference",
                self._get_rnaseq_expression_phenotype_reference_nodes,
            ),
            (
                "visual score phenotype reference",
                self._get_visual_score_phenotype_reference_nodes,
            ),
            (
                "metabolite phenotype reference",
                self._get_metabolite_phenotype_reference_nodes,
            ),
            (
                "protein abundance phenotype reference",
                self._get_protein_abundance_phenotype_reference_nodes,
            ),
            ("dataset", self._get_dataset_nodes),
            ("publication (chunked)", self._publication_node),
        ]
        self.edge_methods = [
            (
                "experiment reference to dataset",
                self._get_experiment_reference_to_dataset_edges,
            ),
            ("experiment to dataset (chunked)", self._experiment_to_dataset_edge),
            (
                "experiment reference to experiment (chunked)",
                self._experiment_reference_to_experiment_edge,
            ),
            ("genotype to experiment (chunked)", self._genotype_to_experiment_edge),
            (
                "perturbation to genotype (chunked)",
                self._perturbation_to_genotype_edges,
            ),
            (
                "environment to experiment (chunked)",
                self._environment_to_experiment_edge,
            ),
            (
                "environment to experiment reference",
                self._get_environment_to_experiment_reference_edges,
            ),
            ("phenotype to experiment (chunked)", self._phenotype_to_experiment_edge),
            ("media to environment (chunked)", self._media_to_environment_edge),
            (
                "temperature to environment (chunked)",
                self._temperature_to_environment_edge,
            ),
            (
                "genome to experiment reference",
                self._get_genome_to_experiment_reference_edges,
            ),
            (
                "phenotype to experiment reference",
                self._get_phenotype_to_experiment_reference_edges,
            ),
            (
                "publication to experiment (chunked)",
                self._publication_to_experiment_edge,
            ),
        ]

    def log_method_table(self) -> None:
        """Log a wandb table summarizing the configured node and edge methods."""
        methods = []
        simulated_event_counter = 0
        for method in (
            self.config.cell_adapter.node_methods
            + self.config.cell_adapter.edge_methods
        ):
            simulated_event_counter += 1
            method_name = method["method_name"]
            data_type = (
                "node" if method in self.config.cell_adapter.node_methods else "edge"
            )

            if "(chunked)" in method_name:
                if "memory_reduction_factor" in method:
                    memory_reduction_factor = method["memory_reduction_factor"]
                else:
                    memory_reduction_factor = 1.0
            else:
                memory_reduction_factor = float("nan")

            method_info = [
                simulated_event_counter,
                method_name,
                data_type,
                memory_reduction_factor,
            ]
            methods.append(method_info)

        columns = ["event", "method", "data_type", "memory_reduction_factor"]
        method_table = wandb.Table(columns=columns, data=methods)
        wandb.log({f"{self.dataset.name}_method_table": method_table})

    def get_data_by_type(
        self,
        chunk_processing_func: Callable[..., Any],
        method_name: str,
        is_edge: bool = False,
    ) -> Iterator[Any]:
        """Yield processed data by chunking the dataset and mapping over processes."""
        memory_reduction_factor = self.get_memory_reduction_factor(method_name, is_edge)
        chunk_size = int(self.chunk_size * memory_reduction_factor)
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
            futures = [
                executor.submit(chunk_processing_func, chunk, method_name)
                for chunk in data_chunks
            ]
            for future in futures:
                yield from future.result()

    def data_chunker(  # type: ignore[misc]  # in-class decorator factory; first arg is the wrapped method, not self
        data_creation_logic: Callable[..., Any],
    ) -> Callable[..., list[Any]]:
        """Wrap a chunk handler so it loads, transforms, and collects each item."""

        @wraps(data_creation_logic)
        def decorator(
            self: "CellAdapter", data_chunk: Any, method_name: str
        ) -> list[Any]:
            memory_reduction_factor = self.get_memory_reduction_factor(method_name)
            loader_batch_size = int(self.loader_batch_size * memory_reduction_factor)
            # loader_batch_size = self.loader_batch_size
            data_loader = CpuExperimentLoaderMultiprocessing(
                data_chunk, batch_size=loader_batch_size, num_workers=self.io_workers
            )
            datas = []
            for batch in tqdm(data_loader):
                for data in batch:
                    transformed_data = data_chunk.transform_item(data)
                    data = data_creation_logic(self, transformed_data, method_name)
                    if isinstance(data, list):
                        datas.extend(data)
                    else:
                        datas.append(data)
            return datas

        return decorator

    def get_memory_reduction_factor(
        self, method_name: str, is_edge: bool = False
    ) -> float:
        """Return the configured memory reduction factor for a method (default 1.0)."""
        method_list = (
            self.config.cell_adapter.edge_methods
            if is_edge
            else self.config.cell_adapter.node_methods
        )
        for method in method_list:
            if method["method_name"] == method_name:
                return cast(float, method.get("memory_reduction_factor", 1.0))
        return 1.0

    def get_nodes(self) -> Iterator[BioCypherNode]:
        """Yield BioCypher nodes from every enabled node method in config order."""
        for method_name, method in self.node_methods:
            config_method_names = [
                i["method_name"] for i in self.config.cell_adapter.node_methods
            ]
            if method_name in config_method_names:
                log.info(f"Running: {method_name}")
                if method.__name__.startswith("_get_"):
                    yield from method()
                else:
                    yield from self.get_data_by_type(method, method_name)
                self.event += 1
                wandb.log({"event": self.event, "method": method_name, "type": "node"})

    def get_edges(self) -> Iterator[BioCypherEdge]:
        """Yield BioCypher edges from every enabled edge method in config order."""
        for method_name, method in self.edge_methods:
            config_method_names = [
                i["method_name"] for i in self.config.cell_adapter.edge_methods
            ]
            if method_name in config_method_names:
                log.info(f"Running: {method_name}")
                if method.__name__.startswith("_get_"):
                    yield from method()
                else:
                    yield from self.get_data_by_type(method, method_name, is_edge=True)
                self.event += 1
                wandb.log({"event": self.event, "method": method_name, "type": "edge"})

    @property
    def supported_node_methods(self) -> list[str]:
        """Return the names of all registered node methods."""
        return [method_name for method_name, _ in self.node_methods]

    @property
    def supported_edge_methods(self) -> list[str]:
        """Return the names of all registered edge methods."""
        return [method_name for method_name, _ in self.edge_methods]

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
        seen_node_ids: set[str] = set()
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
    def _experiment_node(self, data: dict[str, Any], method_name: str) -> BioCypherNode:
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
    def _genotype_node(self, data: dict[str, Any], method_name: str) -> BioCypherNode:
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
    def _perturbation_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
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
    def _environment_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
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
    def _media_node(self, data: dict[str, Any], method_name: str) -> BioCypherNode:
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
    def _temperature_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
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
        seen_node_ids: set[str] = set()
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
    def _fitness_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        fitness = phenotype.fitness
        fitness_std = phenotype.fitness_std

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "fitness": fitness,
            "fitness_std": fitness_std,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="fitness phenotype",
            properties=properties,
        )

    @data_chunker
    def _gene_interaction_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        gene_interaction = phenotype.gene_interaction
        gene_interaction_p_value = phenotype.gene_interaction_p_value

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "gene_interaction": gene_interaction,
            "gene_interaction_p_value": gene_interaction_p_value,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="gene interaction phenotype",
            properties=properties,
        )

    @data_chunker
    def _gene_essentiality_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        is_essential = phenotype.is_essential

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "is_essential": is_essential,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="gene essentiality phenotype",
            properties=properties,
        )

    @data_chunker
    def _synthetic_lethality_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        is_synthetic_lethal = phenotype.is_synthetic_lethal
        synthetic_lethality_statistic_score = (
            phenotype.synthetic_lethality_statistic_score
        )

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "is_synthetic_lethal": is_synthetic_lethal,
            "synthetic_lethality_statistic_score": synthetic_lethality_statistic_score,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="synthetic lethality phenotype",
            properties=properties,
        )

    @data_chunker
    def _synthetic_rescue_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        is_synthetic_rescue = phenotype.is_synthetic_rescue
        synthetic_rescue_statistic_score = phenotype.synthetic_rescue_statistic_score

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "is_synthetic_rescue": is_synthetic_rescue,
            "synthetic_rescue_statistic_score": synthetic_rescue_statistic_score,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="synthetic rescue phenotype",
            properties=properties,
        )

    def _get_fitness_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            fitness = phenotype.fitness
            fitness_std = phenotype.fitness_std

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "fitness": fitness,
                "fitness_std": fitness_std,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="fitness phenotype",
                node_label="fitness phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_gene_interaction_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            gene_interaction = phenotype.gene_interaction
            gene_interaction_p_value = phenotype.gene_interaction_p_value

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "gene_interaction": gene_interaction,
                "gene_interaction_p_value": gene_interaction_p_value,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="gene interaction phenotype",
                node_label="gene interaction phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_gene_essentiality_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            is_essential = phenotype.is_essential

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "is_essential": is_essential,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="gene essentiality phenotype",
                node_label="gene essentiality phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_synthetic_lethality_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            is_synthetic_lethal = phenotype.is_synthetic_lethal
            statistic_score = phenotype.synthetic_lethality_statistic_score

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "is_synthetic_lethal": is_synthetic_lethal,
                "synthetic_lethality_statistic_score": statistic_score,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="synthetic lethality phenotype",
                node_label="synthetic lethality phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_synthetic_rescue_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            is_synthetic_rescue = phenotype.is_synthetic_rescue
            synthetic_rescue_statistic_score = (
                phenotype.synthetic_rescue_statistic_score
            )

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "is_synthetic_rescue": is_synthetic_rescue,
                "synthetic_rescue_statistic_score": synthetic_rescue_statistic_score,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="synthetic rescue phenotype",
                node_label="synthetic rescue phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    @data_chunker
    def _calmorph_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        calmorph = phenotype.calmorph
        calmorph_coefficient_of_variation = phenotype.calmorph_coefficient_of_variation

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "calmorph": json.dumps(calmorph),  # Store as JSON string
            "calmorph_coefficient_of_variation": json.dumps(
                calmorph_coefficient_of_variation
            )
            if calmorph_coefficient_of_variation
            else None,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="calmorph phenotype",
            properties=properties,
        )

    def _get_calmorph_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            calmorph = phenotype.calmorph
            calmorph_coefficient_of_variation = (
                phenotype.calmorph_coefficient_of_variation
            )

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "calmorph": json.dumps(calmorph),  # Store as JSON string
                "calmorph_coefficient_of_variation": json.dumps(
                    calmorph_coefficient_of_variation
                )
                if calmorph_coefficient_of_variation
                else None,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="calmorph phenotype",
                node_label="calmorph phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    # --- Expression / metabolite / visual-score phenotypes (abstract datasets) ---
    # Multi-valued phenotypes (expression, metabolite) serialize their per-key dicts to
    # JSON strings (the CalMorph pattern); the scalar VisualScore stores plain scalars.

    @data_chunker
    def _microarray_expression_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "expression_log2_ratio": json.dumps(phenotype.expression_log2_ratio),
            "expression_log2_ratio_se": (
                json.dumps(phenotype.expression_log2_ratio_se)
                if phenotype.expression_log2_ratio_se is not None
                else None
            ),
            "serialized_data": json.dumps(phenotype.model_dump()),
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="microarray expression phenotype",
            properties=properties,
        )

    def _get_microarray_expression_phenotype_reference_nodes(
        self,
    ) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "expression_log2_ratio": json.dumps(phenotype.expression_log2_ratio),
                "expression_log2_ratio_se": (
                    json.dumps(phenotype.expression_log2_ratio_se)
                    if phenotype.expression_log2_ratio_se is not None
                    else None
                ),
                "serialized_data": json.dumps(phenotype.model_dump()),
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="microarray expression phenotype",
                    node_label="microarray expression phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _rnaseq_expression_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "expression_tpm": json.dumps(phenotype.expression_tpm),
            "measurement_type": phenotype.measurement_type,
            "n_mapped_reads": phenotype.n_mapped_reads,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="rnaseq expression phenotype",
            properties=properties,
        )

    def _get_rnaseq_expression_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "expression_tpm": json.dumps(phenotype.expression_tpm),
                "measurement_type": phenotype.measurement_type,
                "n_mapped_reads": phenotype.n_mapped_reads,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="rnaseq expression phenotype",
                    node_label="rnaseq expression phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _visual_score_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "visual_score": phenotype.visual_score,
            "n_replicates": phenotype.n_replicates,
            "target_product": phenotype.target_product,
            "target_metabolite_id": phenotype.target_metabolite_id,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="visual score phenotype",
            properties=properties,
        )

    def _get_visual_score_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "visual_score": phenotype.visual_score,
                "n_replicates": phenotype.n_replicates,
                "target_product": phenotype.target_product,
                "target_metabolite_id": phenotype.target_metabolite_id,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="visual score phenotype",
                    node_label="visual score phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _metabolite_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "metabolite_level": json.dumps(phenotype.metabolite_level),
            "metabolite_level_se": (
                json.dumps(phenotype.metabolite_level_se)
                if phenotype.metabolite_level_se is not None
                else None
            ),
            "measurement_type": phenotype.measurement_type,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="metabolite phenotype",
            properties=properties,
        )

    def _get_metabolite_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "metabolite_level": json.dumps(phenotype.metabolite_level),
                "metabolite_level_se": (
                    json.dumps(phenotype.metabolite_level_se)
                    if phenotype.metabolite_level_se is not None
                    else None
                ),
                "measurement_type": phenotype.measurement_type,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="metabolite phenotype",
                    node_label="metabolite phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _protein_abundance_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "protein_abundance": json.dumps(phenotype.protein_abundance),
            "protein_abundance_se": (
                json.dumps(phenotype.protein_abundance_se)
                if phenotype.protein_abundance_se is not None
                else None
            ),
            "measurement_type": phenotype.measurement_type,
            "serialized_data": json.dumps(phenotype.model_dump()),
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="protein abundance phenotype",
            properties=properties,
        )

    def _get_protein_abundance_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "protein_abundance": json.dumps(phenotype.protein_abundance),
                "protein_abundance_se": (
                    json.dumps(phenotype.protein_abundance_se)
                    if phenotype.protein_abundance_se is not None
                    else None
                ),
                "measurement_type": phenotype.measurement_type,
                "serialized_data": json.dumps(phenotype.model_dump()),
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="protein abundance phenotype",
                    node_label="protein abundance phenotype",
                    properties=properties,
                )
            )
        return nodes

    def _get_dataset_nodes(self) -> list[BioCypherNode]:
        nodes = [
            BioCypherNode(
                node_id=self.dataset.name,
                preferred_id=self.dataset.name,
                node_label="dataset",
            )
        ]
        return nodes

    @data_chunker
    def _publication_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        publication = data["publication"]
        publication_id = hashlib.sha256(
            json.dumps(publication.model_dump()).encode("utf-8")
        ).hexdigest()

        return BioCypherNode(
            node_id=publication_id,
            preferred_id=f"publication_{publication.pubmed_id}",
            node_label="publication",
            properties={
                "pubmed_id": publication.pubmed_id,
                "pubmed_url": publication.pubmed_url,
                "doi": publication.doi,
                "doi_url": publication.doi_url,
                "serialized_data": json.dumps(publication.model_dump()),
            },
        )

    # edges
    def _get_experiment_reference_to_dataset_edges(self) -> list[BioCypherEdge]:
        edges = []
        for data in self.dataset.experiment_reference_index:
            reference_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=reference_id,
                target_id=self.dataset.name,
                relationship_label="experiment reference member of",
            )
            edges.append(edge)
        return edges

    @data_chunker
    def _experiment_to_dataset_edge(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=experiment_id,
            target_id=self.dataset.name,
            relationship_label="experiment member of",
        )
        return edge  # type: ignore[no-any-return]  # BioCypherEdge is Any (biocypher untyped)

    @data_chunker
    def _experiment_reference_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
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
    def _genotype_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
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
    def _perturbation_to_genotype_edges(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
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
    def _environment_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
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
        seen_environment_experiment_ref_pairs: set[tuple[str, str]] = set()
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
    def _phenotype_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
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
    def _media_to_environment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
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
    def _temperature_to_environment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
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
        seen_genome_experiment_ref_pairs: set[tuple[str, str]] = set()
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

    def _get_phenotype_to_experiment_reference_edges(self) -> list[BioCypherEdge]:
        edges = []
        seen_phenotype_experiment_ref_pairs: set[tuple[str, str]] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.phenotype_reference.model_dump()).encode(
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

    @data_chunker
    def _publication_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        publication_id = hashlib.sha256(
            json.dumps(data["publication"].model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherEdge(
            source_id=publication_id,
            target_id=experiment_id,
            relationship_label="mentions",
        )


if __name__ == "__main__":
    pass
