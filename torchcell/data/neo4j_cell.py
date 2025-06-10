# torchcell/data/neo4j_cell
# [[torchcell.data.neo4j_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_cell
# Test file: tests/torchcell/data/test_neo4j_cell.py

import json
import logging
import os
import os.path as osp
from collections.abc import Callable
from enum import Enum, auto
from typing import Optional, Type, Any
import fcntl
import time
import random

import hypernetx as hnx
import lmdb
import networkx as nx
import numpy as np
import pandas as pd
import torch
from pydantic import field_validator
from torch_geometric.data import Dataset
from tqdm import tqdm
from torchcell.data.graph_processor import GraphProcessor
from torchcell.data.cell_data import to_cell_data
from torchcell.data.deduplicate import Deduplicator
from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from torchcell.datamodels import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    Converter,
    ModelStrictArbitrary,
    PhenotypeType,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import GeneGraph, GeneMultiGraph
from sortedcontainers import SortedDict

log = logging.getLogger(__name__)


# Do we need this?
# class DatasetIndex(ModelStrict):
#     index: dict[str|int, list[int]]


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @field_validator("gene_set")
    def validate_gene_set(cls, v):
        if not isinstance(v, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(v).__name__}")
        return v


# HACK - normalize embedding start
# Probably should reformat data for this, but would need to resave or recompute
def normalize_tensor_row(x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalizes a tensor to sum to 1, after min subtraction."""
    # Handle list input
    if isinstance(x, list):
        x = x[0]

    # Add batch dimension if needed
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x - x.min()
    x = x / x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return list(x.flatten(-1))


def min_max_normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Forces embedding tensor values into [0,1] range using min-max scaling per feature."""
    # Normalize each feature (column) independently
    normalized_embedding = torch.zeros_like(embedding)
    for i in range(embedding.size(1)):
        feature = embedding[:, i]
        feature_min = feature.min()
        feature_max = feature.max()

        # If feature_min == feature_max, set to 0.5 to avoid div by zero
        if feature_min == feature_max:
            normalized_embedding[:, i] = 0.5
        else:
            normalized_embedding[:, i] = (feature - feature_min) / (
                feature_max - feature_min
            )

    return normalized_embedding


def min_max_normalize_dataset(dataset: BaseEmbeddingDataset) -> None:
    """Normalizes embeddings across the entire dataset to range [0,1] using min-max scaling per feature."""
    first_key = list(dataset._data.embeddings.keys())[0]
    embeddings = dataset._data.embeddings[first_key]
    dataset._data.embeddings[first_key] = min_max_normalize_embedding(embeddings)


def create_embedding_graph(
    gene_set: GeneSet, embeddings: BaseEmbeddingDataset
) -> GeneGraph:
    """Create a GeneGraph from embeddings."""
    # Normalize dataset first
    min_max_normalize_dataset(embeddings)

    G = nx.Graph()
    for item in embeddings:
        keys = item["embeddings"].keys()
        if item.id in gene_set:
            item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
            concatenated_embedding = torch.cat(item_embeddings)
            G.add_node(item.id, embedding=concatenated_embedding)

    # Create and return a GeneGraph
    return GeneGraph(name=embeddings.__class__.__name__, graph=G, max_gene_set=gene_set)


def create_graph_from_gene_set(gene_set: GeneSet) -> GeneGraph:
    """
    Create a GeneGraph where nodes are gene names from the GeneSet.
    Initially, this graph will have no edges.
    """
    G = nx.Graph()
    for gene_name in gene_set:
        G.add_node(gene_name)  # Nodes are gene names
    return GeneGraph(name="base", graph=G, max_gene_set=gene_set)


def parse_genome(genome) -> ParsedGenome:
    if genome is None:
        return None
    else:
        data = {}
        data["gene_set"] = genome.gene_set
        return ParsedGenome(**data)


class ProcessingStep(Enum):
    RAW = auto()
    CONVERSION = auto()
    DEDUPLICATION = auto()
    AGGREGATION = auto()
    PROCESSED = auto()


# TODO implement
class Aggregator:
    pass


class Neo4jCellDataset(Dataset):
    """Dataset for loading cell data from Neo4j with file locking for DDP support."""
    
    @staticmethod
    def _read_json_with_lock(filepath: str, max_retries: int = 10) -> Any:
        """Read JSON file with file locking and retry logic."""
        retry_delay = 0.1  # Start with 100ms
        
        for attempt in range(max_retries):
            try:
                with open(filepath, 'r') as f:
                    # Try to acquire shared lock (non-blocking first)
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except IOError:
                        # If non-blocking fails, wait with exponential backoff
                        if attempt < max_retries - 1:
                            sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                            time.sleep(sleep_time)
                            continue
                        else:
                            # Last attempt: block until we get the lock
                            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    
                    try:
                        # Read the file content
                        content = f.read()
                        if not content.strip():
                            # Empty file, retry
                            raise json.JSONDecodeError("Empty file", "", 0)
                        
                        return json.loads(content)
                    finally:
                        # Always release the lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # File might be in the process of being written, retry
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                    continue
                else:
                    raise ValueError(f"Invalid or empty JSON file after {max_retries} attempts: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                    continue
                else:
                    raise
        
        raise ValueError(f"Failed to read {filepath} after {max_retries} attempts")
    
    @staticmethod
    def _write_json_with_lock(filepath: str, data: Any) -> None:
        """Write JSON file with exclusive lock and atomic rename."""
        # Ensure directory exists
        os.makedirs(osp.dirname(filepath), exist_ok=True)
        
        temp_path = filepath + f".tmp.{os.getpid()}.{time.time()}"
        
        # Write to temporary file first
        try:
            with open(temp_path, 'w') as f:
                # Acquire exclusive lock for writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=0)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force write to disk
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename - this is atomic on POSIX systems
            os.rename(temp_path, filepath)
            
        except Exception:
            # Clean up temp file if something went wrong
            if osp.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise
    
    # @profile
    def __init__(
        self,
        root: str,
        query: str = None,
        gene_set: GeneSet = None,
        graphs: GeneMultiGraph = None,
        incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] = None,
        node_embeddings: dict[str, BaseEmbeddingDataset] = None,
        graph_processor: GraphProcessor = None,
        add_remaining_gene_self_loops: bool = True,
        converter: Optional[Type[Converter]] = None,
        deduplicator: Type[Deduplicator] = None,
        aggregator: Type[Aggregator] = None,
        overwrite_intermediates: bool = False,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "torchcell",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self.env = None
        self.root = root
        # get item processor
        self.process_graph = graph_processor

        # self loops, transform base graph
        self.add_remaining_gene_self_loops = add_remaining_gene_self_loops

        # Needed in get item
        self._phenotype_info = None

        # Cached indices
        self._phenotype_label_index = None
        self._dataset_name_index = None
        self._perturbation_count_index = None
        self._is_any_perturbed_gene_index = None

        # Cached label df for converting regression to classification
        self._label_df = None

        self.gene_set = gene_set

        # raw db processing
        self.overwrite_intermediates = overwrite_intermediates
        self.converter = converter
        self.deduplicator = deduplicator
        self.aggregator = aggregator

        # raw db deps
        self.uri = uri
        self.username = username
        self.password = password
        self.query = query

        super().__init__(root, transform, pre_transform, pre_filter)

        # Initialize a GeneMultiGraph with a base graph
        base_graph = create_graph_from_gene_set(self.gene_set)

        # Set up the GeneMultiGraph
        if graphs is None:
            # Create a new GeneMultiGraph with just the base graph
            graphs_dict = SortedDict({"base": base_graph})
            multigraph = GeneMultiGraph(graphs=graphs_dict)
        else:
            # Create a copy of the provided GeneMultiGraph to avoid modifying the original
            graphs_dict = SortedDict(graphs.graphs.copy())
            # Ensure the copy has a base graph
            if "base" not in graphs_dict:
                graphs_dict["base"] = base_graph
            multigraph = GeneMultiGraph(graphs=graphs_dict)

        # Add embeddings as GeneGraphs to the GeneMultiGraph
        if node_embeddings is not None:
            for name, embedding in node_embeddings.items():
                embedding_graph = create_embedding_graph(self.gene_set, embedding)
                multigraph.graphs[name] = embedding_graph

        # cell graph used in get item
        self.cell_graph = to_cell_data(
            multigraph,
            incidence_graphs,
            add_remaining_gene_self_loops=self.add_remaining_gene_self_loops,
        )

        # Clean up hanging env, for multiprocessing
        self.env = None

        # compute index
        self.phenotype_label_index
        self.dataset_name_index
        self.perturbation_count_index

    def _determine_processing_steps(self):
        steps = [ProcessingStep.RAW]
        if self.converter is not None:
            steps.append(ProcessingStep.CONVERSION)
        if self.deduplicator is not None:
            steps.append(ProcessingStep.DEDUPLICATION)
        if self.aggregator is not None:
            steps.append(ProcessingStep.AGGREGATION)
        steps.append(ProcessingStep.PROCESSED)
        return steps

    def _get_lmdb_path(self, step: ProcessingStep):
        if step == ProcessingStep.RAW:
            return os.path.join(self.root, "raw", "lmdb")
        elif step == ProcessingStep.PROCESSED:
            return os.path.join(self.processed_dir, "lmdb")
        else:
            return os.path.join(self.root, step.name.lower(), "lmdb")

    def get_init_graphs(self, gene_set):
        cell_graph = create_graph_from_gene_set(gene_set)
        return cell_graph

    @property
    def raw_file_names(self) -> list[str]:
        return "lmdb"

    @staticmethod
    def load_raw(uri, username, password, root_dir, query, gene_set):

        cypher_kwargs = {"gene_set": list(gene_set)}
        # cypher_kwargs = {"gene_set": ["YAL004W", "YAL010C", "YAL011W", "YAL017W"]}
        print("================")
        print(f"raw root_dir: {root_dir}")
        print("================")
        raw_db = Neo4jQueryRaw(
            uri=uri,
            username=username,
            password=password,
            root_dir=root_dir,
            query=query,
            io_workers=10,  # IDEA simple for new, might need to parameterize
            num_workers=10,
            cypher_kwargs=cypher_kwargs,
        )
        return raw_db  # break point here

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    @property
    def phenotype_info(self) -> list[PhenotypeType]:
        if self._phenotype_info is None:
            self._phenotype_info = self._load_phenotype_info()
        return self._phenotype_info

    def _load_phenotype_info(self) -> list[PhenotypeType]:
        experiment_types_path = osp.join(self.processed_dir, "experiment_types.json")
        if osp.exists(experiment_types_path):
            experiment_types = self._read_json_with_lock(experiment_types_path)

            phenotype_classes = set()
            for exp_type in experiment_types:
                experiment_class = EXPERIMENT_TYPE_MAP[exp_type]
                phenotype_class = experiment_class.__annotations__["phenotype"]
                phenotype_classes.add(phenotype_class)

            return list(phenotype_classes)
        else:
            raise FileNotFoundError(
                "experiment_types.json not found. Please process the dataset first."
            )

    def compute_phenotype_info(self):
        self._init_lmdb_read()
        experiment_types = set()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                data_list = json.loads(value.decode("utf-8"))
                for item in data_list:
                    experiment_types.add(item["experiment"]["experiment_type"])

        # Save experiment types to a JSON file with file locking
        self._write_json_with_lock(
            osp.join(self.processed_dir, "experiment_types.json"), 
            list(experiment_types)
        )

        self.close_lmdb()

    def process(self):
        # IDEA consider dependency injection for processing steps
        # We don't inject becaue of unique query process.
        raw_db = self.load_raw(
            self.uri, self.username, self.password, self.root, self.query, self.gene_set
        )

        self.converter = (
            self.converter(root=self.root, query=raw_db) if self.converter else None
        )
        self.deduplicator = (
            self.deduplicator(root=self.root) if self.deduplicator else None
        )
        self.aggregator = self.aggregator(root=self.root) if self.aggregator else None

        self.processing_steps = self._determine_processing_steps()

        current_step = ProcessingStep.RAW
        for next_step in self.processing_steps[1:]:
            input_path = self._get_lmdb_path(current_step)
            output_path = self._get_lmdb_path(next_step)

            if next_step == ProcessingStep.CONVERSION:
                self.converter.process(input_path, output_path)
            elif next_step == ProcessingStep.DEDUPLICATION:
                self.deduplicator.process(input_path, output_path)
            elif next_step == ProcessingStep.AGGREGATION:
                self.aggregator.process(input_path, output_path)
            elif next_step == ProcessingStep.PROCESSED:
                self._copy_lmdb(input_path, output_path)

            if self.overwrite_intermediates and next_step != ProcessingStep.PROCESSED:
                os.remove(input_path)

            current_step = next_step

        # Compute phenotype info - used in get item
        self.compute_phenotype_info()
        # Compute and cache label DataFrame explicitly
        self._label_df = self.label_df
        # clean up raw db
        raw_db.env = None

    def _copy_lmdb(self, src_path: str, dst_path: str):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        env_src = lmdb.open(src_path, readonly=True)
        env_dst = lmdb.open(dst_path, map_size=int(1e12))

        with env_src.begin() as txn_src, env_dst.begin(write=True) as txn_dst:
            cursor = txn_src.cursor()
            for key, value in cursor:
                txn_dst.put(key, value)

        env_src.close()
        env_dst.close()

    # TODO change to query_gene_set
    @property
    def gene_set(self):
        gene_set_path = osp.join(self.processed_dir, "gene_set.json")
        
        # Check if file exists
        if not osp.exists(gene_set_path):
            if self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return GeneSet(self._gene_set)
        
        # Read with file locking and retry logic
        max_retries = 10
        retry_delay = 0.1  # Start with 100ms
        
        for attempt in range(max_retries):
            try:
                with open(gene_set_path, 'r') as f:
                    # Try to acquire shared lock (non-blocking first)
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except IOError:
                        # If non-blocking fails, wait with exponential backoff
                        if attempt < max_retries - 1:
                            sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                            time.sleep(sleep_time)
                            continue
                        else:
                            # Last attempt: block until we get the lock
                            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    
                    try:
                        # Read the file content
                        content = f.read()
                        if not content.strip():
                            # Empty file, retry
                            raise json.JSONDecodeError("Empty file", "", 0)
                        
                        self._gene_set = set(json.loads(content))
                        return GeneSet(self._gene_set)
                    finally:
                        # Always release the lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # File might be in the process of being written, retry
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                    continue
                else:
                    raise ValueError(f"Invalid or empty JSON file found after {max_retries} attempts: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                    continue
                else:
                    raise
        
        raise ValueError(f"Failed to read gene_set.json after {max_retries} attempts")

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        gene_set_path = osp.join(self.processed_dir, "gene_set.json")
        temp_path = gene_set_path + f".tmp.{os.getpid()}.{time.time()}"
        
        # Write to temporary file first
        try:
            with open(temp_path, 'w') as f:
                # Acquire exclusive lock for writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(list(sorted(value)), f, indent=0)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force write to disk
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename - this is atomic on POSIX systems
            os.rename(temp_path, gene_set_path)
            self._gene_set = value
            
        except Exception:
            # Clean up temp file if something went wrong
            if osp.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def get(self, idx):
        if self.env is None:
            self._init_lmdb_read()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode("utf-8"))
            if serialized_data is None:
                return None
            data_list = json.loads(serialized_data.decode("utf-8"))

            data = []
            for item in data_list:
                experiment_class = EXPERIMENT_TYPE_MAP[
                    item["experiment"]["experiment_type"]
                ]
                experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                    item["experiment_reference"]["experiment_reference_type"]
                ]
                reconstructed_data = {
                    "experiment": experiment_class(**item["experiment"]),
                    "experiment_reference": experiment_reference_class(
                        **item["experiment_reference"]
                    ),
                }
                data.append(reconstructed_data)

            processed_graph = self.process_graph.process(
                self.cell_graph, self.phenotype_info, data
            )

        return processed_graph

    def _init_lmdb_read(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            max_spare_txns=16,
        )

    def len(self) -> int:
        if self.env is None:
            self._init_lmdb_read()

        with self.env.begin(write=False) as txn:
            length = txn.stat()["entries"]
        self.close_lmdb()
        return length

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def label_df(self) -> pd.DataFrame:
        """Cache and return a DataFrame containing all labels and their indices."""
        label_cache_path = osp.join(self.processed_dir, "label_df.parquet")

        # Return cached DataFrame if already loaded in memory and valid
        if hasattr(self, "_label_df") and isinstance(self._label_df, pd.DataFrame):
            return self._label_df

        # Load from disk if previously cached
        if osp.exists(label_cache_path):
            self._label_df = pd.read_parquet(label_cache_path)
            return self._label_df

        print("Computing label DataFrame...")

        # Get label names from phenotype_info
        label_names = [
            phenotype.model_fields["label_name"].default
            for phenotype in self.phenotype_info
        ]

        # Initialize data dictionary with index and label columns
        data_dict = {"index": [], **{label_name: [] for label_name in label_names}}

        # Open LMDB for reading
        self._init_lmdb_read()

        # Iterate through all entries in the database
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                idx = int(key.decode())
                data_list = json.loads(value.decode())

                # Initialize row with index and NaN for all labels
                row_data = {"index": idx}
                for label_name in label_names:
                    row_data[label_name] = np.nan

                # Check all experiments in data_list
                for data in data_list:
                    experiment = EXPERIMENT_TYPE_MAP[
                        data["experiment"]["experiment_type"]
                    ](**data["experiment"])

                    # Add each label if it exists
                    for label_name in label_names:
                        try:
                            value = getattr(experiment.phenotype, label_name)
                            if not np.isnan(
                                value
                            ):  # Only update if we find a non-NaN value
                                row_data[label_name] = value
                        except AttributeError:
                            continue  # Try next experiment if label doesn't exist in this one

                # Add row data to data_dict
                for key, value in row_data.items():
                    data_dict[key].append(value)

        self.close_lmdb()

        # Create DataFrame
        self._label_df = pd.DataFrame(data_dict)

        # Cache the DataFrame
        self._label_df.to_parquet(label_cache_path)

        return self._label_df

    def compute_phenotype_label_index(self) -> dict[str, list[int]]:
        print("Computing phenotype label index...")
        phenotype_label_index = {}

        self._init_lmdb_read()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    idx = int(key.decode())
                    data_list = json.loads(value.decode())
                    for data in data_list:
                        experiment_class = EXPERIMENT_TYPE_MAP[
                            data["experiment"]["experiment_type"]
                        ]
                        experiment = experiment_class(**data["experiment"])
                        label_name = experiment.phenotype.label_name

                        if label_name not in phenotype_label_index:
                            phenotype_label_index[label_name] = set()
                        phenotype_label_index[label_name].add(idx)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for entry {key}. Skipping this entry.")
                except ValueError:
                    print(
                        f"Error converting key to integer: {key}. Skipping this entry."
                    )
                except Exception as e:
                    print(
                        f"Error processing entry {key}: {str(e)}. Skipping this entry."
                    )

        self.close_lmdb()

        # Convert sets to sorted lists
        for label in phenotype_label_index:
            phenotype_label_index[label] = sorted(list(phenotype_label_index[label]))

        return phenotype_label_index

    @property
    def phenotype_label_index(self) -> dict[str, list[int]]:
        filepath = osp.join(self.processed_dir, "phenotype_label_index.json")
        
        if osp.exists(filepath):
            self._phenotype_label_index = self._read_json_with_lock(filepath)
        else:
            self._phenotype_label_index = self.compute_phenotype_label_index()
            self._write_json_with_lock(filepath, self._phenotype_label_index)
        
        return self._phenotype_label_index

    def compute_dataset_name_index(self) -> dict[str, list[int]]:
        print("Computing dataset name index...")
        dataset_name_index = {}

        self._init_lmdb_read()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    idx = int(key.decode())
                    data_list = json.loads(value.decode())
                    for data in data_list:
                        experiment_class = EXPERIMENT_TYPE_MAP[
                            data["experiment"]["experiment_type"]
                        ]
                        experiment = experiment_class(**data["experiment"])
                        dataset_name = experiment.dataset_name

                        if dataset_name not in dataset_name_index:
                            dataset_name_index[dataset_name] = set()
                        dataset_name_index[dataset_name].add(idx)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for entry {key}. Skipping this entry.")
                except ValueError:
                    print(
                        f"Error converting key to integer: {key}. Skipping this entry."
                    )
                except Exception as e:
                    print(
                        f"Error processing entry {key}: {str(e)}. Skipping this entry."
                    )

        self.close_lmdb()

        # Convert sets to sorted lists
        for name in dataset_name_index:
            dataset_name_index[name] = sorted(list(dataset_name_index[name]))

        return dataset_name_index

    @property
    def dataset_name_index(self) -> dict[str, list[int]]:
        filepath = osp.join(self.processed_dir, "dataset_name_index.json")
        
        if osp.exists(filepath):
            self._dataset_name_index = self._read_json_with_lock(filepath)
        else:
            self._dataset_name_index = self.compute_dataset_name_index()
            self._write_json_with_lock(filepath, self._dataset_name_index)
        
        return self._dataset_name_index

    def compute_perturbation_count_index(self) -> dict[int, list[int]]:
        print("Computing perturbation count index...")
        perturbation_count_index = {}

        self._init_lmdb_read()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    idx = int(key.decode())
                    data_list = json.loads(value.decode())
                    for data in data_list:
                        experiment_class = EXPERIMENT_TYPE_MAP[
                            data["experiment"]["experiment_type"]
                        ]
                        experiment = experiment_class(**data["experiment"])
                        perturbation_count = len(experiment.genotype.perturbations)

                        if perturbation_count not in perturbation_count_index:
                            perturbation_count_index[perturbation_count] = set()
                        perturbation_count_index[perturbation_count].add(idx)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for entry {key}. Skipping this entry.")
                except ValueError:
                    print(
                        f"Error converting key to integer: {key}. Skipping this entry."
                    )
                except Exception as e:
                    print(
                        f"Error processing entry {key}: {str(e)}. Skipping this entry."
                    )

        self.close_lmdb()

        # Convert sets to sorted lists
        for count in perturbation_count_index:
            perturbation_count_index[count] = sorted(
                list(perturbation_count_index[count])
            )

        return perturbation_count_index

    @property
    def perturbation_count_index(self) -> dict[int, list[int]]:
        filepath = osp.join(self.processed_dir, "perturbation_count_index.json")
        
        if osp.exists(filepath):
            self._perturbation_count_index = self._read_json_with_lock(filepath)
            # Convert string keys back to integers
            self._perturbation_count_index = {
                int(k): v for k, v in self._perturbation_count_index.items()
            }
        else:
            self._perturbation_count_index = self.compute_perturbation_count_index()
            # Convert integer keys to strings for JSON serialization
            data_to_write = {str(k): v for k, v in self._perturbation_count_index.items()}
            self._write_json_with_lock(filepath, data_to_write)
        
        return self._perturbation_count_index

    def compute_is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
        print("Computing is any perturbed gene index...")
        is_any_perturbed_gene_index = {}

        self._init_lmdb_read()

        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                # Convert cursor to list to avoid multiple LMDB transactions
                entries = [(key, value) for key, value in cursor]

            # Process entries outside of LMDB transaction
            for key, value in entries:
                try:
                    idx = int(key.decode())
                    # Parse JSON once per entry
                    data_list = json.loads(value.decode())
                    # Process all perturbations in one pass
                    for data in data_list:
                        for pert in data["experiment"]["genotype"]["perturbations"]:
                            gene = pert["systematic_gene_name"]
                            if gene not in is_any_perturbed_gene_index:
                                is_any_perturbed_gene_index[gene] = set()
                            is_any_perturbed_gene_index[gene].add(idx)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error processing entry {key}: {e}")

        finally:
            self.close_lmdb()

        # Convert sets to sorted lists
        return {
            gene: sorted(list(indices))
            for gene, indices in is_any_perturbed_gene_index.items()
        }

    @property
    def is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
        # Memory cache
        if hasattr(self, "_is_any_perturbed_gene_index_cache"):
            return self._is_any_perturbed_gene_index_cache

        cache_path = osp.join(self.processed_dir, "is_any_perturbed_gene_index.json")

        # Try to load from disk cache
        if osp.exists(cache_path):
            self._is_any_perturbed_gene_index_cache = self._read_json_with_lock(cache_path)
            return self._is_any_perturbed_gene_index_cache

        # Compute if no cache exists
        result = self.compute_is_any_perturbed_gene_index()

        # Save to both memory and disk cache
        self._is_any_perturbed_gene_index_cache = result
        self._write_json_with_lock(cache_path, result)

        return result

    # HACK
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Remove the unpicklable lmdb environment.
        state["env"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


def main():
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.data.graph_processor import SubgraphRepresentation

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    ### Add Embeddings
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open("gene_set.json", "w") as f:
        json.dump(list(genome.gene_set), f)

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(len(dataset))
    dataset.label_df
    # Data module testing

    # print(dataset[7])
    print(dataset[3])
    dataset.close_lmdb()
    # print(dataset[10000])

    # Assuming you have already created your dataset and CellDataModule
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        print(batch)
        break

    # for i in tqdm(
    #     range(len(cell_data_module.index_details.train.perturbation_count_index[1].indices))
    # ):
    #     single_pert_index = (
    #         cell_data_module.index_details.train.perturbation_count_index[1].indices[i]
    #     )
    #     if len(dataset[single_pert_index]["gene"].ids_pert) != 1:
    #         train_not_single_pert.append(single_pert_index)

    # print("len train_not_single_pert", len(train_not_single_pert))

    # # Now, instantiate the updated PerturbationSubsetDataModule
    # size = 1e4
    # seed = 42
    # perturbation_subset_data_module = PerturbationSubsetDataModule(
    #     cell_data_module=cell_data_module,
    #     size=int(size),
    #     batch_size=2,
    #     num_workers=2,
    #     pin_memory=True,
    #     prefetch=False,
    #     seed=seed,
    # )

    # # Set up the data module
    # perturbation_subset_data_module.setup()

    # # Use the data loaders
    # for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
    #     # Your training code here
    #     break


def main_incidence():
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.metabolism.yeast_GEM import YeastGEM

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    ### Add Embeddings
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open("gene_set.json", "w") as f:
        json.dump(list(genome.gene_set), f)

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism": YeastGEM().reaction_map},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(len(dataset))
    dataset.label_df
    # Data module testing

    # print(dataset[7])
    print(dataset.cell_graph)
    print(dataset[4])

    # Get perturbed gene indices
    print("first few perturbed gene indices with modified metabolism graph:")
    [
        print(
            dataset[i]["metabolite", "reaction", "metabolite"].hyperedge_index.size()
            != dataset.cell_graph[
                "metabolite", "reaction", "metabolite"
            ].hyperedge_index.size()
        )
        for i in range(10)
    ]

    perturbed_indices = dataset[4]["gene"].cell_graph_idx_pert

    # Check which reactions contained these genes in the original graph
    reactions_with_perturbed = set()
    for rxn_idx, genes in dataset.cell_graph[
        "metabolite", "reaction", "metabolite"
    ].reaction_to_genes_indices.items():
        if any(g in perturbed_indices for g in genes):
            reactions_with_perturbed.add(rxn_idx)

    print(
        f"Number of reactions containing perturbed genes: {len(reactions_with_perturbed)}"
    )

    dataset.close_lmdb()
    # print(dataset[10000])

    # Assuming you have already created your dataset and CellDataModule
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        break


def main_transform_standardization():
    """Test standardization of labels using LabelNormalizationTransform with metabolic network."""
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    from dotenv import load_dotenv
    from torch_geometric.transforms import Compose
    from tqdm import tqdm

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )

    # Import necessary components
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    # from torchcell.transforms.hetero_to_dense import HeteroToDense
    from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask
    from torchcell.transforms.regression_to_classification import (
        LabelNormalizationTransform,
    )

    # Load environment variables
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Load query
    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    # Set up genome and graph
    print("Setting up genome and graph...")
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # Set up node embeddings
    print("Setting up embeddings...")
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    # Create dataset with metabolism network
    print("Creating dataset with metabolism network...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        # incidence_graphs={"metabolism_hypergraph": YeastGEM().reaction_map},
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(f"Dataset size: {len(dataset)}")
    dataset
    # Define the labels we want to standardize
    labels = ["fitness", "gene_interaction"]

    # Print statistics of original data using dataset.label_df
    for label in labels:
        values = dataset.label_df[label].dropna().values
        print(f"\n{label} statistics (original):")
        print(f"  Count: {len(values)}")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")

    # Configure normalization to use standard (z-score) normalization for both labels
    norm_configs = {
        "fitness": {"strategy": "standard"},  # z-score: (x - mean) / std
        "gene_interaction": {"strategy": "standard"},  # z-score: (x - mean) / std
    }

    # Create the normalizer
    print("\nCreating normalization transform...")
    normalizer = LabelNormalizationTransform(dataset, norm_configs)

    # Print the normalization parameters
    for label, stats in normalizer.stats.items():
        print(f"\nNormalization parameters for {label}:")
        for key, value in stats.items():
            if key not in ["bin_edges", "bin_counts", "strategy"]:
                print(f"  {key}: {value:.6f}")
        print(f"  strategy: {stats['strategy']}")

    # Apply the transform to the dataset
    # HACK - start
    dense_transform = HeteroToDenseMask({"gene": len(genome.gene_set)})
    dataset.transform = Compose([normalizer, dense_transform])
    # HACK - end
    # dataset.transform = normalizer

    # Check metabolic network connectivity changes
    print("\nChecking metabolic network changes for a few examples...")
    for i in range(5):
        data = dataset[i]
        if "metabolite" in data.node_types and "reaction" in data.node_types:
            if hasattr(data["metabolite", "reaction", "metabolite"], "hyperedge_index"):
                print(
                    f"Sample {i} - hyperedge size: {data['metabolite', 'reaction', 'metabolite'].hyperedge_index.size()}"
                )

                # Check if perturbed genes affect the metabolic network
                if hasattr(data["gene"], "cell_graph_idx_pert"):
                    perturbed_indices = data["gene"].cell_graph_idx_pert

                    # Count reactions affected by gene perturbations
                    reactions_with_perturbed = set()
                    for rxn_idx, genes in dataset.cell_graph[
                        "metabolite", "reaction", "metabolite"
                    ].reaction_to_genes_indices.items():
                        if any(g in perturbed_indices for g in genes):
                            reactions_with_perturbed.add(rxn_idx)

                    print(f"  Perturbed genes: {len(perturbed_indices)}")
                    print(f"  Reactions affected: {len(reactions_with_perturbed)}")

    # Sample data points for visualization
    print("\nSampling data for visualization...")
    num_samples = 1000
    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    original_values = {label: [] for label in labels}
    normalized_values = {label: [] for label in labels}

    for idx in tqdm(sample_indices):
        data = dataset[idx]
        for label in labels:
            if label in data["gene"] and not torch.isnan(data["gene"][label]).any():
                # Get normalized value
                normalized_values[label].append(data["gene"][label].item())
                # Get original value
                original_values[label].append(data["gene"][f"{label}_original"].item())

    # Create visualization with plots
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Label Distributions: Before vs. After Standardization", fontsize=16)

    for i, label in enumerate(labels):
        # Original distribution
        sns.histplot(original_values[label], bins=50, ax=axes[0, i], kde=True)
        axes[0, i].set_title(f"Original {label}")

        # Normalized distribution
        sns.histplot(normalized_values[label], bins=50, ax=axes[1, i], kde=True)
        axes[1, i].set_title(f"Standardized {label} (z-score)")

        # Add mean and std lines to the normalized plot
        axes[1, i].axvline(x=0, color="r", linestyle="--", label="Mean (0)")
        axes[1, i].axvline(x=1, color="g", linestyle="--", label="+1 Std")
        axes[1, i].axvline(x=-1, color="g", linestyle="--", label="-1 Std")
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig("standardization_with_metabolism_comparison.png")

    # Test with datamodule
    print("\nTesting with CellDataModule...")
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    # Get a batch and check normalized values
    for batch in cell_data_module.train_dataloader():
        for label in labels:
            if label in batch["gene"]:
                print(f"\nBatch {label} statistics:")
                print(f"  Shape: {batch['gene'][label].shape}")
                print(f"  Mean: {batch['gene'][label].mean().item():.4f}")
                print(f"  Std: {batch['gene'][label].std().item():.4f}")
                print(f"  Original values present: {'_original' in batch['gene']}")
        break

    # Clean up
    dataset.close_lmdb()
    print("\nTest completed successfully.")


def main_transform_categorical():
    # Used this in hetero gnn pool when converting categorical to regression
    """Test the label binning transforms on the dataset with proper initialization."""
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    from dotenv import load_dotenv
    from torch_geometric.transforms import Compose

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.neo4j_cell import Neo4jCellDataset, SubgraphRepresentation
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.transforms.regression_to_classification import (
        LabelBinningTransform,
        LabelNormalizationTransform,
    )

    # Dataset setup code unchanged...
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    graphs = {"physical": graph.G_physical, "regulatory": graph.G_regulatory}
    node_embeddings = {
        "fudt_3prime": fudt_3prime_dataset,
        "fudt_5prime": fudt_5prime_dataset,
    }

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Configure transforms
    norm_configs = {
        "fitness": {"strategy": "minmax"},
        "gene_interaction": {"strategy": "minmax"},
    }

    bin_configs = {
        "fitness": {
            "num_bins": 10,
            "strategy": "equal_width",
            "store_continuous": True,
            "sigma": 0.1,
            "label_type": "soft",
        },
        "gene_interaction": {
            "num_bins": 5,
            "strategy": "equal_frequency",
            "store_continuous": True,
            "label_type": "ordinal",
        },
    }

    # Create transforms and compose them
    normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    binning_transform = LabelBinningTransform(dataset, bin_configs)
    transform = Compose([normalize_transform, binning_transform])

    # Apply transform to dataset
    dataset.transform = transform

    # Test transforms
    test_indices = [10, 100, 1000]
    print("\nTesting transforms...")
    for idx in test_indices:
        print(f"\nSample {idx}:")
        data = dataset[idx]  # This will apply the composed transform

        # Get original data (without transform)
        dataset.transform = None
        original_data = dataset[idx]

        # Restore transform
        dataset.transform = transform

        # Print results for each label
        for label in norm_configs.keys():
            print(f"\n{label}:")
            print(f"Original:     {original_data['gene'][label].item():.4f}")
            print(f"Normalized:   {data['gene'][f'{label}_continuous'].item():.4f}")
            print(f"Original (stored): {data['gene'][f'{label}_original'].item():.4f}")

            if bin_configs[label]["label_type"] == "soft":
                print(f"Soft labels shape: {data['gene'][label].shape}")
                print(f"Soft labels sum:   {data['gene'][label].sum().item():.4f}")
            elif bin_configs[label]["label_type"] == "ordinal":
                print(f"Ordinal labels shape: {data['gene'][label].shape}")
                print(f"Ordinal values:      {data['gene'][label].numpy()}")

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Label Distributions: Original, Normalized, and Binned", fontsize=16)

    # Sample data for visualization
    sample_size = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

    # Get original distribution data
    dataset.transform = None
    original_data = {label: [] for label in norm_configs.keys()}
    for idx in sample_indices:
        data = dataset[idx]
        for label in norm_configs.keys():
            if not torch.isnan(data["gene"][label]).any():
                original_data[label].append(data["gene"][label].item())

    # Restore transform and get transformed data
    dataset.transform = transform
    transformed_data = {
        label: {"normalized": [], "binned": []} for label in norm_configs.keys()
    }

    for idx in sample_indices:
        data = dataset[idx]
        for label in norm_configs.keys():
            if not torch.isnan(data["gene"][f"{label}_continuous"]).any():
                transformed_data[label]["normalized"].append(
                    data["gene"][f"{label}_continuous"].item()
                )
                transformed_data[label]["binned"].append(data["gene"][label].numpy())

    # Plot distributions
    for i, label in enumerate(norm_configs.keys()):
        # Original distribution
        sns.histplot(original_data[label], bins=50, ax=axes[0, i], stat="density")
        axes[0, i].set_title(f"Original {label}")

        # Normalized distribution
        sns.histplot(
            transformed_data[label]["normalized"],
            bins=50,
            ax=axes[1, i],
            stat="density",
        )
        axes[1, i].set_title(f"Normalized {label}")

        # Binned distribution
        binned = np.array(transformed_data[label]["binned"])
        if bin_configs[label]["label_type"] == "soft":
            mean_soft = np.mean(binned, axis=0)
            axes[2, i].bar(range(len(mean_soft)), mean_soft)
            axes[2, i].set_title(f"Mean Soft Labels {label}")
        else:
            mean_ordinal = np.mean(binned, axis=0)
            axes[2, i].bar(range(len(mean_ordinal)), mean_ordinal)
            axes[2, i].set_title(f"Mean Ordinal Values {label}")

    plt.tight_layout()
    plt.show()

    dataset.close_lmdb()


def main_transform_categorical_dense():
    """Test label transforms and dense conversion with perturbation subset."""
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from dotenv import load_dotenv
    from torch_geometric.transforms import Compose

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.neo4j_cell import Neo4jCellDataset, SubgraphRepresentation
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.transforms.regression_to_classification import (
        InverseCompose,
        LabelBinningTransform,
        LabelNormalizationTransform,
    )

    # Setup dataset (unchanged)
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    # Dataset setup
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    graphs = {"physical": graph.G_physical, "regulatory": graph.G_regulatory}
    node_embeddings = {
        "fudt_3prime": fudt_3prime_dataset,
        "fudt_5prime": fudt_5prime_dataset,
    }

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    # First create dataset without transforms
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Configure transforms
    norm_configs = {
        "fitness": {"strategy": "minmax"},
        "gene_interaction": {"strategy": "minmax"},
    }

    bin_configs = {
        "fitness": {
            "num_bins": 32,
            "strategy": "equal_frequency",
            "store_continuous": True,
            # "sigma": 0.1,
            "label_type": "soft",
        },
        "gene_interaction": {
            "num_bins": 32,
            "strategy": "equal_frequency",
            "store_continuous": True,
            # "sigma": 0.1,
            "label_type": "soft",
            # "num_bins": 5,
            # "strategy": "equal_frequency",
            # "store_continuous": True,
            # "label_type": "ordinal",
        },
    }

    # Create transforms and compose them with dataset stats
    normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    binning_transform = LabelBinningTransform(dataset, bin_configs, normalize_transform)
    # TODO will need to implement inverse maybe?
    # dense_transform = HeteroToDense({"gene": len(genome.gene_set)})

    # Apply transforms to dataset
    forward_transform = Compose([normalize_transform, binning_transform])
    inverse_transform = InverseCompose(forward_transform)

    # I want to be able to do this
    dataset.transform = forward_transform

    # Create base data module
    base_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=32,
        random_seed=42,
        num_workers=2,
        pin_memory=True,
    )
    base_data_module.setup()

    # Create perturbation subset module
    subset_size = 10000
    subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=base_data_module,
        size=subset_size,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        prefetch=False,
        seed=42,
        dense=True,  # Important for dense format
    )
    subset_data_module.setup()

    # Test transforms on subset
    print("\nTesting transforms on subset...")
    train_loader = subset_data_module.train_dataloader()
    batch = next(iter(train_loader))

    print("\nBatch structure:")
    print(f"Batch keys: {batch.keys}")
    print(f"\nNode features shape: {batch['gene'].x.shape}")
    print(f"Adjacency matrix shapes:")
    # print(f"Physical: {batch['gene', 'physical_interaction', 'gene'].adj.shape}")
    # print(f"Regulatory: {batch['gene', 'regulatory_interaction', 'gene'].adj.shape}")

    # Check label shapes and values
    print("\nLabel information:")
    for label in norm_configs.keys():
        print(f"\n{label}:")
        if bin_configs[label]["label_type"] == "soft":
            print(f"Soft label shape: {batch['gene'][label].shape}")
            # Handle potentially extra dimensions in soft labels
            soft_labels = batch["gene"][label].squeeze()
            if soft_labels.dim() == 3:  # If [batch, 1, num_classes]
                soft_labels = soft_labels.squeeze(1)
            print(f"Soft label sums (first 5):")
            print(soft_labels.sum(dim=-1)[:5])  # Sum over classes
        else:
            print(f"Ordinal label shape: {batch['gene'][label].shape}")
            ordinal_labels = batch["gene"][label].squeeze()
            if ordinal_labels.dim() == 3:  # If [batch, 1, num_thresholds]
                ordinal_labels = ordinal_labels.squeeze(1)
            print(f"Ordinal values (first 5):")
            print(ordinal_labels[:5])

        # Handle continuous and original values
        cont_vals = batch["gene"][f"{label}_continuous"].squeeze()
        orig_vals = batch["gene"][f"{label}_original"].squeeze()
        print(f"Continuous values (first 5): {cont_vals[:5]}")
        print(f"Original values (first 5): {orig_vals[:5]}")

    # Create visualization of distributions in batch
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Label Distributions in Dense Batch", fontsize=16)

    for i, label in enumerate(norm_configs.keys()):
        # Original values
        orig_values = batch["gene"][f"{label}_original"].squeeze().cpu().numpy()
        valid_mask = ~np.isnan(orig_values)
        orig_values = orig_values[valid_mask]
        sns.histplot(orig_values, bins=50, ax=axes[0, i], stat="density")
        axes[0, i].set_title(f"Original {label}")

        # Transformed values
        if bin_configs[label]["label_type"] == "soft":
            soft_labels = batch["gene"][label].squeeze()
            if soft_labels.dim() == 3:
                soft_labels = soft_labels.squeeze(1)
            mean_soft = soft_labels.mean(dim=0).cpu().numpy()
            axes[1, i].bar(range(len(mean_soft)), mean_soft)
            axes[1, i].set_title(f"Mean Soft Labels {label}")
            axes[1, i].set_xlabel("Class")
            axes[1, i].set_ylabel("Mean Probability")
        else:
            ordinal_labels = batch["gene"][label].squeeze()
            if ordinal_labels.dim() == 3:
                ordinal_labels = ordinal_labels.squeeze(1)
            mean_ordinal = ordinal_labels.mean(dim=0).cpu().numpy()
            axes[1, i].bar(range(len(mean_ordinal)), mean_ordinal)
            axes[1, i].set_title(f"Mean Ordinal Values {label}")
            axes[1, i].set_xlabel("Threshold")
            axes[1, i].set_ylabel("Mean Value")

    plt.tight_layout()
    plt.show()

    dataset.close_lmdb()


if __name__ == "__main__":
    # main_transform_categorical_dense()
    # main()
    # main_incidence()
    main_transform_standardization()
