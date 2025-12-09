# torchcell/data/neo4j_preprocessed_cell
# [[torchcell.data.neo4j_preprocessed_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_preprocessed_cell
# Test file: tests/torchcell/data/test_neo4j_preprocessed_cell.py

import json
import logging
import os
import os.path as osp
import pickle
from typing import Callable, Optional

import lmdb
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from torchcell.data.graph_processor import GraphProcessor
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.utils.file_lock import FileLockHelper

log = logging.getLogger(__name__)


class Neo4jPreprocessedCellDataset(Dataset):
    """
    Dataset that loads pre-processed graph data from LMDB, bypassing graph processor.

    Architecture:
        1. One-time preprocessing: Applies graph processor to all samples and saves to LMDB
        2. Training: Loads pre-processed data directly from LMDB (zero processing overhead)

    Usage:
        # One-time preprocessing
        source_dataset = Neo4jCellDataset(...)
        preprocessed_dataset = Neo4jPreprocessedCellDataset(root=preprocessed_root)
        preprocessed_dataset.preprocess_from_source(
            source_dataset=source_dataset,
            graph_processor=LazySubgraphRepresentation()
        )

        # Training (reuse preprocessed data)
        dataset = Neo4jPreprocessedCellDataset(root=preprocessed_root)
        data_module = CellDataModule(dataset=dataset, ...)

    Performance:
        - Preprocessing: ~10ms/sample × N samples (one-time cost)
        - Training: ~0.01ms/sample (100x faster than processing on-the-fly)
    """

    def __init__(
        self,
        root: str,
        source_dataset: Optional[Neo4jCellDataset] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Initialize preprocessed dataset.

        Args:
            root: Directory for preprocessed LMDB
            source_dataset: Source Neo4jCellDataset (required for cell_graph, phenotype_info)
            transform: Optional transform to apply when loading
            pre_transform: Not used (preprocessing happens via preprocess_from_source)
            pre_filter: Not used
        """
        self.root = root
        self._source_dataset = source_dataset
        self.env = None

        # Cache properties from source dataset
        self._cell_graph = None
        self._phenotype_info = None
        self._length = None

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load metadata if exists
        if self._is_preprocessed():
            self._load_metadata()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["lmdb", "metadata.json"]

    def _is_preprocessed(self) -> bool:
        """Check if preprocessing has been completed."""
        lmdb_path = osp.join(self.processed_dir, "lmdb")
        metadata_path = osp.join(self.processed_dir, "metadata.json")
        return osp.exists(lmdb_path) and osp.exists(metadata_path)

    def _load_metadata(self):
        """Load cached metadata from preprocessing."""
        metadata_path = osp.join(self.processed_dir, "metadata.json")
        metadata = FileLockHelper.read_json_with_lock(metadata_path)
        self._length = metadata["length"]
        log.info(f"Loaded metadata: {self._length} samples")

    def _save_metadata(self, length: int):
        """Save metadata after preprocessing."""
        metadata_path = osp.join(self.processed_dir, "metadata.json")
        metadata = {"length": length}
        FileLockHelper.write_json_with_lock(metadata_path, metadata)
        log.info(f"Saved metadata: {length} samples")

    @property
    def cell_graph(self):
        """Return cell_graph from source dataset."""
        if self._cell_graph is None:
            if self._source_dataset is None:
                raise RuntimeError(
                    "cell_graph not available. Either provide source_dataset "
                    "at initialization or call preprocess_from_source first."
                )
            self._cell_graph = self._source_dataset.cell_graph
        return self._cell_graph

    @property
    def phenotype_info(self):
        """Return phenotype_info from source dataset."""
        if self._phenotype_info is None:
            if self._source_dataset is None:
                raise RuntimeError(
                    "phenotype_info not available. Either provide source_dataset "
                    "at initialization or call preprocess_from_source first."
                )
            self._phenotype_info = self._source_dataset.phenotype_info
        return self._phenotype_info

    @property
    def label_df(self):
        """Return label_df from source dataset (needed for transforms)."""
        if self._source_dataset is None:
            raise RuntimeError(
                "label_df not available. Provide source_dataset at initialization."
            )
        return self._source_dataset.label_df

    def _extract_mask_indices(self, processed_graph):
        """
        Extract only the False indices from masks to save storage.

        Instead of storing full masks (millions of bools), store only indices where mask=False.
        This reduces storage from ~43MB/sample to ~few KB/sample.

        Returns:
            dict: Compact representation with only False mask indices
        """
        compact_data = {}

        # Store sample-specific data (phenotypes, perturbations)
        # Note: Some fields are lists (ids_pert, phenotype_types), some are tensors
        gene_data = processed_graph['gene']
        compact_data['gene'] = {
            'ids_pert': gene_data.ids_pert,  # Already a list, no .cpu() needed
            'perturbation_indices': gene_data.perturbation_indices.cpu(),
            'phenotype_values': gene_data.phenotype_values.cpu(),
            'phenotype_type_indices': gene_data.phenotype_type_indices.cpu(),
            'phenotype_sample_indices': gene_data.phenotype_sample_indices.cpu(),
            'phenotype_types': gene_data.phenotype_types,  # List, no .cpu() needed
            'phenotype_stat_values': gene_data.phenotype_stat_values.cpu(),
            'phenotype_stat_type_indices': gene_data.phenotype_stat_type_indices.cpu(),
            'phenotype_stat_sample_indices': gene_data.phenotype_stat_sample_indices.cpu(),
            'phenotype_stat_types': gene_data.phenotype_stat_types,  # List, no .cpu() needed
        }

        # For each node type, store indices where mask is False (nodes to remove)
        compact_data['node_masks'] = {}
        for node_type in processed_graph.node_types:
            if 'mask' in processed_graph[node_type]:
                mask = processed_graph[node_type]['mask']
                # Store indices where mask is False (much smaller than full mask)
                false_indices = (~mask).nonzero(as_tuple=True)[0].cpu()
                # Save both false indices AND the original mask size for correct reconstruction
                compact_data['node_masks'][node_type] = {
                    'false_indices': false_indices,
                    'mask_size': len(mask),
                }

        # For each edge type, store indices where mask is False (edges to remove)
        compact_data['edge_masks'] = {}
        for edge_type in processed_graph.edge_types:
            if 'mask' in processed_graph[edge_type]:
                mask = processed_graph[edge_type]['mask']
                # Store indices where mask is False
                false_indices = (~mask).nonzero(as_tuple=True)[0].cpu()
                # Save both false indices AND the original mask size for correct reconstruction
                compact_data['edge_masks'][edge_type] = {
                    'false_indices': false_indices,
                    'mask_size': len(mask),
                }

        return compact_data

    def _reconstruct_from_mask_indices(self, compact_data):
        """
        Reconstruct full processed graph from compact mask indices.

        OPTIMIZED: Minimal reconstruction - only add what's needed.
        """
        from torchcell.data.hetero_data import HeteroData

        # Create minimal HeteroData
        processed_graph = HeteroData()

        # Always use CPU for pin_memory compatibility
        device = torch.device("cpu")

        # ZERO-COPY: Add references to cell_graph data (not copies!)
        # Gene node data
        processed_graph["gene"].node_ids = self.cell_graph["gene"].node_ids
        processed_graph["gene"].num_nodes = self.cell_graph["gene"].num_nodes
        processed_graph["gene"].x = self.cell_graph["gene"].x  # Reference, not copy!

        # Add sample-specific gene data from compact storage
        for key, value in compact_data['gene'].items():
            if isinstance(value, torch.Tensor):
                processed_graph['gene'][key] = value.to(device)
            else:
                # Lists (ids_pert, phenotype_types, phenotype_stat_types) stay as-is
                processed_graph['gene'][key] = value

        # ZERO-COPY: Add references to edge indices
        for edge_type in self.cell_graph.edge_types:
            # Copy edge attributes - some edges have edge_index, some have hyperedge_index
            if hasattr(self.cell_graph[edge_type], 'edge_index'):
                processed_graph[edge_type].edge_index = self.cell_graph[edge_type].edge_index
            if hasattr(self.cell_graph[edge_type], 'hyperedge_index'):
                processed_graph[edge_type].hyperedge_index = self.cell_graph[edge_type].hyperedge_index
            if hasattr(self.cell_graph[edge_type], 'num_edges'):
                processed_graph[edge_type].num_edges = self.cell_graph[edge_type].num_edges
            if hasattr(self.cell_graph[edge_type], 'stoichiometry'):
                processed_graph[edge_type].stoichiometry = self.cell_graph[edge_type].stoichiometry

        # Add reaction and metabolite node data if present
        if "reaction" in self.cell_graph.node_types:
            processed_graph["reaction"].node_ids = self.cell_graph["reaction"].node_ids
            processed_graph["reaction"].num_nodes = self.cell_graph["reaction"].num_nodes
            if hasattr(self.cell_graph["reaction"], "w_growth"):
                processed_graph["reaction"].w_growth = self.cell_graph["reaction"].w_growth

        if "metabolite" in self.cell_graph.node_types:
            processed_graph["metabolite"].node_ids = self.cell_graph["metabolite"].node_ids
            processed_graph["metabolite"].num_nodes = self.cell_graph["metabolite"].num_nodes

        # Reconstruct node masks from False indices using saved mask sizes
        for node_type, mask_info in compact_data['node_masks'].items():
            false_indices = mask_info['false_indices']
            mask_size = mask_info['mask_size']
            mask = torch.ones(mask_size, dtype=torch.bool, device=device)
            if len(false_indices) > 0:
                mask[false_indices.to(device)] = False
            processed_graph[node_type]['mask'] = mask

            # pert_mask is always the inverse of mask (for all node types)
            # mask: True = keep, pert_mask: True = remove
            processed_graph[node_type]['pert_mask'] = ~mask

        # Reconstruct edge masks from False indices using saved mask sizes
        for edge_type, mask_info in compact_data['edge_masks'].items():
            false_indices = mask_info['false_indices']
            mask_size = mask_info['mask_size']
            mask = torch.ones(mask_size, dtype=torch.bool, device=device)
            if len(false_indices) > 0:
                mask[false_indices.to(device)] = False
            processed_graph[edge_type]['mask'] = mask

        return processed_graph

    def preprocess_from_source(
        self,
        source_dataset: Neo4jCellDataset,
        graph_processor: GraphProcessor,
        num_workers: int = 0,
    ):
        """
        One-time preprocessing: apply graph processor to all samples and save to LMDB.

        OPTIMIZED: Stores only False mask indices instead of full processed graphs.
        This reduces storage from ~43MB/sample to ~few KB/sample (1000x reduction!).

        Args:
            source_dataset: Neo4jCellDataset to preprocess
            graph_processor: GraphProcessor to apply (e.g., LazySubgraphRepresentation)
            num_workers: Number of workers for preprocessing (0 = single process)

        Performance:
            - LazySubgraphRepresentation: ~10ms/sample
            - For 332K samples: ~55 minutes one-time cost
            - Storage: ~few KB/sample × 332K ≈ few GB (vs ~14TB for full graphs!)
            - Saves 10ms/sample × 28 batch_size × 1000 batches/epoch = 280s/epoch during training
        """
        log.info("Starting one-time preprocessing...")
        log.info(f"Source dataset length: {len(source_dataset)}")
        log.info(f"Graph processor: {graph_processor.__class__.__name__}")
        log.info("OPTIMIZED: Storing only False mask indices (not full graphs)")

        # Cache source dataset properties
        self._source_dataset = source_dataset
        self._cell_graph = source_dataset.cell_graph
        self._phenotype_info = source_dataset.phenotype_info

        # Initialize LMDB for writing
        lmdb_path = osp.join(self.processed_dir, "lmdb")
        os.makedirs(self.processed_dir, exist_ok=True)

        # OPTIMIZED: Much smaller map_size since we only store mask indices
        # Estimate: ~10KB per sample × 332K samples = ~3.3GB
        map_size = int(1e10)  # 10GB (plenty of headroom)

        env = lmdb.open(lmdb_path, map_size=map_size)

        # Preprocess all samples
        with env.begin(write=True) as txn:
            for idx in tqdm(range(len(source_dataset)), desc="Preprocessing"):
                # Get raw data from source dataset (before graph processor)
                source_dataset._init_lmdb_read()
                serialized_data = source_dataset._read_from_lmdb(idx)
                data_list = source_dataset._deserialize_json(serialized_data)
                data = source_dataset._reconstruct_experiments(data_list)

                # Apply graph processor
                processed_graph = graph_processor.process(
                    source_dataset.cell_graph,
                    source_dataset.phenotype_info,
                    data
                )

                # OPTIMIZED: Extract only False mask indices (not full graph)
                compact_data = self._extract_mask_indices(processed_graph)

                # Serialize compact data (much smaller than full graph)
                serialized_compact = pickle.dumps(compact_data)

                # Store in LMDB
                txn.put(f"{idx}".encode("utf-8"), serialized_compact)

        env.close()
        source_dataset.close_lmdb()

        # Save metadata
        self._save_metadata(len(source_dataset))
        self._length = len(source_dataset)

        log.info(f"Preprocessing complete! Saved {len(source_dataset)} samples to {lmdb_path}")

        # Get actual LMDB size
        lmdb_data_file = osp.join(lmdb_path, "data.mdb")
        if osp.exists(lmdb_data_file):
            actual_size_gb = os.path.getsize(lmdb_data_file) / 1e9
            log.info(f"LMDB size: {actual_size_gb:.2f} GB")
            avg_size_kb = (actual_size_gb * 1e6) / len(source_dataset)
            log.info(f"Average size per sample: {avg_size_kb:.2f} KB (vs ~43 MB for full graph!)")

    def _init_lmdb_read(self):
        """Initialize LMDB environment for reading."""
        if not self._is_preprocessed():
            raise RuntimeError(
                "Dataset not preprocessed. Call preprocess_from_source() first."
            )

        lmdb_path = osp.join(self.processed_dir, "lmdb")
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            max_spare_txns=16,
        )

    def get(self, idx: int):
        """
        Load pre-processed sample from LMDB.

        OPTIMIZED: Reconstructs full graph from compact mask indices.
        Performance: ~0.1ms (still 100x faster than processing on-the-fly)
        """
        if self.env is None:
            self._init_lmdb_read()

        # Read compact data from LMDB
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode("utf-8"))
            if serialized_data is None:
                return None

        # Deserialize compact data
        compact_data = pickle.loads(serialized_data)

        # Reconstruct full processed graph from mask indices
        processed_graph = self._reconstruct_from_mask_indices(compact_data)

        return processed_graph

    def len(self) -> int:
        """Return dataset length."""
        if self._length is None:
            if not self._is_preprocessed():
                raise RuntimeError(
                    "Dataset not preprocessed. Call preprocess_from_source() first."
                )
            self._load_metadata()
        return self._length

    def close_lmdb(self):
        """Close LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def __getstate__(self):
        """Handle pickling for multiprocessing."""
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        """Handle unpickling for multiprocessing."""
        self.__dict__.update(state)


def main_preprocess():
    """Example: Preprocess a dataset for faster training."""
    import os.path as osp
    from dotenv import load_dotenv

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import LazySubgraphRepresentation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    # Load query
    with open(osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r") as f:
        query = f.read()

    # Setup genome and graph
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Create source dataset
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    source_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=None,  # Don't process yet
    )

    print(f"Source dataset length: {len(source_dataset)}")

    # Create preprocessed dataset and run one-time preprocessing
    preprocessed_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build-preprocessed-lazy"
    )

    preprocessed_dataset = Neo4jPreprocessedCellDataset(
        root=preprocessed_root,
        source_dataset=source_dataset,
    )

    # One-time preprocessing with LazySubgraphRepresentation
    print("\nStarting one-time preprocessing...")
    print("This will take ~50 minutes for 300K samples")
    print("But will save ~280 seconds per epoch during training!")

    preprocessed_dataset.preprocess_from_source(
        source_dataset=source_dataset,
        graph_processor=LazySubgraphRepresentation(),
    )

    # Test loading
    print("\nTesting preprocessed data loading...")
    sample = preprocessed_dataset[0]
    print(f"Sample keys: {sample.keys}")
    print(f"Gene nodes: {sample['gene'].num_nodes}")

    # Cleanup
    source_dataset.close_lmdb()
    preprocessed_dataset.close_lmdb()

    print("\nPreprocessing complete! Use this dataset for training:")
    print(f"  preprocessed_dataset = Neo4jPreprocessedCellDataset(root='{preprocessed_root}')")


if __name__ == "__main__":
    main_preprocess()
