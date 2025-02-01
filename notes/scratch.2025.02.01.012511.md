---
id: m0q8t6lcir70fj78jhfmrsu
title: '012511'
desc: ''
updated: 1738395886374
created: 1738394714327
---
You are trying to to help me implement an updated version of `PerturbationSubsetDataModule`. I want to add an argument `gene_subset`, this will be a gene subset so import `from torchcell.sequence import GeneSet`.

```python
# torchcell/datamodules/perturbation_subset
# [[torchcell.datamodules.perturbation_subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/perturbation_subset
# Test file: tests/torchcell/datamodules/test_perturbation_subset.py

import os
import os.path as osp
import json
import random
from typing import Optional
import lightning as L
from torch.utils.data import Subset
from torchcell.utils import format_scientific_notation
from torchcell.datamodules import (
    IndexSplit,
    DatasetSplit,
    DataModuleIndex,
    DataModuleIndexDetails,
)
from torch_geometric.loader import DataLoader, PrefetchLoader
import torch
from torch_geometric.loader import DenseDataLoader

from torchcell.loader.dense_padding_data_loader import DensePaddingDataLoader


class PerturbationSubsetDataModule(L.LightningDataModule):
    def __init__(
        self,
        cell_data_module,
        size: int,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch: bool = False,
        seed: int = 42,
        dense: bool = False,
    ):
        super().__init__()
        self.cell_data_module = cell_data_module
        self.dataset = cell_data_module.dataset
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.seed = seed
        self.dense = dense

        self.cache_dir = self.cell_data_module.cache_dir
        self.subset_dir = osp.join(
            self.cache_dir, f"perturbation_subset_{format_scientific_notation(size)}"
        )
        os.makedirs(self.subset_dir, exist_ok=True)

        random.seed(self.seed)
        self._index = None
        self._index_details = None

    @property
    def index(self) -> DataModuleIndex:
        if self._index is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index

    @property
    def index_details(self) -> DataModuleIndexDetails:
        if self._index_details is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index_details

    def _load_or_compute_index(self):
        index_file = osp.join(
            self.subset_dir,
            f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        details_file = osp.join(
            self.subset_dir,
            f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )

        if osp.exists(index_file) and osp.exists(details_file):
            with open(index_file, "r") as f:
                self._index = DataModuleIndex(**json.load(f))
            with open(details_file, "r") as f:
                self._index_details = DataModuleIndexDetails(**json.load(f))
        else:
            self._create_subset()
            self._save_index()

    def _create_subset(self):
        cell_index_details = self.cell_data_module.index_details
        cell_index = self.cell_data_module.index

        # Calculate original split ratios
        total_samples = sum(
            len(getattr(cell_index, split)) for split in ["train", "val", "test"]
        )
        original_ratios = {
            split: len(getattr(cell_index, split)) / total_samples
            for split in ["train", "val", "test"]
        }

        # Calculate target sizes for each split
        target_sizes = {
            split: int(self.size * ratio) for split, ratio in original_ratios.items()
        }

        # Adjust for rounding errors to ensure we get exactly self.size samples
        difference = self.size - sum(target_sizes.values())
        target_sizes["train"] += difference  # Add any difference to the train set

        selected_indices = {split: [] for split in ["train", "val", "test"]}

        for split in ["train", "val", "test"]:
            pert_count_index = getattr(
                cell_index_details, split
            ).perturbation_count_index
            remaining_size = target_sizes[split]

            # First, select all perturbation count 1 data
            single_pert_indices = pert_count_index[1].indices
            selected_indices[split].extend(single_pert_indices[:remaining_size])
            remaining_size -= len(selected_indices[split])

            if remaining_size > 0:
                # Equally sample from other perturbation levels
                other_pert_levels = [
                    level for level in pert_count_index.keys() if level != 1
                ]
                while remaining_size > 0 and other_pert_levels:
                    samples_per_level = max(1, remaining_size // len(other_pert_levels))
                    for level in other_pert_levels:
                        available_indices = set(pert_count_index[level].indices) - set(
                            selected_indices[split]
                        )
                        sampled = random.sample(
                            list(available_indices),
                            min(samples_per_level, len(available_indices)),
                        )
                        selected_indices[split].extend(sampled)
                        remaining_size -= len(sampled)
                        if remaining_size <= 0:
                            break
                    other_pert_levels = [
                        level
                        for level in other_pert_levels
                        if set(pert_count_index[level].indices)
                        - set(selected_indices[split])
                    ]

        self._index = DataModuleIndex(
            train=sorted(selected_indices["train"]),
            val=sorted(selected_indices["val"]),
            test=sorted(selected_indices["test"]),
        )
        self._create_index_details()

        # Verify total size
        total_selected = sum(len(indices) for indices in selected_indices.values())
        assert (
            total_selected == self.size
        ), f"Expected {self.size} samples, but got {total_selected}"

    def _create_index_details(self):
        cell_index_details = self.cell_data_module.index_details
        methods = cell_index_details.methods

        self._index_details = DataModuleIndexDetails(
            methods=methods,
            train=self._create_dataset_split(
                self._index.train, cell_index_details.train, methods
            ),
            val=self._create_dataset_split(
                self._index.val, cell_index_details.val, methods
            ),
            test=self._create_dataset_split(
                self._index.test, cell_index_details.test, methods
            ),
        )

    def _create_dataset_split(
        self, indices: list[int], cell_split: DatasetSplit, methods: list[str]
    ) -> DatasetSplit:
        dataset_split = DatasetSplit()
        for method in methods:
            split_data = {}
            method_data = getattr(cell_split, method)
            if method_data is not None:
                for key, index_split in method_data.items():
                    if method == "perturbation_count_index":
                        key = int(
                            key
                        )  # Convert key to int for perturbation_count_index
                    intersect = sorted(list(set(indices) & set(index_split.indices)))
                    split_data[key] = IndexSplit(
                        indices=intersect, count=len(intersect)
                    )
                setattr(dataset_split, method, split_data)
        return dataset_split

    def _save_index(self):
        with open(
            osp.join(
                self.subset_dir,
                f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
            ),
            "w",
        ) as f:
            json.dump(self._index.dict(), f, indent=2)
        with open(
            osp.join(
                self.subset_dir,
                f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
            ),
            "w",
        ) as f:
            json.dump(self._index_details.dict(), f, indent=2)

    def _cached_files_exist(self):
        index_file = osp.join(
            self.subset_dir,
            f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        details_file = osp.join(
            self.subset_dir,
            f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        return osp.exists(index_file) and osp.exists(details_file)

    def setup(self, stage: Optional[str] = None):
        print("Setting up PerturbationSubsetDataModule...")

        if (
            self._index is None
            or self._index_details is None
            or not self._cached_files_exist()
        ):
            self._load_or_compute_index()

        print("Creating subset datasets...")
        self.train_dataset = Subset(self.dataset, self.index.train)
        self.val_dataset = Subset(self.dataset, self.index.val)
        self.test_dataset = Subset(self.dataset, self.index.test)
        print("Setup complete.")

    def _get_dataloader(self, dataset, shuffle=False):
        if self.dense:
            # loader = DenseDataLoader(
            #     dataset,
            #     batch_size=self.batch_size,
            #     shuffle=shuffle,
            #     num_workers=self.num_workers,
            #     pin_memory=self.pin_memory,
            #     # follow_batch=["x", "x_pert"],
            # )
            loader = DensePaddingDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                follow_batch=["x", "x_pert"],
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                follow_batch=["x", "x_pert"],
            )
        if self.prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        return loader

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def all_dataloader(self):
        return self._get_dataloader(self.dataset)

    def test_cell_module_dataloader(self):
        return self._get_dataloader(
            Subset(self.dataset, self.cell_data_module.index.test)
        )
```

This script is what we ultimately want to create but for the metabolism gene subset. Notice how we have multiple sizes. We should be able to loop over these. Also we don't need to plot for dataset only over new metabolism limited perturbation subsets. We need a script to test the new gene subsetting and to make subsets and plot relevant results.

```python
# experiments/003-fit-int/scripts/create_cached_perturbation_subset_modules_and_plot
# [[experiments.003-fit-int.scripts.create_cached_perturbation_subset_modules_and_plot]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/create_cached_perturbation_subset_modules_and_plot
# Test file: experiments/003-fit-int/scripts/test_create_cached_perturbation_subset_modules_and_plot.py

import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
import json
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import SubgraphRepresentation
from tqdm import tqdm
from torchcell.viz.datamodules import plot_dataset_index_split
from torchcell.datamodules.cell import overlap_dataset_index_split
from torchcell.utils import format_scientific_notation


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

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
    # Data module testing

    print(dataset[2])
    dataset.close_lmdb()

    seed = 42
    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=seed,
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        break

    exp_name = "experiments-003"
    query_name = "query-001-small-build"
    dm_name = "cell-data-module"

    ## Cell Data Module - Dataset Index Plotting - Start

    ## Dataset Index Plotting - Start
    size_str = str(len(dataset))
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=cell_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=cell_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.perturbation_count_index,
        data_module_index=cell_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        threshold=0.0,
    )
    # ## Cell Data Module - Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 5e1 Module
    size = 5e1
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.perturbation_count_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        threshold=0.0,
    )
    ## Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 1e2 Module
    size = 1e2
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    ## Dataset Index Plotting - Start
    size_str = format_scientific_notation(size)
    # dataset name index
    ds_index = "dataset-name-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.dataset_name_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # phenotype label index
    ds_index = "phenotype-label-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.phenotype_label_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
    )
    # perturbation count index
    ds_index = "perturbation-count-index"
    title = f"{exp_name}_{query_name}_{dm_name}_size_{size_str}_seed_{seed}_{ds_index}"
    split_index = overlap_dataset_index_split(
        dataset_index=dataset.perturbation_count_index,
        data_module_index=perturbation_subset_data_module.index,
    )
    plot_dataset_index_split(
        split_index=split_index,
        title=title,
        save_path=osp.join(ASSET_IMAGES_DIR, f"{title}.png"),
        threshold=0.0,
    )
    ## Dataset Index Plotting - End
    
    ## Subset
    dm_name = "perturbation-subset-data-module"
    # 5e2 Module
    size = 5e2
```

The way we want to subset is by using the `is_any_perturbed_gene_index`. This is an index where `genes:str` are keys and values are list[int] which are the indices.

```python
# torchcell/data/neo4j_cell
# [[torchcell.data.neo4j_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_cell
# Test file: tests/torchcell/data/test_neo4j_cell.py
import torch
import json
import logging
import os
import os.path as osp
from collections.abc import Callable
import lmdb
import pandas as pd
import networkx as nx
import hypernetx as hnx
import numpy as np
from typing import Any
from pydantic import field_validator
from tqdm import tqdm
from torchcell.data.embedding import BaseEmbeddingDataset
from torch_geometric.data import Dataset
from torch_geometric.utils import add_remaining_self_loops

# from torch_geometric.data import HeteroData
from torchcell.data.hetero_data import HeteroData
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datamodels import Converter
from torchcell.data.deduplicate import Deduplicator
from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datamodels import (
    ExperimentType,
    ExperimentReferenceType,
    PhenotypeType,
    EXPERIMENT_TYPE_MAP,
    EXPERIMENT_REFERENCE_TYPE_MAP,
)
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from typing import Type, Optional
from enum import Enum, auto
from abc import ABC, abstractmethod

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


# @profile
def create_embedding_graph(
    gene_set: GeneSet, embeddings: BaseEmbeddingDataset
) -> nx.Graph:
    """
    Create a NetworkX graph from embeddings.
    """
    # Create an empty NetworkX graph
    G = nx.Graph()

    # Extract and concatenate embeddings for all items in embeddings
    for item in embeddings:
        keys = item["embeddings"].keys()
        if item.id in gene_set:
            item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
            concatenated_embedding = torch.cat(item_embeddings)

            G.add_node(item.id, embedding=concatenated_embedding)

    return G


# @profile
# TODO we could remove is_add_remaining_self_loops and put it in transforms
def to_cell_data(
    graphs: dict[str, nx.Graph],
    incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] = None,
    is_add_remaining_self_loops: bool = True,
) -> HeteroData:
    """Convert networkx graphs and incidence graphs to HeteroData format."""
    hetero_data = HeteroData()

    # Base nodes setup
    base_nodes_list = sorted(list(graphs["base"].nodes()))
    node_idx_mapping = {node: idx for idx, node in enumerate(base_nodes_list)}
    num_nodes = len(base_nodes_list)

    # Initialize gene attributes
    hetero_data["gene"].num_nodes = num_nodes
    hetero_data["gene"].node_ids = base_nodes_list
    hetero_data["gene"].x = torch.zeros((num_nodes, 0), dtype=torch.float)

    # Process each graph for edges and embeddings
    for graph_type, graph in graphs.items():
        if graph.number_of_edges() > 0:
            # Convert edges to tensor
            edge_index = torch.tensor(
                [
                    (node_idx_mapping[src], node_idx_mapping[dst])
                    for src, dst in graph.edges()
                    if src in node_idx_mapping and dst in node_idx_mapping
                ],
                dtype=torch.long,
            ).t()

            # Add interaction edges
            if graph_type != "base":
                edge_type = ("gene", f"{graph_type}_interaction", "gene")
                if is_add_remaining_self_loops:
                    edge_index, _ = add_remaining_self_loops(edge_index)
                hetero_data[edge_type].edge_index = edge_index.cpu()
                hetero_data[edge_type].num_edges = edge_index.size(1)
        else:
            # Process node embeddings
            embeddings = torch.zeros((num_nodes, 0), dtype=torch.float)
            for i, node in enumerate(base_nodes_list):
                if node in graph.nodes and "embedding" in graph.nodes[node]:
                    embedding = graph.nodes[node]["embedding"]
                    if embeddings.shape[1] == 0:
                        embeddings = torch.zeros(
                            (num_nodes, embedding.shape[0]), dtype=torch.float
                        )
                    embeddings[i] = embedding.cpu()  # Ensure CPU tensor
            hetero_data["gene"].x = torch.cat(
                (hetero_data["gene"].x.cpu(), embeddings.cpu()), dim=1
            )

    # Process metabolism hypergraph
    if incidence_graphs is not None and "metabolism" in incidence_graphs:
        hypergraph = incidence_graphs["metabolism"]

        # Get unique metabolites
        metabolites = sorted(
            list({m for edge_id in hypergraph.edges for m in hypergraph.edges[edge_id]})
        )
        metabolite_mapping = {m: idx for idx, m in enumerate(metabolites)}

        hetero_data["metabolite"].num_nodes = len(metabolites)
        hetero_data["metabolite"].node_ids = metabolites

        # Add reaction nodes
        num_reactions = len(hypergraph.edges)
        hetero_data["reaction"].num_nodes = num_reactions
        hetero_data["reaction"].node_ids = list(range(num_reactions))

        # Build indices and coefficients
        node_indices = []
        edge_indices = []
        stoich_coeffs = []
        reaction_to_genes = {}
        reaction_to_genes_indices = {}

        for edge_idx, edge_id in enumerate(hypergraph.edges):
            edge = hypergraph.edges[edge_id]

            # Store gene associations
            if "genes" in edge.properties:
                genes = list(edge.properties["genes"])
                reaction_to_genes[edge_idx] = genes

                # Create gene indices list
                gene_indices = []
                for gene in genes:
                    gene_idx = node_idx_mapping.get(gene, -1)
                    gene_indices.append(gene_idx)
                reaction_to_genes_indices[edge_idx] = gene_indices

            # Process metabolites
            for m in edge:
                node_indices.append(metabolite_mapping[m])
                edge_indices.append(edge_idx)
                stoich_coeffs.append(edge.properties[f"stoich_coefficient-{m}"])

        # Create hyperedge tensors
        hyperedge_index = torch.stack(
            [
                torch.tensor(node_indices, dtype=torch.long),
                torch.tensor(edge_indices, dtype=torch.long),
            ]
        ).cpu()
        stoich_coeffs = torch.tensor(stoich_coeffs, dtype=torch.float).cpu()

        # Store metabolic reaction data
        edge_type = ("metabolite", "reaction", "metabolite")
        hetero_data[edge_type].hyperedge_index = hyperedge_index
        hetero_data[edge_type].stoichiometry = stoich_coeffs
        hetero_data[edge_type].num_edges = len(hyperedge_index[1].unique())
        hetero_data[edge_type].reaction_to_genes = reaction_to_genes
        hetero_data[edge_type].reaction_to_genes_indices = reaction_to_genes_indices

        # Create GPR hyperedge
        gpr_gene_indices = []
        gpr_reaction_indices = []
        for reaction_idx, gene_indices in reaction_to_genes_indices.items():
            for gene_idx in gene_indices:
                if gene_idx != -1:  # Skip invalid gene indices
                    gpr_gene_indices.append(gene_idx)
                    gpr_reaction_indices.append(reaction_idx)

        if gpr_gene_indices:  # Only create if we have valid associations
            gpr_edge_index = torch.stack(
                [
                    torch.tensor(gpr_gene_indices, dtype=torch.long),
                    torch.tensor(gpr_reaction_indices, dtype=torch.long),
                ]
            ).cpu()

            # Store GPR edge
            gpr_type = ("gene", "gpr", "reaction")
            hetero_data[gpr_type].hyperedge_index = gpr_edge_index
            hetero_data[gpr_type].num_edges = len(torch.unique(gpr_edge_index[1]))

    return hetero_data


##

##


# @profile
def create_graph_from_gene_set(gene_set: GeneSet) -> nx.Graph:
    """
    Create a graph where nodes are gene names from the GeneSet.
    Initially, this graph will ha   ve no edges.
    """
    G = nx.Graph()
    for gene_name in gene_set:
        G.add_node(gene_name)  # Nodes are gene names
    return G


class GraphProcessor(ABC):
    @abstractmethod
    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: (
            dict[str, ExperimentType | ExperimentReferenceType]
            | list[dict[str, ExperimentType | ExperimentReferenceType]]
        ),
    ) -> HeteroData:
        pass


# class SubgraphRepresentation(GraphProcessor):
#     """
#     Processes gene knockout data by removing perturbed nodes from the graph, keeping
#     track of their features, and updating edge connectivity.

#     Node Transforms:
#         X ∈ ℝ^(N×d) → X_remain ∈ ℝ^((N-p)×d), X_pert ∈ ℝ^(p×d)
#         where N is total nodes, p is perturbed nodes, d is feature dimension

#     Edge Transforms:
#         E ∈ ℤ^(2×|E|) → E_filtered ∈ ℤ^(2×|E'|)
#         where |E| is original edge count, |E'| is edges after removing perturbed nodes
#     """

#     def process(
#         self,
#         cell_graph: HeteroData,
#         phenotype_info: list[PhenotypeType],
#         data: list[dict[str, ExperimentType | ExperimentReferenceType]],
#     ) -> HeteroData:
#         if not data:
#             raise ValueError("Data list is empty")

#         processed_graph = HeteroData()

#         # Collect all nodes to remove across all experiments
#         nodes_to_remove = set()
#         for item in data:
#             if "experiment" not in item or "experiment_reference" not in item:
#                 raise ValueError(
#                     "Each item in data must contain both 'experiment' and "
#                     "'experiment_reference' keys"
#                 )
#             nodes_to_remove.update(
#                 pert.systematic_gene_name
#                 for pert in item["experiment"].genotype.perturbations
#             )

#         # Process node information
#         processed_graph["gene"].node_ids = [
#             nid for nid in cell_graph["gene"].node_ids if nid not in nodes_to_remove
#         ]
#         processed_graph["gene"].num_nodes = len(processed_graph["gene"].node_ids)
#         processed_graph["gene"].ids_pert = list(nodes_to_remove)
#         processed_graph["gene"].cell_graph_idx_pert = torch.tensor(
#             [cell_graph["gene"].node_ids.index(nid) for nid in nodes_to_remove],
#             dtype=torch.long,
#         )

#         # Populate x and x_pert attributes
#         node_mapping = {nid: i for i, nid in enumerate(cell_graph["gene"].node_ids)}
#         x = cell_graph["gene"].x
#         processed_graph["gene"].x = x[
#             torch.tensor(
#                 [node_mapping[nid] for nid in processed_graph["gene"].node_ids]
#             )
#         ]
#         processed_graph["gene"].x_pert = x[processed_graph["gene"].cell_graph_idx_pert]

#         # add all phenotype fields
#         phenotype_fields = []
#         for phenotype in phenotype_info:
#             phenotype_fields.append(phenotype.model_fields["label_name"].default)
#             phenotype_fields.append(
#                 phenotype.model_fields["label_statistic_name"].default
#             )
#         for field in phenotype_fields:
#             processed_graph["gene"][field] = []

#         # add experiment data if it exists
#         for field in phenotype_fields:
#             field_values = []
#             for item in data:
#                 value = getattr(item["experiment"].phenotype, field, None)
#                 if value is not None:
#                     field_values.append(value)
#             if field_values:
#                 processed_graph["gene"][field] = torch.tensor(field_values)
#             else:
#                 processed_graph["gene"][field] = torch.tensor([float("nan")])

#         # Process edges
#         new_index_map = {
#             nid: i for i, nid in enumerate(processed_graph["gene"].node_ids)
#         }
#         for edge_type in cell_graph.edge_types:
#             src_type, _, dst_type = edge_type
#             edge_index = cell_graph[src_type, _, dst_type].edge_index.numpy()
#             filtered_edges = []

#             for src, dst in edge_index.T:
#                 src_id = cell_graph[src_type].node_ids[src]
#                 dst_id = cell_graph[dst_type].node_ids[dst]

#                 if src_id not in nodes_to_remove and dst_id not in nodes_to_remove:
#                     new_src = new_index_map[src_id]
#                     new_dst = new_index_map[dst_id]
#                     filtered_edges.append([new_src, new_dst])

#             if filtered_edges:
#                 new_edge_index = torch.tensor(filtered_edges, dtype=torch.long).t()
#                 processed_graph[src_type, _, dst_type].edge_index = new_edge_index
#                 processed_graph[src_type, _, dst_type].num_edges = new_edge_index.shape[
#                     1
#                 ]
#             else:
#                 processed_graph[src_type, _, dst_type].edge_index = torch.empty(
#                     (2, 0), dtype=torch.long
#                 )
#                 processed_graph[src_type, _, dst_type].num_edges = 0

#         return processed_graph


class SubgraphRepresentation(GraphProcessor):
    def process_regular_edges(
        self,
        edge_type: tuple[str, str, str],
        cell_graph: HeteroData,
        processed_graph: HeteroData,
        idx_keep_t: torch.Tensor,
    ) -> None:
        """Process physical and regulatory edge indices"""
        ei = cell_graph[edge_type].edge_index
        src, dst = ei
        keep_src = torch.isin(src, idx_keep_t)
        keep_dst = torch.isin(dst, idx_keep_t)
        keep = keep_src & keep_dst
        new_ei = ei[:, keep]

        # Create mapping for new indices
        old2new = {o.item(): n for n, o in enumerate(idx_keep_t)}
        for i in range(new_ei.size(1)):
            new_ei[0, i] = old2new[new_ei[0, i].item()]
            new_ei[1, i] = old2new[new_ei[1, i].item()]

        processed_graph[edge_type].edge_index = new_ei
        processed_graph[edge_type].num_edges = new_ei.size(1)

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        """Process graph data by removing perturbed nodes and their associated edges."""
        # 1) Gather perturbed genes
        perturbed_names: set[str] = set()
        for item in data:
            for p in item["experiment"].genotype.perturbations:
                perturbed_names.add(p.systematic_gene_name)

        # 2) Build keep/remove masks
        node_ids: list[str] = cell_graph["gene"].node_ids
        idx_remove = []
        idx_keep = []
        for i, name in enumerate(node_ids):
            if name in perturbed_names:
                idx_remove.append(i)
            else:
                idx_keep.append(i)

        idx_remove_t = torch.tensor(idx_remove, dtype=torch.long)
        idx_keep_t = torch.tensor(idx_keep, dtype=torch.long)

        # 3) Create new subgraph
        subgraph = HeteroData()
        subgraph["gene"].node_ids = [node_ids[i] for i in idx_keep]
        subgraph["gene"].num_nodes = len(idx_keep)
        subgraph["gene"].ids_pert = list(perturbed_names)
        subgraph["gene"].cell_graph_idx_pert = idx_remove_t
        x_full = cell_graph["gene"].x
        subgraph["gene"].x = x_full[idx_keep_t].clone()
        subgraph["gene"].x_pert = x_full[idx_remove_t].clone()

        # 4) Copy phenotype
        fields: list[str] = []
        for p in phenotype_info:
            fields.append(p.model_fields["label_name"].default)
            fields.append(p.model_fields["label_statistic_name"].default)
        for f in fields:
            vals: list[float] = []
            for item in data:
                val = getattr(item["experiment"].phenotype, f, None)
                if val is not None:
                    vals.append(val)
            subgraph["gene"][f] = torch.tensor(
                vals if vals else [float("nan")], dtype=torch.float
            )

        # 5) Copy metabolite info
        if "metabolite" in cell_graph.node_types:
            subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
            subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes

        # 6) Filter gene-gene edges
        for edge_type in cell_graph.edge_types:
            if edge_type[0] == "gene" and edge_type[2] == "gene":
                if edge_type[1] in ["physical_interaction", "regulatory_interaction"]:
                    self.process_regular_edges(
                        edge_type, cell_graph, subgraph, idx_keep_t
                    )

        # 7) Filter gpr and metabolic reactions
        if ("gene", "gpr", "reaction") in cell_graph.edge_types and (
            "metabolite",
            "reaction",
            "metabolite",
        ) in cell_graph.edge_types:
            gpr_key = ("gene", "gpr", "reaction")
            met_key = ("metabolite", "reaction", "metabolite")

            # Mark invalid reactions
            removed_idx = set(idx_remove)
            invalid_rxns = set()
            for rxn_idx, gene_idxs in cell_graph[
                met_key
            ].reaction_to_genes_indices.items():
                if any(g in removed_idx for g in gene_idxs):
                    invalid_rxns.add(rxn_idx)

            # Filter gpr edges
            gpr = cell_graph[gpr_key].hyperedge_index
            keep_g = torch.isin(gpr[0], idx_keep_t)
            keep_r = ~torch.isin(
                gpr[1], torch.tensor(list(invalid_rxns), device=gpr.device)
            )
            keep_mask = keep_g & keep_r
            new_gpr = gpr[:, keep_mask]

            # Remap gene indices
            old2new_g = {o.item(): n for n, o in enumerate(idx_keep_t)}
            for i in range(new_gpr.size(1)):
                new_gpr[0, i] = old2new_g[new_gpr[0, i].item()]

            # Remap reaction indices
            valid_rxns = torch.unique(new_gpr[1])
            valid_rxns_list = valid_rxns.tolist()
            old2new_r = {o: n for n, o in enumerate(valid_rxns_list)}
            new_gpr_1 = torch.tensor(
                [old2new_r[r.item()] for r in new_gpr[1]], device=gpr.device
            )
            new_gpr[1] = new_gpr_1

            # Add reaction nodes
            subgraph["reaction"].num_nodes = len(valid_rxns_list)
            subgraph["reaction"].node_ids = valid_rxns_list

            # Update gpr edges
            subgraph[gpr_key].hyperedge_index = new_gpr
            subgraph[gpr_key].num_edges = len(valid_rxns_list)

            # Filter metabolic edges
            met_edges = cell_graph[met_key].hyperedge_index
            stoich = cell_graph[met_key].stoichiometry
            keep_rxn = ~torch.isin(
                met_edges[1], torch.tensor(list(invalid_rxns), device=met_edges.device)
            )
            new_met = met_edges[:, keep_rxn]
            new_stoich = stoich[keep_rxn]

            # Remap reaction indices for metabolite edges
            new_met_1 = torch.tensor(
                [old2new_r[r.item()] for r in new_met[1]], device=met_edges.device
            )
            new_met[1] = new_met_1

            # Update metabolic edges
            subgraph[met_key].hyperedge_index = new_met
            subgraph[met_key].stoichiometry = new_stoich
            subgraph[met_key].num_edges = len(valid_rxns_list)
        return subgraph


class Unperturbed(GraphProcessor):
    """
    Processes graph data by preserving the original graph structure and storing perturbation
    and phenotype data alongside it for later processing.

    This processor:
    1. Keeps the original graph structure intact
    2. Stores perturbation information separately
    3. Records phenotype data without modifying the graph
    4. Can be used as a base for applying perturbations later in the pipeline

    Attributes remain unchanged:
        - Node features: X ∈ ℝ^(N×d) stays as X ∈ ℝ^(N×d)
        - Edge structure: E ∈ ℤ^(2×|E|) stays as E ∈ ℤ^(2×|E|)
    """

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        processed_graph = HeteroData()

        # Copy graph structure and features
        processed_graph["gene"].x = cell_graph["gene"].x
        processed_graph["gene"].node_ids = cell_graph["gene"].node_ids
        processed_graph["gene"].num_nodes = cell_graph["gene"].num_nodes

        # Store perturbation information
        perturbed_genes = set()
        for item in data:
            if "experiment" not in item or "experiment_reference" not in item:
                raise ValueError(
                    "Each item in data must contain both 'experiment' and "
                    "'experiment_reference' keys"
                )
            perturbed_genes.update(
                pert.systematic_gene_name
                for pert in item["experiment"].genotype.perturbations
            )

        processed_graph["gene"].perturbed_genes = list(perturbed_genes)
        processed_graph["gene"].perturbation_indices = torch.tensor(
            [cell_graph["gene"].node_ids.index(nid) for nid in perturbed_genes],
            dtype=torch.long,
        )

        # Add phenotype fields
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.append(phenotype.model_fields["label_name"].default)
            phenotype_fields.append(
                phenotype.model_fields["label_statistic_name"].default
            )

        # Add experiment data
        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)
            if field_values:
                processed_graph["gene"][field] = torch.tensor(field_values)
            else:
                processed_graph["gene"][field] = torch.tensor([float("nan")])

        # Copy edge information
        for edge_type in cell_graph.edge_types:
            if edge_type[1] in ["physical_interaction", "regulatory_interaction"]:
                processed_graph[edge_type].edge_index = cell_graph[edge_type].edge_index
                processed_graph[edge_type].num_edges = cell_graph[edge_type].num_edges

        # Handle metabolite data
        if "metabolite" in cell_graph.node_types:
            processed_graph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
            processed_graph["metabolite"].node_ids = cell_graph["metabolite"].node_ids

            edge_type = ("metabolite", "reactions", "metabolite")
            if any(e_type == edge_type for e_type in cell_graph.edge_types):
                # Copy hypergraph structure
                processed_graph[edge_type].hyperedge_index = cell_graph[
                    edge_type
                ].hyperedge_index
                processed_graph[edge_type].stoichiometry = cell_graph[
                    edge_type
                ].stoichiometry
                processed_graph[edge_type].num_edges = cell_graph[edge_type].num_edges

                # Only create reaction_to_genes_indices mapping
                node_id_to_idx = {
                    nid: idx for idx, nid in enumerate(cell_graph["gene"].node_ids)
                }
                reaction_to_genes_indices = {}

                for reaction_idx, genes in cell_graph[
                    edge_type
                ].reaction_to_genes.items():
                    gene_indices = []
                    for gene in genes:
                        gene_idx = node_id_to_idx.get(gene, -1)
                        gene_indices.append(gene_idx)
                    reaction_to_genes_indices[reaction_idx] = gene_indices

                processed_graph[edge_type].reaction_to_genes_indices = (
                    reaction_to_genes_indices
                )

        return processed_graph


class Perturbation(GraphProcessor):
    """
    Processes graph data by storing only perturbation-specific information without duplicating
    the base graph structure. This allows sharing a single base graph across instances while
    only tracking what changes between instances (perturbations and associated measurements).

    This processor:
    1. Stores perturbation information and measurements
    2. Does not duplicate the base graph structure
    3. Intended to be used with a shared base graph stored at the dataset level

    Key differences from Identity processor:
    - Does not store complete graph structure
    - Only tracks instance-specific perturbation data
    - Reduces memory usage by avoiding graph duplication
    """

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        if not data:
            raise ValueError("Data list is empty")

        # Create a minimal HeteroData object to store perturbation data
        processed_data = HeteroData()

        # Store perturbation information
        perturbed_genes = set()
        for item in data:
            if "experiment" not in item or "experiment_reference" not in item:
                raise ValueError(
                    "Each item in data must contain both 'experiment' and "
                    "'experiment_reference' keys"
                )
            perturbed_genes.update(
                pert.systematic_gene_name
                for pert in item["experiment"].genotype.perturbations
            )

        # Store perturbation indices
        processed_data["gene"].perturbed_genes = list(perturbed_genes)
        processed_data["gene"].perturbation_indices = torch.tensor(
            [cell_graph["gene"].node_ids.index(nid) for nid in perturbed_genes],
            dtype=torch.long,
        )

        # Add phenotype fields
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.append(phenotype.model_fields["label_name"].default)
            phenotype_fields.append(
                phenotype.model_fields["label_statistic_name"].default
            )

        # Add experiment data
        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)
            if field_values:
                processed_data["gene"][field] = torch.tensor(field_values)
            else:
                processed_data["gene"][field] = torch.tensor([float("nan")])

        return processed_data


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
    # @profile
    def __init__(
        self,
        root: str,
        query: str = None,
        gene_set: GeneSet = None,
        graphs: dict[str, nx.Graph] = None,
        incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] = None,
        node_embeddings: list[BaseEmbeddingDataset] = None,
        graph_processor: GraphProcessor = None,
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

        # init graph for building cell graph
        base_graph = self.get_init_graphs(self.gene_set)

        # graphs
        if graphs is not None:
            # remove edge data from graphs
            for graph in graphs.values():
                [graph.edges[edge].clear() for edge in graph.edges()]
            # remove node data from graphs
            for graph in graphs.values():
                [graph.nodes[node].clear() for node in graph.nodes()]
            graphs["base"] = base_graph
        else:
            graphs = {"base": base_graph}

        # embeddings
        if node_embeddings is not None:
            for name, embedding in node_embeddings.items():
                graphs[name] = create_embedding_graph(self.gene_set, embedding)
                # Integrate node embeddings into graphs

        # cell graph used in get item
        self.cell_graph = to_cell_data(graphs, incidence_graphs)

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
            with open(experiment_types_path, "r") as f:
                experiment_types = json.load(f)

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

        # Save experiment types to a JSON file
        with open(osp.join(self.processed_dir, "experiment_types.json"), "w") as f:
            json.dump(list(experiment_types), f)

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
        try:
            if osp.exists(osp.join(self.processed_dir, "gene_set.json")):
                with open(osp.join(self.processed_dir, "gene_set.json")) as f:
                    self._gene_set = set(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return GeneSet(self._gene_set)
        except json.JSONDecodeError:
            raise ValueError("Invalid or empty JSON file found.")

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        with open(osp.join(self.processed_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

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
        if osp.exists(osp.join(self.processed_dir, "phenotype_label_index.json")):
            with open(
                osp.join(self.processed_dir, "phenotype_label_index.json"), "r"
            ) as file:
                self._phenotype_label_index = json.load(file)
        else:
            self._phenotype_label_index = self.compute_phenotype_label_index()
            with open(
                osp.join(self.processed_dir, "phenotype_label_index.json"), "w"
            ) as file:
                json.dump(self._phenotype_label_index, file)
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
        if osp.exists(osp.join(self.processed_dir, "dataset_name_index.json")):
            with open(
                osp.join(self.processed_dir, "dataset_name_index.json"), "r"
            ) as file:
                self._dataset_name_index = json.load(file)
        else:
            self._dataset_name_index = self.compute_dataset_name_index()
            with open(
                osp.join(self.processed_dir, "dataset_name_index.json"), "w"
            ) as file:
                json.dump(self._dataset_name_index, file)
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
        if osp.exists(osp.join(self.processed_dir, "perturbation_count_index.json")):
            with open(
                osp.join(self.processed_dir, "perturbation_count_index.json"), "r"
            ) as file:
                self._perturbation_count_index = json.load(file)
                # Convert string keys back to integers
                self._perturbation_count_index = {
                    int(k): v for k, v in self._perturbation_count_index.items()
                }
        else:
            self._perturbation_count_index = self.compute_perturbation_count_index()
            with open(
                osp.join(self.processed_dir, "perturbation_count_index.json"), "w"
            ) as file:
                # Convert integer keys to strings for JSON serialization
                json.dump(
                    {str(k): v for k, v in self._perturbation_count_index.items()}, file
                )
        return self._perturbation_count_index

    def compute_is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
        print("Computing is any perturbed gene index...")
        is_any_perturbed_gene_index = {}

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
                        # Get perturbed genes for this experiment
                        perturbed_genes = {
                            pert.systematic_gene_name
                            for pert in experiment.genotype.perturbations
                        }

                        # Add index to each perturbed gene's list
                        for gene in perturbed_genes:
                            if gene not in is_any_perturbed_gene_index:
                                is_any_perturbed_gene_index[gene] = set()
                            is_any_perturbed_gene_index[gene].add(idx)
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
        for gene in is_any_perturbed_gene_index:
            is_any_perturbed_gene_index[gene] = sorted(
                list(is_any_perturbed_gene_index[gene])
            )

        return is_any_perturbed_gene_index

    @property
    def is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
        if osp.exists(osp.join(self.processed_dir, "is_any_perturbed_gene_index.json")):
            with open(
                osp.join(self.processed_dir, "is_any_perturbed_gene_index.json"), "r"
            ) as file:
                self._is_any_perturbed_gene_index = json.load(file)
        else:
            self._is_any_perturbed_gene_index = (
                self.compute_is_any_perturbed_gene_index()
            )
            with open(
                osp.join(self.processed_dir, "is_any_perturbed_gene_index.json"), "w"
            ) as file:
                json.dump(self._is_any_perturbed_gene_index, file)
        return self._is_any_perturbed_gene_index


def main():
    # genome
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.data import MeanExperimentDeduplicator
    from torchcell.data import GenotypeAggregator
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule

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
    print(dataset[183])
    dataset.close_lmdb()
    # print(dataset[10000])

    # Assuming you have already created your dataset and CellDataModule
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
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
    #     num_workers=4,
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
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.data import MeanExperimentDeduplicator
    from torchcell.data import GenotypeAggregator
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.datasets import CodonFrequencyDataset

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
    # fudt_3prime_dataset = FungalUpDownTransformerDataset(
    #     root="data/scerevisiae/fudt_embedding",
    #     genome=genome,
    #     model_name="species_downstream",
    # )
    # fudt_5prime_dataset = FungalUpDownTransformerDataset(
    #     root="data/scerevisiae/fudt_embedding",
    #     genome=genome,
    #     model_name="species_downstream",
    # )
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    from torchcell.metabolism.yeast_GEM import YeastGEM

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism": YeastGEM().reaction_map},
        node_embeddings={"codon_frequency": codon_frequency},
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
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        break


def main_transform():
    """Test the label binning transforms on the dataset with proper initialization."""
    import os.path as osp
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import copy
    import torch
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
    from torchcell.data.neo4j_cell import Neo4jCellDataset, SubgraphRepresentation
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.transforms.regression_to_classification import (
        LabelBinningTransform,
        LabelNormalizationTransform,
    )
    from torch_geometric.transforms import Compose

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


def main_transform_dense():
    """Test label transforms and dense conversion with perturbation subset."""
    import os.path as osp
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import copy
    import torch
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
    from torchcell.data.neo4j_cell import Neo4jCellDataset, SubgraphRepresentation
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.transforms.regression_to_classification import (
        LabelBinningTransform,
        LabelNormalizationTransform,
        InverseCompose,
    )
    from torchcell.transforms.hetero_to_dense import HeteroToDense
    from torch_geometric.transforms import Compose
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule

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
        num_workers=4,
        pin_memory=True,
    )
    base_data_module.setup()

    # Create perturbation subset module
    subset_size = 10000
    subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=base_data_module,
        size=subset_size,
        batch_size=2,
        num_workers=4,
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
    # main_transform_dense()
    # main()
    main_incidence()

```

These are some of the key data structures.

```python
# torchcell/datamodules/cell.py
# [[torchcell.datamodules.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/cell.py
# Test file: torchcell/datamodules/test_cell.py
import json
import os
from typing import List, Dict, Union, Optional, Set
import pandas as pd
import lightning as L
import torch
import random
from collections import defaultdict
import logging
import os.path as osp
from torch_geometric.loader import DataLoader, PrefetchLoader
from torchcell.datamodels import ModelStrict
from pydantic import BaseModel, model_validator, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class IndexSplit(ModelStrict):
    indices: List[int] = Field(..., description="Must be sorted in ascending order")
    count: int

    @model_validator(mode="before")
    @classmethod
    def check_sorted_indices(cls, values):
        indices = values.get("indices")
        if indices and not all(
            indices[i] <= indices[i + 1] for i in range(len(indices) - 1)
        ):
            raise ValueError("Indices must be sorted in ascending order")
        return values

    def __repr__(self):
        max_indices = 3
        indices_str = (
            f"[{', '.join(map(str, self.indices[:max_indices]))}"
            f"{', ...' if len(self.indices) > max_indices else ''}]"
        )
        return f"IndexSplit(indices={indices_str}, count={self.count})"


class DatasetSplit(BaseModel):
    phenotype_label_index: Optional[Dict[str, IndexSplit]] = None
    perturbation_count_index: Optional[Dict[int, IndexSplit]] = None
    dataset_name_index: Optional[Dict[str, IndexSplit]] = None


class DataModuleIndexDetails(ModelStrict):
    methods: List[str]
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit

    def df_summary(self):
        data = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)

        for split in ["train", "val", "test"]:
            split_data = getattr(self, split)
            for index_type, index_data in split_data.dict().items():
                if index_data is not None:
                    for key, index_split in index_data.items():
                        # Handle both IndexSplit objects and dictionaries
                        if isinstance(index_split, dict):
                            count = index_split.get("count")
                        else:
                            count = index_split.count

                        data[(index_type, key)][split] = count
                        totals[(index_type, key)] += count

        summary_data = []
        for (index_type, key), splits in data.items():
            total = totals[(index_type, key)]
            for split in ["train", "val", "test"]:
                count = splits[split]
                ratio = count / total if total > 0 else 0
                summary_data.append(
                    {
                        "split": split,
                        "index_type": index_type,
                        "key": key,
                        "count": count,
                        "ratio": ratio,
                        "total": total,
                    }
                )

        df = pd.DataFrame(summary_data)

        # Create a categorical column for 'split' with the desired order
        df["split"] = pd.Categorical(
            df["split"], categories=["train", "val", "test"], ordered=True
        )

        # Sort the DataFrame
        df = df.sort_values(["split", "index_type", "key"])

        df["ratio"] = df["ratio"].round(3)
        df = df.reset_index(drop=True)

        return df

    def __str__(self):
        df = self.df_summary()
        if df.empty:
            return "DataModuleIndexDetails(empty)"
        return df.to_string()


class DataModuleIndex(ModelStrict):
    train: List[int] = Field(..., description="Must be sorted in ascending order")
    val: List[int] = Field(..., description="Must be sorted in ascending order")
    test: List[int] = Field(..., description="Must be sorted in ascending order")

    @model_validator(mode="before")
    @classmethod
    def check_sorted_and_unique_indices(cls, values):
        for split in ["train", "val", "test"]:
            indices = values.get(split, [])
            if not all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1)):
                raise ValueError(f"{split} indices must be sorted in ascending order")

        all_indices = (
            values.get("train", []) + values.get("val", []) + values.get("test", [])
        )
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Indices in train, val, and test must not overlap")

        return values

    def __repr__(self):
        max_indices = 3
        train_str_index = f"[{', '.join(map(str, self.train[:max_indices]))}{', ...' if len(self.train) > max_indices else ''}]"
        val_str_index = f"[{', '.join(map(str, self.val[:max_indices]))}{', ...' if len(self.val) > max_indices else ''}]"
        test_str_index = f"[{', '.join(map(str, self.test[:max_indices]))}{', ...' if len(self.test) > max_indices else ''}]"
        return f"DataModuleIndex(train={train_str_index}, val={val_str_index}, test={test_str_index})"

    def __str__(self):
        max_indices = 3
        train_str_index = f"[{', '.join(map(str, self.train[:max_indices]))}{', ...' if len(self.train) > max_indices else ''}]"
        val_str_index = f"[{', '.join(map(str, self.val[:max_indices]))}{', ...' if len(self.val) > max_indices else ''}]"
        test_str_index = f"[{', '.join(map(str, self.test[:max_indices]))}{', ...' if len(self.test) > max_indices else ''}]"
        train_str = train_str_index + f" ({len(self.train)} indices)"
        val_str = val_str_index + f" ({len(self.val)} indices)"
        test_str = test_str_index + f" ({len(self.test)} indices)"
        return f"DataModuleIndex(train={train_str}, val={val_str}, test={test_str})"


class DatasetIndexSplit(ModelStrict):
    train: dict[Union[str, int], list[int]] = None
    val: dict[Union[str, int], list[int]] = None
    test: dict[Union[str, int], list[int]] = None


def overlap_dataset_index_split(
    dataset_index: dict[str | int, list[int]], data_module_index: DataModuleIndex
) -> DatasetIndexSplit:
    train_set = set(data_module_index.train)
    val_set = set(data_module_index.val)
    test_set = set(data_module_index.test)

    train_dict = {}
    val_dict = {}
    test_dict = {}

    for dataset_name, indices in dataset_index.items():
        train_indices = sorted(list(set(indices) & train_set))
        val_indices = sorted(list(set(indices) & val_set))
        test_indices = sorted(list(set(indices) & test_set))

        if train_indices:
            train_dict[dataset_name] = train_indices
        if val_indices:
            val_dict[dataset_name] = val_indices
        if test_indices:
            test_dict[dataset_name] = test_indices

    return DatasetIndexSplit(
        train=train_dict if train_dict else None,
        val=val_dict if val_dict else None,
        test=test_dict if test_dict else None,
    )


class CellDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        cache_dir: str = "cache",
        batch_size: int = 32,
        random_seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch: bool = False,
        split_indices: Union[str, List[str], None] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.split_indices = (
            split_indices
            if isinstance(split_indices, list)
            else [split_indices] if split_indices else []
        )
        self._index = None
        self._index_details = None

        # Compute index during initialization
        self.index
        self.index_details

    @property
    def index(self) -> DataModuleIndex:
        if self._index is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index

    @property
    def index_details(self) -> DataModuleIndexDetails:
        if self._index_details is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index_details

    def _load_or_compute_index(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        index_file = osp.join(self.cache_dir, f"index_seed_{self.random_seed}.json")
        details_file = osp.join(
            self.cache_dir, f"index_details_seed_{self.random_seed}.json"
        )
        if osp.exists(index_file) and osp.exists(details_file):
            try:
                with open(index_file, "r") as f:
                    log.info(f"Loading index from {index_file}")
                    index_dict = json.load(f)
                    self._index = DataModuleIndex(**index_dict)
                with open(details_file, "r") as f:
                    log.info(f"Loading index details from {details_file}")
                    details_dict = json.load(f)
                    self._index_details = DataModuleIndexDetails(**details_dict)
            except Exception as e:
                print(f"Error loading index or details: {e}. Regenerating...")
                self._compute_and_save_index(index_file, details_file)
        else:
            self._compute_and_save_index(index_file, details_file)

    def _compute_and_save_index(self, index_file, details_file):
        log.info("Generating detailed index...")
        random.seed(self.random_seed)

        all_indices = set(range(len(self.dataset)))
        split_data = {
            "train": defaultdict(set),
            "val": defaultdict(set),
            "test": defaultdict(set),
        }

        # First, split each index independently
        for index_name in self.split_indices:
            original_index = getattr(self.dataset, index_name)
            for key, indices in original_index.items():
                indices = list(indices)
                random.shuffle(indices)
                num_samples = len(indices)
                num_train = int(self.train_ratio * num_samples)
                num_val = int(self.val_ratio * num_samples)

                split_data["train"][index_name].update(indices[:num_train])
                split_data["val"][index_name].update(
                    indices[num_train : num_train + num_val]
                )
                split_data["test"][index_name].update(indices[num_train + num_val :])

        # Then, create initial final splits
        final_splits = {
            "train": all_indices.intersection(
                *[split_data["train"][index] for index in self.split_indices]
            ),
            "val": all_indices.intersection(
                *[split_data["val"][index] for index in self.split_indices]
            ),
            "test": all_indices.intersection(
                *[split_data["test"][index] for index in self.split_indices]
            ),
        }

        # Sophisticated assignment of remaining indices
        remaining = all_indices - (
            final_splits["train"] | final_splits["val"] | final_splits["test"]
        )
        target_ratios = {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": 1 - self.train_ratio - self.val_ratio,
        }

        for index_name in self.split_indices:
            original_index = getattr(self.dataset, index_name)
            for key, indices in original_index.items():
                key_remaining = set(indices) & remaining
                if not key_remaining:
                    continue

                current_counts = {
                    split: len(set(indices) & final_splits[split])
                    for split in ["train", "val", "test"]
                }
                total_count = sum(current_counts.values()) + len(key_remaining)

                for idx in key_remaining:
                    target_counts = {
                        split: int(total_count * ratio)
                        for split, ratio in target_ratios.items()
                    }
                    best_split = min(
                        ["train", "val", "test"],
                        key=lambda x: (current_counts[x] - target_counts[x])
                        / target_counts[x],
                    )
                    final_splits[best_split].add(idx)
                    current_counts[best_split] += 1
                    remaining.remove(idx)

        # Create DataModuleIndexDetails object
        self._index_details = DataModuleIndexDetails(
            methods=self.split_indices,
            train=DatasetSplit(),
            val=DatasetSplit(),
            test=DatasetSplit(),
        )

        for split in ["train", "val", "test"]:
            for index_name in self.split_indices:
                original_index = getattr(self.dataset, index_name)
                split_data = {}
                for key, indices in original_index.items():
                    intersect = sorted(list(set(indices) & final_splits[split]))
                    split_data[key] = IndexSplit(
                        indices=intersect, count=len(intersect)
                    )
                setattr(getattr(self._index_details, split), index_name, split_data)

        # Create DataModuleIndex object
        self._index = DataModuleIndex(
            train=sorted(list(final_splits["train"])),
            val=sorted(list(final_splits["val"])),
            test=sorted(list(final_splits["test"])),
        )

        # Save the index and details separately
        with open(index_file, "w") as f:
            json.dump(self._index.dict(), f, indent=2)
        with open(details_file, "w") as f:
            json.dump(self._index_details.dict(), f, indent=2)

    def _cached_files_exist(self):
        index_file = osp.join(self.cache_dir, f"index_seed_{self.random_seed}.json")
        details_file = osp.join(
            self.cache_dir, f"index_details_seed_{self.random_seed}.json"
        )
        return osp.exists(index_file) and osp.exists(details_file)

    def setup(self, stage=None):
        train_index = self.index.train
        val_index = self.index.val
        test_index = self.index.test

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_index)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_index)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_index)

    def _get_dataloader(self, dataset, shuffle=False):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x", "x_pert"],
        )
        if self.prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        return loader

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def all_dataloader(self):
        return self._get_dataloader(self.dataset)


if __name__ == "__main__":
    pass

```

Update `PerturbationSubsetDataModule` so that we give the `gene_subset` arg and subset the instances by any values in the `is_any_perturbed_gene_index`. We only want instances that are associated with genes from metabolism in our `PerturbationSubsetDataModule.`

Also, this was our work in progress in trying to make a script to test the updated `PerturbationSubsetDataModule.` But remember we want to plot like we did for the previous script shown earlier.

```python
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
import json
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import SubgraphRepresentation
from tqdm import tqdm
from torchcell.utils import format_scientific_notation
from torchcell.datasets import CodonFrequencyDataset
from torchcell.metabolism.yeast_GEM import YeastGEM


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Initialize genome and GEM
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    # genome.drop_chrmt()
    genome.drop_empty_go()

    # Initialize GEM and get metabolism genes
    gem = YeastGEM()
    metabolism_genes = gem.gene_set
    print(f"Number of metabolism genes: {len(metabolism_genes)}")

    # Save metabolism genes
    with open("metabolism_genes.json", "w") as f:
        json.dump(list(metabolism_genes), f)

    # Setup graph and embeddings
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    # Load query and setup dataset
    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={"codon_frequency": codon_frequency},
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    seed = 42
    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=seed,
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    # Process different subset sizes with metabolism genes priority
    sizes = [5e1]
    # sizes = [5e1, 1e2, 5e2, 1e3, 5e3, 7e3, 1e4, 5e4, 1e5]

    for size in sizes:
        print(f"\nProcessing size {format_scientific_notation(size)}")

        perturbation_subset_data_module = PerturbationSubsetDataModule(
            cell_data_module=cell_data_module,
            size=int(size),
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            prefetch=False,
            seed=seed,
            priority_genes=metabolism_genes,
        )
        perturbation_subset_data_module.setup()

        # Print metabolism gene statistics
        train_data = [dataset[i] for i in perturbation_subset_data_module.index.train]
        metabolism_count = sum(
            1
            for idx in perturbation_subset_data_module.index.train
            if any(
                gene in metabolism_genes
                for gene, indices in dataset.is_any_perturbed_gene_index.items()
                if idx in indices
            )
        )
        print(
            f"Train samples with metabolism genes: {metabolism_count}/{len(train_data)} "
            f"({metabolism_count/len(train_data)*100:.2f}%)"
        )

        # Test dataloader
        for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
            break

        # Save the indices for this size
        save_dir = osp.join(
            dataset_root, f"subset_size_{format_scientific_notation(size)}"
        )
        os.makedirs(save_dir, exist_ok=True)

        with open(osp.join(save_dir, "index.json"), "w") as f:
            json.dump(perturbation_subset_data_module.index.dict(), f)


if __name__ == "__main__":
    main()
```