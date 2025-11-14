# torchcell/scratch/load_neigbor_batch_006
# [[torchcell.scratch.load_neigbor_batch_006]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/load_neigbor_batch_006

"""
Load data with NeighborSubgraphRepresentation for k-hop induced subgraphs.

Creates small induced subgraphs around perturbed genes using k-hop sampling.
"""

import os
import os.path as osp
import random
import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.data import HeteroData
from torch_geometric.utils import k_hop_subgraph
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import GraphProcessor
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
)
from tqdm import tqdm
from typing import Literal


class NeighborSubgraphRepresentation(GraphProcessor):
    """
    GraphProcessor that creates k-hop induced subgraphs around perturbed genes.

    Unlike SubgraphRepresentation (which filters out perturbed genes) or
    LazySubgraphRepresentation (which keeps full graph with masks), this processor
    creates small induced subgraphs containing only k-hop neighborhoods around
    perturbed genes.

    Benefits:
    - Smaller graphs for faster message passing
    - Reduced data loading overhead
    - Preserves original node indices for easy mapping back to full graph

    Args:
        num_hops: Number of hops to include in neighborhood (default: 2)
    """

    def __init__(self, num_hops: int = 2):
        self.num_hops = num_hops
        self.device = torch.device("cpu")  # For DataLoader pin_memory compatibility
        self.masks = {}

    def _initialize_masks(self, cell_graph: HeteroData):
        """Initialize boolean masks for all nodes and edges."""
        self.masks = {}

        # Gene node masks
        num_genes = cell_graph["gene"].num_nodes
        self.masks["gene"] = {
            "perturbed": torch.zeros(num_genes, dtype=torch.bool, device=self.device),
            "kept": torch.zeros(num_genes, dtype=torch.bool, device=self.device),
        }

    def _process_gene_info(self, cell_graph: HeteroData, data):
        """
        Identify perturbed genes from experiment data.

        Returns dict with:
            - perturbed_names: Set of perturbed gene IDs
            - perturbed_indices: Tensor of perturbed gene indices
            - perturbed_node_ids: List of perturbed gene IDs
        """
        # Extract perturbed gene systematic names from all experiments
        perturbed_names = set()
        for item in data:
            for p in item["experiment"].genotype.perturbations:
                perturbed_names.add(p.systematic_gene_name)

        node_ids = cell_graph["gene"].node_ids

        # Find indices of perturbed genes
        perturbed_indices = []
        for i, name in enumerate(node_ids):
            if name in perturbed_names:
                perturbed_indices.append(i)

        perturbed_indices = torch.tensor(perturbed_indices, dtype=torch.long)

        # Update masks
        self.masks["gene"]["perturbed"][perturbed_indices] = True

        return {
            "perturbed_names": perturbed_names,
            "perturbed_indices": perturbed_indices,
            "perturbed_node_ids": [node_ids[i] for i in perturbed_indices],
        }

    def _build_khop_subgraph(
        self,
        cell_graph: HeteroData,
        gene_info: dict
    ) -> dict:
        """
        Build k-hop induced subgraph around perturbed genes.

        For each edge type:
        1. Use k_hop_subgraph to find k-hop neighbors of perturbed genes
        2. Union all neighbor nodes across edge types
        3. Filter edges to only those within the union subset

        Returns dict with:
            - subset_nodes: Tensor of node indices in subgraph
            - subset_node_ids: List of gene IDs in subgraph
            - edge_data: Dict mapping edge_type -> (edge_index, edge_mask)
            - perturbed_mask: Boolean mask for perturbed nodes in subgraph
        """
        perturbed_indices = gene_info["perturbed_indices"]
        num_genes = cell_graph["gene"].num_nodes

        # Collect all k-hop neighbors across all edge types
        all_neighbors = set(perturbed_indices.tolist())
        edge_data = {}

        for et in cell_graph.edge_types:
            if et[0] == "gene" and et[2] == "gene":
                # Get k-hop subgraph for this edge type
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=perturbed_indices,
                    num_hops=self.num_hops,
                    edge_index=cell_graph[et].edge_index,
                    relabel_nodes=False,  # Keep original node indices
                    num_nodes=num_genes,
                )

                # Union neighbors from this edge type
                all_neighbors.update(subset.tolist())

                # Store edge data for this type
                edge_data[et] = {
                    "edge_index": edge_index,
                    "edge_mask": edge_mask,
                    "subset": subset,
                }

        # Convert union of neighbors to sorted tensor
        subset_nodes = torch.tensor(sorted(all_neighbors), dtype=torch.long)
        subset_node_ids = [cell_graph["gene"].node_ids[i] for i in subset_nodes]

        # Create mask for perturbed nodes in the subgraph
        perturbed_mask = torch.zeros(len(subset_nodes), dtype=torch.bool)
        for i, node_idx in enumerate(subset_nodes):
            if self.masks["gene"]["perturbed"][node_idx]:
                perturbed_mask[i] = True

        # Filter edges to only those within the union subset
        # We need to rebuild edge indices to only include edges where both
        # source and target are in subset_nodes
        subset_set = set(subset_nodes.tolist())
        filtered_edge_data = {}

        for et, data_dict in edge_data.items():
            edge_index = cell_graph[et].edge_index

            # Filter edges: both source and target must be in subset
            mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                if src in subset_set and dst in subset_set:
                    mask[i] = True

            filtered_edge_index = edge_index[:, mask]

            filtered_edge_data[et] = {
                "edge_index": filtered_edge_index,
                "num_edges": filtered_edge_index.size(1),
            }

        return {
            "subset_nodes": subset_nodes,
            "subset_node_ids": subset_node_ids,
            "edge_data": filtered_edge_data,
            "perturbed_mask": perturbed_mask,
        }

    def _process_metabolism(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        subgraph_info: dict,
    ):
        """
        Add metabolism edges for genes in the k-hop subgraph.

        For genes in the subgraph:
        1. Include all GPR (gene-reaction) edges involving those genes
        2. Include reactions connected to those genes
        3. Include all metabolites
        4. Include RMR (reaction-metabolite) edges for included reactions
        """
        if "reaction" not in cell_graph.node_types:
            return  # No metabolism in this graph

        subset_nodes = subgraph_info["subset_nodes"]
        subset_set = set(subset_nodes.tolist())

        # Process GPR edges (gene, gpr, reaction)
        if ("gene", "gpr", "reaction") in cell_graph.edge_types:
            gpr_et = ("gene", "gpr", "reaction")
            gpr_hyperedge_index = cell_graph[gpr_et].hyperedge_index

            # Filter GPR edges: keep edges where gene is in subset
            gpr_mask = torch.zeros(gpr_hyperedge_index.size(1), dtype=torch.bool)
            included_reactions = set()

            for i in range(gpr_hyperedge_index.size(1)):
                gene_idx = gpr_hyperedge_index[0, i].item()
                reaction_idx = gpr_hyperedge_index[1, i].item()

                if gene_idx in subset_set:
                    gpr_mask[i] = True
                    included_reactions.add(reaction_idx)

            # Store filtered GPR edges (keep full hyperedge_index, use mask)
            integrated_subgraph[gpr_et].hyperedge_index = gpr_hyperedge_index
            integrated_subgraph[gpr_et].num_edges = gpr_hyperedge_index.size(1)
            integrated_subgraph[gpr_et].mask = gpr_mask

            # Add reaction nodes
            if included_reactions:
                reaction_indices = torch.tensor(sorted(included_reactions), dtype=torch.long)
                integrated_subgraph["reaction"].node_ids = [
                    cell_graph["reaction"].node_ids[i] for i in reaction_indices
                ]
                integrated_subgraph["reaction"].num_nodes = len(reaction_indices)

                # Reaction mask (all reactions are valid in subgraph context)
                integrated_subgraph["reaction"].pert_mask = torch.zeros(
                    len(reaction_indices), dtype=torch.bool
                )
                integrated_subgraph["reaction"].mask = torch.ones(
                    len(reaction_indices), dtype=torch.bool
                )

                # Process RMR edges (reaction, rmr, metabolite)
                if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
                    rmr_et = ("reaction", "rmr", "metabolite")
                    rmr_hyperedge_index = cell_graph[rmr_et].hyperedge_index
                    rmr_stoichiometry = cell_graph[rmr_et].stoichiometry

                    # Filter RMR edges: keep edges where reaction is included
                    rmr_mask = torch.zeros(rmr_hyperedge_index.size(1), dtype=torch.bool)

                    for i in range(rmr_hyperedge_index.size(1)):
                        reaction_idx = rmr_hyperedge_index[0, i].item()
                        if reaction_idx in included_reactions:
                            rmr_mask[i] = True

                    # Store RMR edges
                    integrated_subgraph[rmr_et].hyperedge_index = rmr_hyperedge_index
                    integrated_subgraph[rmr_et].stoichiometry = rmr_stoichiometry
                    integrated_subgraph[rmr_et].num_edges = rmr_hyperedge_index.size(1)
                    integrated_subgraph[rmr_et].mask = rmr_mask

                    # Add all metabolites (we keep all metabolites)
                    integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
                    integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes
                    integrated_subgraph["metabolite"].pert_mask = torch.zeros(
                        cell_graph["metabolite"].num_nodes, dtype=torch.bool
                    )
                    integrated_subgraph["metabolite"].mask = torch.ones(
                        cell_graph["metabolite"].num_nodes, dtype=torch.bool
                    )

    def _add_gene_data(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        gene_info: dict,
        subgraph_info: dict,
    ):
        """Add gene node data to the subgraph."""
        subset_nodes = subgraph_info["subset_nodes"]

        # Node IDs
        integrated_subgraph["gene"].node_ids = subgraph_info["subset_node_ids"]
        integrated_subgraph["gene"].num_nodes = len(subset_nodes)

        # Perturbed gene info
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].perturbation_indices = gene_info["perturbed_indices"]

        # Node features (subset)
        integrated_subgraph["gene"].x = cell_graph["gene"].x[subset_nodes]

        # Perturbation mask (relative to subgraph)
        integrated_subgraph["gene"].pert_mask = subgraph_info["perturbed_mask"]

        # Create x_pert (zero features for perturbed nodes)
        integrated_subgraph["gene"].x_pert = integrated_subgraph["gene"].x.clone()
        integrated_subgraph["gene"].x_pert[subgraph_info["perturbed_mask"]] = 0.0

    def _add_phenotype_data(
        self,
        integrated_subgraph: HeteroData,
        phenotype_info: list,
        data: list,
    ):
        """Add phenotype data in COO (coordinate) format."""
        # Collect phenotype values and indices
        all_values = []
        all_type_indices = []
        all_sample_indices = []
        phenotype_types = []

        all_stat_values = []
        all_stat_type_indices = []
        all_stat_sample_indices = []
        stat_types = []

        # Extract phenotype type information
        for phenotype_class in phenotype_info:
            label_name = phenotype_class.model_fields["label_name"].default
            stat_name = phenotype_class.model_fields["label_statistic_name"].default

            phenotype_types.append(label_name)
            if stat_name:
                stat_types.append(stat_name)

        # Process each experimental data point
        for item_idx, item in enumerate(data):
            # Get phenotype object from experiment
            phenotype = item["experiment"].phenotype

            # Process each phenotype type
            for type_idx, field_name in enumerate(phenotype_types):
                value = getattr(phenotype, field_name, None)
                if value is not None:
                    # Convert single values to lists for consistent handling
                    values = [value] if not isinstance(value, (list, tuple)) else value

                    # Add all values with their type indices and sample indices
                    all_values.extend(values)
                    all_type_indices.extend([type_idx] * len(values))
                    all_sample_indices.extend([item_idx] * len(values))

            # Process statistics in the same way
            for stat_type_idx, stat_field_name in enumerate(stat_types):
                stat_value = getattr(phenotype, stat_field_name, None)
                if stat_value is not None:
                    # Convert single values to lists for consistent handling
                    stat_values = (
                        [stat_value]
                        if not isinstance(stat_value, (list, tuple))
                        else stat_value
                    )

                    # Add all statistic values
                    all_stat_values.extend(stat_values)
                    all_stat_type_indices.extend([stat_type_idx] * len(stat_values))
                    all_stat_sample_indices.extend([item_idx] * len(stat_values))

        # Convert to tensors
        if all_values:
            integrated_subgraph["gene"]["phenotype_values"] = torch.tensor(
                all_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_type_indices"] = torch.tensor(
                all_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_sample_indices"] = torch.tensor(
                all_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_types"] = phenotype_types

        if all_stat_values:
            integrated_subgraph["gene"]["phenotype_stat_values"] = torch.tensor(
                all_stat_values, dtype=torch.float, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_type_indices"] = torch.tensor(
                all_stat_type_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_sample_indices"] = torch.tensor(
                all_stat_sample_indices, dtype=torch.long, device=self.device
            )
            integrated_subgraph["gene"]["phenotype_stat_types"] = stat_types

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list,
        data: list,
    ) -> HeteroData:
        """
        Main processing method.

        Creates k-hop induced subgraph around perturbed genes.

        Returns:
            HeteroData with k-hop neighborhood subgraph
        """
        # Initialize
        self._initialize_masks(cell_graph)
        integrated_subgraph = HeteroData()

        # Identify perturbed genes
        gene_info = self._process_gene_info(cell_graph, data)

        # Build k-hop subgraph
        subgraph_info = self._build_khop_subgraph(cell_graph, gene_info)

        # Add gene data
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info, subgraph_info)

        # Add edge data
        for et, edge_data in subgraph_info["edge_data"].items():
            integrated_subgraph[et].edge_index = edge_data["edge_index"]
            integrated_subgraph[et].num_edges = edge_data["num_edges"]

        # Process metabolism edges (GPR and RMR)
        self._process_metabolism(integrated_subgraph, cell_graph, subgraph_info)

        # Add phenotype data
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        return integrated_subgraph


def load_sample_data_batch(
    batch_size=2,
    num_workers=2,
    num_hops=2,
    config: Literal["hetero_cell_bipartite"] = "hetero_cell_bipartite",
):
    """
    Load a sample data batch with NeighborSubgraphRepresentation.

    Args:
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        num_hops: Number of hops for neighborhood sampling
        config: Model configuration (currently only "hetero_cell_bipartite" supported)

    Returns:
        Tuple of (dataset, batch, input_channels, max_num_nodes)
    """
    # Set all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    # Initialize genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    # Initialize graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Configuration - matching experiment 006
    config_options = {
        "hetero_cell_bipartite": {
            "graph_names": [
                "physical",
                "regulatory",
                "tflink",
                "string12_0_neighborhood",
                "string12_0_fusion",
                "string12_0_cooccurence",
                "string12_0_coexpression",
                "string12_0_experimental",
                "string12_0_database",
            ],
            "use_metabolism": True,
            "follow_batch": ["perturbation_indices"],
        },
    }

    selected_config = config_options[config]

    # Build gene multigraph
    gene_multigraph = build_gene_multigraph(
        graph=graph,
        graph_names=selected_config["graph_names"]
    )

    # Prepare incidence graphs
    incidence_graphs = {}
    if selected_config["use_metabolism"]:
        yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
        incidence_graphs["metabolism_bipartite"] = yeast_gem.bipartite_graph

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Use same dataset root
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Create dataset with NeighborSubgraphRepresentation
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=NeighborSubgraphRepresentation(num_hops=num_hops),
        transform=None,
    )

    # Normalization transform
    norm_configs = {
        "gene_interaction": {"strategy": "standard"}
    }
    normalizer = LabelNormalizationTransform(dataset, norm_configs)

    # Print normalization parameters
    for label, stats in normalizer.stats.items():
        print(f"\nNormalization parameters for {label}:")
        for key, value in stats.items():
            if key not in ["bin_edges", "bin_counts", "strategy"]:
                print(f"  {key}: {value:.6f}")
        print(f"  strategy: {stats['strategy']}")

    dataset.transform = normalizer

    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=[
            "phenotype_label_index",
            "perturbation_count_index",
        ],
        batch_size=8,
        random_seed=42,
        num_workers=4,
        pin_memory=False,
        train_shuffle=False,
    )

    cell_data_module.setup()

    # Subset Module
    size = 5e4

    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch=False,
        seed=seed,
        follow_batch=selected_config["follow_batch"],
        train_shuffle=False,
    )
    perturbation_subset_data_module.setup()

    max_num_nodes = len(dataset.gene_set)

    # Get a batch
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    input_channels = dataset.cell_graph["gene"].x.size()[-1]

    return dataset, batch, input_channels, max_num_nodes


def inspect_data():
    """Inspect the structure of NeighborSubgraphRepresentation data."""
    # Set all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    print("="*80)
    print("NeighborSubgraphRepresentation Data Structure Inspection")
    print("="*80)
    print()

    # Initialize genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    # Initialize graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Use all graph types from experiment 006
    graph_names = [
        "physical",
        "regulatory",
        "tflink",
        "string12_0_neighborhood",
        "string12_0_fusion",
        "string12_0_cooccurence",
        "string12_0_coexpression",
        "string12_0_experimental",
        "string12_0_database",
    ]

    gene_multigraph = build_gene_multigraph(
        graph=graph,
        graph_names=graph_names
    )

    # Load metabolism
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {
        "metabolism_bipartite": yeast_gem.bipartite_graph
    }

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Test with different num_hops values
    for num_hops in [1, 2, 3]:
        print(f"\n{'='*80}")
        print(f"Testing with num_hops={num_hops}")
        print(f"{'='*80}\n")

        # Create dataset with NeighborSubgraphRepresentation
        print(f"Creating dataset with {num_hops}-hop neighborhood sampling...")
        dataset = Neo4jCellDataset(
            root=dataset_root,
            query=query,
            gene_set=genome.gene_set,
            graphs=gene_multigraph,
            incidence_graphs=incidence_graphs,
            node_embeddings=None,
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=NeighborSubgraphRepresentation(num_hops=num_hops),
            transform=None,
        )

        print(f"Dataset length: {len(dataset)}")
        print(f"Full gene set size: {len(genome.gene_set)}")
        print()

        # Get sample 0
        print("Loading sample 0...")
        sample = dataset[0]
        print()

        # Inspect node data
        print("-" * 80)
        print(f"Sample 0 - Node Data ({num_hops}-hop neighborhood)")
        print("-" * 80)
        print()

        print("Gene node attributes:")
        print(f"  node_ids: {len(sample['gene'].node_ids)} genes (in {num_hops}-hop neighborhood)")
        print(f"  ids_pert: {sample['gene'].ids_pert} ({len(sample['gene'].ids_pert)} perturbed)")
        print(f"  perturbation_indices: {sample['gene'].perturbation_indices} (original indices)")
        print(f"  x: {sample['gene'].x.shape}")
        print(f"  x_pert: {sample['gene'].x_pert.shape}")
        print(f"  pert_mask: {sample['gene'].pert_mask.shape}, {sample['gene'].pert_mask.sum().item()} True (perturbed)")
        print()

        # Inspect edge data
        print("-" * 80)
        print(f"Sample 0 - Edge Data ({num_hops}-hop induced subgraph)")
        print("-" * 80)
        print()

        total_edges = 0
        for et in sample.edge_types:
            if et[0] == "gene" and et[2] == "gene":
                edge_index = sample[et].edge_index
                num_edges = sample[et].num_edges

                print(f"{et}:")
                print(f"  edge_index: {edge_index.shape}")
                print(f"  num_edges: {num_edges}")
                print()

                total_edges += num_edges

        # Inspect metabolism data if present
        if "reaction" in sample.node_types:
            print("-" * 80)
            print("Metabolism Data")
            print("-" * 80)
            print()

            print("Reaction nodes:")
            print(f"  num_nodes: {sample['reaction'].num_nodes}")
            print()

            if ("gene", "gpr", "reaction") in sample.edge_types:
                gpr_et = ("gene", "gpr", "reaction")
                gpr_mask = sample[gpr_et].mask
                print(f"{gpr_et}:")
                print(f"  hyperedge_index: {sample[gpr_et].hyperedge_index.shape}")
                print(f"  mask: {gpr_mask.sum().item()} / {len(gpr_mask)} edges kept")
                print()

            if ("reaction", "rmr", "metabolite") in sample.edge_types:
                rmr_et = ("reaction", "rmr", "metabolite")
                rmr_mask = sample[rmr_et].mask
                print(f"{rmr_et}:")
                print(f"  hyperedge_index: {sample[rmr_et].hyperedge_index.shape}")
                print(f"  stoichiometry: {sample[rmr_et].stoichiometry.shape}")
                print(f"  mask: {rmr_mask.sum().item()} / {len(rmr_mask)} edges kept")
                print()

            if "metabolite" in sample.node_types:
                print("Metabolite nodes:")
                print(f"  num_nodes: {sample['metabolite'].num_nodes} (all metabolites kept)")
                print()

        # Size comparison
        print("-" * 80)
        print("Size Comparison")
        print("-" * 80)
        print()

        full_graph_nodes = len(genome.gene_set)
        subgraph_nodes = len(sample['gene'].node_ids)
        reduction = (1 - subgraph_nodes / full_graph_nodes) * 100

        print(f"Full cell graph: {full_graph_nodes} nodes")
        print(f"{num_hops}-hop subgraph: {subgraph_nodes} nodes")
        print(f"Node reduction: {reduction:.1f}%")
        print()

        # Rough edge count comparison
        full_graph_edges = sum(
            dataset.cell_graph[et].edge_index.size(1)
            for et in dataset.cell_graph.edge_types
            if et[0] == "gene" and et[2] == "gene"
        )
        print(f"Full graph edges: {full_graph_edges:,}")
        print(f"{num_hops}-hop subgraph edges: {total_edges:,}")
        edge_reduction = (1 - total_edges / full_graph_edges) * 100
        print(f"Edge reduction: {edge_reduction:.1f}%")
        print()

    print("="*80)
    print("Inspection complete!")
    print("="*80)


if __name__ == "__main__":
    # inspect_data()
    load_sample_data_batch()
