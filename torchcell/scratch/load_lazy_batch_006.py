# torchcell/scratch/load_lazy_batch_006
# [[torchcell.scratch.load_lazy_batch_006]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/load_lazy_batch_006

"""
Load data with LazySubgraphRepresentation for masked message passing.

Demonstrates zero-copy edge handling with boolean masks.
"""

import os
import os.path as osp
import random
import sys
import numpy as np
import torch
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
)
from torchcell.datamodules.lazy_collate import lazy_collate_hetero
from tqdm import tqdm
from typing import Literal


def load_sample_data_batch(
    batch_size=2,
    num_workers=2,
    config: Literal["hetero_cell_bipartite"] = "hetero_cell_bipartite",
    is_dense: bool = False,
    use_custom_collate: bool = True,
):
    """
    Load a sample data batch with LazySubgraphRepresentation for masked message passing.

    Args:
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        config: Model configuration (currently only "hetero_cell_bipartite" supported)
        is_dense: Whether to use dense representation (not supported with lazy)
        use_custom_collate: Whether to use custom collate function for zero-copy batching

    Returns:
        Tuple of (dataset, batch, input_channels, max_num_nodes)
    """
    if is_dense:
        raise ValueError("Dense representation not supported with LazySubgraphRepresentation")

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

    # Configuration
    config_options = {
        "hetero_cell_bipartite": {
            "graph_names": ["physical", "regulatory"],
            "use_metabolism": True,
            "follow_batch": ["perturbation_indices"],  # Masks batch automatically
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

    # Load query - same as in hetero_cell_bipartite_dango_gi.py
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Use same dataset root as existing dataset - graph processor is just a transformation
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Create dataset with LazySubgraphRepresentation (same dataset, different processor)
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
        graph_processor=LazySubgraphRepresentation(),
        transform=None,
    )

    # Normalization transform
    norm_configs = {
        "gene_interaction": {"strategy": "standard"}  # z-score: (x - mean) / std
    }
    normalizer = LabelNormalizationTransform(dataset, norm_configs)

    # Print the normalization parameters
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
    if use_custom_collate:
        # Create custom DataLoader with LazyCollater for Lightning compatibility
        # PyG's DataLoader respects collate_fn when it's a Collater instance
        from torch_geometric.loader import DataLoader
        from torchcell.datamodules.lazy_collate import LazyCollater

        loader = DataLoader(
            perturbation_subset_data_module.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=LazyCollater(perturbation_subset_data_module.train_dataset),
        )
        for batch in tqdm(loader):
            break
    else:
        # Use standard dataloader from module
        for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
            break

    input_channels = dataset.cell_graph["gene"].x.size()[-1]

    return dataset, batch, input_channels, max_num_nodes


def inspect_data():
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
    print("LazySubgraphRepresentation Data Structure Inspection")
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

    # Use physical and regulatory networks (HeteroCell configuration)
    graph_names = ["physical", "regulatory"]

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

    # Create dataset with LazySubgraphRepresentation
    print("Creating dataset with LazySubgraphRepresentation...")
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
        graph_processor=LazySubgraphRepresentation(),
        transform=None,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Gene set size: {len(genome.gene_set)}")
    print()

    # Get sample 0
    print("Loading sample 0...")
    sample = dataset[0]
    print()

    # Inspect node data
    print("-" * 80)
    print("Sample 0 - Node Data (same as SubgraphRepresentation)")
    print("-" * 80)
    print()

    print("Gene node attributes:")
    print(f"  node_ids: {len(sample['gene'].node_ids)} genes (kept)")
    print(f"  ids_pert: {sample['gene'].ids_pert} ({len(sample['gene'].ids_pert)} perturbed)")
    print(f"  perturbation_indices: {sample['gene'].perturbation_indices}")
    print(f"  x: {sample['gene'].x.shape}")
    print(f"  x_pert: {sample['gene'].x_pert.shape}")
    print(f"  pert_mask: {sample['gene'].pert_mask.shape}, {sample['gene'].pert_mask.sum().item()} True (perturbed)")
    print(f"  mask: {sample['gene'].mask.shape}, {sample['gene'].mask.sum().item()} True (kept)")
    print(f"  ✓ Masks are inverse: {torch.equal(sample['gene'].mask, ~sample['gene'].pert_mask)}")
    print()

    # Inspect edge data
    print("-" * 80)
    print("Sample 0 - Edge Data (ZERO-COPY with masks)")
    print("-" * 80)
    print()

    for et in sample.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            edge_index = sample[et].edge_index
            num_edges = sample[et].num_edges
            edge_mask = sample[et].mask

            num_kept = edge_mask.sum().item()
            num_removed = (~edge_mask).sum().item()

            print(f"{et}:")
            print(f"  edge_index: {edge_index.shape} [FULL graph, not filtered]")
            print(f"  num_edges: {num_edges} [FULL count, not filtered]")
            print(f"  mask: {edge_mask.shape}")
            print(f"    - {num_kept} True (keep)")
            print(f"    - {num_removed} False (remove)")
            print()

    # Inspect GPR and reaction data (Phase 4.1)
    print("-" * 80)
    print("Sample 0 - GPR Edges and Reactions (Phase 4.1)")
    print("-" * 80)
    print()

    if ("gene", "gpr", "reaction") in sample.edge_types:
        gpr_et = ("gene", "gpr", "reaction")
        gpr_hyperedge_index = sample[gpr_et].hyperedge_index
        gpr_num_edges = sample[gpr_et].num_edges
        gpr_mask = sample[gpr_et].mask

        num_gpr_kept = gpr_mask.sum().item()
        num_gpr_removed = (~gpr_mask).sum().item()

        print(f"{gpr_et}:")
        print(f"  hyperedge_index: {gpr_hyperedge_index.shape} [FULL graph, not filtered]")
        print(f"  num_edges: {gpr_num_edges} [FULL count]")
        print(f"  mask: {gpr_mask.shape}")
        print(f"    - {num_gpr_kept} True (gene not deleted)")
        print(f"    - {num_gpr_removed} False (gene deleted)")
        print()

    # Inspect reaction nodes
    if "reaction" in sample.node_types:
        print("Reaction nodes:")
        print(f"  num_nodes: {sample['reaction'].num_nodes}")
        print(f"  node_ids: {len(sample['reaction'].node_ids)} reactions")

        if hasattr(sample['reaction'], 'pert_mask'):
            num_reaction_invalid = sample['reaction'].pert_mask.sum().item()
            num_reaction_valid = sample['reaction'].mask.sum().item()
            print(f"  pert_mask: {sample['reaction'].pert_mask.shape}")
            print(f"    - {num_reaction_invalid} True (reaction invalid - required genes deleted)")
            print(f"  mask: {sample['reaction'].mask.shape}")
            print(f"    - {num_reaction_valid} True (reaction valid)")
            print(f"  ✓ Masks are inverse: {torch.equal(sample['reaction'].mask, ~sample['reaction'].pert_mask)}")
        print()

    # Inspect metabolite nodes
    if "metabolite" in sample.node_types:
        print("Metabolite nodes:")
        print(f"  num_nodes: {sample['metabolite'].num_nodes}")
        print(f"  node_ids: {len(sample['metabolite'].node_ids)} metabolites")

        if hasattr(sample['metabolite'], 'pert_mask'):
            num_metabolite_removed = sample['metabolite'].pert_mask.sum().item()
            num_metabolite_kept = sample['metabolite'].mask.sum().item()
            print(f"  pert_mask: {sample['metabolite'].pert_mask.shape}")
            print(f"    - {num_metabolite_removed} True (removed)")
            print(f"  mask: {sample['metabolite'].mask.shape}")
            print(f"    - {num_metabolite_kept} True (kept)")
            print(f"  ✓ All metabolites kept: {num_metabolite_removed == 0}")
        print()

    # Inspect RMR edges (Phase 4.2)
    if ("reaction", "rmr", "metabolite") in sample.edge_types:
        print("-" * 80)
        print("Sample 0 - RMR Edges (Phase 4.2)")
        print("-" * 80)
        print()

        rmr_et = ("reaction", "rmr", "metabolite")
        rmr_hyperedge_index = sample[rmr_et].hyperedge_index
        rmr_num_edges = sample[rmr_et].num_edges
        rmr_mask = sample[rmr_et].mask
        rmr_stoichiometry = sample[rmr_et].stoichiometry

        num_rmr_kept = rmr_mask.sum().item()
        num_rmr_removed = (~rmr_mask).sum().item()

        print(f"{rmr_et}:")
        print(f"  hyperedge_index: {rmr_hyperedge_index.shape} [FULL graph, not filtered]")
        print(f"  stoichiometry: {rmr_stoichiometry.shape} [FULL attributes, not filtered]")
        print(f"  num_edges: {rmr_num_edges} [FULL count]")
        print(f"  mask: {rmr_mask.shape}")
        print(f"    - {num_rmr_kept} True (reaction is valid)")
        print(f"    - {num_rmr_removed} False (reaction is invalid)")
        print()

    # Zero-copy verification
    print("-" * 80)
    print("Zero-Copy Verification")
    print("-" * 80)
    print()

    cell_graph = dataset.cell_graph

    # Verify gene-gene edges
    for et in sample.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            is_reference = id(sample[et].edge_index) == id(cell_graph[et].edge_index)
            print(f"{et}:")
            print(f"  edge_index is reference: {is_reference}")
            if is_reference:
                print(f"  ✓ Zero-copy confirmed!")
            else:
                print(f"  ✗ WARNING: edge_index was copied")
            print()

    # Verify GPR edges (Phase 4.1)
    if ("gene", "gpr", "reaction") in sample.edge_types:
        gpr_et = ("gene", "gpr", "reaction")
        is_gpr_reference = id(sample[gpr_et].hyperedge_index) == id(cell_graph[gpr_et].hyperedge_index)
        print(f"{gpr_et}:")
        print(f"  hyperedge_index is reference: {is_gpr_reference}")
        if is_gpr_reference:
            print(f"  ✓ Zero-copy confirmed!")
        else:
            print(f"  ✗ WARNING: hyperedge_index was copied")
        print()

    # Verify RMR edges (Phase 4.2)
    if ("reaction", "rmr", "metabolite") in sample.edge_types:
        rmr_et = ("reaction", "rmr", "metabolite")
        is_rmr_hyperedge_reference = id(sample[rmr_et].hyperedge_index) == id(cell_graph[rmr_et].hyperedge_index)
        is_rmr_stoich_reference = id(sample[rmr_et].stoichiometry) == id(cell_graph[rmr_et].stoichiometry)

        print(f"{rmr_et}:")
        print(f"  hyperedge_index is reference: {is_rmr_hyperedge_reference}")
        print(f"  stoichiometry is reference: {is_rmr_stoich_reference}")
        if is_rmr_hyperedge_reference and is_rmr_stoich_reference:
            print(f"  ✓ Zero-copy confirmed for both tensors!")
        else:
            if not is_rmr_hyperedge_reference:
                print(f"  ✗ WARNING: hyperedge_index was copied")
            if not is_rmr_stoich_reference:
                print(f"  ✗ WARNING: stoichiometry was copied")
        print()

    # Memory estimation
    print("-" * 80)
    print("Memory Comparison Estimate")
    print("-" * 80)
    print()

    total_edges = 0
    for et in sample.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            total_edges += sample[et].num_edges

    # SubgraphRepresentation would copy edge_index (2 × num_edges × 8 bytes per edge)
    # plus relabeled indices, etc.
    subgraph_edge_memory_mb = (total_edges * 2 * 8) / (1024 ** 2)

    # LazySubgraphRepresentation only stores masks (1 bit per edge, but stored as bool = 1 byte)
    lazy_edge_memory_mb = total_edges / (1024 ** 2)

    print(f"Total edges across all gene-gene edge types: {total_edges:,}")
    print()
    print(f"SubgraphRepresentation (estimated):")
    print(f"  Edge tensors: ~{subgraph_edge_memory_mb:.1f} MB (filtered + relabeled copies)")
    print()
    print(f"LazySubgraphRepresentation (actual):")
    print(f"  Edge masks: ~{lazy_edge_memory_mb:.1f} MB (boolean masks only)")
    print()
    print(f"Memory savings: ~{subgraph_edge_memory_mb - lazy_edge_memory_mb:.1f} MB")
    print(f"Reduction: {(1 - lazy_edge_memory_mb / subgraph_edge_memory_mb) * 100:.1f}%")
    print()

    # Edge mask application example
    print("-" * 80)
    print("Example: Recovering Filtered Edge Index")
    print("-" * 80)
    print()

    # Pick first gene-gene edge type
    for et in sample.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            print(f"Edge type: {et}")
            print()

            edge_index_full = sample[et].edge_index
            edge_mask = sample[et].mask

            # Apply mask to get filtered edges
            edge_index_filtered = edge_index_full[:, edge_mask]

            print(f"Full edge_index: {edge_index_full.shape}")
            print(f"Edge mask: {edge_mask.sum().item()} / {edge_mask.shape[0]} edges kept")
            print(f"Filtered edge_index: {edge_index_filtered.shape}")
            print()
            print("Sample edges (first 5):")
            print(f"  Full graph: {edge_index_full[:, :5]}")
            print(f"  Filtered: {edge_index_filtered[:, :5]}")
            print()
            print("NOTE: Filtered edges use ORIGINAL node numbering from cell_graph")
            print("      Model must either:")
            print("      1. Filter edges before message passing, OR")
            print("      2. Use mask-aware message passing")
            break

    print("="*80)
    print("Inspection complete!")
    print("="*80)


if __name__ == "__main__":
    # inspect_data()
    load_sample_data_batch()
