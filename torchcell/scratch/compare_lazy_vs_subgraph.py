# torchcell/scratch/compare_lazy_vs_subgraph
# [[torchcell.scratch.compare_lazy_vs_subgraph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/compare_lazy_vs_subgraph

"""
Compare LazySubgraphRepresentation vs SubgraphRepresentation.

Verifies that:
1. Node data is identical
2. Edge index can be recovered by applying masks
3. Memory usage is significantly reduced
"""

import os
import os.path as osp
import random
import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.utils import sort_edge_index
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.graph_processor import SubgraphRepresentation, LazySubgraphRepresentation
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph


def load_dataset_with_processor(graph_processor):
    """Load dataset with specified graph processor."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    graph_names = ["physical", "regulatory"]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

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
        graph_processor=graph_processor,
        transform=None,
    )

    return dataset


def main():
    print("="*80)
    print("Comparing LazySubgraphRepresentation vs SubgraphRepresentation")
    print("="*80)
    print()

    # Load datasets
    print("Loading dataset with SubgraphRepresentation...")
    dataset_subgraph = load_dataset_with_processor(SubgraphRepresentation())
    sample_subgraph = dataset_subgraph[0]

    print("Loading dataset with LazySubgraphRepresentation...")
    dataset_lazy = load_dataset_with_processor(LazySubgraphRepresentation())
    sample_lazy = dataset_lazy[0]
    print()

    # Compare node data
    print("-" * 80)
    print("Node Data Comparison")
    print("-" * 80)
    print()

    print("Checking gene node attributes:")

    # Check x (kept nodes features)
    if torch.equal(sample_subgraph["gene"].x, sample_lazy["gene"].x):
        print("  ✓ gene.x: identical")
    else:
        print(f"  ✗ gene.x: MISMATCH")

    # Check x_pert (perturbed nodes features)
    if torch.equal(sample_subgraph["gene"].x_pert, sample_lazy["gene"].x_pert):
        print("  ✓ gene.x_pert: identical")
    else:
        print(f"  ✗ gene.x_pert: MISMATCH")

    # Check ids_pert (perturbed gene IDs)
    if set(sample_subgraph["gene"].ids_pert) == set(sample_lazy["gene"].ids_pert):
        print("  ✓ gene.ids_pert: identical")
    else:
        print(f"  ✗ gene.ids_pert: MISMATCH")

    # Check perturbation_indices
    if torch.equal(sample_subgraph["gene"].perturbation_indices, sample_lazy["gene"].perturbation_indices):
        print("  ✓ gene.perturbation_indices: identical")
    else:
        print(f"  ✗ gene.perturbation_indices: MISMATCH")

    # Check pert_mask
    if torch.equal(sample_subgraph["gene"].pert_mask, sample_lazy["gene"].pert_mask):
        print("  ✓ gene.pert_mask: identical")
    else:
        print(f"  ✗ gene.pert_mask: MISMATCH")

    print()

    # Compare edge data
    print("-" * 80)
    print("Edge Index Recovery Verification")
    print("-" * 80)
    print()

    print("Attempting to recover SubgraphRepresentation edge_index from LazySubgraphRepresentation...")
    print()

    all_match = True

    # Build node mapping from original indices to relabeled indices
    # SubgraphRepresentation relabels kept nodes to 0-indexed
    kept_node_indices = sample_lazy["gene"].mask.nonzero().squeeze(1)  # Original indices of kept nodes
    node_map = torch.zeros(sample_lazy["gene"].mask.size(0), dtype=torch.long)
    for new_idx, orig_idx in enumerate(kept_node_indices):
        node_map[orig_idx] = new_idx

    for et in sample_subgraph.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            # SubgraphRepresentation: filtered and relabeled edge_index
            edge_index_subgraph = sample_subgraph[et].edge_index

            # LazySubgraphRepresentation: full edge_index + mask
            edge_index_full = sample_lazy[et].edge_index
            edge_mask = sample_lazy[et].mask

            # Apply mask to get filtered edges (still in original numbering)
            edge_index_filtered = edge_index_full[:, edge_mask]

            # Relabel to match SubgraphRepresentation's node mapping
            edge_index_relabeled = torch.stack([
                node_map[edge_index_filtered[0]],
                node_map[edge_index_filtered[1]]
            ])

            print(f"{et}:")
            print(f"  SubgraphRepresentation edges: {edge_index_subgraph.size(1)}")
            print(f"  LazySubgraphRepresentation (after mask): {edge_index_filtered.size(1)}")

            # Check if counts match
            if edge_index_subgraph.size(1) != edge_index_relabeled.size(1):
                print(f"  ✗ Edge count MISMATCH")
                all_match = False
                continue

            # Sort both for comparison (order may differ)
            ei_subgraph_sorted = sort_edge_index(edge_index_subgraph, sort_by_row=True)
            ei_relabeled_sorted = sort_edge_index(edge_index_relabeled, sort_by_row=True)

            # Compare
            if torch.equal(ei_subgraph_sorted, ei_relabeled_sorted):
                print(f"  ✓ Edge indices match after relabeling and sorting!")
            else:
                print(f"  ✗ Edge indices MISMATCH after relabeling")
                all_match = False

                # Show first few edges for debugging
                print(f"    SubgraphRepresentation: {ei_subgraph_sorted[:, :5]}")
                print(f"    LazySubgraphRepresentation (relabeled): {ei_relabeled_sorted[:, :5]}")

                # Find first mismatch
                if ei_subgraph_sorted.shape == ei_relabeled_sorted.shape:
                    diff_mask = ~torch.all(ei_subgraph_sorted == ei_relabeled_sorted, dim=0)
                    if diff_mask.any():
                        first_diff = diff_mask.nonzero()[0].item()
                        print(f"    First difference at edge {first_diff}:")
                        print(f"      SubgraphRepresentation: {ei_subgraph_sorted[:, first_diff]}")
                        print(f"      LazySubgraphRepresentation: {ei_relabeled_sorted[:, first_diff]}")

            print()

    if all_match:
        print("="*80)
        print("SUCCESS: LazySubgraphRepresentation is fully equivalent!")
        print("="*80)
    else:
        print("="*80)
        print("FAILURE: Some edge types do not match")
        print("="*80)

    print()

    # Memory comparison
    print("-" * 80)
    print("Memory Comparison")
    print("-" * 80)
    print()

    import sys

    # Estimate SubgraphRepresentation memory
    subgraph_memory = 0
    for et in sample_subgraph.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            subgraph_memory += sample_subgraph[et].edge_index.element_size() * sample_subgraph[et].edge_index.nelement()

    # Estimate LazySubgraphRepresentation memory
    lazy_memory = 0
    for et in sample_lazy.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            # Full edge_index is shared (zero-copy), only count masks
            lazy_memory += sample_lazy[et].mask.element_size() * sample_lazy[et].mask.nelement()

    print(f"SubgraphRepresentation edge tensors: ~{subgraph_memory / (1024**2):.1f} MB")
    print(f"LazySubgraphRepresentation edge masks: ~{lazy_memory / (1024**2):.1f} MB")
    print(f"Memory reduction: ~{(subgraph_memory - lazy_memory) / (1024**2):.1f} MB")
    print(f"Reduction percentage: {(1 - lazy_memory / subgraph_memory) * 100:.1f}%")
    print()

    print("="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
