# experiments/005-kuzmin2018-tmi/scripts/test_go_topological_batching
# [[experiments.005-kuzmin2018-tmi.scripts.test_go_topological_batching]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/test_go_topological_batching
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_test_go_topological_batching.py


import os
import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dotenv import load_dotenv
import torch
import numpy as np
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph.graph import (
    SCerevisiaeGraph, 
    filter_by_contained_genes, 
    filter_redundant_terms,
    filter_go_IGI
)
from torchcell.timestamp import timestamp
from torchcell.scratch.load_batch_005 import load_sample_data_batch

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "assets/images")

def topological_batch_assignment(graph):
    """
    Assigns each node to a batch based on a modified topological sort.
    Nodes in the same batch can be processed in parallel.
    
    Returns:
        - batch_assignments: Dictionary mapping node -> batch number
        - batches: List of lists, where each inner list contains nodes in the same batch
    """
    # Initialize tracking variables
    in_degree = {}
    for node in graph.nodes():
        in_degree[node] = graph.in_degree(node)
    
    # Start with nodes that have no incoming edges (in_degree = 0)
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    
    # Initialize batch assignments
    batch_assignments = {}
    current_batch = 0
    batches = []
    
    # Process nodes in batches
    while queue:
        # Start a new batch
        current_nodes = []
        batches.append([])
        
        # Process all nodes currently in the queue
        for _ in range(len(queue)):
            node = queue.popleft()
            batch_assignments[node] = current_batch
            current_nodes.append(node)
            batches[current_batch].append(node)
        
        # Update in-degrees based on processed nodes and add new nodes to queue
        for node in current_nodes:
            for successor in graph.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Move to next batch
        current_batch += 1
    
    # Check if all nodes are assigned to batches
    if len(batch_assignments) != graph.number_of_nodes():
        unprocessed = set(graph.nodes()) - set(batch_assignments.keys())
        print(f"Warning: {len(unprocessed)} nodes not assigned to batches!")
        print(f"This means the graph has cycles that prevent complete topological sorting.")
    
    return batch_assignments, batches

def analyze_batch_efficiency(graph, batches):
    """Analyze the efficiency of the batch assignment"""
    num_nodes = graph.number_of_nodes()
    num_batches = len(batches)
    
    # Calculate batch sizes
    batch_sizes = [len(batch) for batch in batches]
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)
    max_batch_size = max(batch_sizes)
    min_batch_size = min(batch_sizes)
    
    # Calculate efficiency metrics
    # Note: Perfect parallelization would have all nodes in a single batch
    parallel_efficiency = sum(batch_sizes) / (num_batches * max_batch_size)
    sequential_ratio = num_batches / num_nodes
    
    print(f"\nBatch efficiency analysis:")
    print(f"  Total nodes: {num_nodes}")
    print(f"  Total batches: {num_batches}")
    print(f"  Average batch size: {avg_batch_size:.2f}")
    print(f"  Maximum batch size: {max_batch_size}")
    print(f"  Minimum batch size: {min_batch_size}")
    print(f"  Parallel efficiency: {parallel_efficiency:.2f}")
    print(f"  Sequential ratio: {sequential_ratio:.4f}")
    
    # Plot batch size distribution
    plt.figure(figsize=(12, 6))
    
    # Main bar chart
    batch_indices = list(range(num_batches))
    plt.bar(batch_indices, batch_sizes)
    plt.title('GO Node Batch Sizes for Parallel Processing')
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Nodes')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add cumulative percentage line
    cumulative_nodes = np.cumsum(batch_sizes)
    cumulative_percentage = 100 * cumulative_nodes / num_nodes
    
    ax2 = plt.twinx()
    ax2.plot(batch_indices, cumulative_percentage, 'r-', linewidth=2)
    ax2.set_ylabel('Cumulative % of Nodes', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, 110])  # Give a little space above 100%
    
    # Save plot
    plot_path = osp.join(ASSET_IMAGES_DIR, f"go_batch_distribution_{timestamp()}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nBatch size distribution plot saved to: {plot_path}")
    
    return {
        'num_nodes': num_nodes,
        'num_batches': num_batches,
        'avg_batch_size': avg_batch_size,
        'max_batch_size': max_batch_size,
        'min_batch_size': min_batch_size,
        'parallel_efficiency': parallel_efficiency,
        'sequential_ratio': sequential_ratio,
        'batch_sizes': batch_sizes,
    }

def validate_batch_assignments(graph, batches):
    """Validates that nodes in the same batch don't depend on each other"""
    for batch_idx, batch in enumerate(batches):
        # Skip batches with only one node
        if len(batch) <= 1:
            continue
        
        # Create a subgraph of nodes in this batch
        subgraph = graph.subgraph(batch)
        
        # Check if there are any edges in the subgraph
        if subgraph.number_of_edges() > 0:
            print(f"\nWarning: Batch {batch_idx} has {subgraph.number_of_edges()} internal dependencies!")
            print(f"  Sample edge: {list(subgraph.edges())[0]}")
            return False
    
    print("\nAll batches validated: Nodes within each batch have no dependencies on each other.")
    return True

def apply_dcell_filters(go_graph, min_genes=5):
    """Apply the filters used by DCellGraphProcessor to the GO graph"""
    print("\nApplying DCellGraphProcessor filters...")
    orig_nodes = go_graph.number_of_nodes()
    orig_edges = go_graph.number_of_edges()
    
    # Apply filters
    filtered_graph = filter_by_contained_genes(go_graph, n=min_genes, gene_set=None)
    filtered_graph = filter_redundant_terms(filtered_graph)
    filtered_graph = filter_go_IGI(filtered_graph)
    
    print(f"  Original graph: {orig_nodes} nodes, {orig_edges} edges")
    print(f"  Filtered graph: {filtered_graph.number_of_nodes()} nodes, {filtered_graph.number_of_edges()} edges")
    
    return filtered_graph

def analyze_level_batch_comparison(level_grouped, batch_grouped):
    """Compare level-based vs batch-based grouping"""
    # Count nodes by level
    level_counts = {level: len(nodes) for level, nodes in level_grouped.items()}
    total_levels = len(level_counts)
    max_level_size = max(level_counts.values())
    avg_level_size = sum(level_counts.values()) / total_levels
    
    # Compare with batch approach
    level_vs_batch = {
        'total_groups': (total_levels, len(batch_grouped)),
        'max_group_size': (max_level_size, max(len(batch) for batch in batch_grouped)),
        'avg_group_size': (avg_level_size, sum(len(batch) for batch in batch_grouped) / len(batch_grouped)),
    }
    
    print("\nLevel-based vs Batch-based grouping comparison:")
    print(f"  Total groups (levels vs batches): {total_levels} vs {len(batch_grouped)}")
    print(f"  Max group size (level vs batch): {max_level_size} vs {max(len(batch) for batch in batch_grouped)}")
    print(f"  Avg group size (level vs batch): {avg_level_size:.2f} vs {sum(len(batch) for batch in batch_grouped) / len(batch_grouped):.2f}")
    
    # Create comparison visualization
    plt.figure(figsize=(15, 8))
    
    # Set up the axis for level-based grouping
    ax1 = plt.subplot(2, 1, 1)
    level_sizes = [level_counts[level] for level in sorted(level_counts.keys())]
    level_labels = [f"Level {level}" for level in sorted(level_counts.keys())]
    ax1.bar(level_labels, level_sizes)
    ax1.set_title('Level-Based Grouping')
    ax1.set_ylabel('Number of Nodes')
    ax1.tick_params(axis='x', rotation=45)
    
    # Set up the axis for batch-based grouping
    ax2 = plt.subplot(2, 1, 2)
    batch_sizes = [len(batch) for batch in batch_grouped]
    batch_labels = [f"Batch {i}" for i in range(len(batch_grouped))]
    # Limit to first 20 batches if there are too many
    if len(batch_sizes) > 20:
        batch_sizes = batch_sizes[:20]
        batch_labels = batch_labels[:20]
        batch_labels[-1] = f"Batch {len(batch_grouped)-20}+"
    ax2.bar(batch_labels, batch_sizes)
    ax2.set_title('Batch-Based Grouping')
    ax2.set_ylabel('Number of Nodes')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    # Save comparison plot
    plot_path = osp.join(ASSET_IMAGES_DIR, f"go_level_vs_batch_comparison_{timestamp()}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nComparison plot saved to: {plot_path}")
    
    return level_vs_batch

def main():
    # Load genome and graph
    print("Loading genome and graph...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    
    # Get the GO graph
    go_graph = graph.G_go
    print(f"Original GO graph has {go_graph.number_of_nodes()} nodes and {go_graph.number_of_edges()} edges")
    
    # Apply filters that would be used by the DCellGraphProcessor
    filtered_go_graph = apply_dcell_filters(go_graph)
    
    # Group nodes by level (for comparison)
    nodes_by_level = defaultdict(list)
    for node, data in filtered_go_graph.nodes(data=True):
        level = data.get('level', -100)  # Use a default level if not specified
        nodes_by_level[level].append(node)
    
    print(f"\nAfter filtering, found {len(nodes_by_level)} different levels in the GO hierarchy")
    for level in sorted(nodes_by_level.keys()):
        print(f"Level {level}: {len(nodes_by_level[level])} nodes")
    
    # Create a reversed graph where edges point from parent to child
    # This is because the topological sort should process parent nodes before children
    reversed_graph = filtered_go_graph.reverse(copy=True)
    
    # Assign nodes to batches based on topological ordering
    batch_assignments, batches = topological_batch_assignment(reversed_graph)
    
    # Print batch statistics
    print(f"\nAssigned nodes to {len(batches)} batches for parallel processing")
    print(f"First 10 batches:")
    for i, batch in enumerate(batches[:10]):
        print(f"  Batch {i}: {len(batch)} nodes")
    
    # Analyze batch efficiency
    efficiency_metrics = analyze_batch_efficiency(reversed_graph, batches)
    
    # Validate that batch assignments respect dependencies
    is_valid = validate_batch_assignments(reversed_graph, batches)
    
    # Compare level-based approach with batch-based approach
    level_batch_comparison = analyze_level_batch_comparison(nodes_by_level, batches)
    
    # Check the actual dataset to see what processing we're doing
    print("\nLoading sample data to analyze actual graph processor operations...")
    try:
        dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
            batch_size=4,
            num_workers=0,
            config="dcell",
            is_dense=False,
        )
        print(f"Loaded dataset with {len(dataset)} samples and batch with {batch.num_graphs} graphs")
        print(f"Processed GO graph in dataset has {dataset.cell_graph['gene_ontology'].num_nodes} nodes")
        
        # If mutant_state is available, analyze it
        if hasattr(batch['gene_ontology'], 'mutant_state'):
            mutant_state = batch['gene_ontology'].mutant_state
            print(f"Mutant state tensor shape: {mutant_state.shape}")
            print(f"Number of unique GO terms in mutant_state: {len(torch.unique(mutant_state[:, 0]))}")
            print(f"Number of unique genes in mutant_state: {len(torch.unique(mutant_state[:, 1]))}")
    except Exception as e:
        print(f"Error loading sample data: {e}")
    
    # Final summary and recommendations
    print("\n=== OPTIMIZATION STRATEGY SUMMARY ===")
    print(f"1. Original level-based approach would require {len(nodes_by_level)} sequential processing steps")
    print(f"2. Topological batch approach requires {len(batches)} sequential processing steps")
    improvement = (len(nodes_by_level) - len(batches)) / len(nodes_by_level) if len(nodes_by_level) > 0 else 0
    
    if improvement > 0:
        print(f"3. Topological batching reduces sequential steps by {improvement:.1%}")
    else:
        print(f"3. Topological batching increases sequential steps by {-improvement:.1%}")
    
    # Recommendations for implementation
    print("\n=== IMPLEMENTATION RECOMMENDATIONS ===")
    print("1. Replace the current sequential processing with batch processing:")
    print("   - Group GO terms by topological ranks (as done in this script)")
    print("   - Process each batch in a single GPU operation")
    print("   - Update the forward pass to handle batch-by-batch processing")
    
    print("\n2. Implementation approach:")
    print("   a. During model initialization, compute and store the batch assignments")
    print("   b. In forward pass, iterate over batches instead of individual nodes")
    print("   c. For each batch, concat inputs, process together, then split outputs")
    
    # Return the relevant data for further analysis if needed
    return {
        'filtered_graph': filtered_go_graph,
        'batches': batches,
        'batch_assignments': batch_assignments,
        'efficiency_metrics': efficiency_metrics,
        'nodes_by_level': nodes_by_level,
    }

if __name__ == "__main__":
    main()