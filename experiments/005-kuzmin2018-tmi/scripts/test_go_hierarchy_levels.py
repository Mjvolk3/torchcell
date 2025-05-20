# experiments/005-kuzmin2018-tmi/scripts/test_go_hierarchy_levels
# [[experiments.005-kuzmin2018-tmi.scripts.test_go_hierarchy_levels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/test_go_hierarchy_levels
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_test_go_hierarchy_levels.py


import os
import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dotenv import load_dotenv
import torch
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.timestamp import timestamp

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "assets/images")

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
    
    # Access the GO graph
    go_graph = graph.G_go
    print(f"GO graph has {go_graph.number_of_nodes()} nodes and {go_graph.number_of_edges()} edges")
    
    # Group nodes by level
    nodes_by_level = defaultdict(list)
    for node, data in go_graph.nodes(data=True):
        level = data.get('level', None)
        if level is not None:
            nodes_by_level[level].append(node)
    
    print(f"Found {len(nodes_by_level)} different levels in the GO hierarchy")
    for level in sorted(nodes_by_level.keys()):
        print(f"Level {level}: {len(nodes_by_level[level])} nodes")
    
    # Test for cross-level connections
    has_cross_level_edges = False
    cross_level_edges = []
    same_level_edges = []
    
    for u, v in go_graph.edges():
        u_level = go_graph.nodes[u].get('level', None)
        v_level = go_graph.nodes[v].get('level', None)
        
        if u_level is None or v_level is None:
            print(f"Warning: Edge ({u}, {v}) has a node with no level")
            continue
            
        if u_level == v_level:
            same_level_edges.append((u, v))
        else:
            cross_level_edges.append((u, v, u_level, v_level))
            has_cross_level_edges = True
    
    print(f"\nTotal edges: {go_graph.number_of_edges()}")
    print(f"Same-level edges: {len(same_level_edges)}")
    print(f"Cross-level edges: {len(cross_level_edges)}")
    
    if has_cross_level_edges:
        print("\nFound cross-level connections!")
        print("Sample of cross-level edges (node1, node2, level1, level2):")
        for i, (u, v, u_level, v_level) in enumerate(cross_level_edges[:10]):
            print(f"  {i+1}. ({u}, {v}, {u_level}, {v_level})")
            print(f"     - {go_graph.nodes[u]['name']} -> {go_graph.nodes[v]['name']}")
        
        # Analyze the pattern of cross-level connections
        level_diffs = [abs(e[2] - e[3]) for e in cross_level_edges]
        print(f"\nLevel difference statistics:")
        print(f"  Min difference: {min(level_diffs)}")
        print(f"  Max difference: {max(level_diffs)}")
        print(f"  Average difference: {sum(level_diffs) / len(level_diffs):.2f}")
        
        # Count edges by level difference
        diff_counts = defaultdict(int)
        for diff in level_diffs:
            diff_counts[diff] += 1
        
        print("\nEdge counts by level difference:")
        for diff in sorted(diff_counts.keys()):
            print(f"  Difference {diff}: {diff_counts[diff]} edges")
        
        # Analyze direction of edges (higher to lower or vice versa)
        higher_to_lower = sum(1 for u, v, u_level, v_level in cross_level_edges if u_level > v_level)
        lower_to_higher = sum(1 for u, v, u_level, v_level in cross_level_edges if u_level < v_level)
        
        print(f"\nEdge direction analysis:")
        print(f"  Higher level to lower level: {higher_to_lower} edges")
        print(f"  Lower level to higher level: {lower_to_higher} edges")
        
        # Create visualizations
        plt.figure(figsize=(10, 6))
        plt.bar(sorted(diff_counts.keys()), [diff_counts[k] for k in sorted(diff_counts.keys())])
        plt.xlabel('Level Difference')
        plt.ylabel('Number of Edges')
        plt.title('GO Cross-Level Edge Distribution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = osp.join(ASSET_IMAGES_DIR, f"go_cross_level_edges_{timestamp()}.png")
        plt.savefig(plot_path, dpi=300)
        print(f"\nPlot saved to: {plot_path}")
        
        # For our optimization strategy, check if levels can be processed in order
        # We need to ensure nodes at level L only depend on nodes at levels < L
        can_process_in_order = (lower_to_higher == 0)
        print(f"\nCan process levels in order (higher to lower): {can_process_in_order}")
        
        if not can_process_in_order:
            # Find problematic edges (lower level to higher level)
            problematic_edges = [(u, v, u_level, v_level) for u, v, u_level, v_level in cross_level_edges 
                                 if u_level < v_level]
            print(f"\nFound {len(problematic_edges)} problematic edges (lower level to higher level)")
            print("Sample of problematic edges:")
            for i, (u, v, u_level, v_level) in enumerate(problematic_edges[:5]):
                print(f"  {i+1}. ({u}, {v}, {u_level}, {v_level})")
                print(f"     - {go_graph.nodes[u]['name']} -> {go_graph.nodes[v]['name']}")
    else:
        print("\nNo cross-level connections found in the GO hierarchy.")
        print("This confirms we can safely batch process GO nodes by level.")
    
    # Check for connections between nodes at the same level
    if same_level_edges:
        print(f"\nFound {len(same_level_edges)} same-level connections")
        print("Sample of same-level edges:")
        for i, (u, v) in enumerate(same_level_edges[:5]):
            u_level = go_graph.nodes[u].get('level')
            print(f"  {i+1}. ({u}, {v}) at level {u_level}")
            print(f"     - {go_graph.nodes[u]['name']} -> {go_graph.nodes[v]['name']}")
        
        # Count edges by level
        level_edge_counts = defaultdict(int)
        for u, v in same_level_edges:
            level = go_graph.nodes[u].get('level')
            level_edge_counts[level] += 1
        
        print("\nSame-level edge counts by level:")
        for level in sorted(level_edge_counts.keys()):
            print(f"  Level {level}: {level_edge_counts[level]} edges")
            
        # This would affect our ability to batch nodes at the same level independently
        print("\nNOTE: Same-level connections mean nodes at the same level might depend on each other.")
        print("For optimization, we would need to check if these create cycles within a level.")
    else:
        print("\nNo same-level connections found in the GO hierarchy.")
        print("This confirms we can batch process all nodes within the same level independently.")
    
    # Create a DAG from the original graph to check for cycles within levels
    has_cycles_in_levels = False
    
    for level, nodes in nodes_by_level.items():
        if len(nodes) > 1:  # Only check levels with multiple nodes
            # Create a subgraph of just this level
            level_subgraph = go_graph.subgraph(nodes)
            
            # Check for cycles
            try:
                cycles = list(nx.simple_cycles(level_subgraph))
                if cycles:
                    has_cycles_in_levels = True
                    print(f"\nFound {len(cycles)} cycles within level {level}")
                    print("Sample cycle:", cycles[0])
            except nx.NetworkXNoCycle:
                # No cycles found in this level
                pass
    
    if not has_cycles_in_levels:
        print("\nNo cycles found within any level.")
        print("This confirms we can safely process all nodes within a level in a single batch.")
    
    # Summarize results for optimization strategy
    print("\n=== OPTIMIZATION STRATEGY SUMMARY ===")
    
    if has_cross_level_edges:
        if can_process_in_order:
            print("1. GO nodes at different levels are connected, but all connections go from higher to lower levels.")
            print("2. We can process levels in descending order (highest level first).")
            print("3. Each level can be processed in a single batch after all its dependencies (higher levels) are processed.")
        else:
            print("1. GO nodes at different levels are connected, with some connections going from lower to higher levels.")
            print("2. This creates cycles in the level dependency graph, preventing simple level-by-level processing.")
            print("3. For optimization, we would need to use a more complex approach like strongly connected components.")
    else:
        print("1. No cross-level connections found - each level is independent of other levels.")
        print("2. We can process each level independently and in any order.")
    
    if same_level_edges:
        if has_cycles_in_levels:
            print("3. Some levels contain cycles, so nodes within a level depend on each other recursively.")
            print("4. For optimization, nodes within a level would need sequential processing.")
        else:
            print("3. Nodes within the same level may have connections but no cycles.")
            print("4. For optimization, we can use topological sorting within each level.")
    else:
        print("3. No connections between nodes at the same level.")
        print("4. All nodes within a level can be processed in a single parallel batch.")
    
    return go_graph

if __name__ == "__main__":
    main()