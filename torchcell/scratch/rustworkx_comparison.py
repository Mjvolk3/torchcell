import rustworkx as rx
import networkx as nx
import timeit
import numpy as np


# Function to create and remove nodes in rustworkx graphs
def rustworkx_batch_operation():
    graphs = []
    for _ in range(32):
        graph = rx.PyGraph()
        nodes = [graph.add_node(i) for i in range(1000)]
        graph.add_edges_from(
            [(i, (i + 1) % 1000, 1.0) for i in range(1000)]
        )  # Added weight
        graphs.append((graph, nodes))

    for i, (graph, nodes) in enumerate(graphs):
        graph.remove_node(nodes[i])


# Function to create and remove nodes in networkx graphs
def networkx_batch_operation():
    graphs = []
    for _ in range(32):
        graph = nx.Graph()
        graph.add_nodes_from(range(1000))
        graph.add_edges_from([(i, (i + 1) % 1000) for i in range(1000)])
        graphs.append(graph)

    for i, graph in enumerate(graphs):
        graph.remove_node(i)


# Measure rustworkx performance
rustworkx_time = timeit.timeit(
    "rustworkx_batch_operation()",
    setup="from __main__ import rustworkx_batch_operation",
    number=100,
)
print(f"rustworkx batch operation time: {rustworkx_time:.6f} seconds")

# Measure networkx performance
networkx_time = timeit.timeit(
    "networkx_batch_operation()",
    setup="from __main__ import networkx_batch_operation",
    number=100,
)
print(f"networkx batch operation time: {networkx_time:.6f} seconds")


# The improved vectorized function for rustworkx, focusing on efficient graph operations
def rustworkx_vectorized_optimized():
    graphs = []
    nodes = list(range(1000))
    edges = [(i, (i + 1) % 1000, 1.0) for i in range(1000)]

    for _ in range(32):
        g = rx.PyGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        graphs.append((g, nodes))

    for i, (graph, nodes) in enumerate(graphs):
        graph.remove_node(nodes[i])


# The improved vectorized function for networkx, focusing on efficient graph operations
def networkx_vectorized_optimized():
    graphs = []
    nodes = list(range(1000))
    edges = [(i, (i + 1) % 1000) for i in range(1000)]

    for _ in range(32):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        graphs.append(graph)

    for i, graph in enumerate(graphs):
        graph.remove_node(i)


# Measure optimized rustworkx vectorized performance
rustworkx_vectorized_optimized_time = timeit.timeit(
    "rustworkx_vectorized_optimized()",
    setup="from __main__ import rustworkx_vectorized_optimized",
    number=100,
)
print(
    f"rustworkx optimized vectorized operation time: {rustworkx_vectorized_optimized_time:.6f} seconds"
)

# Measure optimized networkx vectorized performance
networkx_vectorized_optimized_time = timeit.timeit(
    "networkx_vectorized_optimized()",
    setup="from __main__ import networkx_vectorized_optimized",
    number=100,
)
print(
    f"networkx optimized vectorized operation time: {networkx_vectorized_optimized_time:.6f} seconds"
)
