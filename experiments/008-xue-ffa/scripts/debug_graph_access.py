#!/usr/bin/env python3
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# Initialize
genome = SCerevisiaeGenome(
    genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
    go_root=osp.join(DATA_ROOT, "data/go"),
    overwrite=False
)

graph = SCerevisiaeGraph(
    sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
    string_root=osp.join(DATA_ROOT, "data/string"),
    tflink_root=osp.join(DATA_ROOT, "data/tflink"),
    genome=genome
)

# Test gene names from the data
test_genes = ['FKH1', 'GCN5', 'MED4']

print("Testing graph access...")
print(f"Test genes: {test_genes}")
print()

# Test physical graph
print("=" * 60)
print("PHYSICAL GRAPH")
print("=" * 60)
physical = graph.G_physical
print(f"Type: {type(physical)}")
print(f"Graph type: {type(physical.graph)}")
print(f"Nodes: {physical.graph.number_of_nodes()}")
print(f"Edges: {physical.graph.number_of_edges()}")

# Check if test genes are in the graph
for gene in test_genes:
    in_graph = gene in physical.graph.nodes()
    print(f"  {gene} in graph: {in_graph}")

# Check for edges
print(f"\nEdge checks:")
print(f"  FKH1-GCN5: {physical.graph.has_edge('FKH1', 'GCN5') or physical.graph.has_edge('GCN5', 'FKH1')}")
print(f"  GCN5-MED4: {physical.graph.has_edge('GCN5', 'MED4') or physical.graph.has_edge('MED4', 'GCN5')}")
print(f"  MED4-FKH1: {physical.graph.has_edge('MED4', 'FKH1') or physical.graph.has_edge('FKH1', 'MED4')}")

# Sample some nodes
print(f"\nSample nodes: {list(physical.graph.nodes())[:10]}")

# Test STRING 12.0 experimental
print()
print("=" * 60)
print("STRING 12.0 EXPERIMENTAL")
print("=" * 60)
string_exp = graph.G_string12_0_experimental
print(f"Type: {type(string_exp)}")
print(f"Graph type: {type(string_exp.graph)}")
print(f"Nodes: {string_exp.graph.number_of_nodes()}")
print(f"Edges: {string_exp.graph.number_of_edges()}")

for gene in test_genes:
    in_graph = gene in string_exp.graph.nodes()
    print(f"  {gene} in graph: {in_graph}")

# Check for edges
print(f"\nEdge checks:")
print(f"  FKH1-GCN5: {string_exp.graph.has_edge('FKH1', 'GCN5') or string_exp.graph.has_edge('GCN5', 'FKH1')}")
print(f"  GCN5-MED4: {string_exp.graph.has_edge('GCN5', 'MED4') or string_exp.graph.has_edge('MED4', 'GCN5')}")
print(f"  MED4-FKH1: {string_exp.graph.has_edge('MED4', 'FKH1') or string_exp.graph.has_edge('FKH1', 'MED4')}")

print(f"\nSample nodes: {list(string_exp.graph.nodes())[:10]}")
