# torchcell/data/cell_data
# [[torchcell.data.cell_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/cell_data
# Test file: tests/torchcell/data/test_cell_data.py


import torch
import networkx as nx
import hypernetx as hnx
from torchcell.data.hetero_data import HeteroData
from torch_geometric.utils import add_remaining_self_loops


def to_cell_data(
    graphs: dict[str, nx.Graph],
    incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] = None,
    add_remaining_gene_self_loops: bool = True,
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
                if add_remaining_gene_self_loops:
                    edge_index, _ = add_remaining_self_loops(
                        edge_index, num_nodes=num_nodes
                    )
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

    # Process metabolism graphs if provided
    if incidence_graphs is not None:
        # Process hypergraph representation
        if "metabolism_hypergraph" in incidence_graphs:
            hypergraph = incidence_graphs["metabolism_hypergraph"]
            _process_metabolism_hypergraph(hetero_data, hypergraph, node_idx_mapping)

        # Process bipartite representation
        if "metabolism_bipartite" in incidence_graphs:
            bipartite = incidence_graphs["metabolism_bipartite"]
            _process_metabolism_bipartite(hetero_data, bipartite, node_idx_mapping)

    return hetero_data


def _process_metabolism_hypergraph(hetero_data, hypergraph, node_idx_mapping):
    """Process hypergraph representation of metabolism."""
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
    hetero_data[edge_type].num_edges = len(hyperedge_index[1].unique()) + 1
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


def _process_metabolism_bipartite(hetero_data, bipartite, node_idx_mapping):
    """Process bipartite representation of metabolism with signed stoichiometry values."""
    # Collect nodes by type efficiently
    node_data = {n: d for n, d in bipartite.nodes(data=True)}
    reaction_nodes = [n for n, d in node_data.items() if d["node_type"] == "reaction"]
    metabolite_nodes = [
        n for n, d in node_data.items() if d["node_type"] == "metabolite"
    ]

    # Create mappings
    reaction_mapping = {r: i for i, r in enumerate(sorted(reaction_nodes))}
    metabolite_mapping = {m: i for i, m in enumerate(sorted(metabolite_nodes))}

    # Store nodes
    hetero_data["metabolite"].num_nodes = len(metabolite_nodes)
    hetero_data["metabolite"].node_ids = sorted(metabolite_nodes)
    hetero_data["reaction"].num_nodes = len(reaction_nodes)
    hetero_data["reaction"].node_ids = sorted(reaction_nodes)

    # Create w_growth tensor for reactions (1 for Growth subsystem, 0 otherwise)
    w_growth = torch.zeros(len(reaction_nodes), dtype=torch.float)
    sorted_reaction_nodes = sorted(reaction_nodes)

    # Populate w_growth based on subsystem information
    for i, reaction_node in enumerate(sorted_reaction_nodes):
        subsystem = node_data[reaction_node].get("subsystem", "")
        if subsystem == "Growth":
            w_growth[i] = 1.0

    # Add w_growth tensor to reaction nodes
    hetero_data["reaction"].w_growth = w_growth.cpu()

    # Create reaction-gene mapping in one pass
    reaction_to_genes = {}
    reaction_to_genes_indices = {}
    for reaction_node in reaction_nodes:
        reaction_idx = reaction_mapping[reaction_node]
        genes = node_data[reaction_node].get("genes", set())
        if genes:
            genes_list = list(genes)
            reaction_to_genes[reaction_idx] = genes_list
            reaction_to_genes_indices[reaction_idx] = [
                node_idx_mapping.get(gene, -1) for gene in genes_list
            ]

    # Batch process edges
    src_indices = []
    dst_indices = []
    signed_stoich_values = []

    # Process all edges in a single pass with minimal lookups
    edge_data = [(u, v, d) for u, v, d in bipartite.edges(data=True)]
    for u, v, data in edge_data:
        u_type = node_data[u]["node_type"]
        v_type = node_data[v]["node_type"]
        edge_type = data["edge_type"]
        stoich = data["stoichiometry"]

        # Both cases expect u to be reaction and v to be metabolite
        if u_type == "reaction" and v_type == "metabolite":
            if u in reaction_mapping and v in metabolite_mapping:
                src_indices.append(reaction_mapping[u])
                dst_indices.append(metabolite_mapping[v])

                # Apply sign based on edge_type (reactant or product)
                if edge_type == "reactant":
                    signed_stoich_values.append(-stoich)  # Negative for reactants
                else:  # product
                    signed_stoich_values.append(stoich)  # Positive for products

    if src_indices:  # Only create if we have edges
        # Create tensors in one batch
        hyperedge_index = torch.stack(
            [
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(dst_indices, dtype=torch.long),
            ]
        ).cpu()

        stoich_tensor = torch.tensor(signed_stoich_values, dtype=torch.float).cpu()

        # Store edge data - no longer storing edge_type separately
        edge_type = ("reaction", "rmr", "metabolite")
        hetero_data[edge_type].hyperedge_index = hyperedge_index
        hetero_data[edge_type].stoichiometry = stoich_tensor
        hetero_data[edge_type].num_edges = len(src_indices)
        hetero_data[edge_type].reaction_to_genes = reaction_to_genes
        hetero_data[edge_type].reaction_to_genes_indices = reaction_to_genes_indices

    # Create gene-reaction relationships in one operation
    if reaction_to_genes_indices:
        gpr_gene_indices = []
        gpr_reaction_indices = []

        for reaction_idx, gene_indices in reaction_to_genes_indices.items():
            for gene_idx in gene_indices:
                if gene_idx != -1:  # Skip invalid gene indices
                    gpr_gene_indices.append(gene_idx)
                    gpr_reaction_indices.append(reaction_idx)

        if gpr_gene_indices:
            gpr_edge_index = torch.stack(
                [
                    torch.tensor(gpr_gene_indices, dtype=torch.long),
                    torch.tensor(gpr_reaction_indices, dtype=torch.long),
                ]
            ).cpu()

            hetero_data["gene", "gpr", "reaction"].hyperedge_index = gpr_edge_index
            hetero_data["gene", "gpr", "reaction"].num_edges = len(
                torch.unique(gpr_edge_index[1])
            )
