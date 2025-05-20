# torchcell/data/cell_data
# [[torchcell.data.cell_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/cell_data
# Test file: tests/torchcell/data/test_cell_data.py


import torch
import networkx as nx
import hypernetx as hnx
from collections import defaultdict, deque
from torchcell.data.hetero_data import HeteroData
from torch_geometric.utils import add_remaining_self_loops
from torchcell.graph import GeneMultiGraph


def to_cell_data(
    multigraph: GeneMultiGraph,
    incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] = None,
    add_remaining_gene_self_loops: bool = True,
) -> HeteroData:
    """Convert GeneMultiGraph and incidence graphs to HeteroData format."""
    hetero_data = HeteroData()

    # Extract NetworkX graphs from GeneMultiGraph
    graph_dict = {name: gene_graph.graph for name, gene_graph in multigraph.items()}

    # Ensure there's a "base" graph with all nodes
    if "base" not in graph_dict:
        raise ValueError("GeneMultiGraph must contain a 'base' graph")

    # Base nodes setup
    base_nodes_list = sorted(list(graph_dict["base"].nodes()))
    node_idx_mapping = {node: idx for idx, node in enumerate(base_nodes_list)}
    num_nodes = len(base_nodes_list)

    # Initialize gene attributes
    hetero_data["gene"].num_nodes = num_nodes
    hetero_data["gene"].node_ids = base_nodes_list
    hetero_data["gene"].x = torch.zeros((num_nodes, 0), dtype=torch.float)

    # Process each graph for edges and embeddings
    for graph_type, graph in graph_dict.items():
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

            # Add edges with simplified type names (without "_interaction" suffix)
            if graph_type != "base":
                edge_type = ("gene", f"{graph_type}", "gene")
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

    # Process incidence graphs if provided
    if incidence_graphs is not None:
        # Process metabolism bipartite representation
        if "metabolism_bipartite" in incidence_graphs:
            bipartite = incidence_graphs["metabolism_bipartite"]
            _process_metabolism_bipartite(hetero_data, bipartite, node_idx_mapping)

        # Process gene ontology graph
        if "gene_ontology" in incidence_graphs:
            go_graph = incidence_graphs["gene_ontology"]
            _process_gene_ontology(hetero_data, go_graph, node_idx_mapping)

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


def compute_strata(go_graph):
    """
    Compute strata (topological levels) for GO graph from leaves to root.
    A stratum contains nodes that can be processed in parallel.
    
    Args:
        go_graph: NetworkX DiGraph with GO terms as nodes
        
    Returns:
        Dictionary mapping node -> stratum number
    """
    # Build a reversed graph where edges point from child to parent
    # This matches the natural flow of information from specific to general terms
    reversed_graph = go_graph.reverse(copy=True)
    
    # Initialize tracking variables
    in_degree = {}
    for node in reversed_graph.nodes():
        in_degree[node] = reversed_graph.in_degree(node)
    
    # Start with nodes that have no incoming edges (in_degree = 0)
    # These are the leaf nodes in the original graph
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    
    # Initialize strata assignments
    strata = {}
    current_stratum = 0
    
    # Process nodes in batches (breadth-first)
    while queue:
        # All nodes in the current queue get the same stratum
        current_nodes = list(queue)  # Make a copy
        queue.clear()
        
        # Assign current stratum to all nodes in this batch
        for node in current_nodes:
            strata[node] = current_stratum
            
            # Update in-degrees of parent nodes
            for parent in reversed_graph.successors(node):
                in_degree[parent] -= 1
                # If all children of this parent have been processed (in_degree = 0),
                # add it to the queue for the next stratum
                if in_degree[parent] == 0:
                    queue.append(parent)
        
        # Move to next stratum
        current_stratum += 1
    
    # Check if all nodes have been assigned to strata
    if len(strata) != go_graph.number_of_nodes():
        unassigned = set(go_graph.nodes()) - set(strata.keys())
        print(f"Warning: {len(unassigned)} nodes not assigned to strata due to cycles in the GO graph.")
        
        # Handle cycles by assigning remaining nodes to strata higher than any existing one
        if unassigned:
            # Use a simple topological sort algorithm to handle the remaining nodes
            remaining_graph = go_graph.subgraph(unassigned).copy()
            
            # Iteratively find nodes with no outgoing edges and assign them to strata
            while remaining_graph.nodes():
                # Find nodes with no outgoing edges in the remaining graph
                sinks = [n for n in remaining_graph.nodes() if remaining_graph.out_degree(n) == 0]
                
                if not sinks:  # If there are no sinks, there must be a cycle
                    # Just assign all remaining nodes to the next stratum and break
                    for node in remaining_graph.nodes():
                        strata[node] = current_stratum
                    break
                
                # Assign current stratum to these sink nodes
                for node in sinks:
                    strata[node] = current_stratum
                    remaining_graph.remove_node(node)
                
                # Move to next stratum
                current_stratum += 1
    
    return strata

def _process_gene_ontology(hetero_data, go_graph, node_idx_mapping):
    """Process gene ontology graph for DCell model and compute strata for parallel processing."""
    # Extract GO terms as nodes, preserving the hierarchical structure
    go_nodes = list(sorted(go_graph.nodes()))
    go_mapping = {term: idx for idx, term in enumerate(go_nodes)}

    # Store gene ontology nodes
    hetero_data["gene_ontology"].num_nodes = len(go_nodes)
    hetero_data["gene_ontology"].node_ids = go_nodes

    # Create a tensor for gene ontology term features
    # Each node will have a feature indicating the number of genes annotated with it
    x_features = torch.zeros((len(go_nodes), 1), dtype=torch.float)

    # Process gene annotations and parent-child relationships
    gene_to_go_src = []  # Gene indices
    gene_to_go_dst = []  # GO term indices

    go_to_go_src = []  # Child GO term indices
    go_to_go_dst = []  # Parent GO term indices

    # Store gene sets for each GO term for DCell perturbation in tensor format
    term_to_genes_list = []  # Will store [term_idx, gene_idx] pairs
    max_genes_per_term = 0  # Track maximum number of genes per term
    term_gene_counts = torch.zeros(len(go_nodes), dtype=torch.int64)

    # Track which terms have which genes for more efficient lookups
    term_to_gene_dict = {}  # Maps term_idx to list of gene_idx values

    # Process each GO term
    for term, data in go_graph.nodes(data=True):
        term_idx = go_mapping[term]
        term_to_gene_dict[term_idx] = []

        # Store gene set for each GO term
        if 'gene_set' in data:
            gene_set = data['gene_set']
            x_features[term_idx, 0] = len(gene_set)
            term_gene_count = 0

            # Create gene-to-GO edges and track genes for each GO term
            for gene in gene_set:
                if gene in node_idx_mapping:
                    gene_idx = node_idx_mapping[gene]
                    gene_to_go_src.append(gene_idx)
                    gene_to_go_dst.append(term_idx)

                    # Add to term-gene pairs list
                    term_to_genes_list.append([term_idx, gene_idx])
                    term_to_gene_dict[term_idx].append(gene_idx)
                    term_gene_count += 1

            # Store the count of genes for this term
            term_gene_counts[term_idx] = term_gene_count
            max_genes_per_term = max(max_genes_per_term, term_gene_count)

    # Process GO hierarchy (parent-child relationships)
    for u, v in go_graph.edges():
        if u in go_mapping and v in go_mapping:
            # In NetworkX edge (u,v) means u is a child of v
            child_idx = go_mapping[u]
            parent_idx = go_mapping[v]
            go_to_go_src.append(child_idx)
            go_to_go_dst.append(parent_idx)

    # Store GO term features
    hetero_data["gene_ontology"].x = x_features.cpu()

    # Store gene-term associations in tensor format for efficient DCell perturbations
    if term_to_genes_list:
        term_to_genes_tensor = torch.tensor(term_to_genes_list, dtype=torch.long).cpu()
        hetero_data["gene_ontology"].term_gene_mapping = term_to_genes_tensor
        hetero_data["gene_ontology"].term_gene_counts = term_gene_counts.cpu()
        hetero_data["gene_ontology"].term_to_gene_dict = term_to_gene_dict

    # Track maximum genes per term for tensor allocation
    hetero_data["gene_ontology"].max_genes_per_term = max_genes_per_term

    # Create a reverse mapping from indices to term IDs
    term_id_list = list(go_mapping.keys())
    hetero_data["gene_ontology"].term_ids = term_id_list

    # Store gene-to-GO edges
    if gene_to_go_src:
        gene_to_go_edge_index = torch.stack([
            torch.tensor(gene_to_go_src, dtype=torch.long),
            torch.tensor(gene_to_go_dst, dtype=torch.long)
        ]).cpu()

        hetero_data["gene", "has_annotation", "gene_ontology"].edge_index = gene_to_go_edge_index
        hetero_data["gene", "has_annotation", "gene_ontology"].num_edges = len(gene_to_go_src)

    # Store GO-to-GO edges (hierarchy)
    if go_to_go_src:
        go_to_go_edge_index = torch.stack([
            torch.tensor(go_to_go_src, dtype=torch.long),
            torch.tensor(go_to_go_dst, dtype=torch.long)
        ]).cpu()

        hetero_data["gene_ontology", "is_child_of", "gene_ontology"].edge_index = go_to_go_edge_index
        hetero_data["gene_ontology", "is_child_of", "gene_ontology"].num_edges = len(go_to_go_src)
        
    # Compute strata for parallel processing
    strata_dict = compute_strata(go_graph)
    
    # Create tensor of strata (one per GO term)
    strata_tensor = torch.zeros(len(go_nodes), dtype=torch.int64)
    for term, idx in go_mapping.items():
        strata_tensor[idx] = strata_dict.get(term, 0)
    
    # Store strata information in the graph
    hetero_data["gene_ontology"].strata = strata_tensor.cpu()
    
    # Also store mapping from stratum -> terms for efficient lookup
    stratum_to_terms = defaultdict(list)
    for term, stratum in strata_dict.items():
        term_idx = go_mapping.get(term)
        if term_idx is not None:
            stratum_to_terms[stratum].append(term_idx)
    
    hetero_data["gene_ontology"].stratum_to_terms = {
        stratum: torch.tensor(terms, dtype=torch.long).cpu() 
        for stratum, terms in stratum_to_terms.items()
    }
    
    # Print strata statistics
    print(f"Computed {len(stratum_to_terms)} strata for {len(go_nodes)} GO terms")
    for stratum, terms in sorted(stratum_to_terms.items())[:5]:
        print(f"  Stratum {stratum}: {len(terms)} terms")
    if len(stratum_to_terms) > 5:
        print(f"  ... and {len(stratum_to_terms)-5} more strata")


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
