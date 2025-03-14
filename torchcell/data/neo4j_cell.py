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
from torch_geometric.transforms import NormalizeFeatures

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


# HACK - normalize embedding start
# Probably should reformat data for this, but would need to resave or recompute
def normalize_tensor_row(x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalizes a tensor to sum to 1, after min subtraction."""
    # Handle list input
    if isinstance(x, list):
        x = x[0]

    # Add batch dimension if needed
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x - x.min()
    x = x / x.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return list(x.flatten(-1))


# HACK - normalize embedding end


# @profile
# def create_embedding_graph(
#     gene_set: GeneSet, embeddings: BaseEmbeddingDataset
# ) -> nx.Graph:
#     """
#     Create a NetworkX graph from embeddings.
#     """

#     # Create an empty NetworkX graph
#     G = nx.Graph()

#     # Extract and concatenate embeddings for all items in embeddings
#     for item in embeddings:
#         keys = item["embeddings"].keys()
#         if item.id in gene_set:
#             item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
#             # HACK - normalize embedding - start
#             item_embeddings = normalize_tensor_row(item_embeddings)
#             # HACK - normalize embedding - end
#             concatenated_embedding = torch.cat(item_embeddings)

#             G.add_node(item.id, embedding=concatenated_embedding)

#     return G


def min_max_normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Forces embedding tensor values into [0,1] range using min-max scaling per feature."""
    # Normalize each feature (column) independently
    normalized_embedding = torch.zeros_like(embedding)
    for i in range(embedding.size(1)):
        feature = embedding[:, i]
        feature_min = feature.min()
        feature_max = feature.max()

        # If feature_min == feature_max, set to 0.5 to avoid div by zero
        if feature_min == feature_max:
            normalized_embedding[:, i] = 0.5
        else:
            normalized_embedding[:, i] = (feature - feature_min) / (
                feature_max - feature_min
            )

    return normalized_embedding


def min_max_normalize_dataset(dataset: BaseEmbeddingDataset) -> None:
    """Normalizes embeddings across the entire dataset to range [0,1] using min-max scaling per feature."""
    first_key = list(dataset._data.embeddings.keys())[0]
    embeddings = dataset._data.embeddings[first_key]
    dataset._data.embeddings[first_key] = min_max_normalize_embedding(embeddings)


def create_embedding_graph(
    gene_set: GeneSet, embeddings: BaseEmbeddingDataset
) -> nx.Graph:
    """Create a NetworkX graph from embeddings."""
    # Normalize dataset first
    min_max_normalize_dataset(embeddings)

    G = nx.Graph()
    for item in embeddings:
        keys = item["embeddings"].keys()
        if item.id in gene_set:
            item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
            concatenated_embedding = torch.cat(item_embeddings)
            G.add_node(item.id, embedding=concatenated_embedding)

    return G


# @profile
# TODO we could remove add_remaining_gene_self_loops and put it in transforms
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
    """Process bipartite representation of metabolism with a single directed edge type."""
    # Collect nodes by type
    reaction_nodes = [
        n for n, d in bipartite.nodes(data=True) if d["node_type"] == "reaction"
    ]
    metabolite_nodes = [
        n for n, d in bipartite.nodes(data=True) if d["node_type"] == "metabolite"
    ]

    # Create mappings
    reaction_mapping = {r: i for i, r in enumerate(sorted(reaction_nodes))}
    metabolite_mapping = {m: i for i, m in enumerate(sorted(metabolite_nodes))}

    # Store metabolite nodes
    hetero_data["metabolite"].num_nodes = len(metabolite_nodes)
    hetero_data["metabolite"].node_ids = sorted(metabolite_nodes)

    # Store reaction nodes
    hetero_data["reaction"].num_nodes = len(reaction_nodes)
    hetero_data["reaction"].node_ids = sorted(reaction_nodes)

    # Create reaction-gene mapping
    reaction_to_genes = {}
    reaction_to_genes_indices = {}

    for reaction_node in reaction_nodes:
        # Get reaction index
        reaction_idx = reaction_mapping[reaction_node]
        # Get gene set
        genes = bipartite.nodes[reaction_node].get("genes", set())

        if genes:  # If there are genes associated with this reaction
            reaction_to_genes[reaction_idx] = list(genes)

            # Get gene indices
            gene_indices = []
            for gene in genes:
                gene_idx = node_idx_mapping.get(gene, -1)
                gene_indices.append(gene_idx)

            reaction_to_genes_indices[reaction_idx] = gene_indices

    # Create unified edge lists for the single rmr edge type
    src_indices = []  # Reaction or metabolite index
    dst_indices = []  # Metabolite or reaction index
    edge_types = []  # "reactant" or "product"
    stoich_values = []

    # Process all edges from the bipartite graph
    for u, v, data in bipartite.edges(data=True):
        u_type = bipartite.nodes[u]["node_type"]
        v_type = bipartite.nodes[v]["node_type"]
        edge_type = data["edge_type"]
        stoich = data["stoichiometry"]

        # Handle reactant edges (metabolite -> reaction in original graph)
        if edge_type == "reactant" and u_type == "metabolite" and v_type == "reaction":
            # Convert to (reaction, rmr, metabolite) with role=reactant
            if v in reaction_mapping and u in metabolite_mapping:
                src_indices.append(reaction_mapping[v])  # Reaction is source
                dst_indices.append(metabolite_mapping[u])  # Metabolite is target
                edge_types.append("reactant")
                stoich_values.append(stoich)

        # Handle product edges (reaction -> metabolite in original graph)
        elif edge_type == "product" and u_type == "reaction" and v_type == "metabolite":
            # Already in (reaction, rmr, metabolite) format with role=product
            if u in reaction_mapping and v in metabolite_mapping:
                src_indices.append(reaction_mapping[u])  # Reaction is source
                dst_indices.append(metabolite_mapping[v])  # Metabolite is target
                edge_types.append("product")
                stoich_values.append(stoich)

    if src_indices:  # Only create if we have edges
        # Create edge tensors
        hyperedge_index = torch.stack(
            [
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(dst_indices, dtype=torch.long),
            ]
        ).cpu()

        stoich_tensor = torch.tensor(stoich_values, dtype=torch.float).cpu()
        edge_type_tensor = torch.tensor(
            [1 if t == "product" else 0 for t in edge_types], dtype=torch.long
        ).cpu()

        # Store in a single edge type - use hyperedge_index for bipartite relationships
        hetero_data["reaction", "rmr", "metabolite"].hyperedge_index = hyperedge_index
        hetero_data["reaction", "rmr", "metabolite"].stoichiometry = stoich_tensor
        hetero_data["reaction", "rmr", "metabolite"].edge_type = (
            edge_type_tensor  # 0=reactant, 1=product
        )
        hetero_data["reaction", "rmr", "metabolite"].num_edges = len(src_indices)

    # Store gene-reaction relationships
    if reaction_to_genes_indices:
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

            # Store GPR edge - use hyperedge_index for association
            hetero_data["gene", "gpr", "reaction"].hyperedge_index = gpr_edge_index
            hetero_data["gene", "gpr", "reaction"].num_edges = len(
                torch.unique(gpr_edge_index[1])
            )

    # Store reaction-to-genes mappings
    if ("reaction", "rmr", "metabolite") in hetero_data.edge_types:
        hetero_data["reaction", "rmr", "metabolite"].reaction_to_genes = (
            reaction_to_genes
        )
        hetero_data["reaction", "rmr", "metabolite"].reaction_to_genes_indices = (
            reaction_to_genes_indices
        )


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


from typing import Optional, List, Dict, Union, Tuple
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils._subgraph import (
    subgraph,
    bipartite_subgraph,
    k_hop_subgraph,
    hyper_subgraph,
)
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter


class SubgraphRepresentation(GraphProcessor):
    def __init__(self):
        super().__init__()
        self.device = None
        self.masks = None

    def _initialize_masks(self, cell_graph: HeteroData):
        """Initialize masks for all node types."""
        self.masks = {
            "gene": {
                "kept": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
                "perturbed": torch.zeros(
                    cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
                ),
            },
            "reaction": {
                "kept": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["reaction"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            "metabolite": {
                "kept": torch.zeros(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
                "removed": torch.zeros(
                    cell_graph["metabolite"].num_nodes,
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        }

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[PhenotypeType],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],
    ) -> HeteroData:
        """Process a cell graph by removing perturbed genes and their associated reactions."""
        self.device = cell_graph["gene"].x.device
        self._initialize_masks(cell_graph)

        # Initialize output graph
        integrated_subgraph = HeteroData()

        # Process steps with mask tracking
        gene_info = self._process_gene_info(cell_graph, data)
        self._add_gene_data(integrated_subgraph, cell_graph, gene_info)
        self._process_gene_interactions(integrated_subgraph, cell_graph, gene_info)
        reaction_info = self._process_reaction_info(
            cell_graph, gene_info, integrated_subgraph
        )
        self._add_reaction_data(integrated_subgraph, reaction_info)
        self._process_metabolic_network(integrated_subgraph, cell_graph, reaction_info)
        self._add_phenotype_data(integrated_subgraph, phenotype_info, data)

        # Add masks to their respective nodes
        integrated_subgraph["gene"].pert_mask = self.masks["gene"]["perturbed"]
        integrated_subgraph["reaction"].pert_mask = self.masks["reaction"]["removed"]
        integrated_subgraph["metabolite"].pert_mask = self.masks["metabolite"][
            "removed"
        ]
        return integrated_subgraph

    def _process_gene_info(self, cell_graph: HeteroData, data) -> dict:
        """Process gene information and create masks."""
        perturbed_names = {
            p.systematic_gene_name
            for item in data
            for p in item["experiment"].genotype.perturbations
        }

        node_ids = cell_graph["gene"].node_ids
        keep_idx = [i for i, name in enumerate(node_ids) if name not in perturbed_names]
        remove_idx = [i for i, name in enumerate(node_ids) if name in perturbed_names]

        # Update masks
        self.masks["gene"]["kept"][keep_idx] = True
        self.masks["gene"]["perturbed"][remove_idx] = True

        return {
            "perturbed_names": perturbed_names,
            "keep_subset": torch.tensor(keep_idx, dtype=torch.long, device=self.device),
            "remove_subset": torch.tensor(
                remove_idx, dtype=torch.long, device=self.device
            ),
            "keep_node_ids": [node_ids[i] for i in keep_idx],
        }

    def _add_gene_data(
        self, integrated_subgraph: HeteroData, cell_graph: HeteroData, gene_info: dict
    ):
        """Add gene data to integrated subgraph."""
        integrated_subgraph["gene"].node_ids = gene_info["keep_node_ids"]
        integrated_subgraph["gene"].num_nodes = len(gene_info["keep_node_ids"])
        integrated_subgraph["gene"].ids_pert = list(gene_info["perturbed_names"])
        integrated_subgraph["gene"].cell_graph_idx_pert = gene_info["remove_subset"]

        x_full = cell_graph["gene"].x
        integrated_subgraph["gene"].x = x_full[gene_info["keep_subset"]]
        integrated_subgraph["gene"].x_pert = x_full[gene_info["remove_subset"]]

    def _process_gene_interactions(
        self, integrated_subgraph: HeteroData, cell_graph: HeteroData, gene_info: dict
    ):
        """Process gene-gene interaction edges."""
        edge_types = [
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
        ]

        for edge_type in cell_graph.edge_types:
            if edge_type in edge_types:
                # Get edge information and create edge mask
                orig_edge_index = cell_graph[edge_type].edge_index
                edge_index, _, edge_mask = subgraph(
                    subset=gene_info["keep_subset"],
                    edge_index=orig_edge_index,
                    relabel_nodes=True,
                    num_nodes=cell_graph["gene"].num_nodes,
                    return_edge_mask=True,
                )

                # Store edge data
                integrated_subgraph[edge_type].edge_index = edge_index
                integrated_subgraph[edge_type].num_edges = edge_index.size(1)
                integrated_subgraph[edge_type].pert_mask = ~edge_mask

    def _process_reaction_info(
        self, cell_graph: HeteroData, gene_info: dict, integrated_subgraph: HeteroData
    ) -> dict:
        """Process reaction information and create masks, including reactions affected by gene perturbations."""
        max_reaction_idx = cell_graph["reaction"].num_nodes

        # Initialize all reactions as having no genes
        has_genes = torch.zeros(max_reaction_idx, dtype=torch.bool, device=self.device)

        # Handle case with no GPR relationships
        if ("gene", "gpr", "reaction") not in cell_graph.edge_types:
            # If no GPR exists, all reactions are valid (no gene associations)
            valid_reactions = torch.arange(max_reaction_idx, device=self.device)

            # Update reaction masks
            self.masks["reaction"]["kept"].fill_(True)
            self.masks["reaction"]["removed"].fill_(False)

            # Create mappings for relabeling
            gene_map = torch.full(
                (cell_graph["gene"].num_nodes,),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            gene_map[gene_info["keep_subset"]] = torch.arange(
                len(gene_info["keep_subset"]), device=self.device
            )

            reaction_map = torch.arange(max_reaction_idx, device=self.device)

            # No GPR edges to add to integrated_subgraph
            return {
                "valid_reactions": valid_reactions,
                "gene_map": gene_map.tolist(),
                "reaction_map": reaction_map.tolist(),
            }

        # Move edge index to correct device
        gpr_edge_index = cell_graph["gene", "gpr", "reaction"].hyperedge_index.to(
            self.device
        )

        # Create gene mask on correct device
        gene_mask = torch.zeros(
            cell_graph["gene"].num_nodes, dtype=torch.bool, device=self.device
        )
        gene_mask[gene_info["keep_subset"]] = True

        # Add bounds checking
        max_gene_idx = cell_graph["gene"].num_nodes

        gene_indices = gpr_edge_index[0]
        reaction_indices = gpr_edge_index[1]

        # Add bounds checking
        valid_gene_mask = gene_indices < max_gene_idx
        valid_reaction_mask = reaction_indices < max_reaction_idx
        valid_mask = valid_gene_mask & valid_reaction_mask

        gene_indices = gene_indices[valid_mask]
        reaction_indices = reaction_indices[valid_mask]

        # Mark reactions that have gene associations
        has_genes[reaction_indices] = True

        # Group genes by reaction using scatter operations
        reaction_gene_mask = scatter(
            gene_mask[gene_indices].float(),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )

        total_genes_per_reaction = scatter(
            torch.ones_like(gene_indices, dtype=torch.float),
            reaction_indices,
            dim=0,
            dim_size=max_reaction_idx,
            reduce="sum",
        )

        # Reactions are valid if:
        # 1. They have genes and all those genes are kept, OR
        # 2. They have no genes (always kept)

        # Valid reactions with genes: where all genes are kept
        valid_with_genes_mask = (
            reaction_gene_mask == total_genes_per_reaction
        ) & has_genes

        # Valid reactions without genes
        valid_without_genes_mask = ~has_genes

        # Combined valid reactions mask
        valid_mask = valid_with_genes_mask | valid_without_genes_mask
        valid_reactions = torch.where(valid_mask)[0]

        # Update reaction masks
        self.masks["reaction"]["kept"] = valid_mask
        self.masks["reaction"]["removed"] = ~valid_mask

        # Create edge mask using validated indices
        edge_mask = torch.isin(
            reaction_indices, torch.where(valid_with_genes_mask)[0]
        ) & torch.isin(gene_indices, gene_info["keep_subset"])

        # Filter and relabel edges
        new_gpr_edge_index = gpr_edge_index[:, edge_mask].clone()

        # Create mappings for relabeling
        gene_map = torch.full(
            (cell_graph["gene"].num_nodes,), -1, dtype=torch.long, device=self.device
        )
        gene_map[gene_info["keep_subset"]] = torch.arange(
            len(gene_info["keep_subset"]), device=self.device
        )

        reaction_map = torch.full(
            (max_reaction_idx,), -1, dtype=torch.long, device=self.device
        )
        reaction_map[valid_reactions] = torch.arange(
            len(valid_reactions), device=self.device
        )

        # Relabel indices
        new_gpr_edge_index[0] = gene_map[new_gpr_edge_index[0]]
        new_gpr_edge_index[1] = reaction_map[new_gpr_edge_index[1]]

        # Store edge data
        integrated_subgraph["gene", "gpr", "reaction"].hyperedge_index = (
            new_gpr_edge_index
        )
        integrated_subgraph["gene", "gpr", "reaction"].num_edges = (
            new_gpr_edge_index.size(1)
        )  # Use actual number of edges
        integrated_subgraph["gene", "gpr", "reaction"].pert_mask = ~edge_mask

        return {
            "valid_reactions": valid_reactions,
            "gene_map": gene_map.tolist(),
            "reaction_map": reaction_map.tolist(),
        }

    def _add_reaction_data(self, integrated_subgraph: HeteroData, reaction_info: dict):
        """Add reaction data and gene-reaction edges."""
        if not reaction_info:
            return

        valid_reactions = reaction_info["valid_reactions"]
        integrated_subgraph["reaction"].num_nodes = len(valid_reactions)
        integrated_subgraph["reaction"].node_ids = valid_reactions.tolist()

    def _add_phenotype_data(
        self, integrated_subgraph: HeteroData, phenotype_info: list[PhenotypeType], data
    ):
        """Add phenotype data to the integrated subgraph."""
        phenotype_fields = []
        for phenotype in phenotype_info:
            phenotype_fields.extend(
                [
                    phenotype.model_fields["label_name"].default,
                    phenotype.model_fields["label_statistic_name"].default,
                ]
            )

        for field in phenotype_fields:
            field_values = []
            for item in data:
                value = getattr(item["experiment"].phenotype, field, None)
                if value is not None:
                    field_values.append(value)

            integrated_subgraph["gene"][field] = torch.tensor(
                field_values if field_values else [float("nan")],
                dtype=torch.float,
                device=self.device,
            )

    def _process_metabolism_bipartite(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: dict,
    ):
        """Process bipartite metabolic network for perturbations."""
        if not reaction_info or "reaction" not in cell_graph.node_types:
            return

        # Add metabolite data initially
        integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
        integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes

        # Process valid reactions using reaction_info
        valid_reactions = reaction_info["valid_reactions"]
        max_reaction_idx = cell_graph["reaction"].num_nodes

        # Check for unified RMR edge type
        if ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
            rmr_edges = cell_graph["reaction", "rmr", "metabolite"]
            hyperedge_index = rmr_edges.hyperedge_index.to(
                self.device
            )  # Use hyperedge_index
            edge_types = rmr_edges.edge_type.to(self.device)
            stoichiometry = rmr_edges.stoichiometry.to(self.device)

            # STEP 1: Identify all product relationships in the original graph
            original_products = (
                {}
            )  # metabolite_idx -> list of reaction_idx that produce it
            for i in range(hyperedge_index.size(1)):
                if edge_types[i] == 1:  # This is a product edge
                    rxn_idx = hyperedge_index[0, i].item()
                    met_idx = hyperedge_index[1, i].item()
                    if met_idx not in original_products:
                        original_products[met_idx] = []
                    original_products[met_idx].append(rxn_idx)

            # STEP 2: Filter edges connected to valid reactions
            valid_mask = torch.isin(hyperedge_index[0], valid_reactions)

            # STEP 3: Keep only valid edges
            new_hyperedge_index = hyperedge_index[:, valid_mask].clone()
            new_stoich = stoichiometry[valid_mask].to(self.device)
            new_edge_types = edge_types[valid_mask].to(self.device)

            # STEP 4: Identify remaining product relationships after perturbation
            remaining_products = {}
            for i in range(new_hyperedge_index.size(1)):
                if new_edge_types[i] == 1:  # This is a product edge
                    rxn_idx = new_hyperedge_index[0, i].item()
                    met_idx = new_hyperedge_index[1, i].item()
                    if met_idx not in remaining_products:
                        remaining_products[met_idx] = []
                    remaining_products[met_idx].append(rxn_idx)

            # STEP 5: Find metabolites that lost ALL producing reactions
            isolated_products = []
            for met_idx in original_products:
                if met_idx not in remaining_products:
                    isolated_products.append(met_idx)

            # STEP 6: Relabel reaction indices for the filtered graph
            reaction_map = torch.full(
                (max_reaction_idx,), -1, dtype=torch.long, device=self.device
            )
            reaction_map[valid_reactions] = torch.arange(
                len(valid_reactions), device=self.device
            )
            new_hyperedge_index[0] = reaction_map[new_hyperedge_index[0]]

            # STEP 7: Store processed edges in output graph
            edge_type = ("reaction", "rmr", "metabolite")
            integrated_subgraph[edge_type].hyperedge_index = new_hyperedge_index
            integrated_subgraph[edge_type].stoichiometry = new_stoich
            integrated_subgraph[edge_type].edge_type = new_edge_types
            integrated_subgraph[edge_type].num_edges = new_hyperedge_index.size(1)
            integrated_subgraph[edge_type].pert_mask = ~valid_mask

            # Copy over any other attributes
            if hasattr(rmr_edges, "reaction_to_genes"):
                integrated_subgraph[edge_type].reaction_to_genes = (
                    rmr_edges.reaction_to_genes
                )
            if hasattr(rmr_edges, "reaction_to_genes_indices"):
                integrated_subgraph[edge_type].reaction_to_genes_indices = (
                    rmr_edges.reaction_to_genes_indices
                )

            # STEP 8: Create mask - only flag isolated product metabolites as removed
            metabolite_mask = torch.ones(
                integrated_subgraph["metabolite"].num_nodes,
                dtype=torch.bool,
                device=self.device,
            )
            if isolated_products:  # Only mark isolated products as removed
                metabolite_mask[isolated_products] = False

            # STEP 9: Update masks
            self.masks["metabolite"]["kept"] = metabolite_mask
            self.masks["metabolite"]["removed"] = ~metabolite_mask
            integrated_subgraph["metabolite"].pert_mask = ~metabolite_mask

            # STEP 10: Update node_ids to exclude isolated product metabolites
            integrated_subgraph["metabolite"].node_ids = [
                integrated_subgraph["metabolite"].node_ids[i]
                for i in range(len(metabolite_mask))
                if metabolite_mask[i]
            ]
            integrated_subgraph["metabolite"].num_nodes = metabolite_mask.sum().item()
        else:
            # Handle legacy case - set default masks (no isolated products)
            self.masks["metabolite"]["kept"].fill_(True)
            self.masks["metabolite"]["removed"].fill_(False)
            integrated_subgraph["metabolite"].pert_mask = torch.zeros(
                integrated_subgraph["metabolite"].num_nodes,
                dtype=torch.bool,
                device=self.device,
            )

        # Update reaction node information
        integrated_subgraph["reaction"].num_nodes = len(valid_reactions)
        integrated_subgraph["reaction"].node_ids = valid_reactions.tolist()

    def _process_metabolic_network(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: dict,
    ):
        """Process metabolic network based on its representation type."""
        # Skip if no valid reactions
        if not reaction_info:
            return

        # Check for hypergraph representation
        if ("metabolite", "reaction", "metabolite") in cell_graph.edge_types:
            self._process_metabolism_hypergraph(
                integrated_subgraph, cell_graph, reaction_info
            )

        # Check for unified bipartite representation (new format)
        elif ("reaction", "rmr", "metabolite") in cell_graph.edge_types:
            self._process_metabolism_bipartite(
                integrated_subgraph, cell_graph, reaction_info
            )

        # Check for split bipartite representation (legacy format)
        elif ("metabolite", "reactant", "reaction") in cell_graph.edge_types or (
            "reaction",
            "product",
            "metabolite",
        ) in cell_graph.edge_types:
            self._process_metabolism_bipartite(
                integrated_subgraph, cell_graph, reaction_info
            )

    def _process_metabolism_hypergraph(
        self,
        integrated_subgraph: HeteroData,
        cell_graph: HeteroData,
        reaction_info: dict,
    ):
        """Process metabolic network as hypergraph."""
        if ("metabolite", "reaction", "metabolite") not in cell_graph.edge_types:
            return

        # Add metabolite data
        integrated_subgraph["metabolite"].node_ids = cell_graph["metabolite"].node_ids
        integrated_subgraph["metabolite"].num_nodes = cell_graph["metabolite"].num_nodes

        met_edges = cell_graph["metabolite", "reaction", "metabolite"]
        edge_index = met_edges.hyperedge_index.to(self.device)
        stoichiometry = met_edges.stoichiometry.to(self.device)

        # Original production relationships (positive stoichiometry)
        original_products = {}  # metabolite_idx -> list of reaction_idx that produce it
        for i in range(edge_index.size(1)):
            met_idx = edge_index[0, i].item()
            rxn_idx = edge_index[1, i].item()
            stoich = stoichiometry[i].item()

            if stoich > 0:  # Product relationship (positive stoichiometry)
                if met_idx not in original_products:
                    original_products[met_idx] = []
                original_products[met_idx].append(rxn_idx)

        # Filter valid edges
        max_metabolite_idx = integrated_subgraph["metabolite"].num_nodes
        valid_metabolite_mask = edge_index[0] < max_metabolite_idx
        valid_reaction_mask = torch.isin(
            edge_index[1], reaction_info["valid_reactions"]
        )
        valid_mask = valid_metabolite_mask & valid_reaction_mask

        # Filter edges
        new_edge_index = edge_index[:, valid_mask]
        new_stoich = stoichiometry[valid_mask].to(self.device)

        # Remaining production relationships after perturbation
        remaining_products = (
            {}
        )  # metabolite_idx -> list of reaction_idx that produce it
        for i in range(new_edge_index.size(1)):
            met_idx = new_edge_index[0, i].item()
            rxn_idx = new_edge_index[1, i].item()
            stoich = new_stoich[i].item()

            if stoich > 0:  # Product relationship (positive stoichiometry)
                if met_idx not in remaining_products:
                    remaining_products[met_idx] = []
                remaining_products[met_idx].append(rxn_idx)

        # Find isolated product metabolites
        isolated_products = []
        for met_idx in original_products:
            if met_idx not in remaining_products:
                isolated_products.append(met_idx)

        # Create mask for metabolites (only remove isolated products)
        metabolite_mask = torch.ones(
            integrated_subgraph["metabolite"].num_nodes,
            dtype=torch.bool,
            device=self.device,
        )
        if isolated_products:
            metabolite_mask[isolated_products] = False

        # Update masks
        self.masks["metabolite"]["kept"] = metabolite_mask
        self.masks["metabolite"]["removed"] = ~metabolite_mask

        # Update perturbation mask
        integrated_subgraph["metabolite"].pert_mask = ~metabolite_mask

        # Update node_ids to exclude isolated product metabolites
        integrated_subgraph["metabolite"].node_ids = [
            integrated_subgraph["metabolite"].node_ids[i]
            for i in range(len(metabolite_mask))
            if metabolite_mask[i]
        ]
        integrated_subgraph["metabolite"].num_nodes = metabolite_mask.sum().item()

        # Relabel reaction indices
        reaction_map = torch.full(
            (cell_graph["reaction"].num_nodes,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        reaction_map[reaction_info["valid_reactions"]] = torch.arange(
            len(reaction_info["valid_reactions"]), device=self.device
        )
        new_edge_index = new_edge_index.clone()
        new_edge_index[1] = reaction_map[new_edge_index[1]]

        # Store edge data
        edge_type = ("metabolite", "reaction", "metabolite")
        integrated_subgraph[edge_type].hyperedge_index = new_edge_index
        integrated_subgraph[edge_type].stoichiometry = new_stoich
        integrated_subgraph[edge_type].num_edges = (
            len(torch.unique(new_edge_index[1])) + 1
        )
        integrated_subgraph[edge_type].pert_mask = ~valid_mask

        # Update reaction node information
        integrated_subgraph["reaction"].num_nodes = len(
            reaction_info["valid_reactions"]
        )
        integrated_subgraph["reaction"].node_ids = reaction_info[
            "valid_reactions"
        ].tolist()


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
        add_remaining_gene_self_loops: bool = True,
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

        # self loops, transform base graph
        self.add_remaining_gene_self_loops = add_remaining_gene_self_loops

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
        self.cell_graph = to_cell_data(
            graphs,
            incidence_graphs,
            add_remaining_gene_self_loops=self.add_remaining_gene_self_loops,
        )

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

        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                # Convert cursor to list to avoid multiple LMDB transactions
                entries = [(key, value) for key, value in cursor]

            # Process entries outside of LMDB transaction
            for key, value in entries:
                try:
                    idx = int(key.decode())
                    # Parse JSON once per entry
                    data_list = json.loads(value.decode())
                    # Process all perturbations in one pass
                    for data in data_list:
                        for pert in data["experiment"]["genotype"]["perturbations"]:
                            gene = pert["systematic_gene_name"]
                            if gene not in is_any_perturbed_gene_index:
                                is_any_perturbed_gene_index[gene] = set()
                            is_any_perturbed_gene_index[gene].add(idx)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error processing entry {key}: {e}")

        finally:
            self.close_lmdb()

        # Convert sets to sorted lists
        return {
            gene: sorted(list(indices))
            for gene, indices in is_any_perturbed_gene_index.items()
        }

    @property
    def is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
        # Memory cache
        if hasattr(self, "_is_any_perturbed_gene_index_cache"):
            return self._is_any_perturbed_gene_index_cache

        cache_path = osp.join(self.processed_dir, "is_any_perturbed_gene_index.json")

        # Try to load from disk cache
        if osp.exists(cache_path):
            with open(cache_path, "r") as file:
                self._is_any_perturbed_gene_index_cache = json.load(file)
                return self._is_any_perturbed_gene_index_cache

        # Compute if no cache exists
        result = self.compute_is_any_perturbed_gene_index()

        # Save to both memory and disk cache
        self._is_any_perturbed_gene_index_cache = result
        with open(cache_path, "w") as file:
            json.dump(result, file)

        return result

    # HACK
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Remove the unpicklable lmdb environment.
        state["env"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


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
    print(dataset[3])
    dataset.close_lmdb()
    # print(dataset[10000])

    # Assuming you have already created your dataset and CellDataModule
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        print(batch)
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
    #     num_workers=2,
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
    from torchcell.metabolism.yeast_GEM import YeastGEM

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
        incidence_graphs={"metabolism": YeastGEM().reaction_map},
        node_embeddings={
            "codon_frequency": codon_frequency,
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
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        break


def main_transform_standardization():
    """Test standardization of labels using LabelNormalizationTransform with metabolic network."""
    import os.path as osp
    from dotenv import load_dotenv
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torch
    import json
    from tqdm import tqdm

    # Import necessary components
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
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.transforms.regression_to_classification import (
        LabelNormalizationTransform,
    )
    from torchcell.datamodules import CellDataModule
    from torch_geometric.transforms import Compose
    # from torchcell.transforms.hetero_to_dense import HeteroToDense
    from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask

    # Load environment variables
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Load query
    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    # Set up genome and graph
    print("Setting up genome and graph...")
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # Set up node embeddings
    print("Setting up embeddings...")
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
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

    # Create dataset with metabolism network
    print("Creating dataset with metabolism network...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        # incidence_graphs={"metabolism_hypergraph": YeastGEM().reaction_map},
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(f"Dataset size: {len(dataset)}")

    # Define the labels we want to standardize
    labels = ["fitness", "gene_interaction"]

    # Print statistics of original data using dataset.label_df
    for label in labels:
        values = dataset.label_df[label].dropna().values
        print(f"\n{label} statistics (original):")
        print(f"  Count: {len(values)}")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")

    # Configure normalization to use standard (z-score) normalization for both labels
    norm_configs = {
        "fitness": {"strategy": "standard"},  # z-score: (x - mean) / std
        "gene_interaction": {"strategy": "standard"},  # z-score: (x - mean) / std
    }

    # Create the normalizer
    print("\nCreating normalization transform...")
    normalizer = LabelNormalizationTransform(dataset, norm_configs)

    # Print the normalization parameters
    for label, stats in normalizer.stats.items():
        print(f"\nNormalization parameters for {label}:")
        for key, value in stats.items():
            if key not in ["bin_edges", "bin_counts", "strategy"]:
                print(f"  {key}: {value:.6f}")
        print(f"  strategy: {stats['strategy']}")

    # Apply the transform to the dataset
    # HACK - start
    dense_transform = HeteroToDenseMask({"gene": len(genome.gene_set)})
    dataset.transform = Compose([normalizer, dense_transform])
    # HACK - end
    # dataset.transform = normalizer

    # Check metabolic network connectivity changes
    print("\nChecking metabolic network changes for a few examples...")
    for i in range(5):
        data = dataset[i]
        if "metabolite" in data.node_types and "reaction" in data.node_types:
            if hasattr(data["metabolite", "reaction", "metabolite"], "hyperedge_index"):
                print(
                    f"Sample {i} - hyperedge size: {data['metabolite', 'reaction', 'metabolite'].hyperedge_index.size()}"
                )

                # Check if perturbed genes affect the metabolic network
                if hasattr(data["gene"], "cell_graph_idx_pert"):
                    perturbed_indices = data["gene"].cell_graph_idx_pert

                    # Count reactions affected by gene perturbations
                    reactions_with_perturbed = set()
                    for rxn_idx, genes in dataset.cell_graph[
                        "metabolite", "reaction", "metabolite"
                    ].reaction_to_genes_indices.items():
                        if any(g in perturbed_indices for g in genes):
                            reactions_with_perturbed.add(rxn_idx)

                    print(f"  Perturbed genes: {len(perturbed_indices)}")
                    print(f"  Reactions affected: {len(reactions_with_perturbed)}")

    # Sample data points for visualization
    print("\nSampling data for visualization...")
    num_samples = 1000
    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    original_values = {label: [] for label in labels}
    normalized_values = {label: [] for label in labels}

    for idx in tqdm(sample_indices):
        data = dataset[idx]
        for label in labels:
            if label in data["gene"] and not torch.isnan(data["gene"][label]).any():
                # Get normalized value
                normalized_values[label].append(data["gene"][label].item())
                # Get original value
                original_values[label].append(data["gene"][f"{label}_original"].item())

    # Create visualization with plots
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Label Distributions: Before vs. After Standardization", fontsize=16)

    for i, label in enumerate(labels):
        # Original distribution
        sns.histplot(original_values[label], bins=50, ax=axes[0, i], kde=True)
        axes[0, i].set_title(f"Original {label}")

        # Normalized distribution
        sns.histplot(normalized_values[label], bins=50, ax=axes[1, i], kde=True)
        axes[1, i].set_title(f"Standardized {label} (z-score)")

        # Add mean and std lines to the normalized plot
        axes[1, i].axvline(x=0, color="r", linestyle="--", label="Mean (0)")
        axes[1, i].axvline(x=1, color="g", linestyle="--", label="+1 Std")
        axes[1, i].axvline(x=-1, color="g", linestyle="--", label="-1 Std")
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig("standardization_with_metabolism_comparison.png")

    # Test with datamodule
    print("\nTesting with CellDataModule...")
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    # Get a batch and check normalized values
    for batch in cell_data_module.train_dataloader():
        for label in labels:
            if label in batch["gene"]:
                print(f"\nBatch {label} statistics:")
                print(f"  Shape: {batch['gene'][label].shape}")
                print(f"  Mean: {batch['gene'][label].mean().item():.4f}")
                print(f"  Std: {batch['gene'][label].std().item():.4f}")
                print(f"  Original values present: {'_original' in batch['gene']}")
        break

    # Clean up
    dataset.close_lmdb()
    print("\nTest completed successfully.")


def main_transform_categorical():
    # Used this in hetero gnn pool when converting categorical to regression
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


def main_transform_categorical_dense():
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
        num_workers=2,
        pin_memory=True,
    )
    base_data_module.setup()

    # Create perturbation subset module
    subset_size = 10000
    subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=base_data_module,
        size=subset_size,
        batch_size=2,
        num_workers=2,
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
    # main_transform_categorical_dense()
    # main()
    # main_incidence()
    main_transform_standardization()
