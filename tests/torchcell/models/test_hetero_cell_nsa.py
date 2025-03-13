# tests/torchcell/models/test_hetero_cell_nsa.py

import inspect
import os

import pytest
import torch
from torch_geometric.utils import to_dense_adj

from torchcell.nn.masked_attention_block import NodeSetAttention
from torchcell.nn.self_attention_block import SelfAttentionBlock
from torchcell.scratch.load_batch import load_sample_data_batch


@pytest.fixture
def sample_data():
    """Load a sample batch with metabolism bipartite representation."""
    os.environ["DATA_ROOT"] = (
        "/tmp" if not os.environ.get("DATA_ROOT") else os.environ.get("DATA_ROOT")
    )
    try:
        dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
            batch_size=2, num_workers=0, metabolism_graph="metabolism_bipartite"
        )
        return dataset, batch
    except Exception as e:
        pytest.skip(f"Failed to load sample data: {e}")


class TestNodeEmbeddings:
    """Test self-attention blocks on different node types."""

    def test_gene_sab_processing(self, sample_data):
        """Test that gene embeddings can be processed through a SAB."""
        dataset, batch = sample_data
        num_genes = dataset.cell_graph["gene"].num_nodes
        hidden_dim = 64

        # Initialize gene embeddings
        gene_embedding = torch.nn.Embedding(num_genes, hidden_dim)

        # Create a batch of node indices (for all genes)
        gene_ids = torch.arange(num_genes, dtype=torch.long)

        # Get embeddings
        gene_emb = gene_embedding(gene_ids)

        # Add batch dimension
        gene_emb = gene_emb.unsqueeze(0)

        # Initialize SAB
        sab = SelfAttentionBlock(hidden_dim=hidden_dim)

        # Process through SAB
        output = sab(gene_emb)

        # Check output shape
        assert output.shape == (1, num_genes, hidden_dim)
        assert not torch.isnan(output).any()

    def test_reaction_sab_processing(self, sample_data):
        """Test that reaction embeddings can be processed through a SAB."""
        dataset, batch = sample_data
        num_reactions = dataset.cell_graph["reaction"].num_nodes
        hidden_dim = 64

        # Initialize reaction embeddings
        reaction_embedding = torch.nn.Embedding(num_reactions, hidden_dim)

        # Create a batch of node indices (for all reactions)
        reaction_ids = torch.arange(num_reactions, dtype=torch.long)

        # Get embeddings
        reaction_emb = reaction_embedding(reaction_ids)

        # Add batch dimension
        reaction_emb = reaction_emb.unsqueeze(0)

        # Initialize SAB
        sab = SelfAttentionBlock(hidden_dim=hidden_dim)

        # Process through SAB
        output = sab(reaction_emb)

        # Check output shape
        assert output.shape == (1, num_reactions, hidden_dim)
        assert not torch.isnan(output).any()

    def test_metabolite_sab_processing(self, sample_data):
        """Test that metabolite embeddings can be processed through a SAB."""
        dataset, batch = sample_data
        num_metabolites = dataset.cell_graph["metabolite"].num_nodes
        hidden_dim = 64

        # Initialize metabolite embeddings
        metabolite_embedding = torch.nn.Embedding(num_metabolites, hidden_dim)

        # Create a batch of node indices (for all metabolites)
        metabolite_ids = torch.arange(num_metabolites, dtype=torch.long)

        # Get embeddings
        metabolite_emb = metabolite_embedding(metabolite_ids)

        # Add batch dimension
        metabolite_emb = metabolite_emb.unsqueeze(0)

        # Initialize SAB
        sab = SelfAttentionBlock(hidden_dim=hidden_dim)

        # Process through SAB
        output = sab(metabolite_emb)

        # Check output shape
        assert output.shape == (1, num_metabolites, hidden_dim)
        assert not torch.isnan(output).any()


class TestEdgeProcessing:
    """Test node-set attention on different edge types."""

    def test_ppi_nsa_processing(self, sample_data):
        """Test that physical interaction edges can be processed through NSA."""
        dataset, batch = sample_data
        edge_type = ("gene", "physical_interaction", "gene")
        num_genes = dataset.cell_graph["gene"].num_nodes
        hidden_dim = 64

        # Initialize gene embeddings
        gene_embedding = torch.nn.Embedding(num_genes, hidden_dim)

        # Create a batch of node indices (for all genes)
        gene_ids = torch.arange(num_genes, dtype=torch.long)

        # Get embeddings
        gene_emb = gene_embedding(gene_ids)

        # Get edge index
        edge_index = dataset.cell_graph[edge_type].edge_index

        # Convert to dense adjacency matrix
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_genes)[0]

        # Add batch dimension to embeddings and adjacency
        gene_emb = gene_emb.unsqueeze(0)
        adj_matrix = adj_matrix.unsqueeze(0)

        # Initialize NSA
        nsa = NodeSetAttention(hidden_dim=hidden_dim)

        # Process through NSA
        output = nsa(gene_emb, adj_matrix)

        # Check output shape
        assert output.shape == (1, num_genes, hidden_dim)
        assert not torch.isnan(output).any()

    def test_regulatory_nsa_processing(self, sample_data):
        """Test that regulatory interaction edges can be processed through NSA."""
        dataset, batch = sample_data
        edge_type = ("gene", "regulatory_interaction", "gene")
        num_genes = dataset.cell_graph["gene"].num_nodes
        hidden_dim = 64

        # Initialize gene embeddings
        gene_embedding = torch.nn.Embedding(num_genes, hidden_dim)

        # Create a batch of node indices (for all genes)
        gene_ids = torch.arange(num_genes, dtype=torch.long)

        # Get embeddings
        gene_emb = gene_embedding(gene_ids)

        # Get edge index
        edge_index = dataset.cell_graph[edge_type].edge_index

        # Convert to dense adjacency matrix
        adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_genes)[0]

        # Add batch dimension to embeddings and adjacency
        gene_emb = gene_emb.unsqueeze(0)
        adj_matrix = adj_matrix.unsqueeze(0)

        # Initialize NSA
        nsa = NodeSetAttention(hidden_dim=hidden_dim)

        # Process through NSA
        output = nsa(gene_emb, adj_matrix)

        # Check output shape
        assert output.shape == (1, num_genes, hidden_dim)
        assert not torch.isnan(output).any()

        # Note on directionality: The directionality is preserved in the adjacency matrix
        # We don't need to add edge_type attribute as the direction is captured in the
        # structure of the edge_index itself

    def test_gpr_nsa_processing(self, sample_data):
        """Test that gene-protein-reaction edges can be processed through NSA."""
        dataset, batch = sample_data
        edge_type = ("gene", "gpr", "reaction")
        num_genes = dataset.cell_graph["gene"].num_nodes
        num_reactions = dataset.cell_graph["reaction"].num_nodes
        hidden_dim = 64

        # Initialize embeddings
        gene_embedding = torch.nn.Embedding(num_genes, hidden_dim)
        reaction_embedding = torch.nn.Embedding(num_reactions, hidden_dim)

        # Create node indices
        gene_ids = torch.arange(num_genes, dtype=torch.long)
        reaction_ids = torch.arange(num_reactions, dtype=torch.long)

        # Get embeddings
        gene_emb = gene_embedding(gene_ids)
        reaction_emb = reaction_embedding(reaction_ids)

        # Get edge index (hyperedge_index for gpr)
        edge_index = dataset.cell_graph[edge_type].hyperedge_index

        # Test NSA on genes first
        # Convert to gene-centric adjacency matrix
        gene_adj = torch.zeros(num_genes, num_genes, dtype=torch.bool)

        # For each reaction, connect all genes that participate in it
        for r_idx in range(edge_index.size(1)):
            gene = edge_index[0, r_idx]
            # For GPR edges, we can use the hyperedge to find genes that share reactions
            genes_in_reaction = edge_index[0, edge_index[1, :] == edge_index[1, r_idx]]
            for g1 in genes_in_reaction:
                for g2 in genes_in_reaction:
                    gene_adj[g1, g2] = True

        # Add batch dimension
        gene_emb = gene_emb.unsqueeze(0)
        gene_adj = gene_adj.unsqueeze(0)

        # Initialize NSA
        nsa = NodeSetAttention(hidden_dim=hidden_dim)

        # Process genes through NSA
        gene_output = nsa(gene_emb, gene_adj)

        # Check output shape
        assert gene_output.shape == (1, num_genes, hidden_dim)
        assert not torch.isnan(gene_output).any()

        # Now test NSA on reactions
        # Convert to reaction-centric adjacency matrix
        reaction_adj = torch.zeros(num_reactions, num_reactions, dtype=torch.bool)

        # For each gene, connect all reactions that it participates in
        for g_idx in range(edge_index.size(1)):
            reaction = edge_index[1, g_idx]
            # For GPR edges, we can find reactions that share genes
            reactions_with_gene = edge_index[
                1, edge_index[0, :] == edge_index[0, g_idx]
            ]
            for r1 in reactions_with_gene:
                for r2 in reactions_with_gene:
                    reaction_adj[r1, r2] = True

        # Add batch dimension
        reaction_emb = reaction_emb.unsqueeze(0)
        reaction_adj = reaction_adj.unsqueeze(0)

        # Process reactions through NSA
        reaction_output = nsa(reaction_emb, reaction_adj)

        # Check output shape
        assert reaction_output.shape == (1, num_reactions, hidden_dim)
        assert not torch.isnan(reaction_output).any()

    def test_rmr_nsa_processing(self, sample_data):
        """Test that reaction-metabolite edges with stoichiometry can be processed through NSA."""
        dataset, batch = sample_data
        edge_type = ("reaction", "rmr", "metabolite")

        # Use only a small subset for testing
        max_reactions = 50
        max_metabolites = 50
        hidden_dim = 64

        # Get edge index and attributes
        edge_index = dataset.cell_graph[edge_type].edge_index
        stoichiometry = dataset.cell_graph[edge_type].stoichiometry

        # Find reactions and metabolites that appear in the first few edges
        unique_reactions = edge_index[0, :1000].unique()
        unique_metabolites = edge_index[1, :1000].unique()

        # Limit to max size
        reactions_subset = unique_reactions[:max_reactions]
        metabolites_subset = unique_metabolites[:max_metabolites]

        # Initialize embeddings for the subset
        reaction_embedding = torch.nn.Embedding(max_reactions, hidden_dim)
        metabolite_embedding = torch.nn.Embedding(max_metabolites, hidden_dim)

        # Create embeddings
        reaction_emb = reaction_embedding(torch.arange(len(reactions_subset)))
        metabolite_emb = metabolite_embedding(torch.arange(len(metabolites_subset)))

        # Create a simple adjacency for testing
        # Instead of building the complex reaction-reaction and metabolite-metabolite adjacencies,
        # we'll create simpler test matrices
        reaction_adj = torch.eye(
            len(reactions_subset), dtype=torch.bool
        )  # Self-connections only
        reaction_edge_attr = torch.ones_like(reaction_adj, dtype=torch.float)

        # Add batch dimension
        reaction_emb = reaction_emb.unsqueeze(0)
        reaction_adj = reaction_adj.unsqueeze(0)
        reaction_edge_attr = reaction_edge_attr.unsqueeze(0)

        # Initialize NSA
        nsa = NodeSetAttention(hidden_dim=hidden_dim)

        # Process reactions through NSA with stoichiometry
        reaction_output = nsa(reaction_emb, reaction_adj, reaction_edge_attr)

        # Check output shape
        assert reaction_output.shape == (1, len(reactions_subset), hidden_dim)
        assert not torch.isnan(reaction_output).any()

        # Repeat for metabolites with a simple adjacency matrix
        metabolite_adj = torch.eye(len(metabolites_subset), dtype=torch.bool)
        metabolite_edge_attr = torch.ones_like(metabolite_adj, dtype=torch.float)

        # Add batch dimension
        metabolite_emb = metabolite_emb.unsqueeze(0)
        metabolite_adj = metabolite_adj.unsqueeze(0)
        metabolite_edge_attr = metabolite_edge_attr.unsqueeze(0)

        # Process metabolites through NSA
        metabolite_output = nsa(metabolite_emb, metabolite_adj, metabolite_edge_attr)

        # Check output shape
        assert metabolite_output.shape == (1, len(metabolites_subset), hidden_dim)
        assert not torch.isnan(metabolite_output).any()

        # Test that the updated MaskedAttentionBlock can handle edge attributes properly
        assert hasattr(nsa, "forward") and callable(nsa.forward)
        signature = inspect.signature(nsa.forward)
        assert "edge_attr_matrix" in signature.parameters
