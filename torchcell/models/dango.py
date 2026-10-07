# torchcell/models/dango
# [[torchcell.models.dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/dango
# Test file: tests/torchcell/models/test_dango.py

"""DANGO model: PPI-network pretraining, embedding integration, and HyperSAGNN."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_mean

# Zhang et al. (2020): lambda 0.1 where zero entries fell by more than 1% from STRING
# v9.1 to v11.0 (neighborhood, coexpression, experimental), else 1.0.
PAPER_LAMBDA_VALUES: dict[str, float] = {
    f"{version}_{network}": value
    for version in ("string9_1", "string11_0")
    for network, value in (
        ("neighborhood", 0.1),
        ("fusion", 1.0),
        ("cooccurence", 1.0),
        ("coexpression", 0.1),
        ("experimental", 0.1),
        ("database", 1.0),
    )
}


class DangoPreTrain(nn.Module):
    """
    GNN pre-training component of DANGO model that learns node embeddings from
    protein-protein interaction networks in S. cerevisiae.

    As described in the paper, this module:
    1. Processes PPI networks from the STRING database
    2. Uses a 2-layer GNN for each network to reconstruct graph structure
    3. Shares an initial embedding layer across all networks
    4. Uses the output embeddings for downstream tasks
    """

    def __init__(self, gene_num: int, edge_types: list[str], hidden_channels: int = 64):
        """Build the shared embedding and per-network GNN and reconstruction layers.

        Args:
            gene_num: Number of genes (embedding table size).
            edge_types: PPI network edge types, one GNN stack per type.
            hidden_channels: Embedding and hidden dimension.
        """
        super().__init__()

        # Initialize model parameters
        self.gene_num = gene_num
        self.hidden_channels = hidden_channels
        self.edge_types = edge_types

        # Shared embedding layer across all GNNs (H^(0))
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)

        # Two layers of GNN for each network
        self.layer1_convs = nn.ModuleDict()
        self.layer2_convs = nn.ModuleDict()

        # Reconstruction layers for each network
        self.recon_layers = nn.ModuleDict()

        # Initialize GNN layers and reconstruction layers for each edge type
        for edge_type in self.edge_types:
            # First layer GNN
            self.layer1_convs[edge_type] = SAGEConv(
                hidden_channels, hidden_channels, normalize=False, project=False
            )

            # Second layer GNN
            self.layer2_convs[edge_type] = SAGEConv(
                hidden_channels, hidden_channels, normalize=False, project=False
            )

            # Reconstruction layer to predict adjacency matrix row
            self.recon_layers[edge_type] = nn.Linear(hidden_channels, gene_num)

        # Initialize weights
        self.reset_parameters()

    @property
    def lambda_values(self) -> dict[str, float]:
        """Paper lambda (weight of the zero entries in the weighted MSE) per network.

        Zhang et al. (2020) set lambda = 0.1 for the networks whose zero entries fell
        by more than 1% from STRING v9.1 to v11.0 (neighborhood, coexpression,
        experimental) and 1.0 for the rest (fusion, cooccurence, database). The table
        is defined only for those six STRING v9.1 and v11.0 network names; any other
        name (for example a string12_0 network) is refused rather than given 1.0.
        The 005/006 training scripts do not read this table: they pass
        ``determine_lambda_values()`` to ``DangoLoss``.

        Raises:
            ValueError: if an edge type is not a STRING v9.1 or v11.0 network name.
        """
        unknown = [e for e in self.edge_types if e not in PAPER_LAMBDA_VALUES]
        if unknown:
            raise ValueError(
                f"DangoPreTrain.lambda_values is defined only for the STRING v9.1 and "
                f"v11.0 networks {sorted(PAPER_LAMBDA_VALUES)}; no paper lambda for "
                f"{unknown}. Pass lambda values to DangoLoss explicitly (the 005/006 "
                f"scripts use determine_lambda_values())."
            )
        return {e: PAPER_LAMBDA_VALUES[e] for e in self.edge_types}

    def reset_parameters(self) -> None:
        """Initialize model parameters"""
        nn.init.normal_(self.gene_embedding.weight, mean=0, std=0.1)

        for edge_type in self.edge_types:
            cast(SAGEConv, self.layer1_convs[edge_type]).reset_parameters()
            cast(SAGEConv, self.layer2_convs[edge_type]).reset_parameters()
            recon = cast(nn.Linear, self.recon_layers[edge_type])
            nn.init.normal_(recon.weight, mean=0, std=0.1)
            nn.init.zeros_(recon.bias)

    def forward(
        self, cell_graph: HeteroData
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """
        Forward pass for the DangoPreTrain model

        Args:
            cell_graph: The cell graph containing multiple edge types

        Returns:
            Dictionary containing:
                - 'embeddings': Node embeddings for each edge type (H_i^(2))
                - 'reconstructions': Reconstructed adjacency matrix rows for each edge type (FC(H_i^(2)))
        """
        device = self.gene_embedding.weight.device

        # Get gene node indices
        gene_data = cell_graph["gene"]
        num_nodes = gene_data.num_nodes
        node_indices = torch.arange(num_nodes, device=device)

        # Get initial node embeddings (H^(0)) - shared across all networks
        x_init = self.gene_embedding(node_indices)

        # Process each network separately
        embeddings = {}
        reconstructions = {}

        for edge_type in self.edge_types:
            edge_key = ("gene", edge_type, "gene")

            if edge_key in cell_graph.edge_types:
                edge_index = cell_graph[edge_key].edge_index

                # First layer (H^(1))
                # SAGEConv internally handles the neighborhood aggregation and concatenation
                h1 = self.layer1_convs[edge_type](x_init, edge_index)
                h1 = F.relu(h1)

                # Second layer (H^(2))
                h2 = self.layer2_convs[edge_type](h1, edge_index)
                h2 = F.relu(h2)

                # Store final embeddings (H^(2))
                embeddings[edge_type] = h2

                # Reconstruction to predict adjacency matrix row
                recon = self.recon_layers[edge_type](h2)
                reconstructions[edge_type] = recon
            else:
                # If edge type not in graph, use zeros
                embeddings[edge_type] = torch.zeros_like(x_init)
                reconstructions[edge_type] = torch.zeros(
                    num_nodes, self.gene_num, device=device
                )

        return {
            "embeddings": embeddings,
            "reconstructions": reconstructions,
            "initial_embeddings": x_init,
        }


class MetaEmbedding(nn.Module):
    """
    Meta-embedding module to integrate embeddings from multiple networks.

    As described in the paper, this module:
    1. Takes embeddings from 6 different PPI networks for each node
    2. Uses an MLP to compute attention weights for each embedding
    3. Combines embeddings using a weighted sum based on learned attention
    """

    def __init__(self, hidden_channels: int):
        """Build the attention MLP used to weight per-network embeddings.

        Args:
            hidden_channels: Input embedding dimension.
        """
        super().__init__()
        # MLP for attention weights (two fully-connected layers)
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for better training stability"""
        for m in self.attention_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to integrate embeddings from multiple networks

        Args:
            embeddings_dict: Dictionary mapping edge types to node embeddings
                             Each value has shape [num_nodes, hidden_channels]

        Returns:
            Integrated embeddings with shape [num_nodes, hidden_channels]
        """
        # Get list of embeddings from the dictionary
        embeddings_list = list(embeddings_dict.values())

        # Stack embeddings along new dimension
        # Shape: [num_nodes, num_networks, hidden_channels]
        stacked_embeddings = torch.stack(embeddings_list, dim=1)

        # Compute attention scores for each embedding
        # First, reshape for the MLP
        num_nodes, num_networks, hidden_channels = stacked_embeddings.shape
        reshaped_embeddings = stacked_embeddings.view(-1, hidden_channels)

        # Apply MLP
        attention_scores = self.attention_mlp(reshaped_embeddings)
        attention_scores = attention_scores.view(num_nodes, num_networks)

        # Apply softmax to get normalized weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Expand weights for broadcasting
        # Shape: [num_nodes, num_networks, 1]
        attention_weights = attention_weights.unsqueeze(-1)

        # Compute weighted sum
        # Shape: [num_nodes, hidden_channels]
        meta_embeddings = (stacked_embeddings * attention_weights).sum(dim=1)

        return meta_embeddings


class HyperSAGNN(nn.Module):
    """
    Fully vectorized Hypergraph Self-Attention Graph Neural Network
    that handles all perturbation sets in a single forward pass.
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4):
        """Build the static embedding and two multi-head self-attention layers.

        Args:
            hidden_channels: Embedding and hidden dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads

        # Static embedding layer
        self.static_embedding = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU()
        )

        # Attention layer parameters
        # Layer 1
        self.Q1 = nn.Linear(hidden_channels, hidden_channels)
        self.K1 = nn.Linear(hidden_channels, hidden_channels)
        self.V1 = nn.Linear(hidden_channels, hidden_channels)
        self.O1 = nn.Linear(hidden_channels, hidden_channels)
        self.beta1 = nn.Parameter(torch.zeros(1))

        # Layer 2
        self.Q2 = nn.Linear(hidden_channels, hidden_channels)
        self.K2 = nn.Linear(hidden_channels, hidden_channels)
        self.V2 = nn.Linear(hidden_channels, hidden_channels)
        self.O2 = nn.Linear(hidden_channels, hidden_channels)
        self.beta2 = nn.Parameter(torch.zeros(1))

        # Final prediction layer
        self.prediction_layer = nn.Linear(hidden_channels, 1)

    def forward(
        self, embeddings: torch.Tensor, batch_indices: torch.Tensor, num_sets: int
    ) -> torch.Tensor:
        """
        Forward pass processing all nodes at once with masked attention.

        Args:
            embeddings: Tensor of shape [total_nodes, hidden_channels]
            batch_indices: Tensor of shape [total_nodes] indicating set membership,
                with ids in [0, num_sets)
            num_sets: Number of sets (genotypes) in the batch; the output has one
                score per set, ordered by set id.

        Returns:
            Predicted interaction scores with shape [num_sets]

        Raises:
            ValueError: if a set id is outside [0, num_sets), or a set has no node.
                A set with no perturbed gene has nothing to attend over and no
                interaction score, so it is refused rather than scored 0.
        """
        device = embeddings.device
        total_nodes = embeddings.size(0)

        out_of_range = batch_indices[(batch_indices < 0) | (batch_indices >= num_sets)]
        if out_of_range.numel() > 0:
            raise ValueError(
                f"HyperSAGNN got set ids {torch.unique(out_of_range).tolist()} for "
                f"num_sets={num_sets}; set ids must lie in [0, {num_sets})"
            )
        set_sizes = torch.bincount(batch_indices, minlength=num_sets)
        empty_sets = (set_sizes == 0).nonzero().view(-1).tolist()
        if empty_sets:
            raise ValueError(
                f"HyperSAGNN needs at least one gene per set: sets {empty_sets} of "
                f"{num_sets} have no perturbation indices, and a genotype with no "
                f"perturbed gene has no interaction score to predict"
            )

        # Compute static embeddings for all nodes
        static_embeddings = self.static_embedding(embeddings)

        # Create attention mask where nodes can only attend to others in same set
        # mask[i,j] = True if nodes i and j are in the same set, False otherwise
        same_set_mask = batch_indices.unsqueeze(-1) == batch_indices.unsqueeze(0)

        # Add self-mask to prevent nodes from attending to themselves
        self_mask = torch.eye(total_nodes, dtype=torch.bool, device=device)
        valid_attention_mask = same_set_mask & ~self_mask

        # Apply first attention layer with masked attention
        dynamic_embeddings = self._global_attention_layer(
            embeddings,
            valid_attention_mask,
            self.Q1,
            self.K1,
            self.V1,
            self.O1,
            self.beta1,
        )

        # Apply second attention layer
        dynamic_embeddings = self._global_attention_layer(
            dynamic_embeddings,
            valid_attention_mask,
            self.Q2,
            self.K2,
            self.V2,
            self.O2,
            self.beta2,
        )

        # Compute element-wise squared differences
        squared_diff = (dynamic_embeddings - static_embeddings) ** 2

        # Compute node scores
        node_scores = self.prediction_layer(squared_diff).squeeze(-1)

        # Aggregate scores for each set using scatter_mean
        interaction_scores = scatter_mean(
            node_scores, batch_indices, dim=0, dim_size=num_sets
        )

        return cast(torch.Tensor, interaction_scores)

    def _global_attention_layer(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        Q_proj: nn.Linear,
        K_proj: nn.Linear,
        V_proj: nn.Linear,
        O_proj: nn.Linear,
        beta: nn.Parameter,
    ) -> torch.Tensor:
        """
        Apply global masked multi-head attention.

        Args:
            x: Input tensor with shape [total_nodes, hidden_dim]
            attention_mask: Binary mask with shape [total_nodes, total_nodes]
                           True where attention is allowed, False elsewhere
            Q_proj: Query linear projection.
            K_proj: Key linear projection.
            V_proj: Value linear projection.
            O_proj: Output linear projection.
            beta: ReZero parameter

        Returns:
            Output tensor with shape [total_nodes, hidden_dim]
        """
        total_nodes = x.size(0)

        # Linear projections
        Q = Q_proj(x)  # [total_nodes, hidden_dim]
        K = K_proj(x)  # [total_nodes, hidden_dim]
        V = V_proj(x)  # [total_nodes, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(total_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = K.view(total_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = V.view(total_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        # Shape: [num_heads, total_nodes, head_dim]

        # Calculate attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        # Shape: [num_heads, total_nodes, total_nodes]

        # Expand attention_mask for multi-head attention
        expanded_mask = attention_mask.unsqueeze(0).expand(self.num_heads, -1, -1)

        # Apply attention mask - set masked-out values to -inf before softmax
        attention.masked_fill_(~expanded_mask, -float("inf"))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=-1)

        # Handle potential NaNs from empty rows (if a node can't attend to any others)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        # Shape: [num_heads, total_nodes, head_dim]

        # Reshape back to [total_nodes, hidden_dim]
        out = out.permute(1, 0, 2).contiguous().view(total_nodes, self.hidden_channels)

        # Apply output projection
        out = O_proj(out)

        # Apply ReZero connection
        return cast(torch.Tensor, beta * out + x)


class Dango(nn.Module):
    """
    DANGO model for predicting higher-order genetic interactions

    Implements:
    1. GNN pre-training component
    2. Meta-embedding integration
    3. Hypergraph self-attention for prediction
    """

    def __init__(
        self,
        gene_num: int,
        edge_types: list[str],
        hidden_channels: int = 64,
        num_heads: int = 4,
    ):
        """Assemble the pretraining, integration, and HyperSAGNN submodules.

        Args:
            gene_num: Number of genes.
            edge_types: PPI network edge types.
            hidden_channels: Embedding and hidden dimension.
            num_heads: Number of attention heads in the HyperSAGNN.
        """
        super().__init__()
        self.hidden_channels = hidden_channels

        # GNN pre-training component
        self.pretrain_model = DangoPreTrain(
            gene_num=gene_num, edge_types=edge_types, hidden_channels=hidden_channels
        )

        # Meta-embedding integration module
        self.meta_embedding = MetaEmbedding(hidden_channels=hidden_channels)

        # Hypergraph self-attention network
        self.hyper_sagnn = HyperSAGNN(
            hidden_channels=hidden_channels, num_heads=num_heads
        )

        # Initialize weights with better defaults
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights for better training stability"""
        # Initialize HyperSAGNN linear layers
        for name, module in self.hyper_sagnn.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("relu")
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize ReZero parameters to small positive values instead of zero
        # This can help with better gradient flow early in training
        nn.init.constant_(self.hyper_sagnn.beta1, 0.01)
        nn.init.constant_(self.hyper_sagnn.beta2, 0.01)

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for the DANGO model

        Args:
            cell_graph: The cell graph containing multiple edge types
            batch: HeteroDataBatch containing perturbation information

        Returns:
            Tuple containing:
                - predictions: Predicted interaction scores, one per genotype
                  (``batch.num_graphs``)
                - outputs: Dictionary containing node embeddings and intermediate values

        Raises:
            ValueError: if a genotype lists the same gene twice (the HyperSAGNN self
                mask is positional, so a duplicate would attend to its own copy), or
                if a genotype has no perturbed gene (see ``HyperSAGNN.forward``).
        """
        perturbation_indices = batch["gene"].perturbation_indices
        set_ids = batch["gene"].perturbation_indices_batch
        gene_num = self.pretrain_model.gene_num
        keys, counts = torch.unique(
            set_ids * gene_num + perturbation_indices, return_counts=True
        )
        repeated = keys[counts > 1]
        if repeated.numel() > 0:
            pairs = [(k // gene_num, k % gene_num) for k in repeated.tolist()]
            raise ValueError(
                f"Dango needs distinct genes within a genotype; (genotype, gene index) "
                f"pairs {pairs} appear more than once"
            )

        # Get embeddings from pre-training component
        pretrain_outputs = self.pretrain_model(cell_graph)

        # Extract node embeddings for each network
        network_embeddings = pretrain_outputs["embeddings"]

        # Integrate embeddings using meta-embedding module
        integrated_embeddings = self.meta_embedding(network_embeddings)

        # Base outputs dictionary
        outputs = {
            "network_embeddings": network_embeddings,
            "integrated_embeddings": integrated_embeddings,
            "reconstructions": pretrain_outputs["reconstructions"],
            "initial_embeddings": pretrain_outputs["initial_embeddings"],
        }

        # Directly index into integrated_embeddings to get perturbed gene embeddings
        perturbed_embeddings = integrated_embeddings[perturbation_indices]

        # One score per genotype in the batch, ordered by genotype
        interaction_scores = self.hyper_sagnn(
            perturbed_embeddings, set_ids, num_sets=batch.num_graphs
        )

        # Store results in the outputs dictionary
        outputs["interaction_scores"] = interaction_scores

        # Return both the predictions and the outputs dictionary
        return interaction_scores, outputs

    @property
    def num_parameters(self) -> dict[str, int]:
        """Count the number of trainable parameters in the model."""

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "pretrain_model": count_params(self.pretrain_model),
            "meta_embedding": count_params(self.meta_embedding),
            "hyper_sagnn": count_params(self.hyper_sagnn),
        }

        # Calculate overall total
        counts["total"] = sum(counts.values())

        return counts


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/dango_demo.py on 2026-10-06"
    )
