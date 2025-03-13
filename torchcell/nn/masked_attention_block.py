# torchcell/nn/masked_attention_block
# [[torchckell.nn.masked_attention_block]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/masked_attention_block
# Test file: tests/torchcell/nn/test_masked_attention_block.py

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, Callable, Literal, Dict, Union, Tuple


class MaskedAttentionBlock(nn.Module):
    """
    Masked Attention Block (MAB) using flex_attention for efficient scaling with graph structure.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: Literal["node", "edge"] = "node",
    ) -> None:
        """
        Initialize the Masked Attention Block.

        Args:
            hidden_dim: Dimension of input and output features
            num_heads: Number of attention heads
            dropout: Dropout probability for attention and MLP layers
            activation: Activation function to use in the MLP
            mode: Whether to operate on nodes ('node') or edges ('edge')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch_size, seq_len, hidden_dim] to [batch_size, num_heads, seq_len, head_dim]"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    # Update MaskedAttentionBlock.forward to handle edge attributes without dynamic control flow
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_attr_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the MAB with edge attributes.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            adj_matrix: Adjacency matrix of shape [batch_size, seq_len, seq_len]
            edge_attr_matrix: Optional edge attribute matrix of shape [batch_size, seq_len, seq_len]
                            or [batch_size, seq_len, seq_len, edge_feat_dim]

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Apply normalization
        normed_x = self.norm1(x)

        # Project to queries, keys, values
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        # Create attention mask
        mask = adj_matrix.unsqueeze(1).expand(
            batch_size, self.num_heads, seq_len, seq_len
        )
        attention_mask = torch.zeros_like(mask, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(~mask.bool(), float("-inf"))

        # Pre-compute edge attribute influence
        if edge_attr_matrix is not None:
            # Create edge attribute influence tensor
            if edge_attr_matrix.dim() == 3:  # [batch, seq, seq]
                edge_influence = edge_attr_matrix
            else:  # [batch, seq, seq, feat_dim]
                edge_influence = edge_attr_matrix.mean(dim=-1)

            # Scale and multiply by adjacency to zero out non-edges
            edge_modifier = 0.1 * edge_influence * adj_matrix.float()

            # Expand for all heads
            edge_modifier = edge_modifier.unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, seq_len
            )

            # Apply flex_attention with combined mask and edge attribute influence
            def score_mod(score, b, h, q_idx, kv_idx):
                return (
                    score
                    + attention_mask[b, h, q_idx, kv_idx]
                    + edge_modifier[b, h, q_idx, kv_idx]
                )

        else:
            # Just use the mask without edge attributes
            def score_mod(score, b, h, q_idx, kv_idx):
                return score + attention_mask[b, h, q_idx, kv_idx]

        attn_output = flex_attention(q, k, v, score_mod=score_mod)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # First residual connection
        x = residual + attn_output

        # Save for second residual connection
        residual = x

        # Apply normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = residual + mlp_output

        return output


class NodeSetAttention(MaskedAttentionBlock):
    """
    Node-Set Attention (NSA) layer using FlexAttention for graph masking.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            mode="node",
        )


class EdgeSetAttention(MaskedAttentionBlock):
    """
    Edge-Set Attention (ESA) layer using FlexAttention for graph masking.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            mode="edge",
        )


class NSAEncoder(nn.Module):
    """
    NSA encoder with interleaved MAB and SAB blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pattern: list[Literal["M", "S"]] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """
        Initialize the NSA encoder with a pattern of MAB and SAB blocks.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension for attention layers
            pattern: List of strings specifying the sequence of attention blocks.
                     Each element should be either 'M' for MAB or 'S' for SAB.
                     Example: ['M', 'S', 'M', 'S', 'S']
                     If None, defaults to alternating MAB and SAB: ['M', 'S', 'M', 'S']
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function to use
        """
        super().__init__()

        # Set default pattern if none provided
        if pattern is None:
            pattern = ["M", "S", "M", "S"]  # Default to alternating MAB/SAB

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Create attention blocks according to the pattern
        self.layers = nn.ModuleList()

        from torchcell.nn.self_attention_block import SelfAttentionBlock

        for block_type in pattern:
            if block_type == "M":
                self.layers.append(
                    NodeSetAttention(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            elif block_type == "S":
                self.layers.append(
                    SelfAttentionBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            else:
                raise ValueError(
                    f"Invalid block type '{block_type}'. Must be 'M' for MAB or 'S' for SAB."
                )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the NSA encoder with edge attributes.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_feat_dim]
            batch: Batch assignment for nodes [num_nodes] (optional)

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Handle batching
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Convert edge_index to dense adjacency once at the beginning
        from torch_geometric.utils import to_dense_adj

        # Get unique batch indices and counts for potential padding
        unique_batch, counts = torch.unique(batch, return_counts=True)
        batch_size = len(unique_batch)
        max_nodes = counts.max().item()

        # Create adjacency matrix
        adj = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes)

        # Create edge attribute matrix if edge_attr is provided
        edge_attr_matrix = None
        if edge_attr is not None:
            # Convert edge attributes to dense matrix format
            edge_attr_matrix = to_dense_adj(
                edge_index, batch, edge_attr=edge_attr, max_num_nodes=max_nodes
            )

        # Initialize a padded node feature tensor
        device = x.device
        padded_x = torch.zeros(batch_size, max_nodes, x.size(1), device=device)

        # Fill in the padded tensor with actual node features
        for b in range(batch_size):
            nodes_in_batch = (batch == b).nonzero().squeeze()
            # Handle scalar tensor case (when there's only one node in the batch)
            if nodes_in_batch.dim() == 0:
                nodes_in_batch = nodes_in_batch.unsqueeze(0)
            num_nodes_in_batch = nodes_in_batch.size(0)
            if num_nodes_in_batch > 0:  # Avoid empty batch items
                padded_x[b, :num_nodes_in_batch] = x[nodes_in_batch]

        # Project input features
        h = self.input_proj(padded_x)

        # Apply attention layers according to the pattern
        for i, layer in enumerate(self.layers):
            if isinstance(layer, NodeSetAttention):
                # MAB needs adjacency matrix and potentially edge attributes
                h = layer(h, adj, edge_attr_matrix)
            else:  # SelfAttentionBlock
                # SAB doesn't need adjacency
                h = layer(h)

        # Unbatch the output to get back individual node embeddings
        output = []
        for b in range(batch_size):
            mask = batch == b
            count = mask.sum()
            if count > 0:  # Avoid empty batch items
                # Handle scalar tensor case
                if mask.dim() == 0:
                    output.append(h[b, :1])
                else:
                    output.append(h[b, :count])

        # Concatenate all unbatched items
        return torch.cat(output, dim=0)


class ESAEncoder(nn.Module):
    """
    ESA encoder with interleaved MAB and SAB blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pattern: list[Literal["M", "S"]] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """
        Initialize the ESA encoder with a pattern of MAB and SAB blocks.

        Args:
            input_dim: Dimension of input edge features
            hidden_dim: Hidden dimension for attention layers
            pattern: List of strings specifying the sequence of attention blocks.
                     Each element should be either 'M' for MAB or 'S' for SAB.
                     Example: ['M', 'S', 'M', 'S', 'S']
                     If None, defaults to alternating MAB and SAB: ['M', 'S', 'M', 'S']
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function to use
        """
        super().__init__()

        # Set default pattern if none provided
        if pattern is None:
            pattern = ["M", "S", "M", "S"]  # Default to alternating MAB/SAB

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Create attention blocks according to the pattern
        self.layers = nn.ModuleList()

        from torchcell.nn.self_attention_block import SelfAttentionBlock

        for block_type in pattern:
            if block_type == "M":
                self.layers.append(
                    EdgeSetAttention(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            elif block_type == "S":
                self.layers.append(
                    SelfAttentionBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            else:
                raise ValueError(
                    f"Invalid block type '{block_type}'. Must be 'M' for MAB or 'S' for SAB."
                )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the ESA encoder.

        Args:
            x: Edge features [num_edges, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for edges [num_edges] (optional)

        Returns:
            Edge embeddings [num_edges, hidden_dim]
        """
        # Handle batching
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute edge-to-edge connectivity
        edge_adj = self._compute_edge_adjacency(edge_index, batch)

        # Get unique batch indices and counts for potential padding
        unique_batch, counts = torch.unique(batch, return_counts=True)
        batch_size = len(unique_batch)
        max_edges = counts.max().item()

        # Initialize a padded edge feature tensor
        device = x.device
        padded_x = torch.zeros(batch_size, max_edges, x.size(1), device=device)

        # Fill in the padded tensor with actual edge features
        for b in range(batch_size):
            edges_in_batch = (batch == b).nonzero().squeeze()
            num_edges_in_batch = edges_in_batch.numel()
            if num_edges_in_batch > 0:  # Avoid empty batch items
                padded_x[b, :num_edges_in_batch] = x[edges_in_batch]

        # Project input features
        h = self.input_proj(padded_x)

        # Apply attention layers according to the pattern
        for i, layer in enumerate(self.layers):
            if isinstance(layer, EdgeSetAttention):
                # MAB needs edge-to-edge adjacency
                h = layer(h, edge_adj)
            else:  # SelfAttentionBlock
                # SAB doesn't need adjacency
                h = layer(h)

        # Unbatch the output to get back individual edge embeddings
        output = []
        for b in range(batch_size):
            mask = batch == b
            count = mask.sum()
            if count > 0:  # Avoid empty batch items
                output.append(h[b, :count])

        # Concatenate all unbatched items
        return torch.cat(output, dim=0)

    def _compute_edge_adjacency(
        self, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-to-edge adjacency matrix from edge_index.

        Args:
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for edges [num_edges]

        Returns:
            Edge-to-edge adjacency matrix [batch_size, max_edges, max_edges]
        """
        # Get unique batch indices and counts
        unique_batch, counts = torch.unique(batch, return_counts=True)
        batch_size = len(unique_batch)
        max_edges = counts.max().item()

        device = edge_index.device
        edge_adj = torch.zeros(
            batch_size, max_edges, max_edges, dtype=torch.bool, device=device
        )

        # Extract source and target nodes
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # Process each batch separately
        for b in range(batch_size):
            # Get indices of edges in this batch
            batch_mask = batch == b
            batch_indices = batch_mask.nonzero().squeeze(1)
            num_edges_in_batch = batch_indices.size(0)

            # Skip if no edges in this batch
            if num_edges_in_batch == 0:
                continue

            # Get source and target nodes for edges in this batch
            batch_sources = source_nodes[batch_indices]
            batch_targets = target_nodes[batch_indices]

            # Calculate edge-to-edge connectivity
            for i in range(num_edges_in_batch):
                for j in range(num_edges_in_batch):
                    # Two edges are connected if they share a node
                    if (
                        batch_sources[i] == batch_sources[j]
                        or batch_sources[i] == batch_targets[j]
                        or batch_targets[i] == batch_sources[j]
                        or batch_targets[i] == batch_targets[j]
                    ):
                        edge_adj[b, i, j] = True

        return edge_adj
