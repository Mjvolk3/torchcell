# torchcell/nn/masked_attention_block
# [[torchcell.nn.masked_attention_block]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/masked_attention_block
# Test file: tests/torchcell/nn/test_masked_attention_block.py

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, Dict, Tuple, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Dict


class MaskedAttentionBlock(nn.Module):
    """Memory-efficient Masked Attention Block using FlexAttention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: str = "node",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Edge attributes buffers for score modification
        self.register_buffer("edge_attr_values", None, persistent=False)
        self.register_buffer("edge_attr_indices", None, persistent=False)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(
        self, x: Tensor, adj_mask: Tensor, edge_attr_dict: Optional[Dict] = None
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        normed_x = self.norm1(x)
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        if adj_mask.dtype != torch.bool:
            adj_mask = adj_mask.bool()

        if edge_attr_dict is not None:
            # Prepare edge attributes for use in score modification.
            self.prepare_edge_attributes(edge_attr_dict, seq_len)

            def score_mod(score, b, h, q_idx, kv_idx):
                mask_val = adj_mask[b, q_idx, kv_idx]
                score_masked = torch.where(
                    mask_val,
                    score,
                    torch.tensor(float("-inf"), device=score.device, dtype=score.dtype),
                )
                edge_key = q_idx * seq_len + kv_idx
                edge_exists = (self.edge_attr_indices == edge_key).any()
                edge_idx = torch.where(self.edge_attr_indices == edge_key)[0]
                edge_val = torch.zeros_like(score)
                if edge_exists:
                    edge_val = self.edge_attr_values[edge_idx[0]] * 0.1
                return score_masked + edge_val

            from torch.nn.attention.flex_attention import flex_attention

            attn_output = flex_attention(q, k, v, score_mod=score_mod)
        else:

            def mask_mod(b, h, q_idx, kv_idx):
                return adj_mask[b, q_idx, kv_idx]

            from torch.nn.attention.flex_attention import (
                flex_attention,
                create_block_mask,
            )

            block_mask = create_block_mask(
                mask_mod, B=batch_size, H=self.num_heads, Q_LEN=seq_len, KV_LEN=seq_len
            )
            attn_output = flex_attention(q, k, v, block_mask=block_mask)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        x = residual + attn_output
        residual = x
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)
        output = residual + mlp_output
        return output

    def prepare_edge_attributes(self, edge_attr_dict, seq_len):
        indices = []
        values = []
        for (i, j), val in edge_attr_dict.items():
            idx = i * seq_len + j
            indices.append(idx)
            values.append(val)
        if indices:
            self.edge_attr_indices = torch.tensor(
                indices, device=self.q_proj.weight.device
            )
            self.edge_attr_values = torch.tensor(
                values, device=self.q_proj.weight.device
            )
        else:
            self.edge_attr_indices = torch.tensor(
                [], device=self.q_proj.weight.device, dtype=torch.long
            )
            self.edge_attr_values = torch.tensor([], device=self.q_proj.weight.device)


class NodeSetAttention(nn.Module):
    r"""Implements a Masked Attention Block (MAB) for processing graphs with boolean
    adjacency masks while preserving edge attributes in sparse format.

    This implementation uses efficient boolean masks instead of float masks
    for improved memory efficiency (8x memory savings).

    Args:
        hidden_dim (int): Hidden dimension size
        num_heads (int, optional): Number of attention heads. Default: 8
        dropout (float, optional): Dropout probability. Default: 0.1
        activation (nn.Module, optional): Activation function. Default: nn.GELU()
        mode (str, optional): Mode of operation. Default: "node"
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: str = "node",
    ) -> None:
        super().__init__()

        # Ensure hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode

        # Normalization and projections
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Second residual block components
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            activation,
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        adj_mask: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for masked attention with boolean masks.

        Args:
            x (Tensor): Input node features of shape (batch_size, num_nodes, hidden_dim)
                    or (num_nodes, hidden_dim)
            adj_mask (Tensor): Boolean adjacency mask of shape (batch_size, num_nodes, num_nodes)
                            or (num_nodes, num_nodes)
            edge_attr (Optional[Tensor]): Optional sparse edge attributes
            edge_index (Optional[Tensor]): Optional edge indices for mapping sparse edge_attr

        Returns:
            Tensor: Output node features with same shape as input x
        """
        # Handle input dimensions
        input_dim = x.dim()
        if input_dim == 2:  # (num_nodes, hidden_dim)
            # Add batch dimension
            x = x.unsqueeze(0)
            if adj_mask.dim() == 2:
                adj_mask = adj_mask.unsqueeze(0)

        # Store original input for residual connection
        residual = x

        # Layer normalization
        normed_x = self.norm1(x)

        # Compute query, key, value projections
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        # Get dimensions
        batch_size, seq_len, _ = q.size()

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply boolean mask - set masked positions to a large negative value (safer than -inf)
        if adj_mask is not None:
            # Expand mask for multi-head attention
            mask_expanded = adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Apply mask: False positions become a large negative value (more stable than -inf)
            scores = scores.masked_fill(~mask_expanded, -1e9)

        # If we have edge attributes, use them as additive biases to attention scores
        # This is more numerically stable than multiplicative weights
        if edge_attr is not None and edge_index is not None:
            # Create edge bias tensor based on attribute values
            edge_attr_bias = torch.zeros_like(scores)

            for b in range(batch_size):
                for i in range(edge_index.size(1)):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    if src < seq_len and dst < seq_len:
                        # Only apply to valid edges that are in the mask
                        if adj_mask[b, src, dst]:
                            # Get attribute value (clamp to reasonable range to avoid instability)
                            attr_val = (
                                edge_attr[i].item()
                                if edge_attr.dim() == 1
                                else edge_attr[i].mean().item()
                            )
                            scaled_attr = min(max(attr_val, -5.0), 5.0)

                            # Apply as additive bias to all heads
                            for h in range(self.num_heads):
                                edge_attr_bias[b, h, src, dst] = scaled_attr

            # Add bias to attention scores (safer than multiplication)
            scores = scores + edge_attr_bias

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Check for NaN values and fix if needed
        if torch.isnan(attn_weights).any():
            # Replace NaN values with zeros and re-normalize
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            # Re-normalize each row
            row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn_weights = attn_weights / row_sums

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape and project back
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.out_proj(context)

        # First residual connection
        x = residual + self.dropout(attn_output)

        # Second residual block (MLP)
        residual = x
        x = residual + self.dropout(self.mlp(self.norm2(x)))

        # Remove batch dimension if input didn't have it
        if input_dim == 2:
            x = x.squeeze(0)

        return x


class NSAEncoder(nn.Module):
    """
    NSA encoder with proper FlexAttention support.
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
        data_or_edge_index: Union[torch.Tensor, "HeteroData"],
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the NSA encoder with proper FlexAttention support.

        Args:
            x: Node features [num_nodes, input_dim]
            data_or_edge_index: Either edge_index [2, num_edges] or a HeteroData object
            edge_attr: Optional edge attributes [num_edges, edge_feat_dim]
            batch: Batch assignment for nodes [num_nodes] (optional)

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Handle batching
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Extract adjacency information
        if isinstance(data_or_edge_index, torch.Tensor):
            # It's an edge_index
            edge_index = data_or_edge_index

            # Get unique batch indices and counts for potential padding
            unique_batch, counts = torch.unique(batch, return_counts=True)
            batch_size = len(unique_batch)
            max_nodes = counts.max().item()

            # Convert edge_index to dense boolean adjacency matrix
            from torch_geometric.utils import to_dense_adj

            adj_mask = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes).bool()

        elif hasattr(data_or_edge_index, "adj_mask"):
            # HeteroData with precomputed boolean mask
            adj_mask = data_or_edge_index.adj_mask
            batch_size = adj_mask.size(0)
            max_nodes = adj_mask.size(1)

        elif hasattr(data_or_edge_index, "adj"):
            # HeteroData with precomputed adjacency matrix
            adj_mask = data_or_edge_index.adj.bool()  # Convert to boolean
            batch_size = adj_mask.size(0)
            max_nodes = adj_mask.size(1)

        else:
            # Extract edge_index from HeteroData
            try:
                edge_index = data_or_edge_index.edge_index

                # Get unique batch indices and counts for potential padding
                unique_batch, counts = torch.unique(batch, return_counts=True)
                batch_size = len(unique_batch)
                max_nodes = counts.max().item()

                # Convert edge_index to dense boolean adjacency matrix
                from torch_geometric.utils import to_dense_adj

                adj_mask = to_dense_adj(
                    edge_index, batch, max_num_nodes=max_nodes
                ).bool()

            except AttributeError:
                raise ValueError(
                    "Cannot extract adjacency information from the provided data"
                )

        # Prepare edge attributes dictionary if provided
        edge_attr_dict = None
        if edge_attr is not None:
            # Create mapping from edges to their attribute values
            edge_attr_dict = {}

            if isinstance(data_or_edge_index, torch.Tensor):
                # Direct edge_index tensor
                edge_index = data_or_edge_index
            elif hasattr(data_or_edge_index, "edge_index"):
                # Extract from HeteroData
                edge_index = data_or_edge_index.edge_index
            else:
                # No edge_index available
                edge_index = None

            if edge_index is not None:
                # Process each edge
                for i in range(edge_index.size(1)):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()

                    # Get batch assignment for this edge
                    src_batch = batch[src].item() if src < batch.size(0) else 0

                    # Get this edge's attribute value
                    if edge_attr.dim() > 1:
                        attr_val = (
                            edge_attr[i].mean().item()
                        )  # Average multi-dim attributes
                    else:
                        attr_val = edge_attr[i].item()

                    # Store in the dictionary
                    edge_attr_dict[(src, dst)] = attr_val

        # Initialize padded node feature tensor
        device = x.device
        padded_x = torch.zeros(batch_size, max_nodes, x.size(1), device=device)

        # Fill in the padded tensor with actual node features
        for b in range(batch_size):
            nodes_in_batch = (batch == b).nonzero().squeeze(-1)
            if nodes_in_batch.numel() > 0:  # Avoid empty batch items
                num_nodes_in_batch = nodes_in_batch.size(0)
                padded_x[b, :num_nodes_in_batch] = x[nodes_in_batch]

        # Project input features
        h = self.input_proj(padded_x)

        # Apply attention layers according to the pattern
        for layer in self.layers:
            if isinstance(layer, NodeSetAttention):
                # MAB needs adjacency mask and potentially edge attributes
                h = layer(h, adj_mask, edge_attr_dict)
            else:  # SelfAttentionBlock
                # SAB doesn't need adjacency
                h = layer(h)

        # Unbatch the output to get back individual node embeddings
        output = []
        for b in range(batch_size):
            nodes_in_batch = (batch == b).nonzero().squeeze(-1)
            if nodes_in_batch.numel() > 0:
                num_nodes = nodes_in_batch.size(0)
                output.append(h[b, :num_nodes])

        # Concatenate all unbatched items
        return torch.cat(output, dim=0)
