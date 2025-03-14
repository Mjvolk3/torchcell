# torchcell/nn/nsa_encoder
# [[torchcell.nn.nsa_encoder]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/nsa_encoder
# Test file: tests/torchcell/nn/test_nsa_encoder.py


import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, Dict, Tuple, Literal, Union
from torchcell.nn.masked_attention_block import NodeSetAttention


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
