# torchcell/nn/nsa_encoder
# [[torchcell.nn.nsa_encoder]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/nsa_encoder
# Test file: tests/torchcell/nn/test_nsa_encoder.py

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, Union
from torchcell.nn.masked_attention_block import NodeSelfAttention
from typing import Literal

# Import SelfAttentionBlock from your project.
from torchcell.nn.self_attention_block import SelfAttentionBlock


class NSAEncoder(nn.Module):
    """
    NSA Encoder that applies an input projection followed by a sequence
    of masked (MAB) and self-attention (SAB) blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pattern: list[Union[Literal["M"], Literal["S"]]] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        if pattern is None:
            pattern = ["M", "S", "M", "S"]
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for block_type in pattern:
            if block_type == "M":
                self.layers.append(
                    NodeSelfAttention(
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
                raise ValueError(f"Invalid block type '{block_type}'.")

    def forward(
        self,
        x: torch.Tensor,
        data_or_edge_index: Union[torch.Tensor, "HeteroData"],
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if isinstance(data_or_edge_index, torch.Tensor):
            edge_index = data_or_edge_index
            from torch_geometric.utils import to_dense_adj

            unique_batch, counts = torch.unique(batch, return_counts=True)
            batch_size = len(unique_batch)
            max_nodes = counts.max().item()
            adj_mask = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes).bool()
        elif hasattr(data_or_edge_index, "adj_mask"):
            adj_mask = data_or_edge_index.adj_mask
            batch_size = adj_mask.size(0)
            max_nodes = adj_mask.size(1)
        elif hasattr(data_or_edge_index, "adj"):
            adj_mask = data_or_edge_index.adj.bool()
            batch_size = adj_mask.size(0)
            max_nodes = adj_mask.size(1)
        else:
            try:
                edge_index = data_or_edge_index.edge_index
                from torch_geometric.utils import to_dense_adj

                unique_batch, counts = torch.unique(batch, return_counts=True)
                batch_size = len(unique_batch)
                max_nodes = counts.max().item()
                adj_mask = to_dense_adj(
                    edge_index, batch, max_num_nodes=max_nodes
                ).bool()
            except AttributeError:
                raise ValueError(
                    "Cannot extract adjacency information from provided data."
                )
        edge_attr_dict = None
        if edge_attr is not None:
            edge_attr_dict = {}
            if isinstance(data_or_edge_index, torch.Tensor):
                edge_index = data_or_edge_index
            elif hasattr(data_or_edge_index, "edge_index"):
                edge_index = data_or_edge_index.edge_index
            else:
                edge_index = None
            if edge_index is not None:
                for i in range(edge_index.size(1)):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()
                    attr_val = (
                        edge_attr[i].item()
                        if edge_attr.dim() == 1
                        else edge_attr[i].mean().item()
                    )
                    edge_attr_dict[(src, dst)] = attr_val
        device = x.device
        unique_batch, counts = torch.unique(batch, return_counts=True)
        batch_size = len(unique_batch)
        max_nodes = counts.max().item()
        padded_x = torch.zeros(batch_size, max_nodes, x.size(1), device=device)
        for b in range(batch_size):
            nodes_in_batch = (batch == b).nonzero().squeeze(-1)
            if nodes_in_batch.numel() > 0:
                num_nodes_in_batch = nodes_in_batch.size(0)
                padded_x[b, :num_nodes_in_batch] = x[nodes_in_batch]
        h = self.input_proj(padded_x)
        for layer in self.layers:
            if isinstance(layer, NodeSelfAttention):
                h = layer(h, adj_mask, edge_attr_dict)
            else:
                h = layer(h)
        output = []
        for b in range(batch_size):
            nodes_in_batch = (batch == b).nonzero().squeeze(-1)
            if nodes_in_batch.numel() > 0:
                num_nodes = nodes_in_batch.size(0)
                output.append(h[b, :num_nodes])
        return torch.cat(output, dim=0)
