# torchcell/nn/hetero_nsa
# [[torchcell.nn.hetero_nsa]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/hetero_nsa
# Test file: tests/torchcell/nn/test_hetero_nsa.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Literal, Set
import logging

from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.aggr.attention import AttentionalAggregation

from torchcell.nn.self_attention_block import SelfAttentionBlock
from torchcell.nn.masked_attention_block import NodeSetAttention


class HeteroNSA(nn.Module):
    """
    Heterogeneous Node-Set Attention (HeteroNSA) for processing heterogeneous graphs.

    This module processes different node types and their relationships, applying
    masked and self-attention blocks in a pattern specified for each relationship.
    After processing, embeddings from multiple relationships involving the same entity
    are aggregated into a single representation.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        patterns: Dict[Tuple[str, str, str], List[str]],
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "max", "attention"] = "sum",
    ) -> None:
        """
        Initialize the HeteroNSA module.

        Args:
            hidden_dim: Dimension of node embeddings for all types
            node_types: Set of node types in the heterogeneous graph
            edge_types: Set of edge types (source, relation, target)
            patterns: Dictionary mapping edge types to their attention patterns
                     Each pattern is a list of "M" (MAB) or "S" (SAB)
            num_heads: Number of attention heads for NSA blocks
            dropout: Dropout probability
            activation: Activation function
            aggregation: Method to aggregate multiple embeddings of the same node type
                        ("sum", "mean", "max", or "attention")
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.patterns = patterns
        self.aggregation = aggregation

        # Validate that all edge types have a pattern
        for edge_type in edge_types:
            if edge_type not in patterns:
                raise ValueError(
                    f"Edge type '{edge_type}' does not have a pattern defined"
                )

        # Create attention layers for each edge type according to their patterns
        self.attention_blocks = nn.ModuleDict()

        for edge_type, pattern in patterns.items():
            src, rel, dst = edge_type
            key = f"{src}__{rel}__{dst}"  # Convert tuple to string for ModuleDict key

            # Create a list of attention blocks for this relation
            blocks = nn.ModuleList()

            for block_type in pattern:
                if block_type == "M":
                    blocks.append(
                        NodeSetAttention(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            activation=activation,
                        )
                    )
                elif block_type == "S":
                    blocks.append(
                        SelfAttentionBlock(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            activation=activation,
                        )
                    )
                else:
                    raise ValueError(
                        f"Invalid block type '{block_type}' for {edge_type}. "
                        f"Must be 'M' for MAB or 'S' for SAB."
                    )

            self.attention_blocks[key] = blocks

        # Create aggregation modules for nodes that participate in multiple relations
        if aggregation == "attention":
            self.node_aggregators = nn.ModuleDict()
            for node_type in node_types:
                # Count in how many relations this node type participates
                relations = [
                    (s, r, d)
                    for s, r, d in edge_types
                    if s == node_type or d == node_type
                ]
                if len(relations) > 1:
                    # Create an attentional aggregator
                    # Gate network for attention scores
                    gate_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                    )

                    # Transform network for value transformation
                    transform_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )

                    self.node_aggregators[node_type] = AttentionalAggregation(
                        gate_nn=gate_nn, nn=transform_nn
                    )

    def _process_relation(
        self, embeddings: Tensor, adj_matrix: Tensor, edge_type: Tuple[str, str, str]
    ) -> Tensor:
        """Process node embeddings for a specific relation using its attention pattern."""
        src, rel, dst = edge_type
        key = f"{src}__{rel}__{dst}"

        # If the key doesn't exist, try to find a suitable alternative
        if key not in self.attention_blocks:
            # For bipartite processing of dst nodes, reuse the original edge type's blocks
            for original_key in self.attention_blocks.keys():
                if rel in original_key:
                    # Found an edge type with the same relation name
                    key = original_key
                    break

        # If still no block found, just return the input unchanged
        if key not in self.attention_blocks:
            return embeddings

        x = embeddings

        # Apply attention blocks according to the pattern
        for i, block in enumerate(self.attention_blocks[key]):
            if isinstance(block, NodeSetAttention):
                # MAB needs adjacency matrix
                x = block(x, adj_matrix)
            else:  # SelfAttentionBlock
                # SAB doesn't need adjacency
                x = block(x)

        return x

    def forward(
        self,
        node_embeddings: Dict[str, Tensor],
        data: HeteroData,
        batch_idx: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass of HeteroNSA."""
        # Initialize output with input embeddings
        final_embeddings = {k: v.clone() for k, v in node_embeddings.items()}

        # Track relationship outputs for each node type
        relation_outputs = {node_type: [] for node_type in self.node_types}

        # Process each edge type with its attention sequence
        for edge_type in self.edge_types:
            src, rel, dst = edge_type

            # Skip if source or destination embeddings don't exist
            if src not in node_embeddings or dst not in node_embeddings:
                continue

            key = f"{src}__{rel}__{dst}"
            if key not in self.attention_blocks:
                continue

            # Get embeddings
            src_emb = node_embeddings[src]
            dst_emb = node_embeddings[dst]

            # Check if this edge type exists in the data
            if edge_type in data.edge_types:
                # Handle different edge representations
                if hasattr(data[edge_type], "edge_index"):
                    edge_index = data[edge_type].edge_index
                    # Prepare for processing
                    if src == dst:  # homogeneous case
                        # Create self-attention mask for same node type (square adjacency)
                        adj = torch.eye(src_emb.size(0), device=src_emb.device).bool()

                        # Process through blocks
                        out_src = src_emb
                        for block in self.attention_blocks[key]:
                            if isinstance(block, NodeSetAttention):
                                # Add batch dimension for adjacency if needed
                                adj_3d = adj.unsqueeze(0) if adj.dim() == 2 else adj
                                out_src = block(
                                    (
                                        out_src.unsqueeze(0)
                                        if out_src.dim() == 2
                                        else out_src
                                    ),
                                    adj_3d,
                                )
                                # Remove batch dim if it was added
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )
                            else:  # SelfAttentionBlock
                                out_src = block(
                                    out_src.unsqueeze(0)
                                    if out_src.dim() == 2
                                    else out_src
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )

                        relation_outputs[src].append((out_src, edge_type))

                    else:  # bipartite case
                        # For bipartite relations, process each side separately
                        # Source nodes
                        src_adj = torch.eye(
                            src_emb.size(0), device=src_emb.device
                        ).bool()
                        out_src = src_emb
                        for block in self.attention_blocks[key]:
                            if isinstance(block, NodeSetAttention):
                                src_adj_3d = (
                                    src_adj.unsqueeze(0)
                                    if src_adj.dim() == 2
                                    else src_adj
                                )
                                out_src = block(
                                    (
                                        out_src.unsqueeze(0)
                                        if out_src.dim() == 2
                                        else out_src
                                    ),
                                    src_adj_3d,
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )
                            else:
                                out_src = block(
                                    out_src.unsqueeze(0)
                                    if out_src.dim() == 2
                                    else out_src
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )

                        # Destination nodes
                        dst_adj = torch.eye(
                            dst_emb.size(0), device=dst_emb.device
                        ).bool()
                        out_dst = dst_emb
                        for block in self.attention_blocks[key]:
                            if isinstance(block, NodeSetAttention):
                                dst_adj_3d = (
                                    dst_adj.unsqueeze(0)
                                    if dst_adj.dim() == 2
                                    else dst_adj
                                )
                                out_dst = block(
                                    (
                                        out_dst.unsqueeze(0)
                                        if out_dst.dim() == 2
                                        else out_dst
                                    ),
                                    dst_adj_3d,
                                )
                                out_dst = (
                                    out_dst.squeeze(0)
                                    if out_dst.dim() == 3 and out_dst.size(0) == 1
                                    else out_dst
                                )
                            else:
                                out_dst = block(
                                    out_dst.unsqueeze(0)
                                    if out_dst.dim() == 2
                                    else out_dst
                                )
                                out_dst = (
                                    out_dst.squeeze(0)
                                    if out_dst.dim() == 3 and out_dst.size(0) == 1
                                    else out_dst
                                )

                        relation_outputs[src].append((out_src, edge_type))
                        relation_outputs[dst].append((out_dst, (dst, rel, src)))

                elif hasattr(data[edge_type], "hyperedge_index"):
                    # Handle hyperedge similarly to edge_index
                    # We'll use the identity matrix for simplicity
                    if src == dst:
                        # Use identity matrix for self-attention
                        adj = torch.eye(src_emb.size(0), device=src_emb.device).bool()

                        # Process through blocks
                        out_src = src_emb
                        for block in self.attention_blocks[key]:
                            if isinstance(block, NodeSetAttention):
                                adj_3d = adj.unsqueeze(0) if adj.dim() == 2 else adj
                                out_src = block(
                                    (
                                        out_src.unsqueeze(0)
                                        if out_src.dim() == 2
                                        else out_src
                                    ),
                                    adj_3d,
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )
                            else:
                                out_src = block(
                                    out_src.unsqueeze(0)
                                    if out_src.dim() == 2
                                    else out_src
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )

                        relation_outputs[src].append((out_src, edge_type))
                    else:
                        # For bipartite relations with hyperedges
                        # Process source nodes
                        src_adj = torch.eye(
                            src_emb.size(0), device=src_emb.device
                        ).bool()
                        out_src = src_emb
                        for block in self.attention_blocks[key]:
                            if isinstance(block, NodeSetAttention):
                                src_adj_3d = (
                                    src_adj.unsqueeze(0)
                                    if src_adj.dim() == 2
                                    else src_adj
                                )
                                out_src = block(
                                    (
                                        out_src.unsqueeze(0)
                                        if out_src.dim() == 2
                                        else out_src
                                    ),
                                    src_adj_3d,
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )
                            else:
                                out_src = block(
                                    out_src.unsqueeze(0)
                                    if out_src.dim() == 2
                                    else out_src
                                )
                                out_src = (
                                    out_src.squeeze(0)
                                    if out_src.dim() == 3 and out_src.size(0) == 1
                                    else out_src
                                )

                        # Process destination nodes
                        dst_adj = torch.eye(
                            dst_emb.size(0), device=dst_emb.device
                        ).bool()
                        out_dst = dst_emb
                        for block in self.attention_blocks[key]:
                            if isinstance(block, NodeSetAttention):
                                dst_adj_3d = (
                                    dst_adj.unsqueeze(0)
                                    if dst_adj.dim() == 2
                                    else dst_adj
                                )
                                out_dst = block(
                                    (
                                        out_dst.unsqueeze(0)
                                        if out_dst.dim() == 2
                                        else out_dst
                                    ),
                                    dst_adj_3d,
                                )
                                out_dst = (
                                    out_dst.squeeze(0)
                                    if out_dst.dim() == 3 and out_dst.size(0) == 1
                                    else out_dst
                                )
                            else:
                                out_dst = block(
                                    out_dst.unsqueeze(0)
                                    if out_dst.dim() == 2
                                    else out_dst
                                )
                                out_dst = (
                                    out_dst.squeeze(0)
                                    if out_dst.dim() == 3 and out_dst.size(0) == 1
                                    else out_dst
                                )

                        relation_outputs[src].append((out_src, edge_type))
                        relation_outputs[dst].append((out_dst, (dst, rel, src)))
            else:
                # Edge type not in data, use identity matrices
                # Self-attention case
                if src == dst:
                    adj = torch.eye(src_emb.size(0), device=src_emb.device).bool()

                    # Process
                    out_src = src_emb
                    for block in self.attention_blocks[key]:
                        if isinstance(block, NodeSetAttention):
                            adj_3d = adj.unsqueeze(0) if adj.dim() == 2 else adj
                            out_src = block(
                                out_src.unsqueeze(0) if out_src.dim() == 2 else out_src,
                                adj_3d,
                            )
                            out_src = (
                                out_src.squeeze(0)
                                if out_src.dim() == 3 and out_src.size(0) == 1
                                else out_src
                            )
                        else:
                            out_src = block(
                                out_src.unsqueeze(0) if out_src.dim() == 2 else out_src
                            )
                            out_src = (
                                out_src.squeeze(0)
                                if out_src.dim() == 3 and out_src.size(0) == 1
                                else out_src
                            )

                    relation_outputs[src].append((out_src, edge_type))

        # Aggregate outputs for each node type
        for node_type, outputs in relation_outputs.items():
            if not outputs:
                continue  # Keep original embeddings if no relations processed

            # Aggregate
            if len(outputs) == 1:
                final_embeddings[node_type] = outputs[0][0]
            else:
                embs = [emb for emb, _ in outputs]

                # Perform aggregation
                if self.aggregation == "sum":
                    final_embeddings[node_type] = sum(embs)
                elif self.aggregation == "mean":
                    final_embeddings[node_type] = sum(embs) / len(embs)
                elif self.aggregation == "max":
                    # Make sure all tensors have the same dimension
                    if all(e.dim() == embs[0].dim() for e in embs):
                        stacked = torch.stack(embs)
                        final_embeddings[node_type] = torch.max(stacked, dim=0)[0]
                    else:
                        # Fall back to mean if dimensions don't match
                        final_embeddings[node_type] = sum(embs) / len(embs)
                elif (
                    self.aggregation == "attention"
                    and node_type in self.node_aggregators
                ):
                    try:
                        # Try to use attention aggregation
                        if all(e.dim() == 2 for e in embs):
                            # Reshape for attention aggregation
                            flat_embs = torch.cat(embs, dim=0)
                            node_indices = torch.arange(
                                len(embs), device=embs[0].device
                            ).repeat_interleave(embs[0].size(0))

                            aggregated = self.node_aggregators[node_type](
                                flat_embs, index=node_indices
                            )
                            final_embeddings[node_type] = aggregated
                        else:
                            # Fall back to mean
                            final_embeddings[node_type] = sum(embs) / len(embs)
                    except Exception as e:
                        print(f"Error in attention aggregation for {node_type}: {e}")
                        # Fall back to mean
                        final_embeddings[node_type] = sum(embs) / len(embs)
                else:
                    # Default to mean
                    final_embeddings[node_type] = sum(embs) / len(embs)

        return final_embeddings


class NSAEncoder(nn.Module):
    """
    Full encoder using HeteroNSA with input projections and multiple layers.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        patterns: Dict[Tuple[str, str, str], List[str]],
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "max", "attention"] = "sum",
    ) -> None:
        """Initialize the NSA encoder."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types

        # Input projections
        self.input_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(dim, hidden_dim)
                for node_type, dim in input_dims.items()
            }
        )

        # Stack of HeteroNSA layers
        self.nsa_layers = nn.ModuleList(
            [
                HeteroNSA(
                    hidden_dim=hidden_dim,
                    node_types=node_types,
                    edge_types=edge_types,
                    patterns=patterns,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                    aggregation=aggregation,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms after each NSA layer
        self.layer_norms = nn.ModuleDict(
            {
                node_type: nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
                )
                for node_type in node_types
            }
        )

        # Global pooling for graph-level representation
        self.graph_projections = nn.ModuleDict(
            {node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types}
        )

        # Final aggregation across node types
        self.final_projection = nn.Linear(hidden_dim * len(node_types), hidden_dim)

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward pass for the NSA encoder.

        Args:
            data: PyTorch Geometric HeteroData object

        Returns:
            Tuple of:
            - Dictionary mapping node types to their final embeddings
            - Aggregated graph-level representation
        """
        # Extract node features and batch information
        x_dict = {}
        batch_idx = {}

        # Debug info about available node types
        print(f"Available node types in data: {data.node_types}")
        print(f"Expected node types: {self.node_types}")

        # First, process all available node types
        for node_type in self.node_types:
            if node_type in data.node_types:  # Use node_types attribute
                # Get node features
                node_data = data[node_type]
                if hasattr(node_data, "x") and node_data.x is not None:
                    x = (
                        node_data.x.clone()
                    )  # Make a copy to avoid in-place modifications
                    print(f"Found features for {node_type} with shape {x.shape}")

                    # Project input features
                    if node_type in self.input_projections:
                        x = self.input_projections[node_type](x)

                    x_dict[node_type] = x

                    # Track batch assignment if present
                    if hasattr(node_data, "batch") and node_data.batch is not None:
                        batch_idx[node_type] = node_data.batch
            else:
                print(f"Warning: Node type {node_type} not found in data.node_types")

        # Pre-populate final embeddings with initial projected features
        # This ensures we have embeddings even if layers don't process them
        final_embeddings = {k: v.clone() for k, v in x_dict.items()}

        # Apply NSA layers with residual connections
        for i, nsa_layer in enumerate(self.nsa_layers):
            print(f"Processing NSA layer {i+1}")

            # Process through the layer
            try:
                new_x_dict = nsa_layer(x_dict, data, batch_idx)

                # Apply normalization and residual connections
                for node_type in new_x_dict:
                    if node_type in x_dict:
                        # Add residual connection and normalize
                        residual = x_dict[node_type]
                        # Ensure same dimensionality for broadcasting
                        if residual.dim() != new_x_dict[node_type].dim():
                            if residual.dim() == 2 and new_x_dict[node_type].dim() == 3:
                                residual = residual.unsqueeze(0)
                            elif (
                                residual.dim() == 3 and new_x_dict[node_type].dim() == 2
                            ):
                                residual = residual.squeeze(0)

                        new_x_dict[node_type] = self.layer_norms[node_type][i](
                            new_x_dict[node_type] + residual
                        )

                # Update for next layer
                x_dict = new_x_dict

                # Update final embeddings
                final_embeddings.update(x_dict)
            except Exception as e:
                print(f"Error in layer {i+1}: {e}")
                # Continue with existing embeddings

        print(f"Final node types with embeddings: {list(final_embeddings.keys())}")

        # Generate graph-level representation by mean pooling
        graph_embeddings = {}
        for node_type, embeddings in final_embeddings.items():
            # Ensure batch dimension
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)

            # Apply projection
            projected = self.graph_projections[node_type](embeddings)

            # Mean pooling over nodes
            graph_embeddings[node_type] = projected.mean(dim=1)

        # Concatenate embeddings from all node types
        graph_embs = []
        for node_type in sorted(self.node_types):  # Sort for deterministic order
            if node_type in graph_embeddings:
                graph_embs.append(graph_embeddings[node_type])
            else:
                # Create a zero tensor for missing node types
                device = next(self.parameters()).device
                graph_embs.append(torch.zeros(1, self.hidden_dim, device=device))

        # Final aggregation
        concatenated = torch.cat(graph_embs, dim=-1)
        final_graph_embedding = self.final_projection(concatenated)

        return final_embeddings, final_graph_embedding
