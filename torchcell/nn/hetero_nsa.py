# torchcell/nn/hetero_nsa
# [[torchcell.nn.hetero_nsa]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/hetero_nsa
# Test file: tests/torchcell/nn/test_hetero_nsa.py

from typing import Dict, List, Optional, Tuple, Union, Literal, Set
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torchcell.nn.self_attention_block import SelfAttentionBlock
from torchcell.nn.masked_attention_block import NodeSelfAttention


class HeteroNSA(nn.Module):
    """
    Heterogeneous Node-Set Attention (HeteroNSA) module.
    Applies a unified sequence of masked (MAB) and self-attention (SAB)
    blocks to all node and edge types using the provided attention pattern.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        pattern: List[str],
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "attention"] = "sum",
    ):
        super().__init__()
        if not pattern:
            raise ValueError("Pattern list cannot be empty")
        for block_type in pattern:
            if block_type not in ["M", "S"]:
                raise ValueError(
                    f"Invalid block type '{block_type}'. Must be 'M' for MAB or 'S' for SAB."
                )
        if aggregation not in ["sum", "mean", "attention"]:
            raise ValueError(
                f"Invalid aggregation '{aggregation}'. Must be 'sum', 'mean', or 'attention'."
            )

        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.pattern = pattern
        self.aggregation = aggregation

        # Create masked attention blocks for each edge type.
        self.masked_blocks = nn.ModuleDict()
        for edge_type in edge_types:
            src, rel, dst = edge_type
            key = f"{src}__{rel}__{dst}"
            self.masked_blocks[key] = NodeSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )

        # Create self-attention blocks for each node type.
        self.self_blocks = nn.ModuleDict()
        for node_type in node_types:
            self.self_blocks[node_type] = SelfAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )

        # Optionally create attentional aggregators for nodes in multiple relations.
        if aggregation == "attention":
            self.node_aggregators = nn.ModuleDict()
            for node_type in node_types:
                relations = [
                    (s, r, d)
                    for s, r, d in edge_types
                    if s == node_type or d == node_type
                ]
                if len(relations) > 1:
                    gate_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                    transform_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    self.node_aggregators[node_type] = AttentionalAggregation(
                        gate_nn=gate_nn, nn=transform_nn
                    )

    def _process_with_mask(
        self,
        block: nn.Module,
        embeddings: Tensor,
        mask: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        if mask.dim() == 2 and embeddings.dim() == 2:
            mask = mask.unsqueeze(0)
            embeddings = embeddings.unsqueeze(0)
        elif mask.dim() == 2 and embeddings.dim() == 3:
            mask = mask.unsqueeze(0).expand(embeddings.size(0), -1, -1)
        elif mask.dim() == 3 and embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if isinstance(block, NodeSelfAttention):
            if edge_attr is not None and edge_index is not None:
                x = block(embeddings, mask, edge_attr, edge_index)
            else:
                x = block(embeddings, mask)
        else:
            x = block(embeddings)
        if embeddings.dim() == 2 and x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        return x

    def forward(
        self,
        node_embeddings: Dict[str, Tensor],
        data: HeteroData,
        batch_idx: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        current_embeddings = {k: v.clone() for k, v in node_embeddings.items()}
        for block_type in self.pattern:
            if block_type == "M":
                relation_outputs = {node_type: [] for node_type in self.node_types}
                for edge_type in self.edge_types:
                    src, rel, dst = edge_type
                    if src not in current_embeddings or dst not in current_embeddings:
                        continue
                    key = f"{src}__{rel}__{dst}"
                    if key not in self.masked_blocks:
                        continue
                    block = self.masked_blocks[key]
                    src_emb = current_embeddings[src]
                    dst_emb = current_embeddings[dst]
                    if edge_type in data.edge_types:
                        edge_store = data[edge_type]
                        if (
                            hasattr(edge_store, "adj_mask")
                            and edge_store.adj_mask is not None
                        ):
                            adj_mask = edge_store.adj_mask
                            edge_attr = getattr(edge_store, "edge_attr", None)
                            edge_index = getattr(edge_store, "edge_index", None)
                            if rel == "rmr" and hasattr(edge_store, "stoichiometry"):
                                edge_attr = edge_store.stoichiometry
                                edge_index = edge_store.edge_index
                            if src == dst:
                                out_src = self._process_with_mask(
                                    block, src_emb, adj_mask, edge_attr, edge_index
                                )
                                relation_outputs[src].append(out_src)
                            else:
                                out_src = self._process_with_mask(
                                    block, src_emb, adj_mask, edge_attr, edge_index
                                )
                                relation_outputs[src].append(out_src)
                                adj_mask_t = adj_mask.transpose(-2, -1)
                                out_dst = self._process_with_mask(
                                    block, dst_emb, adj_mask_t, edge_attr, edge_index
                                )
                                relation_outputs[dst].append(out_dst)
                        elif (
                            hasattr(edge_store, "inc_mask")
                            and edge_store.inc_mask is not None
                        ):
                            inc_mask = edge_store.inc_mask
                            edge_attr = None
                            edge_index = None
                            if rel == "rmr" and hasattr(edge_store, "stoichiometry"):
                                edge_attr = edge_store.stoichiometry
                                edge_index = (
                                    getattr(edge_store, "hyperedge_index", None)
                                    or edge_store.edge_index
                                )
                            out_src = self._process_with_mask(
                                block, src_emb, inc_mask, edge_attr, edge_index
                            )
                            relation_outputs[src].append(out_src)
                            if src != dst:
                                inc_mask_t = inc_mask.transpose(-2, -1)
                                out_dst = self._process_with_mask(
                                    block, dst_emb, inc_mask_t, edge_attr, edge_index
                                )
                                relation_outputs[dst].append(out_dst)
                        elif hasattr(edge_store, "edge_index"):
                            edge_index = edge_store.edge_index
                            if src == dst:
                                adj = torch.eye(
                                    src_emb.size(0), device=src_emb.device
                                ).bool()
                                edge_attr = getattr(edge_store, "edge_attr", None)
                                out_src = self._process_with_mask(
                                    block, src_emb, adj, edge_attr, edge_index
                                )
                                relation_outputs[src].append(out_src)
                            else:
                                src_adj = torch.eye(
                                    src_emb.size(0), device=src_emb.device
                                ).bool()
                                dst_adj = torch.eye(
                                    dst_emb.size(0), device=dst_emb.device
                                ).bool()
                                edge_attr = getattr(edge_store, "edge_attr", None)
                                out_src = self._process_with_mask(
                                    block, src_emb, src_adj, edge_attr, edge_index
                                )
                                out_dst = self._process_with_mask(
                                    block, dst_emb, dst_adj, edge_attr, edge_index
                                )
                                relation_outputs[src].append(out_src)
                                relation_outputs[dst].append(out_dst)
                next_embeddings = {}
                for node_type, outputs in relation_outputs.items():
                    if not outputs:
                        next_embeddings[node_type] = current_embeddings[node_type]
                        continue
                    if len(outputs) == 1:
                        next_embeddings[node_type] = outputs[0]
                    else:
                        if self.aggregation == "sum":
                            next_embeddings[node_type] = sum(outputs)
                        elif self.aggregation == "mean":
                            next_embeddings[node_type] = sum(outputs) / len(outputs)
                        elif self.aggregation == "attention" and node_type in getattr(
                            self, "node_aggregators", {}
                        ):
                            flat_embs = torch.cat(outputs, dim=0)
                            node_indices = torch.arange(
                                len(outputs), device=flat_embs.device
                            ).repeat_interleave(outputs[0].size(0))
                            aggregated = self.node_aggregators[node_type](
                                flat_embs, index=node_indices
                            )
                            next_embeddings[node_type] = aggregated
                        else:
                            # Default to mean if attention is requested but not available
                            next_embeddings[node_type] = sum(outputs) / len(outputs)
                current_embeddings.update(next_embeddings)
            elif block_type == "S":
                next_embeddings = {}
                for node_type in self.node_types:
                    if node_type not in current_embeddings:
                        continue
                    if node_type not in self.self_blocks:
                        next_embeddings[node_type] = current_embeddings[node_type]
                        continue
                    block = self.self_blocks[node_type]
                    emb = current_embeddings[node_type]
                    if emb.dim() == 2:
                        emb = emb.unsqueeze(0)
                        out = block(emb).squeeze(0)
                    else:
                        out = block(emb)
                    next_embeddings[node_type] = out
                current_embeddings = next_embeddings
            else:
                raise ValueError(f"Invalid block type '{block_type}'.")
        return current_embeddings


class HeteroNSAEncoder(nn.Module):
    """
    Full encoder using HeteroNSA with input projections and multiple layers.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        node_types: Set[str],
        edge_types: Set[Tuple[str, str, str]],
        pattern: List[str],  # Single pattern list for all edge types
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        aggregation: Literal["sum", "mean", "attention"] = "sum",
    ) -> None:
        """Initialize the NSA encoder."""
        super().__init__()
        if aggregation not in ["sum", "mean", "attention"]:
            raise ValueError(
                f"Invalid aggregation '{aggregation}'. Must be 'sum', 'mean', or 'attention'."
            )

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
                    pattern=pattern,  # Use single pattern for all edge types
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

        # First, process all available node types
        for node_type in self.node_types:
            if node_type in data.node_types:
                # Get node features
                node_data = data[node_type]
                if hasattr(node_data, "x") and node_data.x is not None:
                    x = (
                        node_data.x.clone()
                    )  # Make a copy to avoid in-place modifications

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
            # Process through the layer
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
                        elif residual.dim() == 3 and new_x_dict[node_type].dim() == 2:
                            residual = residual.squeeze(0)

                    new_x_dict[node_type] = self.layer_norms[node_type][i](
                        new_x_dict[node_type] + residual
                    )

            # Update for next layer
            x_dict = new_x_dict

            # Update final embeddings
            final_embeddings.update(x_dict)

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
