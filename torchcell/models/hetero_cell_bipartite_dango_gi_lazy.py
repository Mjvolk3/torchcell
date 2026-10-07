# torchcell/models/hetero_cell_bipartite_dango_gi_lazy
# [[torchcell.models.hetero_cell_bipartite_dango_gi_lazy]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_bipartite_dango_gi_lazy
# Test file: tests/torchcell/models/test_hetero_cell_bipartite_dango_gi_lazy.py
#
# Lazy version adapted for LazySubgraphRepresentation's zero-copy architecture
"""Lazy bipartite Dango gene-interaction model using masked, zero-copy message passing."""

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Additional imports for enhanced plotting
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GATv2Conv, GINConv
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.typing import EdgeType
from torch_scatter import scatter_mean

from torchcell.graph.graph import GeneMultiGraph
from torchcell.models.act import act_register
from torchcell.models.norm import norm_register
from torchcell.nn.masked_gin_conv import MaskedGINConv


class SelfAttentionGraphAggregation(nn.Module):
    """Self-attention mechanism for aggregating multiple graph representations of the same nodes."""

    def __init__(
        self, hidden_dim: int, num_graphs: int, num_heads: int = 4, dropout: float = 0.0
    ):
        """Build the multi-head attention layer and learnable per-graph positional encodings."""
        super().__init__()
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Learnable graph positional encodings
        self.graph_embeddings = nn.Parameter(torch.randn(num_graphs, hidden_dim) * 0.02)

    def forward(
        self, graph_outputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Attend across per-graph node features and mean-pool to one representation.

        Args:
            graph_outputs: Dict mapping graph names to node features [num_nodes, hidden_dim]

        Returns:
            aggregated: Aggregated node features [num_nodes, hidden_dim]
            attention_weights: Attention weights [num_nodes, num_graphs, num_graphs]
        """
        # Sort graph names for consistent ordering
        graph_names = sorted(graph_outputs.keys())
        if not graph_names:
            return None, None

        # Stack graph outputs: [num_nodes, num_graphs, hidden_dim]
        stacked = torch.stack([graph_outputs[name] for name in graph_names], dim=1)

        # Add learnable graph positional encodings
        num_nodes = stacked.size(0)
        num_actual_graphs = len(graph_names)
        graph_emb_expanded = (
            self.graph_embeddings[:num_actual_graphs]
            .unsqueeze(0)
            .expand(num_nodes, -1, -1)
        )
        stacked = stacked + graph_emb_expanded

        # Apply self-attention
        attended, attn_weights = self.multihead_attn(
            query=stacked,
            key=stacked,
            value=stacked,
            need_weights=True,
            average_attn_weights=True,  # Average over heads for visualization
        )

        # Mean pooling across graphs dimension
        aggregated = attended.mean(dim=1)  # [num_nodes, hidden_dim]

        return aggregated, attn_weights


class PairwiseGraphAggregation(nn.Module):
    """Pairwise interaction mechanism for aggregating multiple graph representations."""

    def __init__(
        self,
        hidden_dim: int,
        graph_names: list[str],
        dropout: float = 0.0,
        activation: str = "relu",
        num_layers: int = 2,
        bottleneck_dim: int | None = None,
        norm: str | None = None,
    ):
        """Build per-pair interaction MLPs, a scorer, and optional output normalization."""
        super().__init__()
        self.graph_names = sorted(graph_names)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Default bottleneck to half of hidden_dim if not specified
        self.bottleneck_dim = (
            bottleneck_dim if bottleneck_dim is not None else hidden_dim // 2
        )

        # Get activation layer from register - NO FALLBACK
        if activation not in act_register:
            raise ValueError(
                f"activation '{activation}' not found in act_register. "
                f"Available: {list(act_register.keys())}"
            )
        act_layer = act_register[activation]

        # Pairwise interaction networks with configurable depth and bottleneck
        self.interaction_mlps = nn.ModuleDict()
        for i, g1 in enumerate(self.graph_names):
            for j, g2 in enumerate(self.graph_names[i:], i):
                key = f"{g1}_{g2}"

                # Build MLP layers dynamically
                layers = []
                input_dim = hidden_dim * 2

                # First layer: compress to bottleneck
                layers.extend(
                    [
                        nn.Linear(input_dim, self.bottleneck_dim),
                        act_layer,
                        nn.Dropout(dropout),
                    ]
                )

                # Middle layers: maintain bottleneck dimension
                for _ in range(num_layers - 2):
                    layers.extend(
                        [
                            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
                            act_layer,
                            nn.Dropout(dropout),
                        ]
                    )

                # Final layer: expand back to hidden_dim
                layers.append(nn.Linear(self.bottleneck_dim, hidden_dim))

                self.interaction_mlps[key] = nn.Sequential(*layers)

        # Scoring mechanism for weighting pairwise interactions
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim, self.bottleneck_dim),
            act_layer,
            nn.Linear(self.bottleneck_dim, 1),
        )

        # Normalization after aggregation
        self.norm: nn.Module | None
        if norm is not None:
            self.norm = get_norm_layer(hidden_dim, norm)
        else:
            self.norm = None

    def forward(
        self, graph_outputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute pairwise graph interactions and aggregate them.

        Includes identity option (mean of individual graphs) alongside pairwise combinations.

        Returns:
            aggregated: Aggregated features [num_nodes, hidden_dim]
            pair_weights: Learned importance weights for pairwise interactions + identity [num_nodes, num_interactions+1]
        """
        if not graph_outputs:
            return None, None

        interactions = []
        interaction_pairs = []

        for i, g1 in enumerate(self.graph_names):
            if g1 not in graph_outputs:
                continue
            g1_feat = graph_outputs[g1]

            for j, g2 in enumerate(self.graph_names[i:], i):
                if g2 not in graph_outputs:
                    continue
                g2_feat = graph_outputs[g2]

                # Concatenate features
                if i == j:  # Self-interaction
                    combined = torch.cat([g1_feat, g1_feat], dim=-1)
                else:
                    combined = torch.cat([g1_feat, g2_feat], dim=-1)

                # Compute interaction
                key = f"{g1}_{g2}"
                if key in self.interaction_mlps:
                    interaction = self.interaction_mlps[key](combined)
                    interactions.append(interaction)
                    interaction_pairs.append((g1, g2))

        if not interactions:
            # Fallback to mean if no interactions
            return torch.stack(list(graph_outputs.values())).mean(dim=0), None

        # Add identity option: mean of individual graph representations
        identity = torch.stack(list(graph_outputs.values())).mean(
            dim=0
        )  # [num_nodes, hidden_dim]
        interactions.append(identity)
        interaction_pairs.append(("identity", "identity"))

        # Stack all options: [num_nodes, num_interactions+1, hidden_dim]
        stacked = torch.stack(interactions, dim=1)

        # Compute importance scores for all options (pairwise + identity)
        pair_logits = self.pair_scorer(stacked).squeeze(
            -1
        )  # [num_nodes, num_interactions+1]
        pair_weights = F.softmax(pair_logits, dim=-1)

        # Weighted aggregation over all options
        aggregated = (stacked * pair_weights.unsqueeze(-1)).sum(dim=1)

        # Apply normalization after aggregation if configured
        if self.norm is not None:
            aggregated = self.norm(aggregated)

        return aggregated, pair_weights


class HeteroConvAggregator(nn.Module):
    """HeteroConv wrapper with configurable aggregation strategies.

    Supports: sum, mean, cross_attention, pairwise_interaction
    """

    def __init__(
        self,
        convs: dict[EdgeType, nn.Module],
        hidden_channels: int,
        aggregation_method: str = "cross_attention",
        aggregation_config: dict[str, Any] | None = None,
    ):
        """Store the per-edge-type convs and build the chosen aggregation module."""
        super().__init__()
        self.convs = nn.ModuleDict({str(k): v for k, v in convs.items()})
        self.hidden_channels = hidden_channels
        self.aggregation_method = aggregation_method

        # Extract unique graph names from edge types
        self.graph_names = sorted(
            list(set([edge_type[1] for edge_type in convs.keys()]))
        )

        # Initialize aggregation module based on method
        config = aggregation_config or {}

        self.aggregator: SelfAttentionGraphAggregation | PairwiseGraphAggregation | None
        if aggregation_method == "cross_attention":
            self.aggregator = SelfAttentionGraphAggregation(
                hidden_dim=hidden_channels,
                num_graphs=len(self.graph_names),
                num_heads=config.get("num_heads", 4),
                dropout=config.get("dropout", 0.0),
            )
        elif aggregation_method == "pairwise_interaction":
            self.aggregator = PairwiseGraphAggregation(
                hidden_dim=hidden_channels,
                graph_names=self.graph_names,
                dropout=config.get("dropout", 0.0),
                activation=config.get("activation", "relu"),
                num_layers=config.get("pairwise_num_layers", 2),
                bottleneck_dim=config.get("pairwise_hidden_dim", None),
                norm=config.get("aggregation_norm", None),
            )
        elif aggregation_method in ["sum", "mean"]:
            self.aggregator = None  # Will use simple aggregation
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_mask_dict: dict[EdgeType, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None] | None]:
        """Apply graph convolutions and aggregate using specified method.

        LAZY VERSION: Accepts edge_mask_dict for masking edges during message passing.

        Args:
            x_dict: Node features by node type
            edge_index_dict: Edge indices by edge type
            edge_mask_dict: Optional edge masks by edge type (True = keep edge)

        Returns:
            out_dict: Updated node features
            attention_weights: Graph aggregation weights (if applicable)
        """
        # Default to empty dict if no masks provided
        edge_mask_dict = edge_mask_dict or {}

        # Store outputs by destination node type and graph name
        out_dict: dict[str, torch.Tensor] = {}
        graph_outputs_by_dst: dict[
            str, dict[str, torch.Tensor]
        ] = {}  # {dst_type: {graph_name: tensor}}

        # Apply convolutions for each edge type
        for edge_type_str, conv in self.convs.items():
            # Parse the edge type string back to tuple
            edge_type = (
                eval(edge_type_str) if isinstance(edge_type_str, str) else edge_type_str
            )
            src, rel, dst = edge_type

            # Get the edge index for this edge type
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
            else:
                continue

            # Get edge mask for this edge type (if available)
            edge_mask = edge_mask_dict.get(edge_type, None)

            # Apply convolution with optional edge mask
            out = conv(x_dict[src], edge_index, edge_mask=edge_mask)

            # Organize outputs by destination and graph
            if dst not in graph_outputs_by_dst:
                graph_outputs_by_dst[dst] = {}
            graph_outputs_by_dst[dst][rel] = out

        # Aggregate for each destination node type
        all_attention_weights: dict[str, torch.Tensor | None] = {}

        for dst, graph_outputs in graph_outputs_by_dst.items():
            if not graph_outputs:
                continue

            if self.aggregation_method == "sum":
                # Simple sum aggregation
                out_dict[dst] = cast(torch.Tensor, sum(graph_outputs.values()))
                all_attention_weights[dst] = None

            elif self.aggregation_method == "mean":
                # Simple mean aggregation
                stacked = torch.stack(list(graph_outputs.values()))
                out_dict[dst] = stacked.mean(dim=0)
                all_attention_weights[dst] = None

            elif self.aggregator is not None:
                # Use learned aggregation (cross_attention or pairwise_interaction)
                agg_out, attn_weights = self.aggregator(graph_outputs)
                out_dict[dst] = cast(torch.Tensor, agg_out)
                all_attention_weights[dst] = attn_weights
            else:
                # Fallback to sum
                out_dict[dst] = cast(torch.Tensor, sum(graph_outputs.values()))
                all_attention_weights[dst] = None

        # Return aggregated features and attention weights
        return out_dict, (
            all_attention_weights
            if any(v is not None for v in all_attention_weights.values())
            else None
        )


class AttentionalGraphAggregation(nn.Module):
    """Attentional aggregation pooling node features into graph-level vectors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Build the gating and transform MLPs and the attentional aggregator."""
        super().__init__()
        # Validate activation - NO FALLBACK
        if activation not in act_register:
            raise ValueError(
                f"activation '{activation}' not found in act_register. "
                f"Available: {list(act_register.keys())}"
            )

        act_layer = act_register[activation]
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )
        self.transform_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels), act_layer, nn.Dropout(dropout)
        )
        self.aggregator = AttentionalAggregation(
            gate_nn=self.gate_nn, nn=self.transform_nn
        )

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, dim_size: int | None = None
    ) -> torch.Tensor:
        """Aggregate node features into graph-level vectors using attention weights."""
        return cast(torch.Tensor, self.aggregator(x, index=index, dim_size=dim_size))


class DangoLikeHyperSAGNN(nn.Module):
    """Dango-like HyperSAGNN for local gene interaction prediction.

    Implements multi-layer self-attention with multi-head attention and ReZero connections.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Build the stacked self-attention layers with ReZero connections."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )

        # Validate activation - NO FALLBACK
        if activation not in act_register:
            raise ValueError(
                f"activation '{activation}' not found in act_register. "
                f"Available: {list(act_register.keys())}"
            )

        # Static embedding layer with config-driven activation
        self.static_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act_register[activation]
        )

        # Create multiple attention layers
        self.attention_layers = nn.ModuleList()
        self.beta_params = nn.ParameterList()  # Store ReZero parameters separately

        for i in range(num_layers):
            layer = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(hidden_dim, hidden_dim),
                    "k_proj": nn.Linear(hidden_dim, hidden_dim),
                    "v_proj": nn.Linear(hidden_dim, hidden_dim),
                    "out_proj": nn.Linear(hidden_dim, hidden_dim),
                }
            )
            self.attention_layers.append(layer)
            # Create ReZero parameter for this layer
            beta = nn.Parameter(torch.zeros(1))
            nn.init.constant_(beta, 0.01)  # Initialize to small value like Dango
            self.beta_params.append(beta)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, gene_embeddings: torch.Tensor, batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute static and attention-refined dynamic gene embeddings.

        Args:
            gene_embeddings: Tensor of shape [total_genes, hidden_dim]
            batch: Optional tensor [total_genes] indicating batch assignment
        Returns:
            static_embeddings: Tensor of shape [total_genes, hidden_dim]
            dynamic_embeddings: Tensor of shape [total_genes, hidden_dim]
        """
        # Compute static embeddings
        static_embeddings = self.static_embedding(gene_embeddings)

        # Initialize dynamic embeddings
        dynamic_embeddings = gene_embeddings

        # Apply attention layers
        for i, layer in enumerate(self.attention_layers):
            layer = cast(nn.ModuleDict, layer)
            if batch is not None:
                # VECTORIZED: No .unique(), no Python loop over batches
                # Pre-compute batch sizes and offsets (no GPU sync)
                batch_sizes = torch.bincount(batch)
                dynamic_embeddings = self._apply_attention_layer_batched(
                    dynamic_embeddings, batch, batch_sizes, layer, self.beta_params[i]
                )
            else:
                # Single batch processing
                dynamic_embeddings = self._apply_attention_layer(
                    dynamic_embeddings, layer, self.beta_params[i]
                )

        return static_embeddings, dynamic_embeddings

    def _apply_attention_layer(
        self, x: torch.Tensor, layer: nn.ModuleDict, beta: nn.Parameter
    ) -> torch.Tensor:
        """Apply a single attention layer with multi-head attention."""
        batch_size = x.size(0)

        # Handle special case of single gene (no attention possible)
        if batch_size <= 1:
            return x

        # Linear projections
        q = layer["q_proj"](x)  # [batch_size, hidden_dim]
        k = layer["k_proj"](x)  # [batch_size, hidden_dim]
        v = layer["v_proj"](x)  # [batch_size, hidden_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim).transpose(
            0, 1
        )  # [num_heads, batch_size, head_dim]
        k = k.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        # Shape: [num_heads, batch_size, batch_size]

        # Create mask for self-attention (exclude self)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=x.device)
        self_mask = self_mask.unsqueeze(0).expand(self.num_heads, -1, -1)

        # Apply mask
        attention_scores.masked_fill_(self_mask, -float("inf"))

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Handle potential NaNs from empty rows (single gene case)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        # Shape: [num_heads, batch_size, head_dim]

        # Reshape back to [batch_size, hidden_dim]
        out = out.transpose(0, 1).contiguous().view(batch_size, self.hidden_dim)

        # Apply output projection
        out = layer["out_proj"](out)
        out = self.dropout(out)

        # Apply ReZero connection
        return cast(torch.Tensor, x + beta * out)

    def _apply_attention_layer_batched(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        batch_sizes: torch.Tensor,
        layer: nn.ModuleDict,
        beta: nn.Parameter,
    ) -> torch.Tensor:
        """Apply attention layer across multiple batches using segment operations.

        No .unique(), no Python loops over batches.

        Args:
            x: [total_genes, hidden_dim]
            batch: [total_genes] sorted batch indices [0,0,0,1,1,1,2,2,2,...]
            batch_sizes: [num_batches] count per batch (from bincount)
            layer: attention layer dict
            beta: ReZero parameter

        Returns:
            output: [total_genes, hidden_dim] with attention applied per batch
        """
        device = x.device
        num_batches = len(batch_sizes)

        # Handle batches with single gene (no attention possible)
        # Find which batches have > 1 gene
        multi_gene_mask = batch_sizes > 1

        if not multi_gene_mask.any():
            return x  # All batches have single genes, no attention needed

        # Create output buffer
        output = x.clone()

        # Get cumulative offsets for slicing (no sync)
        offsets = torch.cat([torch.tensor([0], device=device), batch_sizes.cumsum(0)])

        # Process each batch (loop over num_batches ~32, not over genes ~thousands)
        # This is much faster than the previous approach
        for batch_idx in range(num_batches):
            if batch_sizes[batch_idx] <= 1:
                continue  # Skip single-gene batches

            start_idx = offsets[batch_idx]
            end_idx = offsets[batch_idx + 1]

            # Extract batch embeddings and apply attention
            batch_embeddings = x[start_idx:end_idx]
            batch_output = self._apply_attention_layer(batch_embeddings, layer, beta)
            output[start_idx:end_idx] = batch_output

        return output


class GeneInteractionPredictor(nn.Module):
    """Predict gene interaction scores from gene embeddings via a HyperSAGNN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Build the HyperSAGNN encoder and the linear scoring head."""
        super().__init__()
        # Use the new Dango-like HyperSAGNN with config-driven activation
        self.hyper_sagnn = DangoLikeHyperSAGNN(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )
        # Prediction layer to compute scores from squared differences
        self.prediction_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.prediction_layer.weight)

    def forward(
        self, gene_embeddings: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score gene interactions from static/dynamic embedding differences.

        Args:
            gene_embeddings: Tensor of shape [total_genes, hidden_dim]
            batch: Optional tensor [total_genes] indicating batch assignment
        Returns:
            interaction_scores: Tensor of shape [num_batches]
        """
        # Get static and dynamic embeddings from HyperSAGNN
        static_embeddings, dynamic_embeddings = self.hyper_sagnn(gene_embeddings, batch)

        # Calculate the difference and square it (like Dango)
        diff = dynamic_embeddings - static_embeddings
        diff_squared = diff**2

        # Get gene-level scores
        gene_scores = self.prediction_layer(diff_squared).squeeze(-1)  # [total_genes]

        # If batch information is provided, average scores per batch using scatter_mean
        if batch is not None:
            # Use scatter_mean for efficient batched averaging
            # Compute num_batches on GPU first, sync only when needed for scatter_mean
            num_batches = batch.max() + 1  # Keep as tensor
            interaction_scores = scatter_mean(
                gene_scores,
                batch,
                dim=0,
                dim_size=int(num_batches.item()),  # Sync here
            )
            return cast(
                torch.Tensor, interaction_scores.unsqueeze(-1)
            )  # [num_batches, 1]
        else:
            # Single batch case
            return cast(
                torch.Tensor, gene_scores.mean().unsqueeze(0).unsqueeze(-1)
            )  # [1, 1]


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    """Get normalization layer from norm_register - NO FALLBACK.

    Args:
        channels: Number of channels for normalization
        norm: Normalization type string (must be in norm_register)

    Returns:
        Normalization layer instance

    Raises:
        ValueError: If norm not found in norm_register
    """
    if norm not in norm_register:
        raise ValueError(
            f"norm '{norm}' not found in norm_register. "
            f"Available: {list(norm_register.keys())}"
        )
    # PyG norm classes take in_channels as first parameter
    return cast(nn.Module, norm_register[norm](channels))


class PreProcessor(nn.Module):
    """MLP that preprocesses node features with linear, norm, activation, and dropout layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
    ):
        """Build the preprocessing MLP layer stack from the channel and norm settings."""
        super().__init__()
        # Get activation from register - NO FALLBACK
        if activation is None:
            raise ValueError("activation must be specified for PreProcessor")
        if activation not in act_register:
            raise ValueError(
                f"activation '{activation}' not found in act_register. "
                f"Available: {list(act_register.keys())}"
            )
        self.act = act_register[activation]
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(norm_layer)
        layers.append(self.act)
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(norm_layer)
            layers.append(self.act)
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass node features through the preprocessing MLP."""
        return cast(torch.Tensor, self.mlp(x))


class AttentionConvWrapper(nn.Module):
    """Wrapper applying norm, activation, and dropout around a graph convolution."""

    def __init__(
        self,
        conv: nn.Module,
        target_dim: int,
        norm: str | None = None,
        activation: str | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Wrap the conv and set up the norm, activation, and dropout layers."""
        super().__init__()
        self.conv = conv

        # Determine expected output dimension based on conv type
        expected_dim: int
        if isinstance(conv, (GINConv, MaskedGINConv)):
            # For GINConv/MaskedGINConv, get output dim from the last layer of the MLP
            mlp = conv.nn
            if isinstance(mlp, nn.Sequential):
                # Find the last Linear layer in the MLP
                for module in reversed(list(mlp.modules())):
                    if isinstance(module, nn.Linear):
                        expected_dim = module.out_features
                        break
            else:
                expected_dim = target_dim  # fallback
        elif hasattr(conv, "concat"):
            # For GATv2Conv
            expected_dim = (
                cast(int, conv.heads) * cast(int, conv.out_channels)
                if conv.concat
                else cast(int, conv.out_channels)
            )
        else:
            # For other conv types that have out_channels
            expected_dim = cast(int, conv.out_channels)

        self.proj = (
            nn.Identity()
            if expected_dim == target_dim
            else nn.Linear(expected_dim, target_dim)
        )

        # Use norm_register - NO FALLBACK
        self.norm: nn.Module | None
        if norm is not None:
            self.norm = get_norm_layer(target_dim, norm)
        else:
            self.norm = None

        # Get activation from register - NO FALLBACK
        if activation is None:
            raise ValueError("activation must be specified for AttentionConvWrapper")
        if activation not in act_register:
            raise ValueError(
                f"activation '{activation}' not found in act_register. "
                f"Available: {list(act_register.keys())}"
            )
        self.act = act_register[activation]
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the wrapped conv with optional edge masking, then norm/activation/dropout.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_mask: Optional edge mask for MaskedGINConv (True = keep edge)
            **kwargs: Additional arguments for the conv layer
        """
        # Pass edge_mask to conv if it's MaskedGINConv
        if isinstance(self.conv, MaskedGINConv):
            out = self.conv(x, edge_index, edge_mask=edge_mask)
        else:
            # For other conv types (GATv2), pass through kwargs
            out = self.conv(x, edge_index, **kwargs)

        out = self.proj(out)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return cast(torch.Tensor, out)


def create_conv_layer(
    encoder_type: str,
    in_channels: int,
    out_channels: int,
    config: dict[str, Any],
    activation: str,
    edge_dim: int | None = None,
    dropout: float = 0.1,
) -> nn.Module:
    """Create appropriate conv layer based on encoder type.

    LAZY VERSION: Only supports GIN with MaskedGINConv for edge masking.

    Args:
        encoder_type: Type of encoder - only "gin" supported (gatv2 not yet implemented)
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        config: Configuration dict for the encoder
        activation: Activation function name (must be in act_register)
        edge_dim: Edge feature dimension (unused for GIN)
        dropout: Dropout rate for GIN MLP

    Returns:
        MaskedGINConv layer with edge masking support

    Raises:
        ValueError: If activation not found in act_register
    """
    # Validate activation - NO FALLBACK
    if activation not in act_register:
        raise ValueError(
            f"activation '{activation}' not found in act_register. "
            f"Available: {list(act_register.keys())}"
        )

    if encoder_type == "gatv2":
        raise NotImplementedError(
            "GATv2 not yet supported for lazy architecture - use GIN encoder. "
            "To implement: create MaskedGATv2Conv following MaskedGINConv pattern."
        )
    elif encoder_type == "gin":
        # GIN uses MLP for transformation
        gin_hidden = config.get("gin_hidden_dim") or out_channels
        gin_layers = config.get("gin_num_layers", 2)

        # Build MLP with config-driven activation
        mlp_layers = []
        act_layer = act_register[activation]
        for i in range(gin_layers):
            if i == 0:
                mlp_layers.extend(
                    [nn.Linear(in_channels, gin_hidden), act_layer, nn.Dropout(dropout)]
                )
            elif i == gin_layers - 1:
                mlp_layers.append(nn.Linear(gin_hidden, out_channels))
            else:
                mlp_layers.extend(
                    [nn.Linear(gin_hidden, gin_hidden), act_layer, nn.Dropout(dropout)]
                )

        mlp = nn.Sequential(*mlp_layers)
        # LAZY: Use MaskedGINConv instead of GINConv for edge masking
        return MaskedGINConv(mlp, train_eps=True)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class GeneInteractionDango(nn.Module):
    """Lazy Dango-style model predicting gene interactions from a gene multigraph."""

    def __init__(
        self,
        gene_num: int,
        hidden_channels: int,
        num_layers: int,
        gene_multigraph: GeneMultiGraph,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        gene_encoder_config: dict[str, Any] | None = None,
        local_predictor_config: dict[str, Any] | None = None,
    ):
        """Build gene embeddings, the masked graph encoder stack, and the interaction predictor."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gene_multigraph = gene_multigraph

        # Extract graph names from the multigraph
        self.graph_names = list(gene_multigraph.keys())

        # Get combination method from config
        local_predictor_config = local_predictor_config or {}
        self.use_local_predictor = local_predictor_config.get(
            "use_local_predictor", True
        )
        self.combination_method = local_predictor_config.get(
            "combination_method", "gating"
        )

        # Learnable gene embeddings
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)

        # Initialize embedding with better bounds and smaller scale for stability
        # Use smaller initialization to prevent early NaNs
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)

        # Preprocessor for input embeddings
        self.preprocessor = PreProcessor(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        # Default config if not provided
        gene_encoder_config = gene_encoder_config or {}

        # Get graph aggregation configuration
        self.graph_aggregation_method = gene_encoder_config.get(
            "graph_aggregation_method",
            "cross_attention",  # Default to cross_attention
        )
        self.graph_aggregation_config = gene_encoder_config.get(
            "graph_aggregation_config", {}
        )

        # Store aggregation weights from all layers for visualization
        self.layer_aggregation_weights: list[torch.Tensor] = []

        # Graph convolution layers with interaction-based aggregation
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict: dict[EdgeType, nn.Module] = {}

            # Create a conv layer for each graph in the multigraph
            for graph_name in self.graph_names:
                edge_type = ("gene", graph_name, "gene")
                encoder_type = gene_encoder_config.get("encoder_type", "gatv2")

                conv_layer = create_conv_layer(
                    encoder_type=encoder_type,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    config=gene_encoder_config,
                    activation=activation,
                    dropout=dropout,
                )

                # Wrap with AttentionConvWrapper
                conv_dict[edge_type] = AttentionConvWrapper(
                    conv_layer,
                    hidden_channels,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )

            # Use our custom HeteroConvAggregator with configurable aggregation
            self.convs.append(
                HeteroConvAggregator(
                    convs=conv_dict,
                    hidden_channels=hidden_channels,
                    aggregation_method=self.graph_aggregation_method,
                    aggregation_config={
                        **self.graph_aggregation_config,
                        "dropout": dropout,  # Pass model dropout to aggregation
                        "activation": activation,  # Pass model activation to aggregation
                    },
                )
            )

        # Get local predictor config - now as a separate parameter
        local_predictor_config = local_predictor_config or {}

        # Gene interaction predictor for perturbed genes with Dango-like architecture
        # (optional - only create if enabled)
        self.gene_interaction_predictor: GeneInteractionPredictor | None
        if self.use_local_predictor:
            self.gene_interaction_predictor = GeneInteractionPredictor(
                hidden_dim=hidden_channels,
                num_heads=local_predictor_config.get("num_heads", 4),
                num_layers=local_predictor_config.get("num_attention_layers", 2),
                dropout=dropout,
                activation=activation,
            )
        else:
            self.gene_interaction_predictor = None

        # Global aggregator for proper aggregation
        self.global_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
            activation=activation,
        )

        # Global predictor for z_p_global
        self.global_interaction_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            act_register[activation],
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # MLP for gating weights (only if using gating combination method AND local predictor enabled)
        self.gate_mlp: nn.Sequential | None
        if self.combination_method == "gating" and self.use_local_predictor:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2, hidden_channels),
                act_register[activation],
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 2),
            )
        else:
            self.gate_mlp = None

        # Log local predictor mode
        predictor_mode = (
            "enabled" if self.use_local_predictor else "disabled (global-only)"
        )
        print(f"GeneInteractionDango - Local predictor: {predictor_mode}")

        # Initialize all weights properly
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all weights in the model with appropriate initializations."""

        def _init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                # Kaiming initialization for ReLU-based networks
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, GATv2Conv):
                if hasattr(module, "lin_src"):
                    nn.init.kaiming_normal_(
                        module.lin_src.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if module.lin_src.bias is not None:
                        nn.init.zeros_(module.lin_src.bias)
                if hasattr(module, "lin_dst"):
                    nn.init.kaiming_normal_(
                        module.lin_dst.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if module.lin_dst.bias is not None:
                        nn.init.zeros_(module.lin_dst.bias)
                if hasattr(module, "att_src"):
                    nn.init.xavier_normal_(module.att_src)
                if hasattr(module, "att_dst"):
                    nn.init.xavier_normal_(module.att_dst)

        # Apply to all modules
        self.apply(_init_module)

        # Specific initializations for key components
        # ReZero parameters are already initialized in DangoLikeHyperSAGNN.__init__

    def forward_single(self, data: HeteroData | Batch) -> torch.Tensor:
        """Process graph data with full node/edge tensors and masking (lazy version).

        Key differences from original:
        - No node filtering: x references FULL graph (all nodes including perturbed)
        - Masking happens during message passing (via edge masks)
        - Returns full graph embeddings [total_nodes, hidden_dim]

        Args:
            data: HeteroData or Batch with LazySubgraphRepresentation format

        Returns:
            Full graph embeddings [total_nodes, hidden_dim]
        """
        device = self.gene_embedding.weight.device

        # Handle both batch and single graph input
        is_batch = isinstance(data, Batch) or hasattr(data["gene"], "batch")

        if is_batch:
            gene_data = data["gene"]
            batch_size = len(data["gene"].ptr) - 1

            # LAZY APPROACH: Reference FULL graph - no filtering!
            # x is [total_nodes, feat_dim] - already includes all nodes (kept + perturbed)
            # Expand embedding for all genes in the full graph
            x_gene_exp = self.gene_embedding.weight.expand(batch_size, -1, -1)
            x_gene = x_gene_exp.reshape(-1, x_gene_exp.size(-1))

            # Apply preprocessing to ALL nodes
            # Masking happens during message passing and aggregation, not here!
            x_gene = self.preprocessor(x_gene)
        else:
            # Single graph case - x is still full graph
            gene_data = data["gene"]
            gene_idx = torch.arange(gene_data.num_nodes, device=device)
            x_gene = self.preprocessor(self.gene_embedding(gene_idx))

        x_dict = {"gene": x_gene}

        # Extract edge indices AND edge masks
        edge_index_dict = {}
        edge_mask_dict = {}  # NEW!

        for graph_name in self.graph_names:
            edge_type = ("gene", graph_name, "gene")

            # Check if edge type exists in data
            if edge_type not in data.edge_types:
                continue

            edge_data = data[edge_type]

            # Try to get edge_index - handle both lazy and non-lazy formats
            if hasattr(edge_data, "edge_index"):
                edge_index = edge_data.edge_index.to(device)
            elif "edge_index" in edge_data:
                edge_index = edge_data["edge_index"].to(device)
            else:
                # Skip this edge type if no edge_index found
                continue

            edge_index_dict[edge_type] = edge_index

            # Extract edge mask if present (for lazy data)
            if hasattr(edge_data, "mask"):
                edge_mask_dict[edge_type] = edge_data.mask.to(device)  # NEW!
            elif "mask" in edge_data:
                edge_mask_dict[edge_type] = edge_data["mask"].to(device)
            else:
                # Fallback: assume all edges are valid (for non-lazy data like cell_graph)
                num_edges = edge_index.size(1)
                edge_mask_dict[edge_type] = torch.ones(
                    num_edges, dtype=torch.bool, device=device
                )

        # Apply convolution layers with edge masking
        layer_attention_weights = []
        for conv in self.convs:
            x_dict, attn_weights = conv(
                x_dict, edge_index_dict, edge_mask_dict
            )  # NEW: pass edge masks
            if attn_weights is not None:
                layer_attention_weights.append(attn_weights)

        # Store weights from all layers for visualization/analysis
        self.layer_aggregation_weights = layer_attention_weights

        # Return full graph embeddings [total_nodes, hidden_dim]
        return cast(torch.Tensor, x_dict["gene"])

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the forward pass with mask application before aggregation (lazy version).

        Key differences from original:
        - z_w and z_i are full graph embeddings [total_nodes, hidden_dim]
        - Apply pert_mask before global aggregation (only aggregate kept genes)
        - perturbation_indices work unchanged (index into full graph)
        """
        # Process reference graph (wildtype) - returns FULL graph embeddings
        z_w = self.forward_single(cell_graph)  # [total_nodes, hidden_dim]

        # Check for NaNs after processing wildtype
        if torch.isnan(z_w).any():
            raise RuntimeError("NaN detected in wildtype embeddings (z_w)")

        # LAZY: Apply mask before global aggregation (if pert_mask exists)
        # For cell_graph (wildtype reference), there's no pert_mask - aggregate ALL genes
        # For lazy batched data with perturbations, filter to kept genes only
        if hasattr(cell_graph["gene"], "pert_mask"):
            gene_mask = ~cell_graph["gene"].pert_mask  # True for kept genes
            z_w_kept = z_w[gene_mask]  # Filter to kept genes
        else:
            # No pert_mask - use all genes (typical for wildtype cell_graph)
            z_w_kept = z_w

        # Global aggregation on kept genes
        z_w_global = self.global_aggregator(
            z_w_kept,
            index=torch.zeros(z_w_kept.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )

        # Check for NaNs in global wildtype embeddings
        if torch.isnan(z_w_global).any():
            raise RuntimeError(
                "NaN detected in global wildtype embeddings (z_w_global)"
            )

        # Process perturbed batch - returns FULL graph embeddings for batch
        z_i = self.forward_single(batch)  # [total_nodes_batch, hidden_dim]

        # Check for NaNs in perturbed embeddings
        if torch.isnan(z_i).any():
            raise RuntimeError("NaN detected in perturbed embeddings (z_i)")

        # LAZY: Apply mask before global aggregation for batch (if pert_mask exists)
        if hasattr(batch["gene"], "pert_mask"):
            batch_gene_mask = ~batch["gene"].pert_mask  # True for kept genes
            z_i_kept = z_i[batch_gene_mask]  # Filter to kept genes
            batch_idx = batch["gene"].batch[
                batch_gene_mask
            ]  # Batch vector for kept genes only
        else:
            # No pert_mask - use all genes
            z_i_kept = z_i
            batch_idx = batch["gene"].batch

        # Global aggregation on kept genes, respecting batch structure
        z_i_global = self.global_aggregator(z_i_kept, index=batch_idx)

        # Check for NaNs in global perturbed embeddings
        if torch.isnan(z_i_global).any():
            raise RuntimeError(
                "NaN detected in global perturbed embeddings (z_i_global)"
            )

        # Get embeddings of perturbed genes from wildtype
        pert_indices = batch["gene"].perturbation_indices
        pert_gene_embs = z_w[pert_indices]

        # Check for NaNs in perturbed gene embeddings
        if torch.isnan(pert_gene_embs).any():
            raise RuntimeError(
                "NaN detected in perturbed gene embeddings (pert_gene_embs)"
            )

        # Calculate perturbation difference for z_p_global
        batch_size = z_i_global.size(0)
        z_w_exp = z_w_global.expand(batch_size, -1)
        z_p_global = z_w_exp - z_i_global

        # Check for NaNs in perturbation difference
        if torch.isnan(z_p_global).any():
            raise RuntimeError("NaN detected in perturbation difference (z_p_global)")

        # Determine batch assignment for perturbed genes
        batch_assign: torch.Tensor | None
        if hasattr(batch["gene"], "perturbation_indices_ptr"):
            # VECTORIZED: Create batch assignment using repeat_interleave (no loop!)
            ptr = batch["gene"].perturbation_indices_ptr
            counts = ptr[1:] - ptr[:-1]  # Genes per batch
            batch_assign = torch.repeat_interleave(
                torch.arange(len(counts), device=z_w.device), counts
            )
        else:
            # Alternative if perturbation_indices_ptr is not available
            batch_assign = (
                batch["gene"].perturbation_indices_batch
                if hasattr(batch["gene"], "perturbation_indices_batch")
                else None
            )

        # Get gene interaction predictions using the local predictor (if enabled)
        if self.use_local_predictor:
            assert self.gene_interaction_predictor is not None
            local_interaction = self.gene_interaction_predictor(
                pert_gene_embs, batch_assign
            )

            # Check for NaNs in local interaction predictions
            if torch.isnan(local_interaction).any():
                raise RuntimeError("NaN detected in local interaction predictions")
        else:
            local_interaction = None

        # Get gene interaction predictions using the global predictor
        global_interaction = self.global_interaction_predictor(z_p_global)

        # Check for NaNs in global interaction predictions
        if torch.isnan(global_interaction).any():
            raise RuntimeError("NaN detected in global interaction predictions")

        # Combine predictions based on configuration
        if not self.use_local_predictor:
            # Global only mode - weight is 1.0
            gene_interaction = global_interaction
            gate_weights = torch.ones(batch_size, 1, device=global_interaction.device)

        else:
            # Ensure dimensions match for gating
            if local_interaction.size(0) != batch_size:
                # VECTORIZED: Use scatter instead of loop (no .item()!)
                local_interaction_expanded = torch.zeros(
                    batch_size, 1, device=z_w.device
                )
                if batch_assign is not None:
                    # Filter valid indices and scatter in one operation
                    valid_mask = batch_assign < batch_size
                    local_interaction_expanded.scatter_(
                        0,
                        batch_assign[valid_mask].unsqueeze(1),
                        local_interaction[valid_mask],
                    )
                local_interaction = local_interaction_expanded

                # Check for NaNs after dimension adjustment
                if torch.isnan(local_interaction).any():
                    raise RuntimeError(
                        "NaN detected after dimension adjustment of local interaction"
                    )

            # Ensure both tensors have the same number of dimensions before concatenation
            if global_interaction.dim() == 1:
                global_interaction = global_interaction.unsqueeze(1)
            if local_interaction.dim() == 1:
                local_interaction = local_interaction.unsqueeze(1)

            # Combine local and global predictions
            if self.combination_method == "gating":
                assert self.gate_mlp is not None
                # Stack the predictions
                pred_stack = torch.cat([global_interaction, local_interaction], dim=1)

                # Check for NaNs in prediction stack
                if torch.isnan(pred_stack).any():
                    raise RuntimeError("NaN detected in prediction stack")

                # Use MLP to get logits for gating, then apply softmax
                gate_logits = self.gate_mlp(pred_stack)

                # Check for NaNs in gate logits
                if torch.isnan(gate_logits).any():
                    raise RuntimeError("NaN detected in gate logits")

                gate_weights = F.softmax(gate_logits, dim=1)

                # Check for NaNs in gate weights
                if torch.isnan(gate_weights).any():
                    raise RuntimeError("NaN detected in gate weights after softmax")

                # Element-wise product of predictions and weights, then sum
                weighted_preds = pred_stack * gate_weights

                # Check for NaNs in weighted predictions
                if torch.isnan(weighted_preds).any():
                    raise RuntimeError("NaN detected in weighted predictions")

                gene_interaction = weighted_preds.sum(dim=1, keepdim=True)

            elif self.combination_method == "concat":
                # Fixed equal weighting (0.5 each)
                gene_interaction = 0.5 * global_interaction + 0.5 * local_interaction

                # Create fixed gate weights for consistency in logging
                batch_size = global_interaction.size(0)
                gate_weights = (
                    torch.ones(batch_size, 2, device=global_interaction.device) * 0.5
                )

                # Check for NaNs
                if torch.isnan(gene_interaction).any():
                    raise RuntimeError("NaN detected in concatenated gene interaction")

            else:
                raise ValueError(
                    f"Unknown combination method: {self.combination_method}"
                )

        # Final check for NaNs in gene interaction output
        if torch.isnan(gene_interaction).any():
            raise RuntimeError("NaN detected in final gene interaction output")

        # Return both predictions and representations dictionary
        return_dict = {
            "z_w": z_w_global,
            "z_i": z_i_global,
            "z_p": z_p_global,
            "global_interaction": global_interaction,
            "gene_interaction": gene_interaction,
            "gate_weights": gate_weights,
            "pert_gene_embs": pert_gene_embs,
            "layer_aggregation_weights": self.layer_aggregation_weights,  # Weights from all layers
        }
        # Only include local_interaction if predictor was used
        if self.use_local_predictor:
            return_dict["local_interaction"] = local_interaction

        return gene_interaction, return_dict

    @property
    def num_parameters(self) -> dict[str, int]:
        """Return a breakdown of trainable parameter counts per submodule."""

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "preprocessor": count_params(self.preprocessor),
            "convs": count_params(self.convs),
            "global_aggregator": count_params(self.global_aggregator),
            "global_interaction_predictor": count_params(
                self.global_interaction_predictor
            ),
        }

        # Only count if modules exist
        if self.use_local_predictor and self.gene_interaction_predictor is not None:
            counts["gene_interaction_predictor"] = count_params(
                self.gene_interaction_predictor
            )
        if self.gate_mlp is not None:
            counts["gate_mlp"] = count_params(self.gate_mlp)

        counts["total"] = sum(counts.values())
        return counts


def calculate_weight_l2_norm(model: nn.Module) -> float:
    """Calculate L2 norm of all model weights."""
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
    return float(total_norm**0.5)


def calculate_rolling_correlation(
    x: Any, y: Any, window: int = 50
) -> list[float]:  # x/y are array-like series (lists or np.ndarray)
    """Calculate rolling correlation between two series."""
    if len(x) < window:
        return []

    correlations = []
    for i in range(window, len(x) + 1):
        x_window = x[i - window : i]
        y_window = y[i - window : i]
        if np.std(x_window) > 0 and np.std(y_window) > 0:
            corr = np.corrcoef(x_window, y_window)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)
    return correlations


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/hetero_cell_bipartite_dango_gi_lazy_demo.py on 2026-10-06"
    )
