# torchcell/models/hetero_cell_bipartite_dango_gi
# [[torchcell.models.hetero_cell_bipartite_dango_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_bipartite_dango_gi
# Test file: tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py


import math
import os
import os.path as osp
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import BatchNorm, GATv2Conv, GINConv, HeteroConv, LayerNorm
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_scatter import scatter_mean

from torchcell.graph.graph import GeneMultiGraph
from torchcell.models.act import act_register
from typing import List
from torch_geometric.typing import EdgeType

# Additional imports for enhanced plotting
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS
from sklearn.decomposition import PCA


class SelfAttentionGraphAggregation(nn.Module):
    """Self-attention mechanism for aggregating multiple graph representations of the same nodes"""
    def __init__(self, hidden_dim: int, num_graphs: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable graph positional encodings
        self.graph_embeddings = nn.Parameter(torch.randn(num_graphs, hidden_dim) * 0.02)
        
    def forward(self, graph_outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
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
        graph_emb_expanded = self.graph_embeddings[:num_actual_graphs].unsqueeze(0).expand(num_nodes, -1, -1)
        stacked = stacked + graph_emb_expanded
        
        # Apply self-attention
        attended, attn_weights = self.multihead_attn(
            query=stacked,
            key=stacked, 
            value=stacked,
            need_weights=True,
            average_attn_weights=True  # Average over heads for visualization
        )
        
        # Mean pooling across graphs dimension
        aggregated = attended.mean(dim=1)  # [num_nodes, hidden_dim]
        
        return aggregated, attn_weights


class PairwiseGraphAggregation(nn.Module):
    """Pairwise interaction mechanism for aggregating multiple graph representations"""
    def __init__(self, hidden_dim: int, graph_names: List[str], dropout: float = 0.0):
        super().__init__()
        self.graph_names = sorted(graph_names)
        self.hidden_dim = hidden_dim
        
        # Pairwise interaction networks
        self.interaction_mlps = nn.ModuleDict()
        for i, g1 in enumerate(self.graph_names):
            for j, g2 in enumerate(self.graph_names[i:], i):
                key = f"{g1}_{g2}"
                self.interaction_mlps[key] = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
        
        # Attention mechanism for aggregating interactions
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, graph_outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise graph interactions and aggregate them.
        Returns:
            aggregated: Aggregated features [num_nodes, hidden_dim]
            attention_weights: Attention weights for interactions [num_nodes, num_interactions]
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
        
        # Stack interactions: [num_nodes, num_interactions, hidden_dim]
        stacked = torch.stack(interactions, dim=1)
        
        # Compute attention weights
        attn_logits = self.attention(stacked).squeeze(-1)  # [num_nodes, num_interactions]
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Weighted aggregation
        aggregated = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return aggregated, attn_weights


class HeteroConvAggregator(nn.Module):
    """
    HeteroConv wrapper with configurable aggregation strategies.
    Supports: sum, mean, cross_attention, pairwise_interaction
    """
    def __init__(
        self,
        convs: Dict[EdgeType, nn.Module],
        hidden_channels: int,
        aggregation_method: str = "cross_attention",
        aggregation_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.convs = nn.ModuleDict({str(k): v for k, v in convs.items()})
        self.hidden_channels = hidden_channels
        self.aggregation_method = aggregation_method
        
        # Extract unique graph names from edge types
        self.graph_names = sorted(list(set([edge_type[1] for edge_type in convs.keys()])))
        
        # Initialize aggregation module based on method
        config = aggregation_config or {}
        
        if aggregation_method == "cross_attention":
            self.aggregator = SelfAttentionGraphAggregation(
                hidden_dim=hidden_channels,
                num_graphs=len(self.graph_names),
                num_heads=config.get("num_heads", 4),
                dropout=config.get("dropout", 0.0)
            )
        elif aggregation_method == "pairwise_interaction":
            self.aggregator = PairwiseGraphAggregation(
                hidden_dim=hidden_channels,
                graph_names=self.graph_names,
                dropout=config.get("dropout", 0.0)
            )
        elif aggregation_method in ["sum", "mean"]:
            self.aggregator = None  # Will use simple aggregation
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply graph convolutions and aggregate using specified method.
        
        Returns:
            out_dict: Updated node features
            attention_weights: Graph aggregation weights (if applicable)
        """
        # Store outputs by destination node type and graph name
        out_dict = {}
        graph_outputs_by_dst = {}  # {dst_type: {graph_name: tensor}}
        
        # Apply convolutions for each edge type
        for edge_type_str, conv in self.convs.items():
            # Parse the edge type string back to tuple
            edge_type = eval(edge_type_str) if isinstance(edge_type_str, str) else edge_type_str
            src, rel, dst = edge_type
            
            # Get the edge index for this edge type
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
            else:
                continue
            
            # Apply convolution
            out = conv(x_dict[src], edge_index)
            
            # Organize outputs by destination and graph
            if dst not in graph_outputs_by_dst:
                graph_outputs_by_dst[dst] = {}
            graph_outputs_by_dst[dst][rel] = out
        
        # Aggregate for each destination node type
        all_attention_weights = {}
        
        for dst, graph_outputs in graph_outputs_by_dst.items():
            if not graph_outputs:
                continue
                
            if self.aggregation_method == "sum":
                # Simple sum aggregation
                out_dict[dst] = sum(graph_outputs.values())
                all_attention_weights[dst] = None
                
            elif self.aggregation_method == "mean":
                # Simple mean aggregation
                stacked = torch.stack(list(graph_outputs.values()))
                out_dict[dst] = stacked.mean(dim=0)
                all_attention_weights[dst] = None
                
            elif self.aggregator is not None:
                # Use learned aggregation (cross_attention or pairwise_interaction)
                out_dict[dst], attn_weights = self.aggregator(graph_outputs)
                all_attention_weights[dst] = attn_weights
            else:
                # Fallback to sum
                out_dict[dst] = sum(graph_outputs.values())
                all_attention_weights[dst] = None
        
        # Return aggregated features and attention weights
        return out_dict, all_attention_weights if any(v is not None for v in all_attention_weights.values()) else None


class AttentionalGraphAggregation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )
        self.transform_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Dropout(dropout)
        )
        self.aggregator = AttentionalAggregation(
            gate_nn=self.gate_nn, nn=self.transform_nn
        )

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None
    ) -> torch.Tensor:
        return self.aggregator(x, index=index, dim_size=dim_size)


class DangoLikeHyperSAGNN(nn.Module):
    """
    Dango-like HyperSAGNN for local gene interaction prediction.
    Implements multi-layer self-attention with multi-head attention and ReZero connections.
    """

    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads

        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        # Static embedding layer (like Dango)
        self.static_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
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

    def forward(self, gene_embeddings, batch=None):
        """
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
            if batch is not None:
                # Process each batch separately
                unique_batches = batch.unique()
                output_embeddings = torch.zeros_like(dynamic_embeddings)

                for b in unique_batches:
                    mask = batch == b
                    batch_embeddings = dynamic_embeddings[mask]
                    batch_output = self._apply_attention_layer(
                        batch_embeddings, layer, self.beta_params[i]
                    )
                    output_embeddings[mask] = batch_output

                dynamic_embeddings = output_embeddings
            else:
                # Single batch processing
                dynamic_embeddings = self._apply_attention_layer(
                    dynamic_embeddings, layer, self.beta_params[i]
                )

        return static_embeddings, dynamic_embeddings

    def _apply_attention_layer(self, x, layer, beta):
        """Apply a single attention layer with multi-head attention"""
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
        return x + beta * out


class GeneInteractionPredictor(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        # Use the new Dango-like HyperSAGNN
        self.hyper_sagnn = DangoLikeHyperSAGNN(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Prediction layer to compute scores from squared differences
        self.prediction_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.prediction_layer.weight)

    def forward(self, gene_embeddings, batch=None):
        """
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
            num_batches = batch.max().item() + 1
            interaction_scores = scatter_mean(
                gene_scores, batch, dim=0, dim_size=num_batches
            )
            return interaction_scores.unsqueeze(-1)  # [num_batches, 1]
        else:
            # Single batch case
            return gene_scores.mean().unsqueeze(0).unsqueeze(-1)  # [1, 1]


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(
            channels, eps=1e-5
        )  # Increased epsilon for better stability
    elif norm == "batch":
        return nn.BatchNorm1d(channels, eps=1e-5)  # Also increase batch norm epsilon
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


class PreProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers = []
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
        return self.mlp(x)


class AttentionConvWrapper(nn.Module):
    def __init__(
        self,
        conv: nn.Module,
        target_dim: int,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = conv
        
        # Determine expected output dimension based on conv type
        if isinstance(conv, GINConv):
            # For GINConv, get output dim from the last layer of the MLP
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
                conv.heads * conv.out_channels if conv.concat else conv.out_channels
            )
        else:
            # For other conv types that have out_channels
            expected_dim = conv.out_channels
            
        self.proj = (
            nn.Identity()
            if expected_dim == target_dim
            else nn.Linear(expected_dim, target_dim)
        )

        if norm is not None:
            if norm == "batch":
                self.norm = BatchNorm(target_dim)
            elif norm == "layer":
                self.norm = LayerNorm(target_dim)
            else:
                self.norm = None
        else:
            self.norm = None

        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, edge_index, **kwargs):
        out = self.conv(x, edge_index, **kwargs)
        out = self.proj(out)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


def create_conv_layer(
    encoder_type: str,
    in_channels: int,
    out_channels: int,
    config: Dict[str, Any],
    edge_dim: Optional[int] = None,
    dropout: float = 0.1
) -> nn.Module:
    """Create appropriate conv layer based on encoder type.
    
    Args:
        encoder_type: Type of encoder - "gatv2" or "gin"
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        config: Configuration dict for the encoder
        edge_dim: Edge feature dimension (for GATv2)
        dropout: Dropout rate for GIN MLP
        
    Returns:
        Conv layer (GATv2Conv or GINConv)
    """
    if encoder_type == "gatv2":
        return GATv2Conv(
            in_channels,
            out_channels // config.get("heads", 1),
            heads=config.get("heads", 1),
            concat=config.get("concat", True),
            add_self_loops=config.get("add_self_loops", False),
            edge_dim=edge_dim
        )
    elif encoder_type == "gin":
        # GIN uses MLP for transformation
        gin_hidden = config.get("gin_hidden_dim") or out_channels
        gin_layers = config.get("gin_num_layers", 2)
        
        # Build MLP
        mlp_layers = []
        for i in range(gin_layers):
            if i == 0:
                mlp_layers.extend([
                    nn.Linear(in_channels, gin_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            elif i == gin_layers - 1:
                mlp_layers.append(nn.Linear(gin_hidden, out_channels))
            else:
                mlp_layers.extend([
                    nn.Linear(gin_hidden, gin_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
        
        mlp = nn.Sequential(*mlp_layers)
        return GINConv(mlp, train_eps=True)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class GeneInteractionDango(nn.Module):
    def __init__(
        self,
        gene_num: int,
        hidden_channels: int,
        num_layers: int,
        gene_multigraph: GeneMultiGraph,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        gene_encoder_config: Optional[Dict[str, Any]] = None,
        local_predictor_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gene_multigraph = gene_multigraph

        # Extract graph names from the multigraph
        self.graph_names = list(gene_multigraph.keys())

        # Get combination method from config
        local_predictor_config = local_predictor_config or {}
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
            "graph_aggregation_method", "cross_attention"  # Default to cross_attention
        )
        self.graph_aggregation_config = gene_encoder_config.get(
            "graph_aggregation_config", {}
        )
        
        # Store attention weights for visualization
        self.last_layer_attention_weights = []

        # Graph convolution layers with interaction-based aggregation
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict = {}
            
            # Create a conv layer for each graph in the multigraph
            for graph_name in self.graph_names:
                edge_type = ("gene", graph_name, "gene")
                encoder_type = gene_encoder_config.get("encoder_type", "gatv2")
                
                conv_layer = create_conv_layer(
                    encoder_type=encoder_type,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    config=gene_encoder_config,
                    dropout=dropout
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
                        "dropout": dropout  # Pass model dropout to aggregation
                    }
                )
            )

        # Get local predictor config - now as a separate parameter
        local_predictor_config = local_predictor_config or {}

        # Gene interaction predictor for perturbed genes with Dango-like architecture
        self.gene_interaction_predictor = GeneInteractionPredictor(
            hidden_dim=hidden_channels,
            num_heads=local_predictor_config.get("num_heads", 4),
            num_layers=local_predictor_config.get("num_attention_layers", 2),
            dropout=dropout,
        )

        # Global aggregator for proper aggregation
        self.global_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

        # Global predictor for z_p_global
        self.global_interaction_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # MLP for gating weights (only if using gating combination method)
        if self.combination_method == "gating":
            self.gate_mlp = nn.Sequential(
                nn.Linear(2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 2),
            )
        else:
            self.gate_mlp = None

        # Initialize all weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights in the model with appropriate initializations"""

        def _init_module(module):
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
        device = self.gene_embedding.weight.device

        # Handle both batch and single graph input
        is_batch = isinstance(data, Batch) or hasattr(data["gene"], "batch")
        if is_batch:
            gene_data = data["gene"]
            batch_size = len(data["gene"].ptr) - 1

            # Handle perturbation masks if present
            if hasattr(gene_data, "pert_mask"):
                x_gene_exp = self.gene_embedding.weight.expand(batch_size, -1, -1)
                x_gene_comb = x_gene_exp.reshape(-1, x_gene_exp.size(-1))
                x_gene = x_gene_comb[~gene_data.pert_mask]
            else:
                # Default handling without perturbation mask
                gene_idx = torch.arange(gene_data.num_nodes, device=device)
                x_gene = self.gene_embedding(gene_idx)

            x_gene = self.preprocessor(x_gene)
        else:
            gene_data = data["gene"]
            gene_idx = torch.arange(gene_data.num_nodes, device=device)
            x_gene = self.preprocessor(self.gene_embedding(gene_idx))

        x_dict = {"gene": x_gene}

        # Process edge indices dynamically based on graph names
        edge_index_dict = {}

        for graph_name in self.graph_names:
            edge_type = ("gene", graph_name, "gene")
            edge_index = data[edge_type].edge_index.to(device)
            edge_index_dict[edge_type] = edge_index

        # Apply convolution layers with interaction-based aggregation
        layer_attention_weights = []
        for conv in self.convs:
            x_dict, attn_weights = conv(x_dict, edge_index_dict)
            if attn_weights is not None:
                layer_attention_weights.append(attn_weights)
        
        # Store for visualization/analysis
        self.last_layer_attention_weights = layer_attention_weights
        
        return x_dict["gene"]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Process reference graph (wildtype)
        z_w = self.forward_single(cell_graph)

        # Check for NaNs after processing wildtype
        if torch.isnan(z_w).any():
            raise RuntimeError("NaN detected in wildtype embeddings (z_w)")

        # Proper global aggregation for wildtype
        z_w_global = self.global_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )

        # Check for NaNs in global wildtype embeddings
        if torch.isnan(z_w_global).any():
            raise RuntimeError(
                "NaN detected in global wildtype embeddings (z_w_global)"
            )

        # Process perturbed batch if needed
        z_i = self.forward_single(batch)

        # Check for NaNs in perturbed embeddings
        if torch.isnan(z_i).any():
            raise RuntimeError("NaN detected in perturbed embeddings (z_i)")

        # Proper global aggregation for perturbed genes
        z_i_global = self.global_aggregator(z_i, index=batch["gene"].batch)

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
        if hasattr(batch["gene"], "perturbation_indices_ptr"):
            # Create batch assignment using perturbation_indices_ptr
            ptr = batch["gene"].perturbation_indices_ptr
            batch_assign = torch.zeros(
                pert_indices.size(0), dtype=torch.long, device=z_w.device
            )
            for i in range(len(ptr) - 1):
                batch_assign[ptr[i] : ptr[i + 1]] = i
        else:
            # Alternative if perturbation_indices_ptr is not available
            batch_assign = (
                batch["gene"].perturbation_indices_batch
                if hasattr(batch["gene"], "perturbation_indices_batch")
                else None
            )

        # Get gene interaction predictions using the local predictor
        local_interaction = self.gene_interaction_predictor(
            pert_gene_embs, batch_assign
        )

        # Check for NaNs in local interaction predictions
        if torch.isnan(local_interaction).any():
            raise RuntimeError("NaN detected in local interaction predictions")

        # Get gene interaction predictions using the global predictor
        global_interaction = self.global_interaction_predictor(z_p_global)

        # Check for NaNs in global interaction predictions
        if torch.isnan(global_interaction).any():
            raise RuntimeError("NaN detected in global interaction predictions")

        # Ensure dimensions match for gating
        if local_interaction.size(0) != batch_size:
            local_interaction_expanded = torch.zeros(batch_size, 1, device=z_w.device)
            for i in range(local_interaction.size(0)):
                batch_idx = batch_assign[i].item() if batch_assign is not None else 0
                if batch_idx < batch_size:
                    local_interaction_expanded[batch_idx] = local_interaction[i]
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

        # Combine predictions based on combination method
        if self.combination_method == "gating":
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
            raise ValueError(f"Unknown combination method: {self.combination_method}")

        # Final check for NaNs in gene interaction output
        if torch.isnan(gene_interaction).any():
            raise RuntimeError("NaN detected in final gene interaction output")

        # Return both predictions and representations dictionary
        return gene_interaction, {
            "z_w": z_w_global,
            "z_i": z_i_global,
            "z_p": z_p_global,
            "local_interaction": local_interaction,
            "global_interaction": global_interaction,
            "gate_weights": gate_weights,
            "gene_interaction": gene_interaction,
            "pert_gene_embs": pert_gene_embs,
            "graph_attention_weights": self.last_layer_attention_weights,  # Add graph aggregation weights
        }

    @property
    def num_parameters(self) -> Dict[str, int]:
        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "preprocessor": count_params(self.preprocessor),
            "convs": count_params(self.convs),
            "gene_interaction_predictor": count_params(self.gene_interaction_predictor),
            "global_aggregator": count_params(self.global_aggregator),
            "global_interaction_predictor": count_params(
                self.global_interaction_predictor
            ),
        }

        # Only count gate_mlp if it exists
        if self.gate_mlp is not None:
            counts["gate_mlp"] = count_params(self.gate_mlp)

        counts["total"] = sum(counts.values())
        return counts


def calculate_weight_l2_norm(model):
    """Calculate L2 norm of all model weights."""
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def calculate_rolling_correlation(x, y, window=50):
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


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="hetero_cell_bipartite_dango_gi",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    import torch.nn as nn
    from torchcell.timestamp import timestamp
    import numpy as np
    from scipy import stats
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    from torchcell.graph.graph import build_gene_multigraph, SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from sortedcontainers import SortedDict
    from torchcell.graph.graph import GeneMultiGraph, GeneGraph
    from torchcell.losses.logcosh import LogCoshLoss
    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.losses.mle_dist_supcr import MleDistSupCR

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        config="hetero_cell_bipartite",
        is_dense=False,
    )
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Build gene multigraph using the proper function
    # Initialize genome and graph for building the multigraph
    genome = SCerevisiaeGenome()
    sc_graph = SCerevisiaeGraph(genome=genome)

    # Build the multigraph with graph names from config
    gene_multigraph = build_gene_multigraph(
        graph=sc_graph, graph_names=cfg.cell_dataset.graphs
    )

    # Initialize the gene interaction model
    # Ensure configs are properly converted
    gene_encoder_config_dict = (
        OmegaConf.to_container(cfg.model.gene_encoder_config, resolve=True)
        if cfg.model.gene_encoder_config
        else {}
    )
    local_predictor_config_dict = (
        OmegaConf.to_container(cfg.model.local_predictor_config, resolve=True)
        if hasattr(cfg.model, "local_predictor_config")
        and cfg.model.local_predictor_config
        else {}
    )

    model = GeneInteractionDango(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        gene_multigraph=gene_multigraph,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=gene_encoder_config_dict,
        local_predictor_config=local_predictor_config_dict,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

    # Configure loss function based on config
    loss_type = cfg.regression_task.get("loss", "logcosh")
    if loss_type == "logcosh":
        criterion = LogCoshLoss(reduction="mean")
        print("Using LogCosh loss")
    elif loss_type == "icloss":
        # For ICLoss, we need phenotype weights if weighted loss is enabled
        if cfg.regression_task.get("is_weighted_phenotype_loss", False):
            # For gene interaction only dataset, we have just one phenotype
            weights = torch.ones(1).to(device)
        else:
            weights = None

        criterion = ICLoss(
            lambda_dist=cfg.regression_task.get("lambda_dist", 0.1),
            lambda_supcr=cfg.regression_task.get("lambda_supcr", 0.001),
            weights=weights,
        )
        print(
            f"Using ICLoss with lambda_dist={cfg.regression_task.lambda_dist}, lambda_supcr={cfg.regression_task.lambda_supcr}"
        )
    elif loss_type == "mle_dist_supcr":
        # For MleDistSupCR, we need phenotype weights if weighted loss is enabled
        if cfg.regression_task.get("is_weighted_phenotype_loss", False):
            # For gene interaction only dataset, we have just one phenotype
            weights = torch.ones(1).to(device)
        else:
            weights = None
        
        # Get loss configuration
        loss_config = cfg.regression_task.get("loss_config", {})
        
        criterion = MleDistSupCR(
            # Lambda weights
            lambda_mse=cfg.regression_task.get("lambda_mse", 1.0),
            lambda_dist=cfg.regression_task.get("lambda_dist", 0.1),
            lambda_supcr=cfg.regression_task.get("lambda_supcr", 0.001),
            
            # Component-specific parameters
            dist_bandwidth=loss_config.get("dist_bandwidth", 2.0),
            supcr_temperature=loss_config.get("supcr_temperature", 0.1),
            embedding_dim=cfg.model.hidden_channels,  # Use model's hidden_channels
            
            # Buffer configuration
            use_buffer=loss_config.get("use_buffer", True),
            buffer_size=loss_config.get("buffer_size", 256),
            min_samples_for_dist=loss_config.get("min_samples_for_dist", 64),
            min_samples_for_supcr=loss_config.get("min_samples_for_supcr", 64),
            
            # DDP configuration
            use_ddp_gather=loss_config.get("use_ddp_gather", True),
            gather_interval=loss_config.get("gather_interval", 1),
            
            # Adaptive weighting - let it default to dynamic based on max_epochs
            use_adaptive_weighting=loss_config.get("use_adaptive_weighting", True),
            
            # Temperature scheduling
            use_temp_scheduling=loss_config.get("use_temp_scheduling", True),
            init_temperature=loss_config.get("init_temperature", 1.0),
            final_temperature=loss_config.get("final_temperature", 0.1),
            temp_schedule=loss_config.get("temp_schedule", "exponential"),
            
            # Other parameters
            weights=weights,
            max_epochs=cfg.trainer.max_epochs,
        )
        print(
            f"Using MleDistSupCR with lambda_mse={cfg.regression_task.get('lambda_mse', 1.0)}, "
            f"lambda_dist={cfg.regression_task.get('lambda_dist', 0.1)}, "
            f"lambda_supcr={cfg.regression_task.get('lambda_supcr', 0.001)}"
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    # Learning rate scheduler
    lr_scheduler = None
    if (
        hasattr(cfg.regression_task, "lr_scheduler")
        and cfg.regression_task.lr_scheduler
    ):
        from torchcell.scheduler.cosine_annealing_warmup import (
            CosineAnnealingWarmupRestarts,
        )

        scheduler_config = cfg.regression_task.lr_scheduler
        if scheduler_config.type == "CosineAnnealingWarmupRestarts":
            lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=scheduler_config.first_cycle_steps,
                cycle_mult=scheduler_config.get("cycle_mult", 1.0),
                max_lr=scheduler_config.max_lr,
                min_lr=scheduler_config.min_lr,
                warmup_steps=scheduler_config.warmup_steps,
                gamma=scheduler_config.get("gamma", 1.0),
            )
            print(f"Using CosineAnnealingWarmupRestarts scheduler with:")
            print(f"  - first_cycle_steps: {scheduler_config.first_cycle_steps}")
            print(f"  - cycle_mult: {scheduler_config.get('cycle_mult', 1.0)}")
            print(f"  - max_lr: {scheduler_config.max_lr}")
            print(f"  - min_lr: {scheduler_config.min_lr}")
            print(f"  - warmup_steps: {scheduler_config.warmup_steps}")
            print(f"  - gamma: {scheduler_config.get('gamma', 1.0)}")

    # Training target - gene interaction in COO format
    # The phenotype_values contains the gene interaction scores
    y = batch["gene"].phenotype_values

    # Setup directory for plots
    plot_dir = osp.join(
        ASSET_IMAGES_DIR, f"hetero_cell_bipartite_dango_gi_training_{timestamp()}"
    )
    os.makedirs(plot_dir, exist_ok=True)

    def save_intermediate_plot(
        epoch,
        losses,
        correlations,
        mses,
        maes,
        rmses,
        learning_rates,
        gate_weights_history,
        model,
        cell_graph,
        batch,
        y,
        weight_l2_norms,
        cfg,
        loss_type="logcosh",
        loss_components_history=None,
        spearman_correlations=None,
        embedding_norms_history=None,
        lambda_values_history=None,
    ):
        """Save intermediate training plot every print interval."""
        plt.figure(figsize=(25, 20))

        # Determine loss label
        if loss_type == "logcosh":
            loss_label = "LogCosh Loss"
        elif loss_type == "icloss":
            loss_label = "ICLoss"
        else:
            loss_label = "MleDistSupCR"

        # ROW 1: Loss curve, Correlations, and Scatter plot
        # Loss curve
        plt.subplot(5, 3, 1)
        plt.plot(range(1, epoch + 2), losses, "b-", label=loss_label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Correlation evolution
        plt.subplot(5, 3, 2)
        plt.plot(range(1, epoch + 2), correlations, "g-", label="Pearson", linewidth=2)
        if spearman_correlations:
            plt.plot(
                range(1, epoch + 2),
                spearman_correlations,
                "b--",
                label="Spearman",
                linewidth=2,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.title("Correlation Evolution")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)

        # Get current predictions for scatter plot
        model.eval()
        with torch.no_grad():
            current_predictions, current_representations = model(cell_graph, batch)
            true_np = y.cpu().numpy()
            pred_np = current_predictions.squeeze().cpu().numpy()

            # Current correlation
            valid_mask = ~np.isnan(true_np)
            if np.sum(valid_mask) > 0:
                current_corr = np.corrcoef(pred_np[valid_mask], true_np[valid_mask])[
                    0, 1
                ]
            else:
                current_corr = 0.0
        model.train()

        # Correlation scatter plot (swapped axes)
        plt.subplot(5, 3, 3)
        plt.scatter(pred_np[valid_mask], true_np[valid_mask], alpha=0.7)
        min_val = min(true_np[valid_mask].min(), pred_np[valid_mask].min())
        max_val = max(true_np[valid_mask].max(), pred_np[valid_mask].max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect correlation"
        )
        plt.xlabel("Predicted Gene Interaction")
        plt.ylabel("True Gene Interaction")
        plt.title(f"Predictions vs Truth (r={current_corr:.4f})")
        plt.grid(True)
        plt.legend()

        # ROW 2: Combined MSE/MAE/RMSE, Histograms with KDE, Learning Rate
        # Combined MSE, MAE, and RMSE plot
        plt.subplot(5, 3, 4)
        epochs_range = range(1, epoch + 2)

        # Create primary axis for MSE and RMSE
        ax1 = plt.gca()
        ax1.plot(epochs_range, mses, "r-", label="MSE", linewidth=2)
        ax1.plot(epochs_range, rmses, "b-", label="RMSE", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE / RMSE", color="black")
        ax1.set_yscale("log")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, alpha=0.3)

        # Create secondary axis for MAE
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, maes, "orange", label="MAE", linewidth=2)
        ax2.set_ylabel("MAE", color="orange")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title("Error Metrics Evolution")

        # True vs Predicted value distributions with KDE
        plt.subplot(5, 3, 5)
        bins = np.linspace(
            min(true_np[valid_mask].min(), pred_np[valid_mask].min()),
            max(true_np[valid_mask].max(), pred_np[valid_mask].max()),
            30,
        )

        # Plot histograms
        n_true, _, _ = plt.hist(
            true_np[valid_mask],
            bins=bins,
            alpha=0.5,
            label="True",
            color="blue",
            edgecolor="black",
            density=True,
        )
        n_pred, _, _ = plt.hist(
            pred_np[valid_mask],
            bins=bins,
            alpha=0.5,
            label="Predicted",
            color="red",
            edgecolor="black",
            density=True,
        )

        # Add KDE overlay
        kde_true = gaussian_kde(true_np[valid_mask])
        kde_pred = gaussian_kde(pred_np[valid_mask])
        x_range = np.linspace(true_np[valid_mask].min(), true_np[valid_mask].max(), 200)

        plt.plot(x_range, kde_true(x_range), "b-", linewidth=2, label="True KDE")
        plt.plot(x_range, kde_pred(x_range), "r-", linewidth=2, label="Predicted KDE")

        plt.xlabel("Gene Interaction Score")
        plt.ylabel("Density")
        plt.title("Value Distributions with KDE")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning rate evolution
        plt.subplot(5, 3, 6)
        plt.plot(range(1, epoch + 2), learning_rates, "purple", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.yscale("log")

        # ROW 3: Hyperparameters, Embedding norms, Gate weights
        # Model hyperparameters display
        plt.subplot(5, 3, 7)
        # Move title higher to avoid clash
        plt.title("Model Configuration", pad=20)

        # Get model parameters
        param_counts = model.num_parameters
        total_params = param_counts["total"]

        # Display parameters with better spacing
        y_pos = 0.92  # Start lower to avoid title
        params_text = [
            f"Total Parameters: {total_params:,}",
            f"Hidden Channels: {cfg.model.hidden_channels}",
            f"Embedding Dim: {cfg.cell_dataset.learnable_embedding_input_channels}",
            f"Combination Method: {cfg.model.local_predictor_config.combination_method}",
            f"Activation: {cfg.model.activation}",
            f"Norm: {cfg.model.norm}",
            f"Model Dropout: {cfg.model.dropout}",
            f"Head Dropout: {cfg.model.prediction_head_config.dropout if hasattr(cfg.model, 'prediction_head_config') else 'N/A'}",
            f"Weight Decay: {cfg.regression_task.optimizer.weight_decay}",
            f"Num Layers: {cfg.model.num_layers}",
            f"Num Attention Heads: {cfg.model.local_predictor_config.num_heads}",
            f"Num Attention Layers: {cfg.model.local_predictor_config.num_attention_layers}",
        ]

        for i, text in enumerate(params_text):
            plt.text(
                0.05,
                y_pos - i * 0.075,
                text,
                transform=plt.gca().transAxes,
                fontsize=9,
                ha="left",
                va="top",
            )

        # Remove axes
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # Embeddings norm evolution
        plt.subplot(5, 3, 8)
        if embedding_norms_history:
            epochs_range = range(1, len(embedding_norms_history) + 1)
            z_w_norms = [norm["z_w_norm"] for norm in embedding_norms_history]
            z_i_norms = [norm["z_i_norm"] for norm in embedding_norms_history]
            z_p_norms = [norm["z_p_norm"] for norm in embedding_norms_history]

            plt.plot(epochs_range, z_w_norms, "g-", label="Wildtype (z_w)", linewidth=2)
            plt.plot(
                epochs_range, z_i_norms, "orange", label="Perturbed (z_i)", linewidth=2
            )
            plt.plot(
                epochs_range, z_p_norms, "r-", label="Difference (z_p)", linewidth=2
            )

            plt.xlabel("Epoch")
            plt.ylabel("Mean L2 Norm")
            plt.title("Embedding Norms Evolution")
            plt.grid(True)
            plt.legend()

        # Gate weights evolution (or display fixed weights for concat)
        plt.subplot(5, 3, 9)
        if (
            cfg.model.local_predictor_config.combination_method == "gating"
            and gate_weights_history
        ):
            # Plot gate weights evolution for gating method
            global_weights = [gw[0] for gw in gate_weights_history]
            local_weights = [gw[1] for gw in gate_weights_history]
            plt.plot(
                range(1, len(global_weights) + 1),
                global_weights,
                "b-",
                label="Global weight",
        )
            plt.plot(
                range(1, len(local_weights) + 1),
                local_weights,
                "r-",
                label="Local weight",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Gate Weight")
            plt.title("Gate Weights Evolution")
            plt.grid(True)
            plt.legend()
            plt.ylim(0, 1)
        else:
            # For concat method, plot fixed weights as horizontal lines
            epochs_range = range(1, epoch + 2)
            fixed_global_weights = [0.5] * len(epochs_range)
            fixed_local_weights = [0.5] * len(epochs_range)

            plt.plot(
                epochs_range,
                fixed_global_weights,
                "b-",
                label="Global weight (fixed)",
                linewidth=2,
            )
            plt.plot(
                epochs_range,
                fixed_local_weights,
                "r-",
                label="Local weight (fixed)",
                linewidth=2,
            )

            plt.xlabel("Epoch")
            plt.ylabel("Gate Weight")
            plt.title("Gate Weights (Fixed for Concat)")
            plt.grid(True)
            plt.legend()
            plt.ylim(0, 1)

        # ROW 4: ICLoss components with L2, Unweighted losses with L2, Dist-MSE diff
        # Weighted Loss components evolution with L2 norm (for ICLoss/MleDistSupCR)
        plt.subplot(5, 3, 10)
        if loss_components_history and loss_type in ["icloss", "mle_dist_supcr"]:
            epochs_range = range(1, len(loss_components_history) + 1)

            # Extract components
            weighted_mse = [comp["weighted_mse"] for comp in loss_components_history]
            weighted_dist = [comp["weighted_dist"] for comp in loss_components_history]
            weighted_supcr = [
                comp["weighted_supcr"] for comp in loss_components_history
            ]

            # Plot all loss components on same scale
            plt.plot(
                epochs_range, weighted_mse, "b-", label="Weighted MSE", linewidth=2
            )
            plt.plot(
                epochs_range, weighted_dist, "r-", label="Weighted Dist", linewidth=2
            )
            plt.plot(
                epochs_range, weighted_supcr, "g-", label="Weighted SupCR", linewidth=2
            )

            # Add L2 norm if available (scaled by weight decay)
            if len(weight_l2_norms) >= len(epochs_range):
                l2_norms = weight_l2_norms[: len(epochs_range)]
                weight_decay = cfg.regression_task.optimizer.weight_decay
                l2_penalty = [norm * weight_decay for norm in l2_norms]
                plt.plot(
                    epochs_range,
                    l2_penalty,
                    "purple",
                    label=f"L2 Penalty (wd={weight_decay})",
                    linewidth=2,
                    linestyle="--",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Loss Component Value")
            plt.title("Weighted ICLoss Components with L2 Norm")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale("log")

        # Unweighted loss components with L2 norm
        plt.subplot(5, 3, 11)
        if loss_components_history and loss_type in ["icloss", "mle_dist_supcr"]:
            epochs_range = range(1, len(loss_components_history) + 1)

            # Extract unweighted components
            mse_loss = [comp["mse_loss"] for comp in loss_components_history]
            dist_loss = [comp["dist_loss"] for comp in loss_components_history]
            supcr_loss = [comp["supcr_loss"] for comp in loss_components_history]

            # Plot all loss components on same scale
            plt.plot(epochs_range, mse_loss, "b-", label="MSE Loss", linewidth=2)
            plt.plot(epochs_range, dist_loss, "r-", label="Dist Loss", linewidth=2)
            plt.plot(epochs_range, supcr_loss, "g-", label="SupCR Loss", linewidth=2)

            # Add L2 norm if available (without weight decay for unweighted plot)
            if len(weight_l2_norms) >= len(epochs_range):
                l2_norms = weight_l2_norms[: len(epochs_range)]
                plt.plot(
                    epochs_range,
                    l2_norms,
                    "purple",
                    label="L2 Norm",
                    linewidth=2,
                    linestyle="--",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Unweighted Loss Components with L2 Norm")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale("log")

        # Dist Loss vs MSE Loss difference
        plt.subplot(5, 3, 12)
        if loss_components_history and loss_type in ["icloss", "mle_dist_supcr"]:
            epochs_range = range(1, len(loss_components_history) + 1)

            # Calculate difference between unweighted dist and mse losses
            dist_minus_mse = [
                comp["dist_loss"] - comp["mse_loss"] for comp in loss_components_history
            ]

            plt.plot(epochs_range, dist_minus_mse, "purple", linewidth=2)
            plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            plt.xlabel("Epoch")
            plt.ylabel("Dist Loss - MSE Loss")
            plt.title("Distribution vs MSE Loss Difference (Log Scale)")
            plt.grid(True, alpha=0.3)
            plt.yscale("symlog")

            # Add shading to show positive/negative regions
            plt.fill_between(
                epochs_range,
                0,
                dist_minus_mse,
                where=[d > 0 for d in dist_minus_mse],
                color="red",
                alpha=0.2,
                label="Dist > MSE",
            )
            plt.fill_between(
                epochs_range,
                0,
                dist_minus_mse,
                where=[d <= 0 for d in dist_minus_mse],
                color="blue",
                alpha=0.2,
                label="Dist ≤ MSE",
            )

            # Add final value annotation
            final_diff = dist_minus_mse[-1]
            plt.text(
                0.95,
                0.95,
                f"Final: {final_diff:.4f}",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # ROW 5: LR-Loss correlation, 2D PCoA, Lambda display
        # Learning rate vs Loss correlation
        plt.subplot(5, 3, 13)
        if (
            len(losses) > 50 and len(learning_rates) > 50
        ):  # Need enough data for correlation
            rolling_corr = calculate_rolling_correlation(
                learning_rates, losses, window=50
            )
            if rolling_corr:
                corr_epochs = range(50, len(losses) + 1)
                plt.plot(corr_epochs, rolling_corr, "green", linewidth=2)
                plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
                plt.xlabel("Epoch")
                plt.ylabel("Correlation")
                plt.title("Rolling LR-Loss Correlation (window=50)")
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)

                # Add current correlation value
                if rolling_corr:
                    current_corr = rolling_corr[-1]
                    plt.text(
                        0.95,
                        0.95,
                        f"Current: {current_corr:.3f}",
                        transform=plt.gca().transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
        else:
            plt.text(
                0.5,
                0.5,
                "Insufficient data\nfor correlation\n(need 50+ epochs)",
                transform=plt.gca().transAxes,
                fontsize=14,
                ha="center",
                va="center",
            )
            plt.title("Rolling LR-Loss Correlation")
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

        # 2D PCoA of latent embeddings
        plt.subplot(5, 3, 14)
        if "z_p" in current_representations:
            z_p = current_representations["z_p"].cpu().numpy()

            # Apply PCoA using MDS with error handling
            if z_p.shape[0] > 2:  # Need at least 3 points
                try:
                    # Check for variance in embeddings
                    embedding_std = np.std(z_p)
                    if embedding_std < 1e-6:  # Very low variance
                        plt.text(
                            0.5,
                            0.5,
                            "Embeddings too similar\nfor meaningful PCoA\n(low variance)",
                            transform=plt.gca().transAxes,
                            fontsize=14,
                            ha="center",
                            va="center",
                        )
                        plt.title("2D PCoA of Perturbation Embeddings")
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])
                    else:
                        # Try MDS with increased tolerance and max iterations
                        mds = MDS(
                            n_components=2,
                            metric=True,
                            random_state=42,
                            eps=1e-3,
                            max_iter=500,
                        )

                        # Suppress the specific warning
                        import warnings

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                category=RuntimeWarning,
                                message="invalid value encountered",
                            )
                            z_p_2d = mds.fit_transform(z_p)

                        # Create scatter plot colored by label values
                        scatter = plt.scatter(
                            z_p_2d[:, 0],
                            z_p_2d[:, 1],
                            c=true_np[valid_mask],
                            cmap="magma",
                            alpha=0.7,
                            s=50,
                        )
                        plt.colorbar(scatter, label="True Value")
                        plt.xlabel("PCoA 1")
                        plt.ylabel("PCoA 2")
                        plt.title("2D PCoA of Perturbation Embeddings (z_p)")
                        plt.grid(True, alpha=0.3)

                        # Fix xticks to prevent overlap
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(
                            plt.MaxNLocator(nbins=5, prune="both")
                        )
                        ax.yaxis.set_major_locator(
                            plt.MaxNLocator(nbins=5, prune="both")
                        )
                except Exception as e:
                    # Fallback to PCA if MDS fails
                    try:
                        pca = PCA(n_components=2, random_state=42)
                        z_p_2d = pca.fit_transform(z_p)

                        scatter = plt.scatter(
                            z_p_2d[:, 0],
                            z_p_2d[:, 1],
                            c=true_np[valid_mask],
                            cmap="magma",
                            alpha=0.7,
                            s=50,
                        )
                        plt.colorbar(scatter, label="True Value")
                        plt.xlabel("PC 1")
                        plt.ylabel("PC 2")
                        plt.title("2D PCA of Perturbation Embeddings (z_p)")
                        plt.grid(True, alpha=0.3)

                        # Fix xticks to prevent overlap
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(
                            plt.MaxNLocator(nbins=5, prune="both")
                        )
                        ax.yaxis.set_major_locator(
                            plt.MaxNLocator(nbins=5, prune="both")
                        )
                    except:
                        plt.text(
                            0.5,
                            0.5,
                            f"Error in dimensionality reduction\n{str(e)[:50]}",
                            transform=plt.gca().transAxes,
                            fontsize=12,
                            ha="center",
                            va="center",
                        )
                        plt.title("2D Projection Failed")
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])
            else:
                plt.text(
                    0.5,
                    0.5,
                    "Insufficient points\nfor PCoA\n(need 3+ samples)",
                    transform=plt.gca().transAxes,
                    fontsize=14,
                    ha="center",
                    va="center",
                )
                plt.title("2D PCoA of Perturbation Embeddings")
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])

        # Lambda values and loss parameters display
        plt.subplot(5, 3, 15)
        if lambda_values_history and loss_type == "icloss":
            # Display loss function parameters
            lambda_dist = lambda_values_history[0]["lambda_dist"]
            lambda_supcr = lambda_values_history[0]["lambda_supcr"]

            # Better spacing for ICLoss parameters
            plt.title("Loss Function Configuration", pad=15)

            # Lambda parameters section
            plt.text(
                0.5,
                0.96,
                "ICLoss Parameters",
                transform=plt.gca().transAxes,
                fontsize=13,
                ha="center",
                va="top",
                weight="bold",
            )

            plt.text(
                0.5,
                0.85,
                f"λ_dist = {lambda_dist}",
                transform=plt.gca().transAxes,
                fontsize=12,
                ha="center",
                va="center",
            )

            plt.text(
                0.5,
                0.77,
                f"λ_supcr = {lambda_supcr}",
                transform=plt.gca().transAxes,
                fontsize=12,
                ha="center",
                va="center",
            )

            # Add current loss values with better spacing
            if loss_components_history:
                current_loss = losses[-1]
                current_components = loss_components_history[-1]

                # Calculate sum of weighted components
                components_sum = (
                    current_components["weighted_mse"]
                    + current_components["weighted_dist"]
                    + current_components["weighted_supcr"]
                )

                # More lenient tolerance for assertion (accounting for floating point precision)
                tolerance = 1e-4  # Increased from 1e-6
                relative_error = abs(components_sum - current_loss) / (
                    current_loss + 1e-10
                )
                assert_passed = relative_error < tolerance

                # Better formatted display
                plt.text(
                    0.5,
                    0.67,
                    f"Total Loss: {current_loss:.6f}",
                    transform=plt.gca().transAxes,
                    fontsize=11,
                    ha="center",
                    va="center",
                    weight="bold",
                )

                # Component breakdown with spacing
                y_start = 0.57
                component_texts = [
                    f'Weighted MSE: {current_components["weighted_mse"]:.6f}',
                    f'Weighted Dist: {current_components["weighted_dist"]:.6f}',
                    f'Weighted SupCR: {current_components["weighted_supcr"]:.6f}',
                ]

                for i, text in enumerate(component_texts):
                    plt.text(
                        0.5,
                        y_start - i * 0.06,
                        text,
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        ha="center",
                        va="center",
                    )

                # Separator line
                plt.text(
                    0.5,
                    0.35,
                    "─────────────────",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    ha="center",
                    va="center",
                )

                # Sum and assertion
                plt.text(
                    0.5,
                    0.29,
                    f"Sum: {components_sum:.6f}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    ha="center",
                    va="center",
                )

                # Assertion with relative error display
                assert_color = "green" if assert_passed else "orange"
                assert_symbol = "✓" if assert_passed else "≈"
                plt.text(
                    0.5,
                    0.21,
                    f"Assert Sum = Total: {assert_symbol}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    ha="center",
                    va="center",
                    color=assert_color,
                )

                # Show relative error if not exactly matching
                if not assert_passed:
                    plt.text(
                        0.5,
                        0.14,
                        f"(Rel. Error: {relative_error:.2e})",
                        transform=plt.gca().transAxes,
                        fontsize=9,
                        ha="center",
                        va="center",
                        color="gray",
                    )

            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
        else:
            # For LogCosh loss
            plt.text(
                0.5,
                0.5,
                f"{loss_type.upper()} Loss\n\nNo hyperparameters",
                transform=plt.gca().transAxes,
                fontsize=16,
                ha="center",
                va="center",
            )
            plt.title("Loss Function Configuration")
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"training_epoch_{epoch+1:04d}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # Training loop
    model.train()
    print("\nStarting training:")
    losses = []
    correlations = []
    spearman_correlations = []
    mses = []
    maes = []
    rmses = []  # New: RMSE tracking
    learning_rates = []
    gate_weights_history = []
    loss_components_history = []
    embedding_norms_history = []
    lambda_values_history = []
    weight_l2_norms = []  # New: L2 norm tracking
    num_epochs = cfg.trainer.max_epochs
    plot_interval = cfg.regression_task.plot_every_n_epochs

    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()

            # Forward pass
            predictions, representations = model(cell_graph, batch)

            # Handle different loss functions
            if isinstance(criterion, LogCoshLoss):
                loss = criterion(predictions.squeeze(), y)
            elif isinstance(criterion, (ICLoss, MleDistSupCR)):
                # ICLoss and MleDistSupCR expect z_p as third argument
                z_p = representations.get("z_p")
                if z_p is None:
                    raise ValueError("ICLoss/MleDistSupCR requires z_p from model representations")

                # ICLoss expects inputs with shape [batch_size, num_phenotypes]
                # For gene interaction only, we need to add a dummy dimension
                pred_reshaped = predictions.squeeze()
                if pred_reshaped.dim() == 0:
                    pred_reshaped = pred_reshaped.unsqueeze(0)
                pred_reshaped = pred_reshaped.unsqueeze(1)  # [batch_size, 1]

                y_reshaped = y
                if y_reshaped.dim() == 0:
                    y_reshaped = y_reshaped.unsqueeze(0)
                y_reshaped = y_reshaped.unsqueeze(1)  # [batch_size, 1]

                loss_output = criterion(pred_reshaped, y_reshaped, z_p)
                # ICLoss and MleDistSupCR return tuple (loss, loss_dict)
                loss = loss_output[0]
                loss_dict = loss_output[1]
                # You can log additional loss components if needed
                loss_name = "ICLoss" if isinstance(criterion, ICLoss) else "MleDistSupCR"
                print(
                    f"  {loss_name} components: mse={loss_dict['mse_loss']:.4f}, dist={loss_dict['weighted_dist']:.4f}, supcr={loss_dict['weighted_supcr']:.4f}"
                )
                # Track loss components
                # Ensure values are Python scalars, not tensors
                loss_components_history.append(
                    {
                        "weighted_mse": float(loss_dict["weighted_mse"]),
                        "weighted_dist": float(loss_dict["weighted_dist"]),
                        "weighted_supcr": float(loss_dict["weighted_supcr"]),
                        "mse_loss": float(loss_dict["mse_loss"]),
                        "dist_loss": float(loss_dict["dist_loss"]),
                        "supcr_loss": float(loss_dict["supcr_loss"]),
                    }
                )
                # Track lambda values (fixed hyperparameters only for gene interaction)
                lambda_values_history.append(
                    {
                        "lambda_dist": cfg.regression_task.lambda_dist,
                        "lambda_supcr": cfg.regression_task.lambda_supcr,
                    }
                )
            else:
                loss = criterion(predictions.squeeze(), y)

            # Calculate metrics
            with torch.no_grad():
                pred_np = predictions.squeeze().cpu().numpy()
                target_np = y.cpu().numpy()
                valid_mask = ~np.isnan(target_np)

                if np.sum(valid_mask) > 0:
                    correlation = np.corrcoef(
                        pred_np[valid_mask], target_np[valid_mask]
                    )[0, 1]
                    # Calculate Spearman correlation
                    spearman_corr, _ = stats.spearmanr(
                        pred_np[valid_mask], target_np[valid_mask]
                    )
                    mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
                    mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))
                    rmse = np.sqrt(mse)  # Calculate RMSE
                else:
                    correlation = 0.0
                    spearman_corr = 0.0
                    mse = float("inf")
                    mae = float("inf")
                    rmse = float("inf")

                # Extract gate weights
                if "gate_weights" in representations:
                    gate_weights = (
                        representations["gate_weights"].mean(dim=0).cpu().numpy()
                    )
                    gate_weights_history.append(gate_weights)

                # Track embedding norms
                if "z_w" in representations and "z_i" in representations:
                    z_w_norm = representations["z_w"].norm(dim=1).mean().item()
                    z_i_norm = representations["z_i"].norm(dim=1).mean().item()
                    z_p_norm = representations["z_p"].norm(dim=1).mean().item()
                    embedding_norms_history.append(
                        {
                            "z_w_norm": z_w_norm,
                            "z_i_norm": z_i_norm,
                            "z_p_norm": z_p_norm,
                        }
                    )

            losses.append(loss.item())
            correlations.append(correlation)
            spearman_correlations.append(spearman_corr)
            mses.append(mse)
            maes.append(mae)
            rmses.append(rmse)  # Track RMSE
            learning_rates.append(optimizer.param_groups[0]["lr"])

            # Calculate L2 norm of weights (moved outside no_grad context)
            l2_norm = calculate_weight_l2_norm(model)
            weight_l2_norms.append(l2_norm)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Get GPU memory stats if using CUDA
            gpu_memory_str = ""
            if device.type == "cuda":
                allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
                gpu_memory_str = f", GPU: {allocated_gb:.2f}/{reserved_gb:.2f}GB"

            # Logging every plot_interval epochs
            if epoch % plot_interval == 0 or epoch == num_epochs - 1:
                # Store pre-update correlation for comparison
                pre_update_correlation = correlation
                pre_update_spearman = spearman_corr

                # Save intermediate plot
                save_intermediate_plot(
                    epoch,
                    losses,
                    correlations,
                    mses,
                    maes,
                    rmses,
                    learning_rates,
                    gate_weights_history,
                    model,
                    cell_graph,
                    batch,
                    y,
                    weight_l2_norms,
                    cfg,
                    loss_type=loss_type,
                    loss_components_history=loss_components_history,
                    spearman_correlations=spearman_correlations,
                    embedding_norms_history=embedding_norms_history,
                    lambda_values_history=lambda_values_history,
                )

            loss.backward()

            # Optional gradient clipping
            if cfg.regression_task.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.regression_task.clip_grad_norm_max_norm
                )

            optimizer.step()

            # Step the learning rate scheduler if it exists
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Recalculate correlation after optimizer step for consistency
            if epoch % plot_interval == 0 or epoch == num_epochs - 1:
                with torch.no_grad():
                    updated_predictions, _ = model(cell_graph, batch)
                    updated_pred_np = updated_predictions.squeeze().cpu().numpy()
                    if np.sum(valid_mask) > 0:
                        updated_correlation = np.corrcoef(
                            updated_pred_np[valid_mask], target_np[valid_mask]
                        )[0, 1]
                        updated_spearman, _ = stats.spearmanr(
                            updated_pred_np[valid_mask], target_np[valid_mask]
                        )
                        # Update the last values in the lists
                        correlations[-1] = updated_correlation
                        spearman_correlations[-1] = updated_spearman

                        # Recalculate MSE, MAE, RMSE with updated predictions
                        updated_mse = np.mean(
                            (updated_pred_np[valid_mask] - target_np[valid_mask]) ** 2
                        )
                        updated_mae = np.mean(
                            np.abs(updated_pred_np[valid_mask] - target_np[valid_mask])
                        )
                        updated_rmse = np.sqrt(updated_mse)

                        # Print updated metrics
                        print(f"\nEpoch {epoch + 1}/{num_epochs}")
                        print(f"{loss_type.upper()} Loss: {loss.item():.6f}")
                        print(
                            f"Pearson: {updated_correlation:.4f} (was {pre_update_correlation:.4f})"
                        )
                        print(
                            f"Spearman: {updated_spearman:.4f} (was {pre_update_spearman:.4f})"
                        )
                        print(
                            f"MSE: {updated_mse:.6f}, MAE: {updated_mae:.6f}, RMSE: {updated_rmse:.6f}"
                        )
                        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                        print(f"L2 Norm: {l2_norm:.4f}")
                        print(f"Time: {epoch_time:.2f}s{gpu_memory_str}")

    except RuntimeError as e:
        print(f"\nError during training: {e}")
        if device.type == "cuda":
            print("\nThis might be a GPU memory issue. Try:")
            print("1. Reducing batch size")
            print("2. Reducing model size")
            print("3. Using gradient checkpointing")
            print("4. Using mixed precision training")
        raise

    # Final evaluation and comprehensive plot
    print("\n\nFinal evaluation:")
    model.eval()
    with torch.no_grad():
        final_predictions, final_representations = model(cell_graph, batch)

        # Final metrics
        pred_np = final_predictions.squeeze().cpu().numpy()
        target_np = y.cpu().numpy()
        valid_mask = ~np.isnan(target_np)

        if np.sum(valid_mask) > 0:
            final_correlation = np.corrcoef(pred_np[valid_mask], target_np[valid_mask])[
                0, 1
            ]
            final_spearman, _ = stats.spearmanr(
                pred_np[valid_mask], target_np[valid_mask]
            )
            final_mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
            final_mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))
            final_rmse = np.sqrt(final_mse)

            print(f"Final Pearson Correlation: {final_correlation:.6f}")
            print(f"Final Spearman Correlation: {final_spearman:.6f}")
            print(f"Final MSE: {final_mse:.6f}")
            print(f"Final MAE: {final_mae:.6f}")
            print(f"Final RMSE: {final_rmse:.6f}")
            print(f"Final {loss_type.upper()} Loss: {losses[-1]:.6f}")
            if weight_l2_norms:
                print(f"Final L2 Norm: {weight_l2_norms[-1]:.4f}")

            # Create comprehensive final plot
            plt.figure(figsize=(20, 12))

            # Determine loss label for final plots
            loss_label = "LogCosh Loss" if loss_type == "logcosh" else "ICLoss"

            # Loss curve
            plt.subplot(3, 4, 1)
            plt.plot(range(1, len(losses) + 1), losses, "b-", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel(loss_label)
            plt.title("Training Loss Evolution")
            plt.grid(True)
            plt.yscale("log")

            # Correlation evolution
            plt.subplot(3, 4, 2)
            plt.plot(range(1, len(correlations) + 1), correlations, "g-", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Correlation (r)")
            plt.title(f"Correlation Evolution (Final: {final_correlation:.4f})")
            plt.grid(True)
            plt.ylim(0, 1)

            # Final predictions scatter
            plt.subplot(3, 4, 3)
            plt.scatter(target_np[valid_mask], pred_np[valid_mask], alpha=0.6)
            min_val = min(target_np[valid_mask].min(), pred_np[valid_mask].min())
            max_val = max(target_np[valid_mask].max(), pred_np[valid_mask].max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
            plt.xlabel("True Gene Interaction")
            plt.ylabel("Predicted Gene Interaction")
            plt.title(f"Final Predictions (r={final_correlation:.4f})")
            plt.grid(True)

            # Error histogram
            plt.subplot(3, 4, 4)
            errors = pred_np[valid_mask] - target_np[valid_mask]
            plt.hist(errors, bins=30, alpha=0.7, edgecolor="black")
            plt.axvline(x=0, color="r", linestyle="--", linewidth=2)
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(
                f"Error Distribution (μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})"
            )
            plt.grid(True)

            # MSE evolution
            plt.subplot(3, 4, 5)
            plt.plot(range(1, len(mses) + 1), mses, "r-", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title(f"MSE Evolution (Final: {final_mse:.6f})")
            plt.grid(True)
            plt.yscale("log")

            # MAE evolution
            plt.subplot(3, 4, 6)
            plt.plot(range(1, len(maes) + 1), maes, "orange", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.title(f"MAE Evolution (Final: {final_mae:.6f})")
            plt.grid(True)
            plt.yscale("log")

            # Gate weights final distribution
            plt.subplot(3, 4, 7)
            if gate_weights_history:
                final_gate_weights = gate_weights_history[-1]
                plt.bar(["Global", "Local"], final_gate_weights, color=["blue", "red"])
                plt.ylabel("Gate Weight")
                plt.title("Final Gate Weights")
                plt.ylim(0, 1)
                plt.grid(True, axis="y")

            # Gate weights evolution
            plt.subplot(3, 4, 8)
            if gate_weights_history:
                global_weights = [gw[0] for gw in gate_weights_history]
                local_weights = [gw[1] for gw in gate_weights_history]
                plt.plot(
                    range(1, len(global_weights) + 1),
                    global_weights,
                    "b-",
                    label="Global",
                    linewidth=2,
                )
                plt.plot(
                    range(1, len(local_weights) + 1),
                    local_weights,
                    "r-",
                    label="Local",
                    linewidth=2,
                )
                plt.xlabel("Epoch")
                plt.ylabel("Gate Weight")
                plt.title("Gate Weights Evolution")
                plt.grid(True)
                plt.legend()
                plt.ylim(0, 1)

            # Learning rate schedule
            plt.subplot(3, 4, 9)
            plt.plot(
                range(1, len(learning_rates) + 1), learning_rates, "purple", linewidth=2
            )
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
            plt.yscale("log")

            # Embeddings analysis
            plt.subplot(3, 4, 10)
            if "z_w" in final_representations and "z_i" in final_representations:
                z_w_norm = final_representations["z_w"].norm(dim=1).mean().item()
                z_i_norm = final_representations["z_i"].norm(dim=1).mean().item()
                z_p_norm = final_representations["z_p"].norm(dim=1).mean().item()

                categories = [
                    "Wildtype\n(z_w)",
                    "Perturbed\n(z_i)",
                    "Difference\n(z_p)",
                ]
                norms = [z_w_norm, z_i_norm, z_p_norm]
                bars = plt.bar(categories, norms, color=["green", "orange", "red"])
                plt.ylabel("Mean L2 Norm")
                plt.title("Final Embedding Norms")
                plt.grid(True, axis="y")

                # Add value labels on bars
                for bar, norm in zip(bars, norms):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{norm:.3f}",
                        ha="center",
                        va="bottom",
                    )

            # Prediction statistics
            plt.subplot(3, 4, 11)
            plt.boxplot([target_np[valid_mask], pred_np[valid_mask]])
            plt.xticks([1, 2], ["True", "Predicted"])
            plt.ylabel("Gene Interaction Score")
            plt.title("Value Distribution Comparison")
            plt.grid(True, axis="y")

            # Model parameters summary
            plt.subplot(3, 4, 12)
            param_counts = model.num_parameters
            categories = list(param_counts.keys())
            counts = list(param_counts.values())
            plt.bar(categories, counts)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Number of Parameters")
            plt.title(f"Model Parameters (Total: {param_counts['total']:,})")
            plt.grid(True, axis="y")

            # Loss components evolution (for ICLoss) - final plot
            if loss_components_history and loss_type == "icloss":
                plt.figure(figsize=(10, 6))
                epochs_range = range(1, len(loss_components_history) + 1)

                # Extract components
                weighted_mse = [
                    comp["weighted_mse"] for comp in loss_components_history
                ]
                weighted_dist = [
                    comp["weighted_dist"] for comp in loss_components_history
                ]
                weighted_supcr = [
                    comp["weighted_supcr"] for comp in loss_components_history
                ]

                plt.plot(
                    epochs_range,
                    weighted_mse,
                    "b-",
                    label="Weighted MSE",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    markevery=max(1, len(epochs_range) // 20),
                )
                plt.plot(
                    epochs_range,
                    weighted_dist,
                    "r-",
                    label="Weighted Dist",
                    linewidth=2,
                    marker="s",
                    markersize=4,
                    markevery=max(1, len(epochs_range) // 20),
                )
                plt.plot(
                    epochs_range,
                    weighted_supcr,
                    "g-",
                    label="Weighted SupCR",
                    linewidth=2,
                    marker="^",
                    markersize=4,
                    markevery=max(1, len(epochs_range) // 20),
                )

                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Loss Component Value", fontsize=12)
                plt.title("ICLoss Weighted Components Evolution", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=11)
                plt.yscale("log")

                # Add final values in legend
                plt.text(
                    0.02,
                    0.98,
                    f"Final MSE: {weighted_mse[-1]:.6f}",
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
                plt.text(
                    0.02,
                    0.90,
                    f"Final Dist: {weighted_dist[-1]:.6f}",
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
                plt.text(
                    0.02,
                    0.82,
                    f"Final SupCR: {weighted_supcr[-1]:.6f}",
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

                plt.tight_layout()
                plt.savefig(
                    osp.join(plot_dir, f"loss_components_evolution_{timestamp()}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Go back to the main figure
                plt.figure(1)

            plt.tight_layout()

            # Add overall title
            plt.suptitle(
                f"HeteroCellBipartiteDangoGI Training Results - {num_epochs} Epochs ({loss_type.upper()} Loss)",
                fontsize=16,
                y=0.998,
            )

            plt.tight_layout()
            plt.savefig(
                osp.join(plot_dir, f"final_results_{timestamp()}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            print(f"\nResults saved to: {plot_dir}")
            print(f"- Training plots: training_epoch_*.png")
            print(f"- Final results: final_results_{timestamp()}.png")

    # Save the model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
