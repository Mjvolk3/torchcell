# torchcell/models/hetero_cell_bipartite_dango_gi
# [[torchcell.models.hetero_cell_bipartite_dango_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_bipartite_dango_gi
# Test file: tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py


import math
import os
import os.path as osp
import time
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import BatchNorm, GATv2Conv, HeteroConv, LayerNorm
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_scatter import scatter_mean

from torchcell.graph.graph import GeneMultiGraph
from torchcell.models.act import act_register


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
        
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        
        # Static embedding layer (like Dango)
        self.static_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Create multiple attention layers
        self.attention_layers = nn.ModuleList()
        self.beta_params = nn.ParameterList()  # Store ReZero parameters separately
        
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'q_proj': nn.Linear(hidden_dim, hidden_dim),
                'k_proj': nn.Linear(hidden_dim, hidden_dim),
                'v_proj': nn.Linear(hidden_dim, hidden_dim),
                'out_proj': nn.Linear(hidden_dim, hidden_dim),
            })
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
                    batch_output = self._apply_attention_layer(batch_embeddings, layer, self.beta_params[i])
                    output_embeddings[mask] = batch_output
                
                dynamic_embeddings = output_embeddings
            else:
                # Single batch processing
                dynamic_embeddings = self._apply_attention_layer(dynamic_embeddings, layer, self.beta_params[i])
        
        return static_embeddings, dynamic_embeddings

    def _apply_attention_layer(self, x, layer, beta):
        """Apply a single attention layer with multi-head attention"""
        batch_size = x.size(0)
        
        # Handle special case of single gene (no attention possible)
        if batch_size <= 1:
            return x
        
        # Linear projections
        q = layer['q_proj'](x)  # [batch_size, hidden_dim]
        k = layer['k_proj'](x)  # [batch_size, hidden_dim]
        v = layer['v_proj'](x)  # [batch_size, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, batch_size, head_dim]
        k = k.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Shape: [num_heads, batch_size, batch_size]
        
        # Create mask for self-attention (exclude self)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=x.device)
        self_mask = self_mask.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        # Apply mask
        attention_scores.masked_fill_(self_mask, -float('inf'))
        
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
        out = layer['out_proj'](out)
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
            dropout=dropout
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
            interaction_scores = scatter_mean(gene_scores, batch, dim=0, dim_size=num_batches)
            return interaction_scores.unsqueeze(-1)  # [num_batches, 1]
        else:
            # Single batch case
            return gene_scores.mean().unsqueeze(0).unsqueeze(-1)  # [1, 1]


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(channels, eps=1e-5)  # Increased epsilon for better stability
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
        if hasattr(conv, "concat"):
            expected_dim = (
                conv.heads * conv.out_channels if conv.concat else conv.out_channels
            )
        else:
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

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}

            # Create a GATv2Conv for each graph in the multigraph
            for graph_name in self.graph_names:
                edge_type = ("gene", graph_name, "gene")
                conv_dict[edge_type] = GATv2Conv(
                    hidden_channels,
                    hidden_channels // gene_encoder_config.get("heads", 1),
                    heads=gene_encoder_config.get("heads", 1),
                    concat=gene_encoder_config.get("concat", True),
                    add_self_loops=gene_encoder_config.get("add_self_loops", False),
                )

            # Wrap each conv with attention wrapper
            for key, conv in conv_dict.items():
                conv_dict[key] = AttentionConvWrapper(
                    conv,
                    hidden_channels,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Get local predictor config - now as a separate parameter
        local_predictor_config = local_predictor_config or {}
        
        # Gene interaction predictor for perturbed genes with Dango-like architecture
        self.gene_interaction_predictor = GeneInteractionPredictor(
            hidden_dim=hidden_channels,
            num_heads=local_predictor_config.get('num_heads', 4),
            num_layers=local_predictor_config.get('num_attention_layers', 2),
            dropout=dropout
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

        # MLP for gating weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

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

        # Apply convolution layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

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
            "gate_mlp": count_params(self.gate_mlp),
        }
        counts["total"] = sum(counts.values())
        return counts


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/005-kuzmin2018-tmi/conf"),
    config_name="hetero_cell_bipartite_dango_gi",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    import torch.nn as nn
    from torchcell.timestamp import timestamp
    import numpy as np
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    from torchcell.graph.graph import build_gene_multigraph, SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from sortedcontainers import SortedDict
    from torchcell.graph.graph import GeneMultiGraph, GeneGraph
    from torchcell.losses.logcosh import LogCoshLoss
    from torchcell.losses.isomorphic_cell_loss import ICLoss

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
    gene_encoder_config_dict = OmegaConf.to_container(cfg.model.gene_encoder_config, resolve=True) if cfg.model.gene_encoder_config else {}
    local_predictor_config_dict = OmegaConf.to_container(cfg.model.local_predictor_config, resolve=True) if hasattr(cfg.model, 'local_predictor_config') and cfg.model.local_predictor_config else {}
    
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
            weights=weights
        )
        print(f"Using ICLoss with lambda_dist={cfg.regression_task.lambda_dist}, lambda_supcr={cfg.regression_task.lambda_supcr}")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

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
        learning_rates,
        gate_weights_history,
        model,
        cell_graph,
        batch,
        y,
        loss_type="logcosh",
    ):
        """Save intermediate training plot every print interval."""
        plt.figure(figsize=(15, 10))

        # Determine loss label
        loss_label = "LogCosh Loss" if loss_type == "logcosh" else "ICLoss"

        # Loss curve
        plt.subplot(3, 3, 1)
        plt.plot(range(1, epoch + 2), losses, "b-", label=loss_label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Correlation evolution
        plt.subplot(3, 3, 2)
        plt.plot(range(1, epoch + 2), correlations, "g-", label="Correlation")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation (r)")
        plt.title("Correlation Evolution")
        plt.grid(True)
        plt.legend()

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

        # Correlation scatter plot
        plt.subplot(3, 3, 3)
        plt.scatter(true_np[valid_mask], pred_np[valid_mask], alpha=0.7)
        min_val = min(true_np[valid_mask].min(), pred_np[valid_mask].min())
        max_val = max(true_np[valid_mask].max(), pred_np[valid_mask].max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect correlation"
        )
        plt.xlabel("True Gene Interaction")
        plt.ylabel("Predicted Gene Interaction")
        plt.title(f"Predictions vs Truth (r={current_corr:.4f})")
        plt.grid(True)
        plt.legend()

        # MSE and MAE evolution
        plt.subplot(3, 3, 4)
        plt.plot(range(1, epoch + 2), mses, "r-", label="MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Mean Squared Error")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        plt.subplot(3, 3, 5)
        plt.plot(range(1, epoch + 2), maes, "orange", label="MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Mean Absolute Error")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Error distribution
        plt.subplot(3, 3, 6)
        errors = pred_np[valid_mask] - true_np[valid_mask]
        plt.hist(errors, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Error Distribution (std={np.std(errors):.4f})")
        plt.grid(True)

        # Learning rate evolution
        plt.subplot(3, 3, 7)
        plt.plot(range(1, epoch + 2), learning_rates, "purple")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.yscale("log")

        # Gate weights evolution (local vs global)
        if gate_weights_history:
            plt.subplot(3, 3, 8)
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

        # Embeddings analysis
        plt.subplot(3, 3, 9)
        if "z_w" in current_representations and "z_i" in current_representations:
            z_w_norm = current_representations["z_w"].norm(dim=1).mean().item()
            z_i_norm = current_representations["z_i"].norm(dim=1).mean().item()
            z_p_norm = current_representations["z_p"].norm(dim=1).mean().item()

            categories = ["Wildtype", "Perturbed", "Difference"]
            norms = [z_w_norm, z_i_norm, z_p_norm]
            plt.bar(categories, norms)
            plt.ylabel("Mean L2 Norm")
            plt.title("Embedding Norms")
            plt.grid(True, axis="y")

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
    mses = []
    maes = []
    learning_rates = []
    gate_weights_history = []
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
            elif isinstance(criterion, ICLoss):
                # ICLoss expects z_p as third argument
                z_p = representations.get("z_p")
                if z_p is None:
                    raise ValueError("ICLoss requires z_p from model representations")
                
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
                # ICLoss returns tuple (loss, loss_dict)
                loss = loss_output[0]
                loss_dict = loss_output[1]
                # You can log additional loss components if needed
                print(f"  ICLoss components: mse={loss_dict['mse_loss']:.4f}, dist={loss_dict['weighted_dist']:.4f}, supcr={loss_dict['weighted_supcr']:.4f}")
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
                    mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
                    mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))
                else:
                    correlation = 0.0
                    mse = float("inf")
                    mae = float("inf")

                # Extract gate weights
                if "gate_weights" in representations:
                    gate_weights = (
                        representations["gate_weights"].mean(dim=0).cpu().numpy()
                    )
                    gate_weights_history.append(gate_weights)

            losses.append(loss.item())
            correlations.append(correlation)
            mses.append(mse)
            maes.append(mae)
            learning_rates.append(optimizer.param_groups[0]["lr"])

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
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"{loss_type.upper()} Loss: {loss.item():.6f}")
                print(f"Correlation: {correlation:.4f}")
                print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
                print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Time: {epoch_time:.2f}s{gpu_memory_str}")

                # Save intermediate plot
                save_intermediate_plot(
                    epoch,
                    losses,
                    correlations,
                    mses,
                    maes,
                    learning_rates,
                    gate_weights_history,
                    model,
                    cell_graph,
                    batch,
                    y,
                    loss_type=loss_type,
                )

            loss.backward()

            # Optional gradient clipping
            if cfg.regression_task.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.regression_task.clip_grad_norm_max_norm
                )

            optimizer.step()

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
            final_mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
            final_mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))

            print(f"Final Correlation: {final_correlation:.6f}")
            print(f"Final MSE: {final_mse:.6f}")
            print(f"Final MAE: {final_mae:.6f}")
            print(f"Final {loss_type.upper()} Loss: {losses[-1]:.6f}")

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
            plt.ylim(-1, 1)

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
            plt.boxplot(
                [target_np[valid_mask], pred_np[valid_mask]],
                labels=["True", "Predicted"],
            )
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
