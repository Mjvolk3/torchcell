from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import os.path as osp
import os
import hydra
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import (
    HeteroConv,
    GCNConv,
    GATv2Conv,
    TransformerConv,
    GINConv,
    BatchNorm,
    LayerNorm,
    GraphNorm,
    InstanceNorm,
    PairNorm,
    MeanSubtractionNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    HypergraphConv,
)
from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv
from typing import Optional, Literal
from torch_geometric.typing import EdgeType
from torchcell.models.act import act_register
from collections import defaultdict

from typing import Any, Union, Optional
from torch_geometric.nn.aggr.attention import AttentionalAggregation
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.typing import EdgeType
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import HeteroData
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn.aggr.attention import AttentionalAggregation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv
from torchcell.models.act import act_register
from typing import Optional, Dict, Any, Tuple
from torch_geometric.data import Batch
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, BatchNorm, LayerNorm
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, BatchNorm, LayerNorm
from torch_geometric.data import HeteroData, Batch
from typing import Optional, Dict, Any, Tuple

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


class GeneInteractionAttention(nn.Module):
    """Self-attention module for gene interaction prediction"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # ReZero parameter (initialized to small value)
        self.beta = nn.Parameter(torch.ones(1) * 0.1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_embeddings, batch=None):
        """
        Args:
            gene_embeddings: Tensor of shape [total_genes, hidden_dim]
            batch: Optional tensor [total_genes] indicating batch assignment
        Returns:
            dynamic_embeddings: Tensor of shape [total_genes, hidden_dim]
        """
        # Process each batch separately if batch information is provided
        if batch is not None:
            unique_batches = batch.unique()
            output_embeddings = torch.zeros_like(gene_embeddings)

            for b in unique_batches:
                mask = batch == b
                batch_embeddings = gene_embeddings[mask]
                batch_output = self._process_batch(batch_embeddings)
                output_embeddings[mask] = batch_output

            return output_embeddings
        else:
            # Single batch processing
            return self._process_batch(gene_embeddings)

    def _process_batch(self, embeddings):
        """Process a single batch of embeddings"""
        num_genes = embeddings.size(0)
        residual = embeddings

        # Handle special case of single gene (no attention possible)
        if num_genes <= 1:
            return embeddings

        # Project to queries, keys, values
        q = self.q_proj(embeddings)
        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)

        # Calculate attention scores (excluding self-attention)
        attention_scores = torch.matmul(q, k.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float, device=embeddings.device)
        )

        # Mask out self-attention
        mask = torch.eye(num_genes, device=embeddings.device)
        attention_scores = attention_scores.masked_fill(mask.bool(), -1e9)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to get dynamic embeddings
        dynamic_embeddings = torch.matmul(attention_weights, v)
        dynamic_embeddings = self.out_proj(dynamic_embeddings)
        dynamic_embeddings = self.dropout(dynamic_embeddings)

        # Apply ReZero
        return residual + self.beta * dynamic_embeddings


class GeneInteractionPredictor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = GeneInteractionAttention(hidden_dim, dropout=dropout)
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
        # Static embeddings (original)
        static_embeddings = gene_embeddings

        # Dynamic embeddings through self-attention
        dynamic_embeddings = self.attention(gene_embeddings, batch)

        # Calculate the difference and square it
        diff = dynamic_embeddings - static_embeddings
        diff_squared = diff**2

        # Get gene-level scores
        gene_scores = self.prediction_layer(diff_squared)

        # If batch information is provided, average scores per batch
        if batch is not None:
            num_batches = batch.max().item() + 1
            batch_scores = torch.zeros(num_batches, 1, device=gene_embeddings.device)

            # For each batch, average the gene scores
            for b in range(num_batches):
                mask = batch == b
                if mask.sum() > 0:  # Ensure there are genes in this batch
                    batch_scores[b] = gene_scores[mask].mean()

            return batch_scores
        else:
            # Single batch case
            return gene_scores.mean().unsqueeze(0)


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
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
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        gene_encoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Learnable gene embeddings
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)

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

            # Gene-gene physical interactions
            conv_dict[("gene", "physical_interaction", "gene")] = GATv2Conv(
                hidden_channels,
                hidden_channels // gene_encoder_config.get("heads", 1),
                heads=gene_encoder_config.get("heads", 1),
                concat=gene_encoder_config.get("concat", True),
                add_self_loops=gene_encoder_config.get("add_self_loops", False),
            )

            # Gene-gene regulatory interactions
            conv_dict[("gene", "regulatory_interaction", "gene")] = GATv2Conv(
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

        # Gene interaction predictor for perturbed genes
        self.gene_interaction_predictor = GeneInteractionPredictor(
            hidden_dim=hidden_channels, dropout=dropout
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

        # Process edge indices
        edge_index_dict = {}

        # Gene-gene physical interactions
        if ("gene", "physical_interaction", "gene") in data.edge_types:
            gene_phys_edge_index = data[
                ("gene", "physical_interaction", "gene")
            ].edge_index.to(device)
            edge_index_dict[("gene", "physical_interaction", "gene")] = (
                gene_phys_edge_index
            )

        # Gene-gene regulatory interactions
        if ("gene", "regulatory_interaction", "gene") in data.edge_types:
            gene_reg_edge_index = data[
                ("gene", "regulatory_interaction", "gene")
            ].edge_index.to(device)
            edge_index_dict[("gene", "regulatory_interaction", "gene")] = (
                gene_reg_edge_index
            )

        # Apply convolution layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict["gene"]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Process reference graph (wildtype)
        z_w = self.forward_single(cell_graph)

        # Proper global aggregation for wildtype
        z_w_global = self.global_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )

        # Process perturbed batch if needed
        z_i = self.forward_single(batch)

        # Proper global aggregation for perturbed genes
        z_i_global = self.global_aggregator(z_i, index=batch["gene"].batch)

        # Get embeddings of perturbed genes from wildtype
        pert_indices = batch["gene"].cell_graph_idx_pert
        pert_gene_embs = z_w[pert_indices]

        # Calculate perturbation difference for z_p_global
        batch_size = z_i_global.size(0)
        z_w_exp = z_w_global.expand(batch_size, -1)
        z_p_global = z_w_exp - z_i_global

        # Determine batch assignment for perturbed genes
        if hasattr(batch["gene"], "x_pert_ptr"):
            # Create batch assignment using x_pert_ptr
            ptr = batch["gene"].x_pert_ptr
            batch_assign = torch.zeros(
                pert_indices.size(0), dtype=torch.long, device=z_w.device
            )
            for i in range(len(ptr) - 1):
                batch_assign[ptr[i] : ptr[i + 1]] = i
        else:
            # Alternative if x_pert_ptr is not available
            batch_assign = (
                batch["gene"].x_pert_batch
                if hasattr(batch["gene"], "x_pert_batch")
                else None
            )

        # Get gene interaction predictions using the local predictor
        local_interaction = self.gene_interaction_predictor(
            pert_gene_embs, batch_assign
        )

        # Get gene interaction predictions using the global predictor
        global_interaction = self.global_interaction_predictor(z_p_global)

        # Ensure dimensions match for gating
        if local_interaction.size(0) != batch_size:
            local_interaction_expanded = torch.zeros(batch_size, 1, device=z_w.device)
            for i in range(local_interaction.size(0)):
                batch_idx = batch_assign[i].item() if batch_assign is not None else 0
                if batch_idx < batch_size:
                    local_interaction_expanded[batch_idx] = local_interaction[i]
            local_interaction = local_interaction_expanded

        # Stack the predictions
        pred_stack = torch.cat([global_interaction, local_interaction], dim=1)
        
        # Use MLP to get logits for gating, then apply softmax
        gate_logits = self.gate_mlp(pred_stack)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        # Element-wise product of predictions and weights, then sum
        weighted_preds = pred_stack * gate_weights
        gene_interaction = weighted_preds.sum(dim=1, keepdim=True)

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
    config_path=osp.join(os.getcwd(), "experiments/004-dmi-tmi/conf"),
    config_name="hetero_cell_bipartite_dango_gi",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    import torch.nn as nn
    from torchcell.timestamp import timestamp
    import numpy as np
    from torchcell.scratch.load_batch_004 import load_sample_data_batch

    class LogCoshLoss(nn.Module):
        def __init__(self, reduction: str = "mean") -> None:
            super().__init__()
            if reduction not in ("none", "mean", "sum"):
                raise ValueError(f"Invalid reduction mode: {reduction}")
            self.reduction = reduction

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            loss = torch.log(torch.cosh(input - target))
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

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
        metabolism_graph="metabolism_bipartite",
    )
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Initialize the gene interaction model
    model = GeneInteractionDango(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=cfg.model.gene_encoder_config,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

    # Simple MSE loss for gene interaction prediction
    criterion = LogCoshLoss(reduction="mean")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    # Training target - only gene interaction
    y = batch["gene"].gene_interaction

    # Setup directory for plots
    os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    # Training loop
    model.train()
    print("\nStarting training:")
    losses = []
    num_epochs = cfg.trainer.max_epochs

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            predictions, representations = model(cell_graph, batch)
            loss = criterion(predictions.squeeze(), y)

            # Logging every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")

                # Calculate correlation for visualization
                with torch.no_grad():
                    pred_np = predictions.squeeze().cpu().numpy()
                    target_np = y.cpu().numpy()
                    valid_mask = ~np.isnan(target_np)
                    if np.sum(valid_mask) > 0:
                        correlation = np.corrcoef(
                            pred_np[valid_mask], target_np[valid_mask]
                        )[0, 1]
                        print(f"Correlation: {correlation:.4f}")

                # Report GPU usage
                if device.type == "cuda":
                    print(
                        f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB"
                    )
                    print(
                        f"GPU memory reserved: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB"
                    )

            losses.append(loss.item())
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

    # Final loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses) + 1), losses, "b-", label="MSE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(
        f"Gene Interaction Training Loss Over Time: "
        f"wd={cfg.regression_task.optimizer.weight_decay}"
    )
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        osp.join(ASSET_IMAGES_DIR, f"gene_interaction_training_loss_{timestamp()}.png")
    )
    plt.close()

    # Save the model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
