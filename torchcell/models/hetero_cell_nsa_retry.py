"""Heterogeneous cell model using Node-Set Attention blocks over gene/reaction/metabolite graphs."""

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn.aggr.attention import AttentionalAggregation

from torchcell.models.act import act_register
from torchcell.nn.hetero_nsa import HeteroNSA


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    """Create a normalization layer based on specified type."""
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


class AttentionalGraphAggregation(nn.Module):
    """Attentional aggregation for graph-level representations."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        """Build the gating and transform MLPs and the attentional aggregator."""
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
        self, x: torch.Tensor, index: torch.Tensor, dim_size: int | None = None
    ) -> torch.Tensor:
        """Aggregate node features into graph-level vectors using attention weights."""
        return self.aggregator(x, index=index, dim_size=dim_size)


class PreProcessor(nn.Module):
    """MLP for preprocessing node features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
    ):
        """Build a stack of linear, norm, activation, and dropout layers."""
        super().__init__()
        act = act_register[activation]
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(norm_layer)
        layers.append(act)
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(norm_layer)
            layers.append(act)
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass node features through the preprocessing MLP."""
        return self.mlp(x)


class HeteroCellNSA(nn.Module):
    """Heterogeneous Cell Model using Node-Set Attention (NSA) blocks.

    This model processes heterogeneous biological cell graphs with gene, reaction,
    and metabolite nodes using attention-based message passing.
    """

    def __init__(
        self,
        gene_num: int,
        reaction_num: int,
        metabolite_num: int,
        hidden_channels: int,
        out_channels: int,
        attention_pattern: list[str] = ["M", "S"],
        num_heads: int = 8,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        prediction_head_config: dict[str, Any] | None = None,
        graph_names: list[str] | None = None,
    ):
        """Build node embeddings, preprocessors, NSA blocks, aggregation, and prediction head."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gene_num = gene_num
        self.reaction_num = reaction_num
        self.metabolite_num = metabolite_num

        # Store graph names (default to standard ones if not provided)
        if graph_names is None:
            self.graph_names = ["physical_interaction", "regulatory_interaction"]
        else:
            self.graph_names = graph_names

        # Validate configurations
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"Hidden dimension ({hidden_channels}) must be divisible by number of heads ({num_heads})"
            )

        # Learnable embeddings for each node type
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)
        self.reaction_embedding = nn.Embedding(reaction_num, hidden_channels)
        self.metabolite_embedding = nn.Embedding(metabolite_num, hidden_channels)

        # init embeddings TODO
        nn.init.orthogonal_(self.gene_embedding.weight)
        nn.init.orthogonal_(self.reaction_embedding.weight)
        nn.init.orthogonal_(self.metabolite_embedding.weight)

        # Preprocessor for gene features
        self.preprocessor = PreProcessor(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        # Define node and edge types for the heterogeneous graph
        node_types: set[str] = {"gene", "reaction", "metabolite"}

        # Dynamically construct edge types based on graph names
        edge_types: set[tuple[str, str, str]] = set()

        # Add gene-gene edges for each graph name
        for graph_name in self.graph_names:
            edge_types.add(("gene", graph_name, "gene"))

        # Add the metabolic edges (always present)
        edge_types.add(("gene", "gpr", "reaction"))
        edge_types.add(("metabolite", "reaction", "metabolite"))

        # NSA layer with the specified attention pattern
        # FIX: Pass the activation class, not an instance
        self.nsa_layer = HeteroNSA(
            hidden_dim=hidden_channels,
            node_types=node_types,
            edge_types=edge_types,
            pattern=attention_pattern,
            num_heads=num_heads,
            dropout=dropout,
            activation=act_register[activation],  # Remove the parentheses here
            aggregation="sum",
        )

        # Layer norm for each node type
        self.layer_norms = nn.ModuleDict(
            {
                node_type: get_norm_layer(hidden_channels, norm)
                for node_type in node_types
            }
        )

        # Global attentional aggregation for graph-level representation
        self.global_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

        # Prediction head for fitness and gene interaction
        pred_config = prediction_head_config or {}
        self.prediction_head = self._build_prediction_head(
            in_channels=hidden_channels,
            hidden_channels=pred_config.get("hidden_channels", hidden_channels),
            out_channels=out_channels,
            num_layers=pred_config.get("head_num_layers", 1),
            dropout=pred_config.get("dropout", dropout),
            activation=pred_config.get("activation", activation),
            norm=pred_config.get("head_norm", norm),
        )

    def _build_prediction_head(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        activation: str,
        norm: str | None = None,
    ) -> nn.Module:
        """Build a multi-layer prediction head for final outputs."""
        if num_layers == 0:
            return nn.Identity()

        act = act_register[activation]
        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                if norm is not None:
                    layers.append(get_norm_layer(dims[i + 1], norm))
                layers.append(act)
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward_single(self, data: HeteroData | Batch) -> torch.Tensor:
        """Process a single graph or batch of graphs through the model."""
        device = self.gene_embedding.weight.device

        # Check if we're handling a batch
        is_batch = isinstance(data, Batch) or hasattr(data["gene"], "batch")

        # Initialize node features
        if is_batch:
            # For batched data, we need to carefully handle the node indices
            batch_size = (
                len(data["gene"].ptr) - 1
                if hasattr(data["gene"], "ptr")
                else int(data["gene"].batch.max()) + 1
            )

            # Create consistent embeddings for all node types
            x_dict = {}

            # Process gene nodes
            gene_idx = torch.arange(self.gene_num, device=device)
            gene_emb = self.gene_embedding(gene_idx)
            # Expand gene embeddings for each batch item
            gene_emb_exp = gene_emb.unsqueeze(0).expand(batch_size, -1, -1)
            # Reshape to [batch_size * gene_num, hidden_dim]
            gene_emb_flat = gene_emb_exp.reshape(-1, self.hidden_channels)

            # Handle gene perturbation mask if available
            if hasattr(data["gene"], "pert_mask") and torch.is_tensor(
                data["gene"].pert_mask
            ):
                # Only keep non-perturbed gene nodes
                x_dict["gene"] = gene_emb_flat[~data["gene"].pert_mask]
            else:
                # Use all gene embeddings
                x_dict["gene"] = gene_emb_flat[: data["gene"].num_nodes]

            # Apply preprocessor to gene features
            x_dict["gene"] = self.preprocessor(x_dict["gene"])

            # Similar approach for reaction nodes
            reaction_idx = torch.arange(self.reaction_num, device=device)
            reaction_emb = self.reaction_embedding(reaction_idx)
            reaction_emb_exp = reaction_emb.unsqueeze(0).expand(batch_size, -1, -1)
            reaction_emb_flat = reaction_emb_exp.reshape(-1, self.hidden_channels)

            if hasattr(data["reaction"], "pert_mask") and torch.is_tensor(
                data["reaction"].pert_mask
            ):
                x_dict["reaction"] = reaction_emb_flat[~data["reaction"].pert_mask]
            else:
                x_dict["reaction"] = reaction_emb_flat[: data["reaction"].num_nodes]

            # And for metabolite nodes
            metabolite_idx = torch.arange(self.metabolite_num, device=device)
            metabolite_emb = self.metabolite_embedding(metabolite_idx)
            metabolite_emb_exp = metabolite_emb.unsqueeze(0).expand(batch_size, -1, -1)
            metabolite_emb_flat = metabolite_emb_exp.reshape(-1, self.hidden_channels)

            if hasattr(data["metabolite"], "pert_mask") and torch.is_tensor(
                data["metabolite"].pert_mask
            ):
                x_dict["metabolite"] = metabolite_emb_flat[
                    ~data["metabolite"].pert_mask
                ]
            else:
                x_dict["metabolite"] = metabolite_emb_flat[
                    : data["metabolite"].num_nodes
                ]
        else:
            # For a single graph, use direct indexing with the full embeddings
            gene_count = min(data["gene"].num_nodes, self.gene_num)
            reaction_count = min(data["reaction"].num_nodes, self.reaction_num)
            metabolite_count = min(data["metabolite"].num_nodes, self.metabolite_num)

            gene_idx = torch.arange(gene_count, device=device)
            reaction_idx = torch.arange(reaction_count, device=device)
            metabolite_idx = torch.arange(metabolite_count, device=device)

            x_dict = {
                "gene": self.preprocessor(self.gene_embedding(gene_idx)),
                "reaction": self.reaction_embedding(reaction_idx),
                "metabolite": self.metabolite_embedding(metabolite_idx),
            }

        # Prepare batch assignment information for NSA layer
        batch_idx = {}
        if is_batch:
            for node_type in ["gene", "reaction", "metabolite"]:
                if hasattr(data[node_type], "batch"):
                    batch_idx[node_type] = data[node_type].batch

        # Process through NSA layer with proper error handling
        try:
            new_x = self.nsa_layer(x_dict, data, batch_idx)

            # Apply layer norm and residual connection
            for node_type in new_x:
                if node_type in x_dict:
                    new_x[node_type] = self.layer_norms[node_type](
                        new_x[node_type] + x_dict[node_type]
                    )

            return new_x["gene"]
        except Exception as e:
            print(f"Error in NSA layer: {e}")
            import traceback

            traceback.print_exc()
            raise

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData | Batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass comparing reference cell with perturbed cell."""
        # Process the reference (wildtype) graph
        z_w = self.forward_single(cell_graph)
        z_w = self.global_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )

        # Force z_w to be 2D [1, hidden_dim]
        z_w = z_w.view(-1, self.hidden_channels)

        # Process the perturbed (intact) batch
        z_i = self.forward_single(batch)

        # Get batch information
        if hasattr(batch["gene"], "batch"):
            batch_idx = batch["gene"].batch
            batch_size = int(batch_idx.max()) + 1
        else:
            # For single sample
            batch_idx = torch.zeros(
                batch["gene"].num_nodes, dtype=torch.long, device=z_i.device
            )
            batch_size = 1

        # Apply global aggregation
        z_i = self.global_aggregator(z_i, index=batch_idx)

        # Force z_i to be 2D [batch_size, hidden_dim]
        z_i = z_i.view(batch_size, self.hidden_channels)

        # Expand z_w to match batch size
        z_w_exp = z_w.expand(batch_size, self.hidden_channels)

        # Calculate difference
        z_p = z_w_exp - z_i

        # Force z_p to be exactly [batch_size, hidden_dim]
        z_p = z_p.view(batch_size, self.hidden_channels)

        # Generate predictions
        predictions = self.prediction_head(z_p)

        # Force predictions to be exactly [batch_size, 1]
        predictions = predictions.view(batch_size, -1)

        # Only gene interaction prediction
        gene_interaction = predictions

        # Print shapes to debug
        # print(f"Final shapes: z_w={z_w.shape}, z_i={z_i.shape}, z_p={z_p.shape}, pred={predictions.shape}")

        return predictions, {
            "z_w": z_w,
            "z_i": z_i,
            "z_p": z_p,  # This should now be [batch_size, hidden_dim]
            "gene_interaction": gene_interaction,
        }

    @property
    def num_parameters(self) -> dict[str, int]:
        """Count the number of parameters in each component of the model."""

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "reaction_embedding": count_params(self.reaction_embedding),
            "metabolite_embedding": count_params(self.metabolite_embedding),
            "preprocessor": count_params(self.preprocessor),
            "nsa_layer": count_params(self.nsa_layer),
            "layer_norms": sum(count_params(ln) for ln in self.layer_norms.values()),
            "global_aggregator": count_params(self.global_aggregator),
            "prediction_head": count_params(self.prediction_head),
        }
        counts["total"] = sum(counts.values())
        return counts


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/hetero_cell_nsa_retry_demo.py on 2026-10-06"
    )
