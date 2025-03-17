from omegaconf import DictConfig
import os
import os.path as osp
import hydra
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch
from typing import Dict, Optional, Any, Tuple, List, Set

from torchcell.nn.hetero_nsa import HeteroNSA
from torchcell.models.act import act_register
from torch_geometric.nn.aggr.attention import AttentionalAggregation


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
        return self.mlp(x)


class HeteroCellNSA(nn.Module):
    """
    Heterogeneous Cell Model using Node-Set Attention (NSA) blocks.

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
        attention_pattern: List[str] = ["M", "S"],
        num_heads: int = 8,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        gene_encoder_config: Optional[Dict[str, Any]] = None,
        metabolism_config: Optional[Dict[str, Any]] = None,
        prediction_head_config: Optional[Dict[str, Any]] = None,
        gpr_conv_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gene_num = gene_num
        self.reaction_num = reaction_num
        self.metabolite_num = metabolite_num

        # Validate configuration
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"Hidden dimension ({hidden_channels}) must be divisible by number of heads ({num_heads})"
            )

        # Learnable embeddings for each node type
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)
        self.reaction_embedding = nn.Embedding(reaction_num, hidden_channels)
        self.metabolite_embedding = nn.Embedding(metabolite_num, hidden_channels)

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
        node_types: Set[str] = {"gene", "reaction", "metabolite"}
        edge_types: Set[Tuple[str, str, str]] = {
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
            ("gene", "gpr", "reaction"),
            ("metabolite", "reaction", "metabolite"),
        }

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
        norm: Optional[str] = None,
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
        """
        Process a single graph or batch of graphs through the model.

        Args:
            data: HeteroData object or batched HeteroData

        Returns:
            torch.Tensor: Gene node embeddings after processing
        """
        device = self.gene_embedding.weight.device

        # Verify required mask attributes for attention
        missing_masks = []
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            if (
                src != dst
                and "adj_mask" not in data[edge_type]
                and "inc_mask" not in data[edge_type]
            ):
                missing_masks.append(edge_type)

        if missing_masks:
            raise ValueError(
                f"Missing mask attributes in data for edge types: {missing_masks}. "
                f"HeteroNSA requires adj_mask or inc_mask attributes for attention. "
                f"Ensure dense mask transform has been applied."
            )

        # Initialize embeddings based on whether we're processing a batch or single graph
        is_batch = isinstance(data, Batch) or hasattr(data["gene"], "batch")

        if is_batch:
            gene_data = data["gene"]
            reaction_data = data["reaction"]
            metabolite_data = data["metabolite"]

            # Determine batch size based on available batch information
            batch_size = (
                len(gene_data.ptr) - 1
                if hasattr(gene_data, "ptr")
                else int(gene_data.batch.max()) + 1
            )

            # Process genes with perturbation mask if available
            if hasattr(gene_data, "pert_mask") and hasattr(
                gene_data.pert_mask, "shape"
            ):
                # Using perturbation mask to select non-perturbed nodes
                x_gene_exp = self.gene_embedding.weight.expand(batch_size, -1, -1)
                x_gene_comb = x_gene_exp.reshape(-1, x_gene_exp.size(-1))
                x_gene = x_gene_comb[~gene_data.pert_mask]
            else:
                # Fallback to using batch assignment
                gene_x = torch.zeros(
                    gene_data.num_nodes, self.hidden_channels, device=device
                )
                for i in range(batch_size):
                    batch_mask = gene_data.batch == i
                    num_nodes = int(batch_mask.sum())
                    gene_x[batch_mask] = self.gene_embedding.weight[:num_nodes]
                x_gene = gene_x

            # Apply preprocessor to gene features
            x_gene = self.preprocessor(x_gene)

            # Process reaction nodes with perturbation mask if available
            if hasattr(reaction_data, "pert_mask") and hasattr(
                reaction_data.pert_mask, "shape"
            ):
                x_reaction_exp = self.reaction_embedding.weight.expand(
                    batch_size, -1, -1
                )
                x_reaction_comb = x_reaction_exp.reshape(-1, x_reaction_exp.size(-1))
                x_reaction = x_reaction_comb[~reaction_data.pert_mask]
            else:
                # Fallback to using batch assignment
                reaction_x = torch.zeros(
                    reaction_data.num_nodes, self.hidden_channels, device=device
                )
                for i in range(batch_size):
                    batch_mask = reaction_data.batch == i
                    num_nodes = int(batch_mask.sum())
                    # Ensure we don't exceed embedding dimensions
                    num_nodes = min(num_nodes, self.reaction_num)
                    reaction_x[batch_mask] = self.reaction_embedding.weight[:num_nodes]
                x_reaction = reaction_x

            # Process metabolite nodes with perturbation mask if available
            if hasattr(metabolite_data, "pert_mask") and hasattr(
                metabolite_data.pert_mask, "shape"
            ):
                x_metabolite_exp = self.metabolite_embedding.weight.expand(
                    batch_size, -1, -1
                )
                x_metabolite_comb = x_metabolite_exp.reshape(
                    -1, x_metabolite_exp.size(-1)
                )
                x_metabolite = x_metabolite_comb[~metabolite_data.pert_mask]
            else:
                # Fallback to using batch assignment
                metabolite_x = torch.zeros(
                    metabolite_data.num_nodes, self.hidden_channels, device=device
                )
                for i in range(batch_size):
                    batch_mask = metabolite_data.batch == i
                    num_nodes = int(batch_mask.sum())
                    # Ensure we don't exceed embedding dimensions
                    num_nodes = min(num_nodes, self.metabolite_num)
                    metabolite_x[batch_mask] = self.metabolite_embedding.weight[
                        :num_nodes
                    ]
                x_metabolite = metabolite_x
        else:
            # For a single graph, use embeddings directly with proper slicing
            gene_count = min(data["gene"].num_nodes, self.gene_num)
            reaction_count = min(data["reaction"].num_nodes, self.reaction_num)
            metabolite_count = min(data["metabolite"].num_nodes, self.metabolite_num)

            x_gene = self.preprocessor(self.gene_embedding.weight[:gene_count])
            x_reaction = self.reaction_embedding.weight[:reaction_count]
            x_metabolite = self.metabolite_embedding.weight[:metabolite_count]

        # Combine all node embeddings
        x_dict = {"gene": x_gene, "reaction": x_reaction, "metabolite": x_metabolite}

        # Prepare batch assignment information if available
        batch_idx = {}
        if is_batch:
            for node_type in ["gene", "reaction", "metabolite"]:
                if hasattr(data[node_type], "batch"):
                    batch_idx[node_type] = data[node_type].batch

        # Process through NSA layer with residual connection
        new_x = self.nsa_layer(x_dict, data, batch_idx)

        # Apply layer norm and residual connection
        for node_type in new_x:
            if node_type in x_dict:
                new_x[node_type] = self.layer_norms[node_type](
                    new_x[node_type] + x_dict[node_type]
                )

        # Return gene embeddings for downstream tasks
        return new_x["gene"]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData | Batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        # Force predictions to be exactly [batch_size, 2]
        predictions = predictions.view(batch_size, 2)

        # Extract individual predictions
        fitness = predictions[:, 0:1]
        gene_interaction = predictions[:, 1:2]

        # Print shapes to debug
        # print(f"Final shapes: z_w={z_w.shape}, z_i={z_i.shape}, z_p={z_p.shape}, pred={predictions.shape}")

        return predictions, {
            "z_w": z_w,
            "z_i": z_i,
            "z_p": z_p,  # This should now be [batch_size, hidden_dim]
            "fitness": fitness,
            "gene_interaction": gene_interaction,
        }

    @property
    def num_parameters(self) -> Dict[str, int]:
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


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/003-fit-int/conf"),
    config_name="hetero_cell_nsa",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.timestamp import timestamp
    from torchcell.scratch.load_batch import load_sample_data_batch
    from torchcell.scratch.cell_batch_overfit_visualization import (
        plot_embeddings,
        plot_correlations,
    )
    from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data with dense mask transformation
    dataset, batch, _, _ = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        metabolism_graph="metabolism_bipartite",
        is_dense=True,  # This applies dense transformation to the dataset
    )

    # Get the cell_graph from the dataset
    cell_graph = dataset.cell_graph

    # Verify cell_graph has the necessary dense mask attributes
    for edge_type in cell_graph.edge_types:
        src, rel, dst = edge_type
        if src != dst:  # Skip self-loops
            if (
                "adj_mask" not in cell_graph[edge_type]
                and "inc_mask" not in cell_graph[edge_type]
            ):
                print(
                    f"Warning: Cell graph edge type {edge_type} missing dense mask attributes."
                )
                # Apply dense transformation explicitly if needed
                dense_transform = HeteroToDenseMask(
                    {
                        "gene": cfg.model.gene_num,
                        "reaction": cfg.model.reaction_num,
                        "metabolite": cfg.model.metabolite_num,
                    }
                )
                cell_graph = dense_transform(cell_graph)
                print("Applied dense mask transformation to cell_graph.")
                break

    # Verify batch also has mask attributes
    for edge_type in batch.edge_types:
        src, rel, dst = edge_type
        if src != dst:  # Skip self-loops
            if (
                "adj_mask" not in batch[edge_type]
                and "inc_mask" not in batch[edge_type]
            ):
                print(
                    f"Warning: Batch edge type {edge_type} missing dense mask attributes!"
                )

    # Move to device
    cell_graph = cell_graph.to(device)
    batch = batch.to(device)

    # Print dimensions for verification
    print("\nVerifying data dimensions:")
    print(f"Cell graph dimensions:")
    for node_type in cell_graph.node_types:
        print(f"  {node_type}: {cell_graph[node_type].num_nodes} nodes")

    print(f"Batch dimensions:")
    for node_type in batch.node_types:
        print(f"  {node_type}: {batch[node_type].num_nodes} nodes")

    print("\nModel configuration:")
    print(f"  gene_num: {cfg.model.gene_num}")
    print(f"  reaction_num: {cfg.model.reaction_num}")
    print(f"  metabolite_num: {cfg.model.metabolite_num}")
    print(f"  hidden_channels: {cfg.model.hidden_channels}")
    print(f"  attention_pattern: {cfg.model.attention_pattern}")

    # Initialize model with verified dimensions
    model = HeteroCellNSA(
        gene_num=cfg.model.gene_num,
        reaction_num=cfg.model.reaction_num,
        metabolite_num=cfg.model.metabolite_num,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        attention_pattern=cfg.model.attention_pattern,
        num_heads=cfg.model.gene_encoder_config.heads,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        prediction_head_config=cfg.model.prediction_head_config,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass before training
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            predictions, representations = model(cell_graph, batch)
            print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return  # Exit if forward pass fails

    # Set up loss and optimizer
    fit_nan_count = batch["gene"].fitness.isnan().sum()
    gi_nan_count = batch["gene"].gene_interaction.isnan().sum()
    total_samples = len(batch["gene"].fitness) * 2
    weights = torch.tensor(
        [1 - (gi_nan_count / total_samples), 1 - (fit_nan_count / total_samples)]
    ).to(device)

    criterion = ICLoss(
        lambda_dist=cfg.regression_task.lambda_dist,
        lambda_supcr=cfg.regression_task.lambda_supcr,
        weights=weights,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Setup directories for plots
    embeddings_dir = osp.join(ASSET_IMAGES_DIR, "embedding_plots")
    os.makedirs(embeddings_dir, exist_ok=True)
    correlation_dir = osp.join(ASSET_IMAGES_DIR, "correlation_plots")
    os.makedirs(correlation_dir, exist_ok=True)

    # Training preparation
    model.train()
    losses = []
    num_epochs = cfg.trainer.max_epochs

    # Initial visualization
    with torch.no_grad():
        predictions, representations = model(cell_graph, batch)
        z_w_np = representations["z_w"].detach().cpu().numpy()
        z_i_np = representations["z_i"].detach().cpu().numpy()
        z_p_np = representations["z_p"].detach().cpu().numpy()
        embedding_fixed_axes = {
            "value_min": min(z_w_np.min(), z_i_np.min(), z_p_np.min()),
            "value_max": max(z_w_np.max(), z_i_np.max(), z_p_np.max()),
            "dim_max": representations["z_w"].shape[1] - 1,
            "z_i_min": z_i_np.min(),
            "z_i_max": z_i_np.max(),
            "z_p_min": z_p_np.min(),
            "z_p_max": z_p_np.max(),
        }
        init_epoch = 0
        correlation_save_path = osp.join(
            correlation_dir, f"correlation_plots_epoch{init_epoch:03d}.png"
        )
        correlation_fixed_axes = plot_correlations(
            predictions.cpu(),
            y.cpu(),
            correlation_save_path,
            lambda_info=f"λ_dist={cfg.regression_task.lambda_dist}, "
            f"λ_supcr={cfg.regression_task.lambda_supcr}",
            weight_decay=cfg.regression_task.optimizer.weight_decay,
            fixed_axes=None,
            epoch=init_epoch,
        )

    # Training loop
    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions, representations = model(cell_graph, batch)
            loss, loss_components = criterion(predictions, y, representations["z_p"])

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print("Loss components:", loss_components)
                try:
                    embedding_fixed_axes = plot_embeddings(
                        representations["z_w"].expand(predictions.size(0), -1),
                        representations["z_i"],
                        representations["z_p"],
                        batch_size=predictions.size(0),
                        save_dir=embeddings_dir,
                        epoch=epoch,
                        fixed_axes=embedding_fixed_axes,
                    )
                    correlation_save_path = osp.join(
                        correlation_dir, f"correlation_plots_epoch{epoch:03d}.png"
                    )
                    plot_correlations(
                        predictions.cpu(),
                        y.cpu(),
                        correlation_save_path,
                        lambda_info=f"λ_dist={cfg.regression_task.lambda_dist}, "
                        f"λ_supcr={cfg.regression_task.lambda_supcr}",
                        weight_decay=cfg.regression_task.optimizer.weight_decay,
                        fixed_axes=correlation_fixed_axes,
                    )
                except Exception as e:
                    print(f"Warning: Plot generation failed: {e}")
                if device.type == "cuda":
                    print(
                        f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB"
                    )
                    print(
                        f"GPU memory reserved: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB"
                    )

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    except RuntimeError as e:
        print(f"\nError during training: {e}")
        if device.type == "cuda":
            print(
                "GPU memory may be insufficient. Consider reducing batch size or model size."
            )
        raise

    # Plot final training loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses) + 1), losses, "b-", label="ICLoss Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(
        f"Training Loss Over Time: λ_dist={cfg.regression_task.lambda_dist}, "
        f"λ_supcr={cfg.regression_task.lambda_supcr}, "
        f"wd={cfg.regression_task.optimizer.weight_decay}"
    )
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        osp.join(ASSET_IMAGES_DIR, f"hetero_cell_nsa_training_loss_{timestamp()}.png")
    )
    plt.close()

    # Clean up GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
