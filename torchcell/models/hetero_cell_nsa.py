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
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


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
        self,
        x: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.aggregator(x, index=index, dim_size=dim_size)


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
        act = act_register[activation]
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers = [
            nn.Linear(in_channels, hidden_channels),
            norm_layer,
            act,
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    norm_layer,
                    act,
                    nn.Dropout(dropout),
                ]
            )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class HeteroCellNSA(nn.Module):
    """
    Heterogeneous Cell Model using Node-Set Attention (NSA) blocks.
    It replaces HeteroConv with interleaved Self-Attention (SAB) and
    Masked-Attention (MAB) blocks. Input HeteroData must be pre-transformed
    with DenseMask so that boolean masks (e.g. adj_mask, inc_mask) are available.
    """

    def __init__(
        self,
        gene_num: int,
        reaction_num: int,
        metabolite_num: int,
        hidden_channels: int,
        out_channels: int,
        attention_pattern: List[str] = ["S", "M", "S", "M"],
        num_heads: int = 8,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        prediction_head_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"Hidden dimension ({hidden_channels}) must be divisible by "
                f"number of attention heads ({num_heads})"
            )

        # Define node and edge types as in the data.
        self.node_types: Set[str] = {"gene", "reaction", "metabolite"}
        self.edge_types: Set[Tuple[str, str, str]] = {
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
            ("gene", "gpr", "reaction"),
            ("reaction", "rmr", "metabolite"),
        }
        # Pass the attention pattern list directly
        self.pattern: List[str] = attention_pattern

        # Learnable node embeddings
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)
        self.reaction_embedding = nn.Embedding(reaction_num, hidden_channels)
        self.metabolite_embedding = nn.Embedding(metabolite_num, hidden_channels)

        # Preprocessing for gene features (you may extend this if needed)
        self.preprocessor = PreProcessor(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        # Create a stack (here one layer; adjust num_layers as needed)
        self.nsa_layers = nn.ModuleList()
        for _ in range(1):
            self.nsa_layers.append(
                HeteroNSA(
                    hidden_dim=hidden_channels,
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    pattern=self.pattern,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=self._get_activation(activation),
                    aggregation="sum",
                )
            )

        # Layer norms for residual connections
        self.layer_norms = nn.ModuleDict(
            {
                node_type: nn.ModuleList(
                    [
                        get_norm_layer(hidden_channels, norm)
                        for _ in range(len(self.nsa_layers))
                    ]
                )
                for node_type in self.node_types
            }
        )

        # Global aggregator for graph-level embedding
        self.global_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

        # Build prediction head
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

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            return nn.ReLU()

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
        if num_layers == 0:
            return nn.Identity()
        act = self._get_activation(activation)
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

    def forward_single(self, data: HeteroData | Batch) -> Dict[str, torch.Tensor]:
        device = self.gene_embedding.weight.device
        # Use all nodes from each type (assuming DenseMask has padded as needed)
        gene_idx = (
            torch.arange(data["gene"].num_nodes, device=device)
            % self.gene_embedding.num_embeddings
        )
        reaction_idx = (
            torch.arange(data["reaction"].num_nodes, device=device)
            % self.reaction_embedding.num_embeddings
        )
        metabolite_idx = (
            torch.arange(data["metabolite"].num_nodes, device=device)
            % self.metabolite_embedding.num_embeddings
        )

        x_dict = {
            "gene": self.preprocessor(self.gene_embedding(gene_idx)),
            "reaction": self.reaction_embedding(reaction_idx),
            "metabolite": self.metabolite_embedding(metabolite_idx),
        }

        # Collect batch indices if available
        batch_idx: Dict[str, torch.Tensor] = {}
        if isinstance(data, Batch) or hasattr(data["gene"], "batch"):
            for node_type in self.node_types:
                if node_type in data and hasattr(data[node_type], "batch"):
                    batch_idx[node_type] = data[node_type].batch

        # Process NSA layers with residual connections
        for i, layer in enumerate(self.nsa_layers):
            try:
                new_x_dict = layer(x_dict, data, batch_idx)
                for node_type in new_x_dict:
                    if node_type in x_dict:
                        new_x_dict[node_type] = self.layer_norms[node_type][i](
                            new_x_dict[node_type] + x_dict[node_type]
                        )
                x_dict = new_x_dict
            except Exception as e:
                print(f"Error in NSA layer {i}: {e}")
        return x_dict

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for the HeteroCellNSA model with explicit reshaping."""
        # Process reference and perturbed graphs
        z_w_dict = self.forward_single(cell_graph)
        z_w_genes = z_w_dict["gene"]
        z_i_dict = self.forward_single(batch)
        z_i_genes = z_i_dict["gene"]

        # Determine batch size from batch data
        batch_size = (
            batch["gene"].batch.max().item() + 1
            if hasattr(batch["gene"], "batch")
            else 1
        )

        # Global pooling for reference graph
        z_w = self.global_aggregator(
            z_w_genes,
            index=torch.zeros(
                z_w_genes.size(0), device=z_w_genes.device, dtype=torch.long
            ),
            dim_size=1,
        )

        # Global pooling for perturbed graph(s)
        if hasattr(batch["gene"], "batch"):
            batch_idx = batch["gene"].batch
            z_i = self.global_aggregator(
                z_i_genes, index=batch_idx, dim_size=batch_size
            )
        else:
            z_i = self.global_aggregator(
                z_i_genes,
                index=torch.zeros(
                    z_i_genes.size(0), device=z_i_genes.device, dtype=torch.long
                ),
                dim_size=1,
            )

        # Reshape both tensors to ensure correct dimensions before operations
        # Force z_w to be [1, hidden_dim]
        if z_w.dim() > 2:
            z_w = z_w.reshape(1, -1)

        # Force z_i to be [batch_size, hidden_dim]
        if z_i.dim() > 2:
            z_i = z_i.reshape(batch_size, -1)

        # Expand reference and compute perturbation
        z_w_exp = z_w.expand(batch_size, -1)

        # Critical fix: explicitly reshape z_p to be exactly [batch_size, hidden_dim]
        z_p = (z_w_exp - z_i).reshape(batch_size, -1)

        # Generate predictions and ensure correct shape
        predictions = self.prediction_head(z_p)

        # Handle any remaining dimension issues in predictions
        if predictions.dim() > 2:
            predictions = predictions.reshape(batch_size, -1)

        # Split predictions
        fitness = predictions[:, 0:1]
        gene_interaction = predictions[:, 1:2]

        return predictions, {
            "z_w": z_w,
            "z_i": z_i,
            "z_p": z_p,  # This is now explicitly [batch_size, hidden_dim]
            "fitness": fitness,
            "gene_interaction": gene_interaction,
        }

    @property
    def num_parameters(self) -> Dict[str, int]:
        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "reaction_embedding": count_params(self.reaction_embedding),
            "metabolite_embedding": count_params(self.metabolite_embedding),
            "preprocessor": count_params(self.preprocessor),
            "nsa_layers": count_params(self.nsa_layers),
            "layer_norms": sum(
                count_params(ln)
                for node_lns in self.layer_norms.values()
                for ln in node_lns
            ),
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

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data (assumes DenseMask has been applied)
    dataset, batch, _, _ = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        metabolism_graph="metabolism_hypergraph",
    )
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

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
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

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

    embeddings_dir = osp.join(ASSET_IMAGES_DIR, "embedding_plots")
    os.makedirs(embeddings_dir, exist_ok=True)
    correlation_dir = osp.join(ASSET_IMAGES_DIR, "correlation_plots")
    os.makedirs(correlation_dir, exist_ok=True)

    model.train()
    losses = []
    num_epochs = cfg.trainer.max_epochs

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

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
