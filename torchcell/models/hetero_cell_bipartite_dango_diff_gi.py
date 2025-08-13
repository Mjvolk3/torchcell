# torchcell/models/hetero_cell_bipartite_dango_diff_gi
# [[torchcell.models.hetero_cell_bipartite_dango_diff_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_bipartite_dango_diff_gi
# Test file: tests/torchcell/models/test_hetero_cell_bipartite_dango_diff_gi.py

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData
from torchcell.models.hetero_cell_bipartite_dango_gi import GeneInteractionDango
from torchcell.models.diffusion_decoder import DiffusionDecoder


class LinearDecoder(nn.Module):
    """Simple linear decoder for baseline comparison.
    
    Maps combined embeddings directly to phenotype predictions.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z_c: Combined embeddings [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        return self.proj(z_c)
    
    def sample(self, context: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample method for compatibility with diffusion decoder interface.
        
        For linear decoder, this is just a forward pass.
        """
        return self.forward(context)


class GeneInteractionDiff(GeneInteractionDango):
    """Gene interaction model with diffusion-based prediction head.

    This model inherits from GeneInteractionDango to reuse the entire encoder pipeline
    (graph processing, convolutions, aggregation) while replacing only the final
    prediction head with a diffusion decoder. This ensures consistency in feature
    extraction between the deterministic and diffusion versions of the model.

    Key inherited components:
    - Gene embeddings and preprocessing
    - Graph convolution layers
    - Global and local aggregation
    - forward_single() for processing graphs

    What's replaced:
    - The MLP prediction head (global_interaction_predictor) is removed
    - A diffusion decoder with cross-attention is used instead
    """

    def __init__(
        self,
        gene_num: int,
        hidden_channels: int,
        num_layers: int,
        gene_multigraph: HeteroData,
        dropout: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        gene_encoder_config: Optional[Dict] = None,
        local_predictor_config: Optional[Dict] = None,
        diffusion_config: Optional[Dict] = None,
        decoder_type: str = "diffusion",  # Add decoder type parameter
    ):
        """Initialize GeneInteractionDiff model.

        Args:
            gene_num: Number of genes
            hidden_channels: Hidden dimension size
            num_layers: Number of graph conv layers
            gene_multigraph: Gene interaction graph
            dropout: Dropout rate
            norm: Normalization type
            activation: Activation function
            gene_encoder_config: Config for gene encoder
            local_predictor_config: Config for local predictor
            diffusion_config: Config for diffusion decoder
        """
        # Initialize parent class to get all encoder components
        super().__init__(
            gene_num=gene_num,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            gene_multigraph=gene_multigraph,
            dropout=dropout,
            norm=norm,
            activation=activation,
            gene_encoder_config=gene_encoder_config,
            local_predictor_config=local_predictor_config,
        )

        # Remove the deterministic MLP prediction head - we'll use different decoder
        del self.global_interaction_predictor

        # Store decoder type
        self.decoder_type = decoder_type
        
        # Create decoder based on type
        if decoder_type == "linear":
            # Simple linear decoder for baseline testing
            self.decoder = LinearDecoder(
                input_dim=hidden_channels * 2,  # Concatenated embeddings
                output_dim=1,  # Single phenotype
            )
        elif decoder_type == "diffusion":
            # Parse diffusion config
            diffusion_config = diffusion_config or {}
            
            # Create diffusion decoder with all configuration options
            self.decoder = DiffusionDecoder(
                input_dim=hidden_channels * 2,  # Concatenated embeddings
                hidden_dim=diffusion_config.get("hidden_dim", hidden_channels),
                output_dim=1,  # Single phenotype
                num_layers=diffusion_config.get("num_layers", 4),
                num_heads=diffusion_config.get("num_heads", 8),
                dropout=diffusion_config.get("dropout", dropout),
                norm=norm,
                num_timesteps=diffusion_config.get("num_timesteps", 1000),
                # New parameters from config
                mlp_ratio=diffusion_config.get("mlp_ratio", 4.0),
                beta_schedule=diffusion_config.get("beta_schedule", "cosine"),
                beta_start=diffusion_config.get("beta_start", 0.0001),
                beta_end=diffusion_config.get("beta_end", 0.02),
                cosine_s=diffusion_config.get("cosine_s", 0.008),
                sampling_steps=diffusion_config.get("sampling_steps", 50),
                parameterization=diffusion_config.get("parameterization", "x0"),
            )
            # Keep alias for backward compatibility
            self.diffusion_decoder = self.decoder
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        self.training_mode = True

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            cell_graph: Full cell graph
            batch: Batch of perturbed genes

        Returns:
            predictions: Phenotype predictions [batch_size, 1]
            representations: Dictionary of intermediate representations
        """
        # Process reference graph (wildtype)
        z_w = self.forward_single(cell_graph)

        # Proper global aggregation for wildtype
        z_w_global = self.global_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )

        # Process perturbed batch
        z_i = self.forward_single(batch)

        # Proper global aggregation for perturbed genes
        z_i_global = self.global_aggregator(z_i, index=batch["gene"].batch)

        # Get embeddings of perturbed genes from wildtype
        pert_indices = batch["gene"].perturbation_indices
        pert_gene_embs_wt = z_w[pert_indices]

        # Determine batch assignment for perturbed genes
        if hasattr(batch["gene"], "perturbation_indices_ptr"):
            ptr = batch["gene"].perturbation_indices_ptr
            batch_assign = torch.zeros(
                pert_indices.size(0), dtype=torch.long, device=z_w.device
            )
            for i in range(len(ptr) - 1):
                batch_assign[ptr[i] : ptr[i + 1]] = i
        else:
            batch_assign = (
                batch["gene"].perturbation_indices_batch
                if hasattr(batch["gene"], "perturbation_indices_batch")
                else None
            )

        # For diffusion, we use the raw embeddings, not the predictions
        # Aggregate perturbed gene embeddings to batch level
        if batch_assign is not None:
            # Average perturbed gene embeddings per batch
            batch_size = z_i_global.size(0)
            pert_gene_embs_aggregated = torch.zeros(
                batch_size, self.hidden_channels, device=z_w.device
            )
            for i in range(batch_size):
                mask = batch_assign == i
                if mask.any():
                    pert_gene_embs_aggregated[i] = pert_gene_embs_wt[mask].mean(dim=0)
                else:
                    # Fallback if no genes for this batch
                    pert_gene_embs_aggregated[i] = pert_gene_embs_wt.mean(dim=0)
        else:
            # Single batch case
            pert_gene_embs_aggregated = pert_gene_embs_wt.mean(dim=0, keepdim=True)
            if pert_gene_embs_aggregated.size(0) != z_i_global.size(0):
                pert_gene_embs_aggregated = pert_gene_embs_aggregated.expand(
                    z_i_global.size(0), -1
                )

        # Concatenate global and local embeddings for conditioning
        z_c = torch.cat([z_i_global, pert_gene_embs_aggregated], dim=-1)  # z_c: combined latent

        # Handle forward pass based on decoder type
        if self.decoder_type == "linear":
            # Linear decoder directly computes predictions
            predictions = self.decoder(z_c)
        elif self.decoder_type == "diffusion":
            # Diffusion decoder behavior
            if self.training:
                # During training, return dummy predictions
                # The diffusion loss will handle the actual training internally
                # via compute_diffusion_loss method
                if hasattr(batch["gene"], "phenotype_values"):
                    # Get ground truth phenotype values just for shape
                    y_true = batch["gene"].phenotype_values
                    if y_true.dim() == 0:
                        y_true = y_true.unsqueeze(0).unsqueeze(0)
                    elif y_true.dim() == 1:
                        y_true = y_true.unsqueeze(1)
                    # Return zeros of the same shape as targets
                    predictions = torch.zeros_like(y_true)
                else:
                    # If no ground truth, just return zeros
                    batch_size = z_i_global.shape[0]
                    predictions = torch.zeros(batch_size, 1, device=z_i_global.device)
            else:
                # During inference, sample from the diffusion model
                predictions = self.decoder.sample(z_c)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        # Build representations dictionary
        representations = {
            "z_i_global": z_i_global,
            "z_i": z_i_global,  # Alias for compatibility
            "z_w": z_w_global,  # Add wildtype global
            "pert_gene_embs": pert_gene_embs_aggregated,
            "z_c": z_c,  # Combined latent embeddings
            "combined_embeddings": z_c,  # Alias for compatibility
            "z_p": z_c,  # Alias for backward compatibility with training code
        }

        # Gate weights are not needed for diffusion model

        return predictions, representations

    def compute_diffusion_loss(
        self, y_true: torch.Tensor, z_c: torch.Tensor, t_mode: str = "random"
    ) -> torch.Tensor:
        """Compute the training loss based on decoder type.

        Args:
            y_true: Ground truth phenotype values [batch_size, 1]
            z_c: Combined latent embeddings for conditioning [batch_size, hidden_dim * 2]
            t_mode: Timestep sampling mode ("zero", "partial", "full") - only used for diffusion

        Returns:
            Loss value
        """
        if self.decoder_type == "linear":
            # For linear decoder, compute simple MSE loss
            predictions = self.decoder(z_c)
            return F.mse_loss(predictions, y_true)
        elif self.decoder_type == "diffusion":
            # For diffusion decoder, use diffusion loss
            return self.decoder.loss(y_true, z_c, t_mode=t_mode)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

    def sample(
        self,
        cell_graph: HeteroData,
        batch: HeteroData,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample phenotype predictions using the diffusion model.

        Note: This method performs a fresh forward pass through the encoder
        to get new embeddings, ensuring we don't reuse stale representations
        from training.

        Args:
            cell_graph: Full cell graph
            batch: Batch of perturbed genes
            num_samples: Number of samples to generate

        Returns:
            Sampled phenotype predictions [batch_size, 1]
        """
        # Get fresh embeddings with a new forward pass
        with torch.no_grad():
            _, representations = self.forward(cell_graph, batch)
            z_c = representations["z_c"]

        # Sample from decoder
        if self.decoder_type == "linear":
            # Linear decoder doesn't sample, just returns prediction
            return self.decoder(z_c)
        else:
            # Diffusion decoder samples
            return self.decoder.sample(z_c, num_samples)

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component."""

        # Get parent class parameter counts
        def count_params(module: torch.nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        param_counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "preprocessor": count_params(self.preprocessor),
            "convs": count_params(self.convs),
            "gene_interaction_predictor": count_params(self.gene_interaction_predictor),
            "global_aggregator": count_params(self.global_aggregator),
            "decoder": count_params(self.decoder),
        }

        # Only count gate_mlp if it exists
        if hasattr(self, "gate_mlp") and self.gate_mlp is not None:
            param_counts["gate_mlp"] = count_params(self.gate_mlp)

        param_counts["total"] = sum(param_counts.values())
        return param_counts


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="hetero_cell_bipartite_dango_diff_gi",
)
def main(cfg: DictConfig) -> None:
    """Test the model with overfitting on a single batch using Hydra config."""
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from torch_geometric.loader import DataLoader
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.data import Neo4jCellDataset
    from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
    from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
    from torchcell.losses.diffusion_loss import DiffusionLoss
    from torchcell.losses.logcosh import LogCoshLoss
    from torchcell.timestamp import timestamp

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    print("Testing GeneInteractionDiff model with overfitting...")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup genome and graph
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph using config
    graph_names = wandb_cfg["cell_dataset"]["graphs"]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Build node embeddings
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=wandb_cfg["cell_dataset"]["node_embeddings"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )

    # Setup dataset
    print("Loading dataset...")
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs={},
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
        transform=None,
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Get a single batch for overfitting using config values
    batch_size = wandb_cfg["data_module"]["batch_size"]
    num_workers = wandb_cfg["data_module"]["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Use the actual batch size from config
        shuffle=False,
        num_workers=num_workers,
        pin_memory=wandb_cfg["data_module"]["pin_memory"],
    )
    batch = next(iter(loader))
    batch = batch.to(device)

    # Get cell graph
    cell_graph = dataset.cell_graph.to(device)

    # Setup gene encoder config
    gene_encoder_config = dict(wandb_cfg["model"]["gene_encoder_config"])
    if any("learnable" in emb for emb in wandb_cfg["cell_dataset"]["node_embeddings"]):
        gene_encoder_config.update(
            {
                "embedding_type": "learnable",
                "max_num_nodes": cell_graph["gene"].num_nodes,
                "learnable_embedding_input_channels": wandb_cfg["cell_dataset"][
                    "learnable_embedding_input_channels"
                ],
            }
        )

    # Get local predictor and diffusion config
    local_predictor_config = dict(wandb_cfg["model"].get("local_predictor_config", {}))
    diffusion_config = dict(wandb_cfg["model"].get("diffusion_config", {}))

    # Create model using config
    print("\nInitializing GeneInteractionDiff model...")
    model = GeneInteractionDiff(
        gene_num=wandb_cfg["model"]["gene_num"],
        hidden_channels=wandb_cfg["model"]["hidden_channels"],
        num_layers=wandb_cfg["model"]["num_layers"],
        gene_multigraph=gene_multigraph,
        dropout=wandb_cfg["model"]["dropout"],
        norm=wandb_cfg["model"]["norm"],
        activation=wandb_cfg["model"]["activation"],
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=local_predictor_config,
        diffusion_config=diffusion_config,
    ).to(device)

    # Print model info
    param_counts = model.num_parameters
    print("\nModel parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")

    # Setup loss based on config
    if wandb_cfg["regression_task"]["loss"] == "diffusion":
        loss_func = DiffusionLoss(
            model=model,
            lambda_diffusion=wandb_cfg["regression_task"].get("lambda_diffusion", 1.0),
        )
    elif wandb_cfg["regression_task"]["loss"] == "logcosh":
        loss_func = LogCoshLoss(reduction="mean")
    else:
        # Default to MSE for testing
        loss_func = nn.MSELoss()

    # Setup optimizer from config
    optimizer_config = wandb_cfg["regression_task"]["optimizer"]
    if optimizer_config["type"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["lr"])

    # Setup directory for plots
    plot_dir = osp.join(
        ASSET_IMAGES_DIR, f"gene_interaction_diff_training_{timestamp()}"
    )
    os.makedirs(plot_dir, exist_ok=True)

    # Multi-sample evaluation function
    def evaluate_with_uncertainty(model, cell_graph, batch, num_samples=10):
        """Evaluate model with multiple samples for uncertainty estimation."""
        model.eval()
        with torch.no_grad():
            # Get embeddings once
            predictions, representations = model(cell_graph, batch)
            z_c = representations.get("z_c")
            
            # Sample multiple times
            samples = []
            for _ in range(num_samples):
                pred = model.diffusion_decoder.sample(context=z_c, num_samples=z_c.shape[0])
                samples.append(pred)
            
            samples = torch.stack(samples, dim=0)  # [num_samples, batch, 1]
            
            # Compute statistics
            mean_pred = samples.mean(dim=0)
            std_pred = samples.std(dim=0)
            
        model.train()
        return mean_pred, std_pred, samples

    # Uncertainty visualization function
    def plot_predictions_with_uncertainty(targets, predictions, std, epoch, save_dir):
        """Plot predictions with error bars showing uncertainty."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        x = targets.cpu().numpy().flatten()
        y = predictions.cpu().numpy().flatten()
        yerr = std.cpu().numpy().flatten()
        
        # Scatter with error bars
        ax.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.6, 
                    capsize=3, capthick=1, elinewidth=1)
        
        # Add diagonal line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', alpha=0.5, label='Perfect prediction')
        
        # Calculate correlation
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 1:
            r = np.corrcoef(x[mask], y[mask])[0, 1]
        else:
            r = 0.0
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values (mean ± std)')
        ax.set_title(f'Epoch {epoch}: Predictions with Uncertainty (r={r:.3f})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(osp.join(save_dir, f'uncertainty_epoch_{epoch}.png'))
        plt.close()

    # Training loop for overfitting with plotting
    print("\nStarting overfitting test on single batch...")
    model.train()
    num_epochs = wandb_cfg["trainer"]["max_epochs"]
    plot_interval = wandb_cfg["regression_task"]["plot_every_n_epochs"]

    # Initialize tracking lists
    losses = []
    correlations = []
    spearman_correlations = []
    mses = []
    maes = []
    rmses = []
    diffusion_losses = []
    x0_mses = []  # Track x0 prediction MSE

    # Get ground truth once
    y_true = batch["gene"].phenotype_values
    if y_true.dim() == 0:
        y_true = y_true.unsqueeze(0).unsqueeze(0)
    elif y_true.dim() == 1:
        y_true = y_true.unsqueeze(1)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        predictions, representations = model(cell_graph, batch)

        # Compute loss
        if wandb_cfg["regression_task"]["loss"] == "diffusion":
            z_c = representations.get("z_c")
            
            # Get t_mode from config (default to "full" if not specified)
            t_mode = wandb_cfg["regression_task"].get("t_mode", "full")
            
            loss, loss_dict = loss_func(predictions, y_true, z_c, t_mode=t_mode)
            if "diffusion_loss" in loss_dict:
                diffusion_losses.append(loss_dict["diffusion_loss"].item())
                
            # Add training-time x0 metric for monitoring
            with torch.no_grad():
                # Sample random timestep
                t = torch.randint(0, model.diffusion_decoder.num_timesteps, (batch_size,), device=device)
                noise = torch.randn_like(y_true)
                
                # Add noise
                x_t, _ = model.diffusion_decoder.forward_diffusion(y_true, t, noise)
                
                # Predict x0 directly
                x0_pred = model.diffusion_decoder.denoise(x_t, z_c, t, predict_x0=True)
                
                # Compute MSE
                x0_mse = F.mse_loss(x0_pred, y_true).item()
                x0_mses.append(x0_mse)
        else:
            loss = loss_func(predictions, y_true)
            loss_dict = {}

        # Calculate metrics from sampled predictions for diffusion models
        if wandb_cfg["regression_task"]["loss"] == "diffusion":
            # For diffusion models, compute metrics from sampled predictions
            model.eval()
            with torch.no_grad():
                # Use fresh forward pass for sampling to avoid stale representations
                sampled_pred = model.sample(cell_graph, batch)
                pred_np = sampled_pred.squeeze().cpu().numpy()
                target_np = y_true.squeeze().cpu().numpy()

                # Handle scalar values
                if pred_np.ndim == 0:
                    pred_np = np.array([pred_np])
                if target_np.ndim == 0:
                    target_np = np.array([target_np])

                valid_mask = ~np.isnan(target_np)

                if np.sum(valid_mask) > 1:
                    correlation = np.corrcoef(
                        pred_np[valid_mask], target_np[valid_mask]
                    )[0, 1]
                    spearman_corr, _ = stats.spearmanr(
                        pred_np[valid_mask], target_np[valid_mask]
                    )

                    # Debug: Check if predictions are constant or have very low variance
                    pred_std = np.std(pred_np[valid_mask])
                    target_std = np.std(target_np[valid_mask])
                    if epoch % 10 == 0:  # Print every 10 epochs
                        print(
                            f"\n  Debug - Pred std: {pred_std:.6f}, Target std: {target_std:.6f}"
                        )
                        print(
                            f"  Pred range: [{pred_np[valid_mask].min():.4f}, {pred_np[valid_mask].max():.4f}]"
                        )
                        print(
                            f"  Target range: [{target_np[valid_mask].min():.4f}, {target_np[valid_mask].max():.4f}]"
                        )
                        print(f"  Sample predictions: {pred_np[valid_mask][:5]}")
                        print(f"  Sample targets: {target_np[valid_mask][:5]}")
                else:
                    correlation = 0.0
                    spearman_corr = 0.0

                mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
                mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))
                rmse = np.sqrt(mse)
            model.train()
        else:
            # For non-diffusion models, use training predictions
            with torch.no_grad():
                pred_np = predictions.squeeze().cpu().numpy()
                target_np = y_true.squeeze().cpu().numpy()

                # Handle scalar values
                if pred_np.ndim == 0:
                    pred_np = np.array([pred_np])
                if target_np.ndim == 0:
                    target_np = np.array([target_np])

                valid_mask = ~np.isnan(target_np)

                if np.sum(valid_mask) > 1:
                    correlation = np.corrcoef(
                        pred_np[valid_mask], target_np[valid_mask]
                    )[0, 1]
                    spearman_corr, _ = stats.spearmanr(
                        pred_np[valid_mask], target_np[valid_mask]
                    )
                else:
                    correlation = 0.0
                    spearman_corr = 0.0

                mse = np.mean((pred_np[valid_mask] - target_np[valid_mask]) ** 2)
                mae = np.mean(np.abs(pred_np[valid_mask] - target_np[valid_mask]))
                rmse = np.sqrt(mse)

        losses.append(loss.item())
        correlations.append(correlation if not np.isnan(correlation) else 0.0)
        spearman_correlations.append(
            spearman_corr if not np.isnan(spearman_corr) else 0.0
        )
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)

        # Backward pass
        loss.backward()

        # Gradient clipping if configured
        if wandb_cfg["regression_task"].get("clip_grad_norm", False):
            max_norm = wandb_cfg["regression_task"].get("clip_grad_norm_max_norm", 1.0)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        # Print and plot progress
        if epoch % plot_interval == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {loss.item():.4f}", end="")
            if "diffusion_loss" in loss_dict:
                print(f" | Diff Loss: {loss_dict['diffusion_loss'].item():.4f}", end="")
            if wandb_cfg["regression_task"]["loss"] == "diffusion" and len(x0_mses) > 0:
                print(f" | x0_MSE: {x0_mses[-1]:.4f}", end="")
            print(f" | Pearson: {correlation:.4f} | MSE: {mse:.4f}")

            # Save the current pred_np and target_np for plotting
            # For diffusion models, pred_np already contains sampled predictions from above
            plot_pred_np = pred_np.copy()
            plot_target_np = target_np.copy()
            plot_valid_mask = valid_mask.copy()
            
            # Perform multi-sample evaluation for uncertainty estimation
            if wandb_cfg["regression_task"]["loss"] == "diffusion" and epoch > 0:
                mean_pred, std_pred, samples = evaluate_with_uncertainty(
                    model, cell_graph, batch, num_samples=10
                )
                # Create uncertainty plot
                plot_predictions_with_uncertainty(
                    y_true, mean_pred, std_pred, epoch, plot_dir
                )

            # Create intermediate plot with noise estimates
            if len(losses) > 1:
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))

                # Loss curve
                axes[0, 0].plot(range(1, len(losses) + 1), losses, "b-", linewidth=2)
                axes[0, 0].set_xlabel("Epoch")
                axes[0, 0].set_ylabel("Total Loss")
                axes[0, 0].set_title("Training Loss")
                axes[0, 0].grid(True)
                axes[0, 0].set_yscale("log")

                # Correlation evolution
                axes[0, 1].plot(
                    range(1, len(correlations) + 1),
                    correlations,
                    "g-",
                    label="Pearson",
                    linewidth=2,
                )
                axes[0, 1].plot(
                    range(1, len(spearman_correlations) + 1),
                    spearman_correlations,
                    "b--",
                    label="Spearman",
                    linewidth=2,
                )
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].set_ylabel("Correlation")
                axes[0, 1].set_title("Correlation Evolution")
                axes[0, 1].grid(True)
                axes[0, 1].legend()
                axes[0, 1].set_ylim(-0.1, 1.1)

                # Predictions scatter - use saved values (predictions on x-axis, true on y-axis)
                axes[0, 2].scatter(
                    plot_pred_np[plot_valid_mask],
                    plot_target_np[plot_valid_mask],
                    alpha=0.6,
                )
                min_val = min(
                    plot_target_np[plot_valid_mask].min(),
                    plot_pred_np[plot_valid_mask].min(),
                )
                max_val = max(
                    plot_target_np[plot_valid_mask].max(),
                    plot_pred_np[plot_valid_mask].max(),
                )
                axes[0, 2].plot(
                    [min_val, max_val], [min_val, max_val], "r--", linewidth=2
                )
                axes[0, 2].set_xlabel("Predicted Values")
                axes[0, 2].set_ylabel("True Values")
                axes[0, 2].set_title(f"Predictions (r={correlation:.3f})")
                axes[0, 2].grid(True)

                # MSE evolution
                axes[1, 0].plot(range(1, len(mses) + 1), mses, "r-", linewidth=2)
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("MSE")
                axes[1, 0].set_title("MSE Evolution")
                axes[1, 0].grid(True)
                axes[1, 0].set_yscale("log")

                # Loss components (for diffusion)
                if diffusion_losses or x0_mses:
                    if diffusion_losses:
                        axes[1, 1].plot(
                            range(1, len(diffusion_losses) + 1),
                            diffusion_losses,
                            "purple",
                            label="Diffusion Loss",
                            linewidth=2,
                        )
                    if x0_mses:
                        axes[1, 1].plot(
                            range(1, len(x0_mses) + 1),
                            x0_mses,
                            "cyan",
                            label="x0 MSE",
                            linewidth=2,
                            linestyle="--",
                        )
                    axes[1, 1].set_xlabel("Epoch")
                    axes[1, 1].set_ylabel("Loss")
                    axes[1, 1].set_title("Diffusion Training Metrics")
                    axes[1, 1].grid(True)
                    axes[1, 1].legend()
                    axes[1, 1].set_yscale("log")
                else:
                    # MAE for non-diffusion models
                    axes[1, 1].plot(
                        range(1, len(maes) + 1), maes, "orange", linewidth=2
                    )
                    axes[1, 1].set_xlabel("Epoch")
                    axes[1, 1].set_ylabel("MAE")
                    axes[1, 1].set_title("MAE Evolution")
                    axes[1, 1].grid(True)
                    axes[1, 1].set_yscale("log")

                # Model info
                axes[1, 2].axis("off")
                info_text = f"Model: GeneInteractionDiff\n"
                info_text += f"Parameters: {param_counts['total']:,}\n"
                info_text += (
                    f"Hidden channels: {wandb_cfg['model']['hidden_channels']}\n"
                )
                info_text += f"Diffusion timesteps: {wandb_cfg['model']['diffusion_config']['num_timesteps']}\n"
                info_text += f"Batch size: {batch_size}\n"
                info_text += f"Loss type: {wandb_cfg['regression_task']['loss']}\n"
                info_text += f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}\n"
                if wandb_cfg["regression_task"]["loss"] == "diffusion":
                    info_text += f"\n*All metrics from sampled predictions"
                axes[1, 2].text(
                    0.1,
                    0.5,
                    info_text,
                    fontsize=10,
                    transform=axes[1, 2].transAxes,
                    verticalalignment="center",
                )

                # Add conditioning check for diffusion model
                if wandb_cfg["regression_task"]["loss"] == "diffusion":
                    # Check if conditioning actually matters
                    model.eval()
                    with torch.no_grad():
                        # Get embeddings
                        _, reps = model(cell_graph, batch)
                        z_c = reps["z_c"]
                        
                        # Sample with real conditioning - use all batch samples
                        preds_on = model.diffusion_decoder.sample(
                            z_c,
                            num_samples=None  # Use batch size from context
                        )
                        
                        # Sample with zero conditioning (ignored)
                        preds_off = model.diffusion_decoder.sample(
                            torch.zeros_like(z_c),
                            num_samples=None  # Use batch size from context
                        )
                        
                        # Calculate difference
                        delta = (preds_on - preds_off).abs().mean().item()
                        delta_std = (preds_on - preds_off).std().item()
                        
                        # Track conditioning effect over time
                        if not hasattr(model, '_cond_deltas'):
                            model._cond_deltas = []
                        model._cond_deltas.append(delta)
                        
                        print(f"\n  [Cond-Check] |preds_on - preds_off| = {delta:.6f} (std: {delta_std:.6f})")
                        if delta < 1e-4:
                            print("  ⚠️ WARNING: Conditioning appears to be ignored!")
                        
                        # Prepare data for other plots
                        test_timesteps = [0, model.diffusion_decoder.num_timesteps // 2, model.diffusion_decoder.num_timesteps - 1]
                        x0_pred_all = []
                        
                        for t_val in test_timesteps:
                            t_batch = torch.full(
                                (y_true.shape[0],),
                                t_val,
                                device=y_true.device,
                                dtype=torch.long,
                            )
                            noise = torch.randn_like(y_true)
                            x_t, _ = model.diffusion_decoder.forward_diffusion(
                                y_true, t_batch, noise
                            )

                            # Get x0 prediction with combined embeddings
                            x0_pred = model.diffusion_decoder.denoise(
                                x_t,
                                z_c,
                                t_batch,
                                predict_x0=True,
                            )
                            
                            x0_pred_all.append(x0_pred.cpu().numpy())

                    model.train()
                    
                    # Plot conditioning effect over epochs
                    if hasattr(model, '_cond_deltas') and len(model._cond_deltas) > 0:
                        cond_epochs = range(max(0, epoch - len(model._cond_deltas) + 1), epoch + 1)
                        axes[2, 0].plot(
                            list(cond_epochs),
                            model._cond_deltas,
                            "g-",
                            linewidth=2,
                            label="Conditioning Effect"
                        )
                        axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                        axes[2, 0].axhline(y=0.01, color='orange', linestyle=':', alpha=0.5, label='Threshold')
                        axes[2, 0].set_xlabel("Epoch")
                        axes[2, 0].set_ylabel("|preds_on - preds_off|")
                        axes[2, 0].set_title("Conditioning Effect Check")
                        axes[2, 0].grid(True)
                        axes[2, 0].legend()
                        
                        # Color background based on conditioning strength
                        current_delta = model._cond_deltas[-1]
                        if current_delta < 1e-4:
                            axes[2, 0].set_facecolor('#ffeeee')  # Light red if ignored
                        elif current_delta < 0.01:
                            axes[2, 0].set_facecolor('#ffffe8')  # Light yellow if weak
                        else:
                            axes[2, 0].set_facecolor('#eeffee')  # Light green if good
                        
                        # Add text annotation
                        axes[2, 0].text(
                            0.95, 0.95,
                            f"Current: {current_delta:.4f}\n" + 
                            ("✓ Active" if current_delta > 0.01 else "⚠ Weak" if current_delta > 1e-4 else "✗ Ignored"),
                            transform=axes[2, 0].transAxes,
                            ha="right", va="top",
                            bbox=dict(
                                boxstyle="round",
                                facecolor="lightgreen" if current_delta > 0.01 else "yellow" if current_delta > 1e-4 else "salmon",
                                alpha=0.7
                            ),
                            fontsize=10
                        )
                    else:
                        axes[2, 0].text(0.5, 0.5, "Conditioning data\nnot yet available", 
                                      ha='center', va='center', transform=axes[2, 0].transAxes)
                        axes[2, 0].set_title("Conditioning Effect Check")

                    # Plot x0 predictions at different timesteps - show actual variation
                    if len(x0_pred_all) > 0:
                        # Show individual predictions to see if they vary
                        for i in range(min(3, x0_pred_all[0].shape[0])):
                            trajectory = [x0[i, 0] for x0 in x0_pred_all]
                            axes[2, 1].plot(
                                test_timesteps, trajectory,
                                alpha=0.4, linewidth=1,
                                label=f"Sample {i}" if i < 3 else None
                            )
                        
                        # Plot mean prediction
                        mean_preds = [x0.mean() for x0 in x0_pred_all]
                        axes[2, 1].plot(
                            test_timesteps,
                            mean_preds,
                            "b-",
                            label="Mean pred",
                            linewidth=3,
                        )
                        
                        # Show true values
                        axes[2, 1].axhline(
                            y=y_true.mean().cpu().item(),
                            color="r",
                            linestyle="--",
                            alpha=0.8,
                            label="True mean",
                            linewidth=2,
                        )
                        
                        axes[2, 1].set_xlabel("Timestep")
                        axes[2, 1].set_ylabel("x0 Prediction")
                        axes[2, 1].set_title("x0 Predictions Across Timesteps")
                        axes[2, 1].legend(loc="best")
                        axes[2, 1].grid(True)
                        
                        # Add text showing if predictions are constant
                        pred_std = np.std([x0.mean() for x0 in x0_pred_all])
                        axes[2, 1].text(
                            0.95, 0.05, 
                            f"Variation: {pred_std:.6f}",
                            transform=axes[2, 1].transAxes,
                            ha="right",
                            bbox=dict(boxstyle="round", facecolor="yellow" if pred_std < 1e-4 else "white", alpha=0.8),
                        )

                    # Plot diffusion sampling consistency - compare two different samples
                    # Generate a second independent sample
                    model.eval()
                    with torch.no_grad():
                        sampled_pred2 = model.sample(cell_graph, batch)
                        sampled_pred2_np = sampled_pred2.squeeze().cpu().numpy()
                    model.train()
                    
                    axes[2, 2].scatter(
                        plot_pred_np[plot_valid_mask],  # First sample
                        sampled_pred2_np[plot_valid_mask],  # Second sample  
                        alpha=0.6,
                    )
                    min_val = min(
                        plot_pred_np[plot_valid_mask].min(),
                        sampled_pred2_np[plot_valid_mask].min(),
                    )
                    max_val = max(
                        plot_pred_np[plot_valid_mask].max(),
                        sampled_pred2_np[plot_valid_mask].max(),
                    )
                    axes[2, 2].plot(
                        [min_val, max_val], [min_val, max_val], "r--", linewidth=2
                    )
                    axes[2, 2].set_xlabel("Sample 1")
                    axes[2, 2].set_ylabel("Sample 2")
                    axes[2, 2].set_title("Sampling Consistency Check")
                    axes[2, 2].grid(True)
                    
                    # Calculate correlation between samples
                    sample_corr = np.corrcoef(
                        plot_pred_np[plot_valid_mask], sampled_pred2_np[plot_valid_mask]
                    )[0, 1]
                    axes[2, 2].text(
                        0.05, 0.95, f"r={sample_corr:.3f}",
                        transform=axes[2, 2].transAxes,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                else:
                    # Empty plots for non-diffusion models
                    for i in range(3):
                        axes[2, i].axis("off")
                        axes[2, i].text(
                            0.5,
                            0.5,
                            "Diffusion-only plots",
                            ha="center",
                            va="center",
                            transform=axes[2, i].transAxes,
                            fontsize=12,
                            color="gray",
                        )

                plt.suptitle(
                    f"GeneInteractionDiff Training - Epoch {epoch + 1}", fontsize=14
                )
                plt.tight_layout()
                plt.savefig(
                    osp.join(plot_dir, f"training_epoch_{epoch + 1:04d}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

    # Test inference
    print("\nTesting inference (sampling)...")
    model.eval()
    with torch.no_grad():
        # Sample from the model
        sampled_predictions = model.sample(cell_graph, batch)

        print(f"Ground truth shape: {y_true.shape}")
        print(f"Ground truth values: {y_true.squeeze().cpu().numpy()}")
        print(f"Sampled predictions shape: {sampled_predictions.shape}")
        print(f"Sampled values: {sampled_predictions.squeeze().cpu().numpy()}")

        # Compare with ground truth
        mse = nn.MSELoss()(sampled_predictions, y_true)
        print(f"\nFinal MSE: {mse.item():.4f}")

    print("\nOverfitting test complete!")

    # Create final comprehensive plot
    print(f"\nPlots saved to: {plot_dir}")

    # Create a final summary plot
    if len(losses) > 1:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Loss curve
        axes[0, 0].plot(range(1, len(losses) + 1), losses, "b-", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title(f"Final Loss: {losses[-1]:.4f}")
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale("log")

        # All correlations
        axes[0, 1].plot(
            range(1, len(correlations) + 1),
            correlations,
            "g-",
            label=f"Pearson (final: {correlations[-1]:.3f})",
            linewidth=2,
        )
        axes[0, 1].plot(
            range(1, len(spearman_correlations) + 1),
            spearman_correlations,
            "b--",
            label=f"Spearman (final: {spearman_correlations[-1]:.3f})",
            linewidth=2,
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Correlation")
        axes[0, 1].set_title("Correlation Evolution")
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(-0.1, 1.1)

        # Final predictions scatter
        with torch.no_grad():
            # For diffusion models, use sampling to get predictions
            if wandb_cfg["regression_task"]["loss"] == "diffusion":
                final_pred = model.sample(cell_graph, batch)
            else:
                final_pred, _ = model(cell_graph, batch)
            final_pred_np = final_pred.squeeze().cpu().numpy()

        if final_pred_np.ndim == 0:
            final_pred_np = np.array([final_pred_np])

        axes[0, 2].scatter(final_pred_np[valid_mask], target_np[valid_mask], alpha=0.6)
        min_val = min(target_np[valid_mask].min(), final_pred_np[valid_mask].min())
        max_val = max(target_np[valid_mask].max(), final_pred_np[valid_mask].max())
        axes[0, 2].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        axes[0, 2].set_xlabel("Predicted Values")
        axes[0, 2].set_ylabel("True Values")
        axes[0, 2].set_title(f"Final Predictions (MSE: {mses[-1]:.4f})")
        axes[0, 2].grid(True)

        # Error metrics
        axes[0, 3].plot(range(1, len(mses) + 1), mses, "r-", label="MSE", linewidth=2)
        axes[0, 3].plot(
            range(1, len(maes) + 1), maes, "orange", label="MAE", linewidth=2
        )
        axes[0, 3].plot(
            range(1, len(rmses) + 1), rmses, "purple", label="RMSE", linewidth=2
        )
        axes[0, 3].set_xlabel("Epoch")
        axes[0, 3].set_ylabel("Error")
        axes[0, 3].set_title("Error Metrics")
        axes[0, 3].grid(True)
        axes[0, 3].legend()
        axes[0, 3].set_yscale("log")

        # Diffusion loss components
        if diffusion_losses or x0_mses:
            if diffusion_losses:
                axes[1, 0].plot(
                    range(1, len(diffusion_losses) + 1),
                    diffusion_losses,
                    "purple",
                    label="Diffusion Loss",
                    linewidth=2,
                )
            if x0_mses:
                axes[1, 0].plot(
                    range(1, len(x0_mses) + 1),
                    x0_mses,
                    "cyan",
                    label="x0 MSE",
                    linewidth=2,
                    linestyle="--",
                )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_title("Diffusion Training Metrics")
            axes[1, 0].grid(True)
            axes[1, 0].legend()
            axes[1, 0].set_yscale("log")
        else:
            axes[1, 0].axis("off")
            axes[1, 0].text(
                0.5,
                0.5,
                "No diffusion loss components",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

        # Residual plot
        residuals = final_pred_np[valid_mask] - target_np[valid_mask]
        axes[1, 1].scatter(target_np[valid_mask], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[1, 1].set_xlabel("True Values")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title(f"Residual Plot (mean: {np.mean(residuals):.4f})")
        axes[1, 1].grid(True)

        # Histogram of residuals
        axes[1, 2].hist(residuals, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 2].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1, 2].set_xlabel("Residuals")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].set_title(f"Residual Distribution (std: {np.std(residuals):.4f})")
        axes[1, 2].grid(True)

        # Summary statistics
        axes[1, 3].axis("off")
        summary_text = f"Training Summary\n"
        summary_text += f"=" * 30 + "\n"
        summary_text += f"Model: GeneInteractionDiff\n"
        summary_text += f"Total Parameters: {param_counts['total']:,}\n"
        summary_text += f"Epochs: {num_epochs}\n"
        summary_text += f"Batch Size: {batch_size}\n"
        summary_text += f"\nFinal Metrics:\n"
        summary_text += f"Loss: {losses[-1]:.6f}\n"
        summary_text += f"Pearson r: {correlations[-1]:.4f}\n"
        summary_text += f"Spearman ρ: {spearman_correlations[-1]:.4f}\n"
        summary_text += f"MSE: {mses[-1]:.6f}\n"
        summary_text += f"MAE: {maes[-1]:.6f}\n"
        summary_text += f"RMSE: {rmses[-1]:.6f}\n"
        if wandb_cfg["regression_task"]["loss"] == "diffusion":
            summary_text += f"\n*All metrics from sampled predictions"

        axes[1, 3].text(
            0.1,
            0.9,
            summary_text,
            fontsize=10,
            transform=axes[1, 3].transAxes,
            verticalalignment="top",
            family="monospace",
        )

        plt.suptitle(f"GeneInteractionDiff Training Summary", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"final_summary_{timestamp()}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Final summary plot saved: final_summary_{timestamp()}.png")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
