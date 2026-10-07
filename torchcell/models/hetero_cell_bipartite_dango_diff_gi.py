"""Gene interaction model variant with a diffusion-based prediction head."""

# torchcell/models/hetero_cell_bipartite_dango_diff_gi
# [[torchcell.models.hetero_cell_bipartite_dango_diff_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_bipartite_dango_diff_gi
# Test file: tests/torchcell/models/test_hetero_cell_bipartite_dango_diff_gi.py

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from torchcell.models.diffusion_decoder import DiffusionDecoder
from torchcell.models.hetero_cell_bipartite_dango_gi import GeneInteractionDango


class LinearDecoder(nn.Module):
    """Simple linear decoder for baseline comparison.

    Maps combined embeddings directly to phenotype predictions.
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        """Create a linear projection from input_dim to output_dim."""
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z_c: Combined embeddings [batch_size, input_dim]

        Returns:
            Predictions [batch_size, output_dim]
        """
        return cast(torch.Tensor, self.proj(z_c))

    def sample(self, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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
        gene_encoder_config: dict[str, Any] | None = None,
        local_predictor_config: dict[str, Any] | None = None,
        diffusion_config: dict[str, Any] | None = None,
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
            decoder_type: Which prediction head to use ("diffusion" or "linear")
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
        self.decoder: LinearDecoder | DiffusionDecoder
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
            self.diffusion_decoder: DiffusionDecoder = self.decoder
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        self.training_mode = True

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        batch_assign: torch.Tensor | None
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
        z_c = torch.cat(
            [z_i_global, pert_gene_embs_aggregated], dim=-1
        )  # z_c: combined latent

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
            decoder = cast(DiffusionDecoder, self.decoder)
            return decoder.loss(y_true, z_c, t_mode=t_mode)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

    def sample(
        self, cell_graph: HeteroData, batch: HeteroData, num_samples: int | None = None
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
            return cast(torch.Tensor, self.decoder(z_c))
        else:
            # Diffusion decoder samples
            decoder = cast(DiffusionDecoder, self.decoder)
            return decoder.sample(z_c, num_samples)

    @property
    def num_parameters(self) -> dict[str, int]:
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


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/hetero_cell_bipartite_dango_diff_gi_demo.py on 2026-10-06"
    )
