# torchcell/models/diffusion_decoder
# [[torchcell.models.diffusion_decoder]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/diffusion_decoder
# Test file: tests/torchcell/models/test_diffusion_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] tensor of timesteps
        Returns:
            [batch_size, dim] tensor of time embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    """Cross-attention module where diffusion state queries graph embeddings."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.scale = self.head_dim**-0.5

        # Q from diffusion state, K/V from graph embeddings
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # Diffusion state [batch, 1, dim]
        context: torch.Tensor,  # Graph embeddings [batch, 1, dim]
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # Debug shape issues
        if context.shape[0] != batch_size:
            # Context batch size doesn't match x batch size
            # This can happen during sampling when we create new samples
            # but reuse the same context
            if context.shape[0] == 1 and batch_size > 1:
                # Expand context to match batch size
                context = context.expand(batch_size, -1, -1)
            elif context.shape[0] > 1 and batch_size == 1:
                # Take first context
                context = context[:1]

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, 1, dim]
        k = self.k_proj(context)  # [batch, 1, dim]
        v = self.v_proj(context)  # [batch, 1, dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        out = self.out_proj(out)

        return out


class DenoisingBlock(nn.Module):
    """Single denoising block with cross-attention and feedforward."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        norm: str = "layer",
    ):
        super().__init__()

        # Normalization layers
        if norm == "layer":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)
        elif norm == "batch":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
            self.norm3 = nn.BatchNorm1d(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        # Cross-attention
        self.cross_attn = CrossAttention(dim, num_heads, dropout)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        # Add time embedding
        time_emb = self.time_mlp(time_emb).unsqueeze(1)  # [batch, 1, dim]
        x = x + time_emb

        # Cross-attention with residual
        if isinstance(self.norm1, nn.BatchNorm1d):
            normed_x = self.norm1(x.squeeze(1)).unsqueeze(1)
        else:
            normed_x = self.norm1(x)
        x = x + self.cross_attn(normed_x, context)

        # MLP with residual
        if isinstance(self.norm2, nn.BatchNorm1d):
            normed_x = self.norm2(x.squeeze(1)).unsqueeze(1)
        else:
            normed_x = self.norm2(x)
        x = x + self.mlp(normed_x)

        return x


class DiffusionDecoder(nn.Module):
    """Diffusion decoder with cross-attention conditioning on graph embeddings."""

    def __init__(
        self,
        input_dim: int,  # Dimension of graph embeddings
        hidden_dim: int = 128,
        output_dim: int = 1,  # Single phenotype prediction
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        norm: str = "layer",
        num_timesteps: int = 1000,
        mlp_ratio: float = 4.0,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        sampling_steps: int = 50,
        parameterization: str = "x0",
    ):
        super().__init__()

        # Safety guard for num_timesteps
        assert num_timesteps >= 1, f"num_timesteps must be >= 1, got {num_timesteps}"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        self.mlp_ratio = mlp_ratio
        self.default_sampling_steps = sampling_steps
        self.parameterization = parameterization

        # Validate parameterization
        assert parameterization in [
            "x0",
            "eps",
        ], f"Unknown parameterization: {parameterization}"

        # Project graph embeddings to hidden dimension
        self.context_proj = nn.Linear(input_dim, hidden_dim)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        # Initial projection of noisy input
        self.input_proj = nn.Linear(output_dim, hidden_dim)

        # Denoising blocks
        self.blocks = nn.ModuleList(
            [
                DenoisingBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Define beta schedule for diffusion
        if beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps, cosine_s)
        elif beta_schedule == "linear":
            betas = self._linear_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod)
        )

    def _linear_beta_schedule(
        self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> torch.Tensor:
        """
        Linear beta schedule.
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        Better for preserving signal especially at the beginning of the process.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward_diffusion(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to data for training.

        Args:
            x_0: Original data [batch, 1]
            t: Timesteps [batch]
            noise: Optional noise [batch, 1]

        Returns:
            x_t: Noisy data
            noise: Noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Pure reconstruction mode: bypass noise when t==0
        # This allows the model to learn pure denoising/reconstruction
        t_is_zero = (t == 0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)

        # Standard diffusion formula
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Override with pure x_0 when t==0 (no noise added)
        if t_is_zero.any():
            x_t = torch.where(t_is_zero.view(-1, 1), x_0, x_t)
            # Also set noise to zero for these samples
            noise = torch.where(t_is_zero.view(-1, 1), torch.zeros_like(noise), noise)

        return x_t, noise

    def denoise(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        predict_x0: bool = True,
    ) -> torch.Tensor:
        """Predict x0 or noise given noisy data and conditioning.

        Args:
            x_t: Noisy data [batch, 1]
            context: Graph embeddings [batch, hidden_dim]
            t: Timesteps [batch]
            predict_x0: If True, predict x0 directly; if False, predict noise

        Returns:
            Predicted x0 or noise [batch, 1]
        """
        # Get time embeddings
        time_emb = self.time_embed(t)  # [batch, hidden_dim]

        # Project context
        context = self.context_proj(context).unsqueeze(1)  # [batch, 1, hidden_dim]

        # Project noisy input
        x = self.input_proj(x_t.unsqueeze(1))  # [batch, 1, hidden_dim]

        # Pass through denoising blocks
        for block in self.blocks:
            x = block(x, context, time_emb)

        # Output projection
        pred = self.output_proj(x).squeeze(1)  # [batch, output_dim]

        # Return prediction based on parameterization
        # Currently only x0 is implemented, eps would need conversion
        if self.parameterization == "x0":
            return pred
        else:
            # For epsilon prediction, network directly predicts noise
            # This would need different training target
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not fully implemented"
            )

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        num_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample from the diffusion model using DDIM-style sampling.

        Args:
            context: Graph embeddings [batch, hidden_dim]
            num_samples: Number of samples (defaults to batch size)
            device: Device to sample on
            sampling_steps: Number of denoising steps (defaults to 50 for faster sampling)

        Returns:
            Sampled phenotypes [batch, 1]
        """
        context_batch_size = context.shape[0]
        batch_size = context_batch_size if num_samples is None else num_samples
        device = context.device if device is None else device

        # Handle context batch size mismatch
        if batch_size != context_batch_size:
            if batch_size == 1 and context_batch_size > 1:
                # If we want 1 sample but have multiple contexts, use first
                context = context[:1]
            elif batch_size > 1 and context_batch_size == 1:
                # If we want multiple samples but have 1 context, expand it
                context = context.expand(batch_size, -1)
            else:
                # Otherwise adjust batch_size to match context
                batch_size = context_batch_size

        # Use configured or default sampling steps
        if sampling_steps is None:
            sampling_steps = min(self.default_sampling_steps, self.num_timesteps)

        # Create a static list of timesteps ONCE before the loop
        ts_list = torch.linspace(
            0, self.num_timesteps - 1, sampling_steps, dtype=torch.long, device=device
        ).tolist()
        ts_list = list(reversed(ts_list))

        # Start from pure noise
        x = torch.randn(batch_size, self.output_dim, device=device)

        # Reverse diffusion process with fewer steps
        for i, t in enumerate(ts_list):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict x0 directly (not noise)
            pred_x0 = self.denoise(x, context, t_batch, predict_x0=True)

            # Note: No clamping - let the model learn appropriate ranges
            # pred_x0 = torch.clamp(pred_x0, -3, 3)  # Removed clamping

            if i < len(ts_list) - 1:
                # Not the last step - move to next timestep
                t_prev = ts_list[i + 1]  # Use the static list
                alpha_cumprod_t = self.alphas_cumprod[t]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]

                # Compute predicted noise from x0 prediction
                noise_pred = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(
                    1 - alpha_cumprod_t
                )

                # DDIM update step
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
                x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
            else:
                # Last step - just return the prediction
                x = pred_x0

        return x

    def loss(
        self, x_0: torch.Tensor, context: torch.Tensor, predict_x0: bool = True,
        t_mode: str = "random"
    ) -> torch.Tensor:
        """Compute diffusion loss using x0-prediction or noise prediction.

        Args:
            x_0: Original phenotype values [batch, 1]
            context: Graph embeddings [batch, hidden_dim]
            predict_x0: If True, predict x0 directly (better for sparse data)
            t_mode: Timestep sampling mode:
                - "zero" or "t0": Always use t=0 (pure reconstruction, no noise)
                - "partial" or "small": Sample from [0, num_timesteps/10)
                - "full" or "random": Sample from full range [0, num_timesteps) (default)

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample timesteps based on mode
        if t_mode == "zero" or t_mode == "t0":
            # Pure reconstruction mode (no noise)
            t = torch.zeros((batch_size,), dtype=torch.long, device=device)
        elif t_mode == "partial" or t_mode == "small":
            # Partial range for easier learning
            max_t = max(1, self.num_timesteps // 10)
            t = torch.randint(0, max_t, (batch_size,), device=device)
        elif t_mode == "full" or t_mode == "random":
            # Full range - standard diffusion training
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        else:
            raise ValueError(f"Unknown t_mode: {t_mode}")

        # Add noise
        noise = torch.randn_like(x_0)
        x_t, actual_noise = self.forward_diffusion(x_0, t, noise)

        # Get prediction
        pred = self.denoise(x_t, context, t, predict_x0=predict_x0)

        # Validate parameterization consistency
        if self.parameterization == "x0":
            assert predict_x0 == True, "x0 parameterization requires predict_x0=True"
            # x0-prediction loss (better for gene expression data)
            loss = F.mse_loss(pred, x_0)
        elif self.parameterization == "eps":
            assert predict_x0 == False, "eps parameterization requires predict_x0=False"
            # Standard noise prediction loss
            # Use actual_noise which is 0 when t==0
            loss = F.mse_loss(pred, actual_noise)
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        return loss
