# torchcell/nn/self_attention_block
# [[torchckell.nn.self_attention_block]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/self_attention_block
# Test file: tests/torchcell/nn/test_self_attention_block.py

# torchcell/nn/self_attention_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention Block (SAB) with device-adaptive implementation.
    Uses flex_attention on GPU and standard attention on CPU.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        """
        Initialize the Self-Attention Block.

        Args:
            hidden_dim: Dimension of input and output features
            num_heads: Number of attention heads
            dropout: Dropout probability for attention and MLP layers
            activation: Activation function to use in the MLP
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Flag for simulating errors in tests
        self.in_simulated_error_test = False

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Projection matrices for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch_size, seq_len, hidden_dim] to [batch_size, num_heads, seq_len, head_dim]"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SAB.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Apply normalization
        normed_x = self.norm1(x)

        # Project to queries, keys, values
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        # Calculate attention - different implementation based on device
        if torch.cuda.is_available() and x.is_cuda:
            # For testing error propagation
            if self.in_simulated_error_test:
                raise RuntimeError("Simulated FlexAttention error")
                
            # Import here for better error handling
            from torch.nn.attention.flex_attention import flex_attention
            
            # Use FlexAttention on GPU
            attn_output = flex_attention(q, k, v)
        else:
            # Standard attention implementation for CPU
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply softmax to get attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Handle NaNs for numerical stability
            if torch.isnan(attn_weights).any():
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_weights = attn_weights / row_sums

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # First residual connection
        x = residual + attn_output

        # Save for second residual connection
        residual = x

        # Apply normalization and MLP
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)

        # Second residual connection
        output = residual + mlp_output

        return output