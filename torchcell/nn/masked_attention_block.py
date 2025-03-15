# torchcell/nn/masked_attention_block
# [[torchcell.nn.masked_attention_block]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/masked_attention_block
# Test file: tests/torchcell/nn/test_masked_attention_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Union, Tuple, Dict


class MaskedAttentionBlock(nn.Module):
    """Memory-efficient Masked Attention Block that uses FlexAttention on GPU."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: str = "node",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Edge attributes buffers for score modification
        self.register_buffer("edge_attr_values", None, persistent=False)
        self.register_buffer("edge_attr_indices", None, persistent=False)

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def prepare_edge_attributes(self, edge_attr_dict, seq_len):
        indices = []
        values = []
        for (i, j), val in edge_attr_dict.items():
            idx = i * seq_len + j
            indices.append(idx)
            values.append(val)
        if indices:
            self.edge_attr_indices = torch.tensor(
                indices, device=self.q_proj.weight.device
            )
            self.edge_attr_values = torch.tensor(
                values, device=self.q_proj.weight.device
            )
        else:
            self.edge_attr_indices = torch.tensor(
                [], device=self.q_proj.weight.device, dtype=torch.long
            )
            self.edge_attr_values = torch.tensor([], device=self.q_proj.weight.device)

    def forward(
        self, x: Tensor, adj_mask: Tensor, edge_attr_dict: Optional[dict] = None
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        normed_x = self.norm1(x)
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        if adj_mask.dtype != torch.bool:
            adj_mask = adj_mask.bool()

        # Check if we're on GPU
        if torch.cuda.is_available() and x.is_cuda:
            # Use FlexAttention on GPU - let errors propagate if it fails
            from torch.nn.attention.flex_attention import (
                flex_attention,
                create_block_mask,
            )

            if edge_attr_dict is not None:
                # Prepare edge attributes for use in score modification
                self.prepare_edge_attributes(edge_attr_dict, seq_len)

                def score_mod(score, b, h, q_idx, kv_idx):
                    mask_val = adj_mask[b, q_idx, kv_idx]
                    score_masked = torch.where(
                        mask_val,
                        score,
                        torch.tensor(
                            float("-inf"), device=score.device, dtype=score.dtype
                        ),
                    )
                    edge_key = q_idx * seq_len + kv_idx
                    edge_exists = (self.edge_attr_indices == edge_key).any()
                    edge_idx = torch.where(self.edge_attr_indices == edge_key)[0]
                    edge_val = torch.zeros_like(score)
                    if edge_exists:
                        edge_val = self.edge_attr_values[edge_idx[0]] * 0.1
                    return score_masked + edge_val

                attn_output = flex_attention(q, k, v, score_mod=score_mod)
            else:

                def mask_mod(b, h, q_idx, kv_idx):
                    b_idx = min(b, adj_mask.size(0) - 1)
                    if q_idx < adj_mask.size(1) and kv_idx < adj_mask.size(2):
                        return adj_mask[b_idx, q_idx, kv_idx]
                    return False

                block_mask = create_block_mask(
                    mask_mod,
                    B=batch_size,
                    H=self.num_heads,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    _compile=False,
                )
                attn_output = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Standard attention implementation on CPU
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Expand mask for attention heads
            expanded_mask = adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Handle dimension mismatches
            if expanded_mask.shape[-2:] != scores.shape[-2:]:
                if expanded_mask.shape[-2] > scores.shape[-2]:
                    expanded_mask = expanded_mask[:, :, : scores.shape[-2], :]
                if expanded_mask.shape[-1] > scores.shape[-1]:
                    expanded_mask = expanded_mask[:, :, :, : scores.shape[-1]]
                if expanded_mask.shape[-2] < scores.shape[-2]:
                    pad_size = scores.shape[-2] - expanded_mask.shape[-2]
                    expanded_mask = F.pad(
                        expanded_mask, (0, 0, 0, pad_size), "constant", False
                    )
                if expanded_mask.shape[-1] < scores.shape[-1]:
                    pad_size = scores.shape[-1] - expanded_mask.shape[-1]
                    expanded_mask = F.pad(
                        expanded_mask, (0, pad_size, 0, 0), "constant", False
                    )

            # Apply mask
            scores = scores.masked_fill(~expanded_mask, -1e9)

            # Apply edge attributes if available
            if edge_attr_dict is not None:
                edge_attr_bias = torch.zeros_like(scores)

                for (i, j), val in edge_attr_dict.items():
                    if i < seq_len and j < seq_len:
                        for h in range(self.num_heads):
                            for b in range(batch_size):
                                if (
                                    b < adj_mask.size(0)
                                    and i < adj_mask.size(1)
                                    and j < adj_mask.size(2)
                                ):
                                    if adj_mask[b, i, j]:
                                        edge_attr_bias[b, h, i, j] = val * 0.1

                # Add to scores
                scores = scores + edge_attr_bias

            # Calculate attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Handle NaNs
            if torch.isnan(attn_weights).any():
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_weights = attn_weights / row_sums

            # Apply attention
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # First residual connection
        x = residual + attn_output

        # Second residual block
        residual = x
        normed_x = self.norm2(x)
        mlp_output = self.mlp(normed_x)
        output = residual + mlp_output

        return output


class NodeSelfAttention(nn.Module):
    """
    Masked attention block with learnable edge attribute projections.
    Uses FlexAttention on GPU, standard attention on CPU. No fallbacks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: str = "node",
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode

        # Standard attention components
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Feedforward components
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            activation,
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # Edge attribute projection layers - one per head
        self.edge_attr_proj = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1, 16), activation, nn.Linear(16, 1))
                for _ in range(num_heads)
            ]
        )

    def forward(
        self,
        x: Tensor,
        adj_mask: Tensor,
        edge_attr: Optional[Union[Tensor, Dict[Tuple[int, int], float]]] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass using FlexAttention on GPU, standard attention on CPU."""
        # Handle input dimensions
        input_dim = x.dim()
        if input_dim == 2:
            x = x.unsqueeze(0)
            if adj_mask.dim() == 2:
                adj_mask = adj_mask.unsqueeze(0)

        # Store for residual connection
        residual = x

        # First normalization and projections
        normed_x = self.norm1(x)
        q = self.q_proj(normed_x)
        k = self.k_proj(normed_x)
        v = self.v_proj(normed_x)

        # Get dimensions
        batch_size, seq_len, _ = q.size()

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Ensure masks are boolean
        if adj_mask.dtype != torch.bool:
            adj_mask = adj_mask.bool()

        # Handle mask size mismatch
        if adj_mask.size(-1) != seq_len or adj_mask.size(-2) != seq_len:
            pad_rows = max(0, seq_len - adj_mask.size(-2))
            pad_cols = max(0, seq_len - adj_mask.size(-1))

            if pad_rows > 0 or pad_cols > 0:
                adj_mask = F.pad(
                    adj_mask, (0, pad_cols, 0, pad_rows), "constant", False
                )

            if adj_mask.size(-2) > seq_len:
                adj_mask = adj_mask[..., :seq_len, :]
            if adj_mask.size(-1) > seq_len:
                adj_mask = adj_mask[..., :seq_len]

        # Check if we're on GPU
        if torch.cuda.is_available() and x.is_cuda:
            # Use FlexAttention on GPU - no fallback if it fails
            from torch.nn.attention.flex_attention import (
                flex_attention,
                create_block_mask,
            )

            # Create mask_mod function
            def mask_mod(b, h, q_idx, kv_idx):
                b_idx = min(b, adj_mask.size(0) - 1)
                if q_idx < adj_mask.size(1) and kv_idx < adj_mask.size(2):
                    return adj_mask[b_idx, q_idx, kv_idx]
                return False

            # Handle edge attributes if provided
            if edge_attr is not None and edge_index is not None:
                # Pre-compute projected values
                edge_projections = {}

                if isinstance(edge_attr, dict):
                    for (src, dst), attr_val in edge_attr.items():
                        if src < seq_len and dst < seq_len:
                            for h in range(self.num_heads):
                                attr_tensor = torch.tensor(
                                    [[attr_val]], device=q.device
                                )
                                with torch.no_grad():
                                    proj_val = self.edge_attr_proj[h](
                                        attr_tensor
                                    ).item()
                                edge_projections[(h, src, dst)] = proj_val
                else:
                    for i in range(min(edge_index.size(1), edge_attr.numel())):
                        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                        if src < seq_len and dst < seq_len:
                            for h in range(self.num_heads):
                                attr_val = (
                                    edge_attr[i].item()
                                    if edge_attr.dim() == 1
                                    else edge_attr[i].mean().item()
                                )
                                attr_tensor = torch.tensor(
                                    [[attr_val]], device=q.device
                                )
                                with torch.no_grad():
                                    proj_val = self.edge_attr_proj[h](
                                        attr_tensor
                                    ).item()
                                edge_projections[(h, src, dst)] = proj_val

                def score_mod(score, b, h, q_idx, kv_idx):
                    key = (h, q_idx, kv_idx)
                    if key in edge_projections:
                        return score + edge_projections[key]
                    return score

                # Create block mask and apply attention with score modification
                block_mask = create_block_mask(
                    mask_mod,
                    B=batch_size,
                    H=self.num_heads,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    _compile=False,
                )
                attn_output = flex_attention(
                    q, k, v, block_mask=block_mask, score_mod=score_mod
                )
            else:
                # Just use the mask without edge attributes
                block_mask = create_block_mask(
                    mask_mod,
                    B=batch_size,
                    H=self.num_heads,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    _compile=False,
                )
                attn_output = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Standard attention implementation on CPU
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Expand mask for heads
            expanded_mask = adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            # Make sure dimensions match
            if expanded_mask.shape[-2:] != scores.shape[-2:]:
                if expanded_mask.shape[-2] > scores.shape[-2]:
                    expanded_mask = expanded_mask[:, :, : scores.shape[-2], :]
                if expanded_mask.shape[-1] > scores.shape[-1]:
                    expanded_mask = expanded_mask[:, :, :, : scores.shape[-1]]
                if expanded_mask.shape[-2] < scores.shape[-2]:
                    pad_size = scores.shape[-2] - expanded_mask.shape[-2]
                    expanded_mask = F.pad(
                        expanded_mask, (0, 0, 0, pad_size), "constant", False
                    )
                if expanded_mask.shape[-1] < scores.shape[-1]:
                    pad_size = scores.shape[-1] - expanded_mask.shape[-1]
                    expanded_mask = F.pad(
                        expanded_mask, (0, pad_size, 0, 0), "constant", False
                    )

            # Apply mask
            scores = scores.masked_fill(~expanded_mask, -1e9)

            # Apply edge attributes if available
            if edge_attr is not None and edge_index is not None:
                edge_attr_bias = torch.zeros_like(scores)

                if isinstance(edge_attr, dict):
                    for (src, dst), attr_val in edge_attr.items():
                        if src < seq_len and dst < seq_len:
                            for h in range(self.num_heads):
                                attr_tensor = torch.tensor(
                                    [[attr_val]], device=scores.device
                                )
                                with torch.no_grad():
                                    proj_val = self.edge_attr_proj[h](
                                        attr_tensor
                                    ).item()
                                for b in range(batch_size):
                                    if b < adj_mask.size(0):
                                        if adj_mask[b, src, dst]:
                                            edge_attr_bias[b, h, src, dst] = proj_val
                else:
                    for i in range(min(edge_index.size(1), edge_attr.numel())):
                        src, dst = edge_index[0, i], edge_index[1, i]
                        if src < seq_len and dst < seq_len:
                            # Get attribute value
                            if edge_attr.dim() == 1:
                                attr_val = edge_attr[i].view(1, 1)
                            else:
                                attr_val = edge_attr[i].mean().view(1, 1)

                            # Project for each head
                            for h in range(self.num_heads):
                                with torch.no_grad():
                                    proj_val = self.edge_attr_proj[h](attr_val).item()
                                for b in range(batch_size):
                                    if b < adj_mask.size(0):
                                        if adj_mask[b, src, dst]:
                                            edge_attr_bias[b, h, src, dst] = proj_val

                # Add to scores
                scores = scores + edge_attr_bias

            # Calculate attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Handle NaNs
            if torch.isnan(attn_weights).any():
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                row_sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_weights = attn_weights / row_sums

            # Apply attention
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )

        # Project and apply dropout
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # First residual connection
        x = residual + attn_output

        # Feed-forward network with second residual connection
        residual = x
        x = residual + self.dropout(self.mlp(self.norm2(x)))

        # Remove batch dimension if input was 2D
        if input_dim == 2:
            x = x.squeeze(0)

        return x
