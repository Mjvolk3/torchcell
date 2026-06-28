"""Scratch examples for PyTorch FlexAttention with causal masking."""

# --------------- WORKS -----------------
# import torch
# from torch.nn.attention.flex_attention import flex_attention


# def noop_score_mod(score, b, h, q_idx, kv_idx):
#     # No modification: return score unchanged.
#     return score


# # Example parameters:
# batch_size = 2
# num_heads = 4
# seq_len = 8
# head_dim = 16
# embed_dim = num_heads * head_dim

# # Create random Q, K, V tensors of shape:
# # [batch_size, num_heads, seq_len, head_dim]
# Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
# K = torch.randn(batch_size, num_heads, seq_len, head_dim)
# V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# # Compute attention using FlexAttention. The function internally compiles
# # an optimized kernel (e.g., FlashAttention) that fuses the computations.
# output = flex_attention(Q, K, V, score_mod=noop_score_mod)

# print("Output shape:", output.shape)

###

from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention


def causal_mask_score_mod(
    score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor
) -> Tensor:
    """Return the score where the query may attend to the key, else -inf."""
    # Compute a boolean condition for causal masking.
    allowed = q_idx >= kv_idx
    # Use torch.as_tensor to create a tensor from the boolean value.
    allowed_tensor = torch.as_tensor(allowed, dtype=torch.bool, device=score.device)
    # Use torch.full to create a tensor with -inf.
    neg_inf = torch.full((), -float("inf"), dtype=score.dtype, device=score.device)
    # Return score if allowed; else -inf.
    return torch.where(allowed_tensor, score, neg_inf)


# Example parameters:
batch_size = 2
num_heads = 4
seq_len = 8
head_dim = 16
embed_dim = num_heads * head_dim

# Create random Q, K, V tensors of shape:
# [batch_size, num_heads, seq_len, head_dim]
Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Compute attention using FlexAttention with causal masking.
output = cast(Tensor, flex_attention(Q, K, V, score_mod=causal_mask_score_mod))
print("Masked output shape:", output.shape)
