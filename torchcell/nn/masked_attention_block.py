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
    """
    Memory-efficient Masked Attention Block that uses FlexAttention on GPU.
    
    This module only handles adjacency masks. For edge attributes, use NodeSelfAttention.
    
    Args:
        hidden_dim: Dimensionality of the input features
        num_heads: Number of attention heads
        dropout: Dropout probability
        activation: Activation function to use in the feed-forward network
        mode: Mode of attention ('node' for node-level attention)
        compile_block_mask: Whether to compile the block mask function
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: str = "node",
        compile_block_mask: bool = True, # FLAG
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode
        self.compile_block_mask = compile_block_mask
        
        # Register a flag for test purposes
        self.in_simulated_error_test = False

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

    def _reshape_for_attention(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x: Tensor, adj_mask: Tensor) -> Tensor:
        """
        Forward pass applying masked attention using adjacency mask.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_dim]
            adj_mask: Adjacency mask tensor with shape [batch_size, seq_len, seq_len]
                      True values indicate allowed attention connections
        
        Returns:
            Output tensor with same shape as input
        """
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
            # Use FlexAttention on GPU
            try:
                from torch.nn.attention.flex_attention import flex_attention
                
                if self.in_simulated_error_test:
                    raise RuntimeError("Simulated FlexAttention error")

                # Create a direct score_mod function to modify based on adjacency mask
                # Using torch.where instead of Python conditional to avoid dynamic control flow
                def score_mod(score, b, h, q_idx, k_idx):
                    mask_val = adj_mask[b, q_idx, k_idx]
                    return torch.where(mask_val, score, torch.tensor(-1e9, device=score.device))
                
                # Use flex_attention with score_mod only
                attn_output = flex_attention(q, k, v, score_mod=score_mod)
            except Exception as e:
                if hasattr(self, 'in_simulated_error_test') and self.in_simulated_error_test:
                    raise RuntimeError("Simulated FlexAttention error")
                
                # Standard attention implementation on CPU as fallback
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Expand mask for attention heads
                expanded_mask = adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                
                # Apply mask
                scores = scores.masked_fill(~expanded_mask, -1e9)
                
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
        else:
            # Standard attention implementation on CPU
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Expand mask for attention heads
            expanded_mask = adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Apply mask
            scores = scores.masked_fill(~expanded_mask, -1e9)
            
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
    Masked attention block with learned edge attribute projections.
    Uses FlexAttention on GPU for efficiency when available.
    
    This module supports:
    1. Adjacency masks to control attention flow
    2. Edge attributes projected through per-head MLPs to modulate attention scores
    
    Args:
        hidden_dim: Dimensionality of the input features
        num_heads: Number of attention heads 
        dropout: Dropout probability
        activation: Activation function for the MLP
        mode: Mode of attention ('node' for node-level attention)
        compile_block_mask: Whether to compile the block mask function
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        mode: str = "node",
        compile_block_mask: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mode = mode
        self.compile_block_mask = compile_block_mask

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
        
        # Register a flag for test purposes
        self.in_simulated_error_test = False
        
        # Edge projections cache
        self.edge_projections = {}

    def _prepare_edge_projections(
        self, 
        edge_attr: Union[Tensor, Dict[Tuple[int, int], float]], 
        edge_index: Optional[Tensor], 
        seq_len: int, 
        device: torch.device
    ) -> Dict[Tuple[int, int, int], float]:
        """
        Prepare edge projections for score modification.
        
        Projects edge attributes through per-head MLPs and caches the results
        for efficient lookup during attention computation.
        
        Args:
            edge_attr: Edge attributes as either a tensor or a dictionary
            edge_index: Edge indices as a tensor with shape [2, num_edges]
            seq_len: Sequence length
            device: Device to place tensors on
            
        Returns:
            Dictionary mapping (head, src_node, dst_node) to projection values
        """
        edge_projections = {}

        # Move all edge_attr_proj modules to the correct device once
        for i in range(len(self.edge_attr_proj)):
            self.edge_attr_proj[i] = self.edge_attr_proj[i].to(device)

        if isinstance(edge_attr, dict):
            # Handle dictionary input
            for (src, dst), attr_val in edge_attr.items():
                if src < seq_len and dst < seq_len:
                    for h in range(self.num_heads):
                        # Make sure attr_tensor is on the correct device
                        attr_tensor = torch.tensor([[attr_val]], device=device)
                        with torch.no_grad():
                            proj_val = self.edge_attr_proj[h](attr_tensor).item()
                        edge_projections[(h, src, dst)] = proj_val
        else:
            # Handle tensor input
            for i in range(min(edge_index.size(1), edge_attr.numel())):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if src < seq_len and dst < seq_len:
                    for h in range(self.num_heads):
                        attr_val = (
                            edge_attr[i].item()
                            if edge_attr.dim() == 1
                            else edge_attr[i].mean().item()
                        )
                        # Make sure attr_tensor is on the correct device
                        attr_tensor = torch.tensor([[attr_val]], device=device)
                        with torch.no_grad():
                            proj_val = self.edge_attr_proj[h](attr_tensor).item()
                        edge_projections[(h, src, dst)] = proj_val
                            
        return edge_projections

    def forward(
        self,
        x: Tensor,
        adj_mask: Tensor,
        edge_attr: Optional[Union[Tensor, Dict[Tuple[int, int], float]]] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with masked attention and edge attribute projections.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_dim]
            adj_mask: Adjacency mask with shape [batch_size, seq_len, seq_len]
            edge_attr: Optional edge attributes as tensor [num_edges] or dictionary
                       mapping (src, dst) tuples to scalar values
            edge_index: Optional edge indices as tensor with shape [2, num_edges]
                        Required if edge_attr is a tensor
            
        Returns:
            Output tensor with same shape as input
        """
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
            if self.in_simulated_error_test:
                raise RuntimeError("Simulated FlexAttention error")
                
            try:
                from torch.nn.attention.flex_attention import flex_attention
                
                # Handle edge attributes if provided
                if edge_attr is not None and edge_index is not None:
                    # Prepare edge projections
                    self.edge_projections = self._prepare_edge_projections(
                        edge_attr, edge_index, seq_len, x.device
                    )
                    
                    # Edge-aware score modification that uses torch.where() instead of if/else
                    def score_mod(score, batch, head, q_idx, k_idx):
                        # Use the mask directly
                        mask_val = adj_mask[batch, q_idx, k_idx]
                        
                        # Look up edge projection if exists
                        h = head.item()
                        q = q_idx.item()
                        k = k_idx.item()
                        
                        # Get edge projection value (0.0 if not found)
                        edge_val = self.edge_projections.get((h, q, k), 0.0)
                        
                        # Apply both mask and edge projection using torch.where
                        return torch.where(
                            mask_val, 
                            score + edge_val, 
                            torch.tensor(-1e9, device=score.device)
                        )
                    
                    # Use flex_attention with score_mod
                    attn_output = flex_attention(q, k, v, score_mod=score_mod)
                else:
                    # Simple score_mod with just masking using torch.where
                    def score_mod(score, batch, head, q_idx, k_idx):
                        mask_val = adj_mask[batch, q_idx, k_idx]
                        return torch.where(
                            mask_val, 
                            score, 
                            torch.tensor(-1e9, device=score.device)
                        )
                    
                    # Use flex_attention with just masking
                    attn_output = flex_attention(q, k, v, score_mod=score_mod)
            except Exception as e:
                # Fall back to CPU attention
                attn_output = self._cpu_attention(
                    q, k, v, adj_mask, edge_attr, edge_index, seq_len
                )
        else:
            # Process on CPU
            attn_output = self._cpu_attention(
                q, k, v, adj_mask, edge_attr, edge_index, seq_len
            )

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
    
    def _cpu_attention(
        self, q, k, v, adj_mask, edge_attr, edge_index, seq_len
    ):
        """Standard attention implementation for CPU."""
        # Ensure everything is on CPU
        device = q.device
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Expand mask for heads
        expanded_mask = adj_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Apply mask
        scores = scores.masked_fill(~expanded_mask, -1e9)
        
        # Apply edge attributes if available
        if edge_attr is not None and edge_index is not None:
            edge_attr_bias = torch.zeros_like(scores)
            
            if isinstance(edge_attr, dict):
                for (src, dst), attr_val in edge_attr.items():
                    if src < seq_len and dst < seq_len:
                        for h in range(self.num_heads):
                            # Make sure tensor is on the same device
                            attr_tensor = torch.tensor([[attr_val]], device=device)
                            # Ensure projection is on the same device
                            proj_module = self.edge_attr_proj[h].to(device)
                            with torch.no_grad():
                                proj_val = proj_module(attr_tensor).item()
                            
                            for b in range(scores.size(0)):
                                if b < adj_mask.size(0) and adj_mask[b, src, dst]:
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
                        
                        # Ensure it's on the right device
                        attr_val = attr_val.to(device)
                        
                        # Project for each head
                        for h in range(self.num_heads):
                            # Ensure projection is on the same device
                            proj_module = self.edge_attr_proj[h].to(device)
                            with torch.no_grad():
                                proj_val = proj_module(attr_val).item()
                                
                            for b in range(scores.size(0)):
                                if b < adj_mask.size(0) and adj_mask[b, src, dst]:
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
        return torch.matmul(attn_weights, v)