# torchcell/nn/masked_gin_conv
# [[torchcell.nn.masked_gin_conv]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/masked_gin_conv

"""
Masked GIN convolution for lazy subgraph representation.

Extends PyG's GINConv to support edge masking WITHOUT filtering edges.
This preserves the speedup from LazySubgraphRepresentation by masking
messages instead of copying/filtering edge tensors.
"""

from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import GINConv
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class MaskedGINConv(GINConv):
    """
    Graph Isomorphism Network with masked message passing.

    **Key Innovation**: Masks messages, NOT edges. This avoids expensive
    edge filtering (`edge_index[:, mask]`) that would negate the speedup
    from LazySubgraphRepresentation.

    **How It Works**:
    1. Process ALL edges (no filtering of edge_index)
    2. Zero out messages from masked edges (element-wise multiply)
    3. Aggregate using PyG's optimized scatter operations
    4. No tensor copying or allocation

    This approach is fast because:
    - No `edge_index[:, mask]` filtering (expensive)
    - Element-wise mask multiplication is GPU-accelerated
    - Same edge_index reused every forward pass (cache-friendly)
    - Memory-efficient (no new tensor allocation)

    Args:
        nn (torch.nn.Module): MLP that transforms aggregated features
        eps (float, optional): Initial epsilon value for self-loop weighting
        train_eps (bool, optional): Whether epsilon should be trainable
        **kwargs: Additional arguments for MessagePassing

    Example:
        >>> import torch.nn as nn
        >>> from torch_geometric.data import Data
        >>>
        >>> # Create MLP for GIN
        >>> mlp = nn.Sequential(
        ...     nn.Linear(64, 64),
        ...     nn.ReLU(),
        ...     nn.Linear(64, 64)
        ... )
        >>>
        >>> # Create masked GIN layer
        >>> conv = MaskedGINConv(mlp, train_eps=True)
        >>>
        >>> # Example data
        >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
        >>> edge_mask = torch.rand(500) > 0.3  # Mask 30% of edges
        >>>
        >>> # Forward with masking - edge_index is NOT copied!
        >>> out = conv(x, edge_index, edge_mask=edge_mask)
        >>> out.shape
        torch.Size([100, 64])
    """

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_mask: Optional[Tensor] = None,
        size: Size = None,
    ) -> Tensor:
        """
        Forward pass with masked message passing.

        **IMPORTANT**: This does NOT filter edge_index. Instead, it masks
        messages in the message() method, preserving lazy speedup.

        Args:
            x: Node features [num_nodes, in_channels] or pair
            edge_index: Edge connectivity [2, num_edges] - NOT filtered!
            edge_mask: Optional boolean mask [num_edges] - True=keep, False=zero
            size: Size of source/destination nodes (bipartite graphs)

        Returns:
            out: Updated node features [num_nodes, out_channels]
        """
        if isinstance(x, Tensor):
            x = (x, x)

        # Pass edge_mask to propagate - PyG routes it to message()
        # propagate_type: (x: OptPairTensor, edge_mask: OptTensor)
        out = self.propagate(edge_index, x=x, edge_mask=edge_mask, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_mask: OptTensor = None) -> Tensor:
        """
        Compute messages with optional masking.

        **This is where masking happens** - messages are zeroed out for
        masked edges, but edge_index itself is never modified.

        Args:
            x_j: Source node features for each edge [num_edges, F]
            edge_mask: Optional mask [num_edges] - True=keep, False=zero

        Returns:
            Masked messages [num_edges, F]
        """
        msg = x_j  # [num_edges, F]

        if edge_mask is not None:
            # Ensure mask is boolean
            if edge_mask.dtype != torch.bool:
                edge_mask = edge_mask.bool()

            # Zero out messages from masked edges
            # Broadcasting: [num_edges, 1] * [num_edges, F] -> [num_edges, F]
            msg = msg * edge_mask.unsqueeze(-1).to(msg.dtype)

        return msg  # PyG aggregates these masked messages

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
