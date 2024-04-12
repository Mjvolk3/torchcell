import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.utils import add_self_loops


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Linear(dim_in, dim_out * num_heads)
        self.key = nn.Linear(dim_in, dim_out * num_heads)
        self.value = nn.Linear(dim_in, dim_out * num_heads)
        self.dim_out = dim_out
        self.num_heads = num_heads

    def forward(self, x):
        Q = self.query(x).view(-1, self.num_heads, self.dim_out).permute(1, 0, 2)
        K = self.key(x).view(-1, self.num_heads, self.dim_out).permute(1, 0, 2)
        V = self.value(x).view(-1, self.num_heads, self.dim_out).permute(1, 0, 2)

        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / self.dim_out**0.5, dim=-1)
        out = (
            (attn_weights @ V)
            .permute(1, 2, 0)
            .reshape(-1, self.dim_out * self.num_heads)
        )
        return out, attn_weights


class SelfAttentionSAG(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads=1,
        dropout_prob=0.2,
        norm="batch",
        activation="relu",
        ratio=0.5,
        min_score=None,
        multiplier=1.0,
        nonlinearity="tanh",
    ):
        super().__init__()
        self.self_attn = SelfAttention(in_channels, hidden_channels, num_heads)
        self.gnn = GCNConv(hidden_channels * num_heads, out_channels)
        self.pool = SAGPooling(
            out_channels,
            ratio=ratio,
            GNN=GCNConv,
            min_score=min_score,
            multiplier=multiplier,
            nonlinearity=nonlinearity,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = getattr(F, activation)

    def forward(self, x, batch):
        x, attn_weights = self.self_attn(x)
        edge_index = (attn_weights > 0.5).nonzero(as_tuple=False).t()

        # Ensure edge_index is properly shaped and not empty
        if edge_index.size(1) == 0:
            edge_index = torch.stack(
                [torch.arange(x.size(0)), torch.arange(x.size(0))], dim=0
            )
        else:
            edge_index = edge_index.view(2, -1)  # Ensure it is 2 x num_edges

        # Adding self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.gnn(x, edge_index)
        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, batch=batch)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x, edge_index, batch, perm, score, attn_weights


def main():
    model = SelfAttentionSAG(10, 32, 8, num_heads=2, norm="layer", activation="relu")
    x = torch.randn(100, 10)  # Dummy data
    batch = torch.cat(
        [torch.full((20,), i, dtype=torch.long) for i in range(5)]
    )  # Batch setup

    x_pooled, edge_index_pooled, batch_pooled, perm, score, attn_weights = model(
        x, batch
    )
    print(f"x_pooled shape: {x_pooled.shape}")
    print(f"edge_index_pooled shape: {edge_index_pooled.shape}")
    print(f"batch_pooled shape: {batch_pooled.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")


if __name__ == "__main__":
    main()
