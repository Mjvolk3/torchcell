import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, GATConv, SAGEConv
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.typing import OptTensor
from torchcell.models.act import act_register


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Linear(dim_in, dim_out * num_heads)
        self.key = nn.Linear(dim_in, dim_out * num_heads)
        self.value = nn.Linear(dim_in, dim_out * num_heads)
        self.dim_out = dim_out
        self.num_heads = num_heads

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(Q.size(0), -1, self.num_heads, self.dim_out).permute(2, 0, 1, 3)
        K = K.view(K.size(0), -1, self.num_heads, self.dim_out).permute(2, 0, 1, 3)
        V = V.view(V.size(0), -1, self.num_heads, self.dim_out).permute(2, 0, 1, 3)

        attn_weights = F.softmax(Q @ K.transpose(-2, -1) / self.dim_out**0.5, dim=-1)
        out = attn_weights @ V
        out = out.permute(1, 2, 0, 3).contiguous()
        return out, attn_weights


class SelfAttentionSAG(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 1,
        dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        gnn_type: nn.Module = GraphConv,
        ratio: float = 0.5,
        min_score: float = None,
        multiplier: float = 1.0,
        nonlinearity: str = "tanh",
        **kwargs,
    ):
        super().__init__()

        assert norm in ["batch", "instance", "layer"], "Invalid norm type"
        assert activation in act_register.keys(), "Invalid activation type"

        self.num_heads = num_heads
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        def create_block(in_dim, out_dim, norm, activation):
            block = [nn.Linear(in_dim, out_dim)]
            if norm == "batch":
                block.append(nn.BatchNorm1d(out_dim))
            elif norm == "instance":
                block.append(nn.InstanceNorm1d(out_dim, affine=True))
            elif norm == "layer":
                block.append(nn.LayerNorm(out_dim))
            block.append(act_register[activation])
            return nn.Sequential(*block)

        self.gnn = gnn_type(in_channels, hidden_channels, **kwargs)
        self.self_attn = SelfAttention(
            dim_in=hidden_channels, dim_out=hidden_channels, num_heads=num_heads
        )
        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.filter_edges = FilterEdges()
        self.fc = create_block(hidden_channels * num_heads, out_channels, norm, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
    ):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = self.gnn(x, edge_index)
        x, attn_weights = self.self_attn(x)
        attn_weights_list = [attn_weights]  # Store attention weights from each layer

        attn_weights = attn_weights.mean(dim=0)  # Average over heads
        attn_weights = attn_weights.view(-1)
        attn_weights_binary = (attn_weights > 0.5).float()

        # Use the binarized attention weights as edges
        edge_index = torch.nonzero(attn_weights_binary).t()
        x = x.view(x.size(0), -1)  # Flatten the node features

        select_out = self.select(x, batch)
        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        # Calculate the number of nodes in each batch manually
        batch_num_nodes = torch.bincount(batch[perm])

        edge_index, edge_attr = self.filter_edges(
            edge_index, edge_attr, perm, num_nodes=batch_num_nodes
        )
        batch = batch[perm]

        x = self.fc(x)
        x = self.dropout(x)

        return x, edge_index, edge_attr, batch, perm, score, attn_weights_list


def main():
    # Model configuration
    in_channels = 128
    hidden_channels = 64
    out_channels = 32
    num_heads = 3
    dropout_prob = 0.2
    norm = "layer"
    activation = "relu"
    gnn_type = GCNConv
    ratio = 0.8
    min_score = 0.5
    multiplier = 1.5
    nonlinearity = "sigmoid"

    model = SelfAttentionSAG(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        norm=norm,
        activation=activation,
        gnn_type=gnn_type,
        ratio=ratio,
        min_score=min_score,
        multiplier=multiplier,
        nonlinearity=nonlinearity,
    )

    # Dummy data
    x = torch.randn(100, in_channels)
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.randint(0, 5, (100,))

    # Forward pass
    x_pooled, edge_index_pooled, _, batch_pooled, perm, score = model(
        x, edge_index, batch=batch
    )
    print("x_pooled shape:", x_pooled.shape)
    print("edge_index_pooled shape:", edge_index_pooled.shape)
    print("batch_pooled shape:", batch_pooled.shape)
    print("perm shape:", perm.shape)
    print("score shape:", score.shape)


if __name__ == "__main__":
    main()
