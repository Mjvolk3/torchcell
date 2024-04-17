# torchcell/models/self_attention_sag
# [[torchcell.models.self_attention_sag]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/self_attention_sag
# Test file: tests/torchcell/models/test_self_attention_sag.py

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

    def forward(self, x, batch):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(-1, self.num_heads, self.dim_out).transpose(0, 1)
        K = K.view(-1, self.num_heads, self.dim_out).transpose(0, 1)
        V = V.view(-1, self.num_heads, self.dim_out).transpose(0, 1)

        attn_weights = F.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.dim_out**0.5, dim=-1
        )
        out = torch.bmm(attn_weights, V)
        out = out.transpose(0, 1).contiguous().view(-1, self.num_heads * self.dim_out)

        # Retrieve attention weights for a single graph
        graph_attn_weights = []
        for i in torch.unique(batch):
            mask = batch == i
            graph_attn_weights.append(attn_weights[:, mask][:, :, mask])

        return out, graph_attn_weights


class SelfAttentionSAG(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads=1,
        num_pooling_layers=1,
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
        self.gnn_input_dim = hidden_channels * num_heads
        self.gnn_output_dim = hidden_channels
        self.gnn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_pooling_layers):
            self.gnn_layers.append(GCNConv(self.gnn_input_dim, self.gnn_output_dim))
            self.pool_layers.append(
                SAGPooling(
                    self.gnn_output_dim,
                    ratio=ratio,
                    GNN=GCNConv,
                    min_score=min_score,
                    multiplier=multiplier,
                    nonlinearity=nonlinearity,
                )
            )
            self.norm_layers.append(nn.LayerNorm(self.gnn_output_dim))
            self.gnn_input_dim = self.gnn_output_dim

        self.prediction_head = nn.Linear(self.gnn_output_dim, out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = getattr(F, activation)

    def forward(self, x, batch):
        x, graph_attn_weights = self.self_attn(x, batch)
        print(f"Self-attention output shape: {x.shape}")
        print(f"Number of graphs: {len(graph_attn_weights)}")
        for i, attn_weights in enumerate(graph_attn_weights):
            print(f"Attention weights shape for graph {i+1}: {attn_weights.shape}")

        # Binarize the attention weights
        attn_weights_binary = (graph_attn_weights[0] > 0.5).float()

        # Create edge_index from binarized attention weights
        edge_index = attn_weights_binary.nonzero(as_tuple=False).t()

        # Ensure edge_index is properly shaped and not empty
        if edge_index.size(1) == 0:
            edge_index = torch.stack(
                [torch.arange(x.size(0)), torch.arange(x.size(0))], dim=0
            )
        else:
            edge_index = edge_index.view(2, -1)  # Ensure it is 2 x num_edges

        # Adding self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print(f"Initial edge_index shape: {edge_index.shape}")

        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
            print(f"GNN layer {i+1} output shape: {x.shape}")
            x, edge_index, _, batch, _, _ = self.pool_layers[i](
                x, edge_index, batch=batch
            )
            print(f"Pooling layer {i+1} output shape: {x.shape}")
            print(f"Pooled edge_index shape: {edge_index.shape}")
            print(f"Pooled batch shape: {batch.shape}")
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        # Perform graph-level prediction
        x = self.prediction_head(x)
        print(f"Prediction output shape: {x.shape}")

        return x


def main():
    in_channels = 10
    hidden_channels = 32
    out_channels = 1  # or 2 for binary classification
    num_heads = 2
    num_pooling_layers = 8
    dropout_prob = 0.2
    norm = "layer"
    activation = "relu"

    model = SelfAttentionSAG(
        in_channels,
        hidden_channels,
        out_channels,
        num_heads=num_heads,
        num_pooling_layers=num_pooling_layers,
        dropout_prob=dropout_prob,
        norm=norm,
        activation=activation,
    )

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    x = torch.randn(12002, in_channels)  # Dummy data for two graphs

    batch = torch.cat(
        [torch.zeros(6000, dtype=torch.long), torch.ones(6002, dtype=torch.long)]
    )  # Batch setup for two graphs

    predictions = model(x, batch)

    print(f"Predictions shape: {predictions.shape}")


if __name__ == "__main__":
    main()
