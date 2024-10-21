# torchcell/models/gat_diffpool
# [[torchcell.models.gat_diffpool]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/gat_diffpool
# Test file: tests/torchcell/models/test_gat_diffpool.py

import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm, InstanceNorm
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from typing import Optional

from torchcell.models.act import act_register


class GatDiffPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        initial_gat_hidden_channels: int,
        initial_gat_out_channels: int,
        diffpool_hidden_channels: int,
        diffpool_out_channels: int,
        num_initial_gat_layers: int,
        num_diffpool_layers: int,
        num_post_pool_gat_layers: int,
        num_graphs: int,
        max_num_nodes: int,
        cluster_size_decay_factor: float = 2.0,
        gat_dropout_prob: float = 0.0,
        last_layer_dropout_prob: float = 0.2,
        norm: str = "instance",
        activation: str = "relu",
        gat_skip_connection: bool = True,
        pruned_max_average_node_degree: Optional[int] = None,  # New parameter
        weight_init: str = "default",
    ):
        super().__init__()
        self.weight_init = weight_init
        self.cluster_size_decay_factor = cluster_size_decay_factor

        assert norm in [
            None,
            "batch",
            "instance",
            "layer",
            "graph",
        ], "Invalid norm type"
        assert activation in act_register.keys(), "Invalid activation type"

        self.num_graphs = num_graphs
        self.gat_skip_connection = gat_skip_connection
        self.activation = act_register[activation]
        self.last_layer_dropout_prob = last_layer_dropout_prob
        self.pruned_max_average_node_degree = pruned_max_average_node_degree
        # Initial GAT layers
        self.initial_gat_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GATv2Conv(
                            in_channels if i == 0 else initial_gat_hidden_channels,
                            (
                                initial_gat_hidden_channels
                                if i < num_initial_gat_layers - 1
                                else initial_gat_out_channels
                            ),
                            heads=1,
                            dropout=gat_dropout_prob,
                        )
                        for i in range(num_initial_gat_layers)
                    ]
                )
                for _ in range(num_graphs)
            ]
        )

        # Norm layers for initial GAT
        self.initial_gat_norm_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self.get_norm_layer(
                            norm,
                            (
                                initial_gat_hidden_channels
                                if i < num_initial_gat_layers - 1
                                else initial_gat_out_channels
                            ),
                        )
                        for i in range(num_initial_gat_layers)
                    ]
                )
                for _ in range(num_graphs)
            ]
        )

        # Calculate cluster sizes
        cluster_sizes = []
        for i in range(1, num_diffpool_layers + 1):
            size = max(1, int(max_num_nodes / (self.cluster_size_decay_factor**i)))
            cluster_sizes.append(size)
        cluster_sizes[-1] = 1  # Ensure the last cluster size is always 1
        print(f"Cluster sizes: {cluster_sizes}")

        # DiffPool layers
        self.diffpool_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(
                            (
                                initial_gat_out_channels
                                if i == 0
                                else diffpool_hidden_channels
                            ),
                            cluster_size,
                        )
                        for i, cluster_size in enumerate(cluster_sizes)
                    ]
                )
                for _ in range(num_graphs)
            ]
        )

        # Post-pooling GAT layers
        self.post_pool_gat_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                GATv2Conv(
                                    diffpool_hidden_channels,
                                    diffpool_hidden_channels,
                                    heads=1,
                                    dropout=gat_dropout_prob,
                                )
                                for _ in range(
                                    num_post_pool_gat_layers
                                )  # Use the new parameter
                            ]
                        )
                        for _ in range(num_diffpool_layers - 1)
                    ]
                )
                for _ in range(num_graphs)
            ]
        )

        # Norm layers for post-pooling GAT
        self.post_pool_gat_norm_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                self.get_norm_layer(norm, diffpool_hidden_channels)
                                for _ in range(
                                    num_post_pool_gat_layers
                                )  # Use the new parameter
                            ]
                        )
                        for _ in range(num_diffpool_layers - 1)
                    ]
                )
                for _ in range(num_graphs)
            ]
        )
        # Final linear layer
        self.final_linear = nn.Linear(
            num_graphs * diffpool_hidden_channels, diffpool_out_channels
        )
        # initialize weights
        self.init_weights()

    def init_weights(self):
        def init_func(module):
            if isinstance(module, nn.Linear):
                if self.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.weight_init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif self.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif self.weight_init != "default":
                    raise ValueError(
                        f"Unsupported initialization method: {self.weight_init}"
                    )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, GATv2Conv):
                if self.weight_init != "default":
                    if hasattr(module, "lin_l"):
                        if self.weight_init == "xavier_uniform":
                            nn.init.xavier_uniform_(module.lin_l.weight)
                            nn.init.xavier_uniform_(module.lin_r.weight)
                        elif self.weight_init == "xavier_normal":
                            nn.init.xavier_normal_(module.lin_l.weight)
                            nn.init.xavier_normal_(module.lin_r.weight)
                        elif self.weight_init == "kaiming_uniform":
                            nn.init.kaiming_uniform_(
                                module.lin_l.weight, nonlinearity="relu"
                            )
                            nn.init.kaiming_uniform_(
                                module.lin_r.weight, nonlinearity="relu"
                            )
                        elif self.weight_init == "kaiming_normal":
                            nn.init.kaiming_normal_(
                                module.lin_l.weight, nonlinearity="relu"
                            )
                            nn.init.kaiming_normal_(
                                module.lin_r.weight, nonlinearity="relu"
                            )
                    if hasattr(module, "att"):
                        nn.init.xavier_uniform_(module.att)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm1d, nn.InstanceNorm1d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.weight_init != "default":
            self.apply(init_func)

    def get_norm_layer(self, norm, channels):
        if norm is None:
            return nn.Identity()
        elif norm == "batch":
            return BatchNorm(channels)
        elif norm == "instance":
            return InstanceNorm(channels)
        elif norm == "layer":
            return LayerNorm(channels)
        elif norm == "graph":
            return GraphNorm(channels)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

    def forward(self, x, edge_indices: list[torch.Tensor], batch):
        graph_outputs = []
        attention_weights = []
        cluster_assignments = []
        link_pred_losses = []
        entropy_losses = []

        for i in range(self.num_graphs):
            edge_index = edge_indices[i]

            # Initial GAT layers
            x_graph = x
            for j, gat_layer in enumerate(self.initial_gat_layers[i]):
                x_out, (edge_index, att_weights) = gat_layer(
                    x_graph, edge_index, return_attention_weights=True
                )
                if self.initial_gat_norm_layers is not None:
                    norm_layer = self.initial_gat_norm_layers[i][j]
                    if isinstance(norm_layer, GraphNorm):
                        x_out = norm_layer(x_out, batch)
                    else:
                        x_out = norm_layer(x_out)
                x_out = self.activation(x_out)
                if self.gat_skip_connection and x_graph.shape == x_out.shape:
                    x_out = x_out + x_graph
                x_graph = x_out

                attention_weights.append((edge_index, att_weights))

            # Convert to dense batch
            x_dense, mask = to_dense_batch(x_graph, batch)
            adj = to_dense_adj(edge_index, batch)

            # DiffPool layers
            x_pool, adj_pool = x_dense, adj
            for k, diffpool_layer in enumerate(self.diffpool_layers[i]):
                s = diffpool_layer(x_pool)
                x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(
                    x_pool, adj_pool, s, mask
                )
                link_pred_losses.append(link_loss)
                entropy_losses.append(ent_loss)
                cluster_assignments.append(s)

                # Update batch information after pooling
                batch_size, num_nodes, _ = x_pool.size()
                new_batch = (
                    torch.arange(batch_size)
                    .repeat_interleave(num_nodes)
                    .to(x_pool.device)
                )

                # HACK Add self-loops by setting diagonal to 1
                # we do this to try to avoid 0 edges in mp which might cause NaNs
                adj_pool.diagonal(dim1=-2, dim2=-1).fill_(1)

                # Prune edges after pooling if specified
                if self.pruned_max_average_node_degree is not None:
                    adj_pool = self.prune_edges_dense(
                        adj_pool, self.pruned_max_average_node_degree
                    )

                # Post-pooling GAT layers (for all but the last DiffPool layer)
                if k < len(self.diffpool_layers[i]) - 1:
                    for l, gat_layer in enumerate(self.post_pool_gat_layers[i][k]):
                        x_pool_flat = x_pool.view(-1, x_pool.size(-1))
                        adj_pool_flat = adj_pool.view(-1, adj_pool.size(-1))
                        edge_index_pool = adj_pool_flat.nonzero().t()
                        x_out, att_weights = gat_layer(
                            x_pool_flat, edge_index_pool, return_attention_weights=True
                        )
                        if self.post_pool_gat_norm_layers is not None:
                            norm_layer = self.post_pool_gat_norm_layers[i][k][l]
                            if isinstance(norm_layer, GraphNorm):
                                x_out = norm_layer(x_out, new_batch)
                            else:
                                x_out = norm_layer(x_out)
                        x_out = self.activation(x_out)
                        if (
                            self.gat_skip_connection
                            and x_pool_flat.shape == x_out.shape
                        ):
                            x_out = x_out + x_pool_flat
                        x_pool = x_out.view(x_pool.size(0), -1, x_out.size(-1))
                        attention_weights.append(att_weights)

                # Update mask for the next iteration
                mask = torch.ones(
                    x_pool.size(0),
                    x_pool.size(1),
                    dtype=torch.bool,
                    device=x_pool.device,
                )

            graph_outputs.append(x_pool)

        # Concatenate outputs from all graphs
        x_concat = torch.cat(graph_outputs, dim=-1)

        # Final linear layer with activation and dropout
        out = self.final_linear(x_concat.squeeze(1))
        out = self.activation(out)
        out = F.dropout(out, p=self.last_layer_dropout_prob, training=self.training)

        return (
            out,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
        )

    def prune_edges_dense(self, adj, k):
        """
        Prune edges in a dense adjacency matrix to keep only the top k*n edges.

        Args:
        adj (torch.Tensor): Dense adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        k (int): Maximum average number of edges to keep per node

        Returns:
        torch.Tensor: Pruned dense adjacency matrix
        """
        batch_size, num_nodes, _ = adj.shape
        max_edges_to_keep = k * num_nodes

        # Create a mask for the upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(adj), diagonal=1)

        # Apply the mask and flatten the adjacency matrix
        adj_masked = adj * mask
        adj_flat = adj_masked.view(batch_size, -1)

        # Count the number of non-zero edges for each batch item
        num_edges = torch.sum(adj_flat != 0, dim=1)

        # Determine the number of edges to keep for each batch item
        edges_to_keep = torch.min(
            num_edges, torch.tensor(max_edges_to_keep, device=adj.device)
        )

        # Create a new adjacency matrix with only the top edges
        new_adj = torch.zeros_like(adj_masked)
        for i in range(batch_size):
            # Sort the edges for this batch item
            values, indices = torch.sort(adj_flat[i], descending=True)
            # Keep only the top k edges (or fewer if there aren't enough edges)
            top_indices = indices[: edges_to_keep[i]]
            new_adj[i].view(-1)[top_indices] = values[: edges_to_keep[i]]

        # Make the adjacency matrix symmetric
        new_adj = new_adj + new_adj.transpose(-1, -2)

        return new_adj


def load_sample_data_batch():
    import os
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.data import MeanExperimentDeduplicator
    from torchcell.data import GenotypeAggregator
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.data import Neo4jCellDataset
    from torchcell.data.neo4j_cell import PhenotypeProcessor
    from tqdm import tqdm

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # fudt_3prime_dataset = FungalUpDownTransformerDataset(
    #     root="data/scerevisiae/fudt_embedding",
    #     genome=genome,
    #     model_name="species_downstream",
    # )
    # fudt_5prime_dataset = FungalUpDownTransformerDataset(
    #     root="data/scerevisiae/fudt_embedding",
    #     genome=genome,
    #     model_name="species_downstream",
    # )
    # node_embeddings={
    #     "fudt_3prime": fudt_3prime_dataset,
    #     "fudt_5prime": fudt_5prime_dataset,
    # },
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={"codon_frequency": codon_frequency},
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=PhenotypeProcessor(),
    )

    seed = 42
    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=seed,
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 1e1
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=4,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()

    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    max_num_nodes = len(dataset.gene_set)
    return batch, max_num_nodes


def main():
    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x
    edge_index_physical = batch["gene", "physical_interaction", "gene"].edge_index
    edge_index_regulatory = batch["gene", "regulatory_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = batch["gene"].fitness.unsqueeze(1)  # Assuming fitness is the target

    # Model configuration
    model = GatDiffPool(
        in_channels=x.size(1),
        initial_gat_hidden_channels=8,
        initial_gat_out_channels=8,
        diffpool_hidden_channels=8,
        diffpool_out_channels=1,
        num_initial_gat_layers=2,
        num_diffpool_layers=3,
        num_post_pool_gat_layers=1,
        num_graphs=2,
        max_num_nodes=max_num_nodes,
        gat_dropout_prob=0.0,
        last_layer_dropout_prob=0.2,
        cluster_size_decay_factor=10.0,
        norm="graph",
        activation="gelu",
        gat_skip_connection=True,
        pruned_max_average_node_degree=None,
        weight_init="xavier_uniform",
    )

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Forward pass
    out, attention_weights, cluster_assignments, link_pred_losses, entropy_losses = (
        model(x, [edge_index_physical, edge_index_regulatory], batch_index)
    )

    # Compute loss
    mse_loss = F.mse_loss(out, y)
    link_pred_loss = sum(link_pred_losses)
    entropy_loss = sum(entropy_losses)
    total_loss = mse_loss + 0.1 * link_pred_loss + 0.1 * entropy_loss

    print(f"Initial loss: {total_loss.item()}")

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradients
    print("\nChecking gradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: grad_norm = {grad_norm}")
            if torch.isnan(grad_norm):
                print(f"NaN gradient detected in {name}")
                print(
                    f"Parameter stats: min={param.min()}, max={param.max()}, mean={param.mean()}"
                )
        else:
            print(f"{name}: No gradient")

    # Optionally, you can also print the model's state_dict to check parameter values
    # print("\nModel state_dict:")
    # for name, param in model.state_dict().items():
    #     print(f"{name}: min={param.min()}, max={param.max()}, mean={param.mean()}")

    print("\nOutput shape:", out.shape)
    print("Number of attention weight tensors:", len(attention_weights))
    print("First attention weight shape:", attention_weights[0][0].shape)
    print("Last attention weight shape:", attention_weights[-1][0].shape)
    print("Number of cluster assignment tensors:", len(cluster_assignments))
    print("First cluster assignment shape:", cluster_assignments[0].shape)
    print("Last cluster assignment shape:", cluster_assignments[-1].shape)
    print("Link prediction losses:", link_pred_losses)
    print("Entropy losses:", entropy_losses)


if __name__ == "__main__":
    main()
