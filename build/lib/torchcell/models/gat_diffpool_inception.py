# torchcell/models/gat_diffpool_inception
# [[torchcell.models.gat_diffpool_inception]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/gat_diffpool_inception
# Test file: tests/torchcell/models/test_gat_diffpool_inception.py

"""GAT-with-DiffPool inception model over multiple graphs for graph-level prediction."""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GraphNorm,
    InstanceNorm,
    LayerNorm,
    MeanSubtractionNorm,
    PairNorm,
    dense_diff_pool,
)
from torch_geometric.utils import to_dense_adj, to_dense_batch

from torchcell.models.act import act_register


class GatDiffPool(nn.Module):
    """GATv2 encoder with hierarchical DiffPool layers over a set of input graphs."""

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
        pruned_max_average_node_degree: int | None = None,
        weight_init: str = "default",
        target_dim: int = 2,
        cluster_reduction: str = "mean",
    ):
        """Build the GAT embedding, DiffPool, post-pool, and prediction layers.

        Args:
            in_channels: Input node feature dimension.
            initial_gat_hidden_channels: Hidden dimension of the initial GAT stack.
            initial_gat_out_channels: Output dimension of the initial GAT stack.
            diffpool_hidden_channels: Hidden dimension of the DiffPool GNNs.
            diffpool_out_channels: Output dimension of the DiffPool embedding GNNs.
            num_initial_gat_layers: Number of GAT layers before pooling.
            num_diffpool_layers: Number of hierarchical DiffPool layers.
            num_post_pool_gat_layers: Number of GAT layers after pooling.
            num_graphs: Number of parallel input graphs (inception branches).
            max_num_nodes: Maximum number of nodes per graph.
            cluster_size_decay_factor: Factor reducing cluster count per pool layer.
            gat_dropout_prob: Dropout probability inside GAT layers.
            last_layer_dropout_prob: Dropout probability before the final head.
            norm: Normalization type for conv layers.
            activation: Activation key from ``act_register``.
            gat_skip_connection: Whether GAT layers use skip connections.
            pruned_max_average_node_degree: Optional cap on average node degree.
            weight_init: Weight initialization scheme.
            target_dim: Output prediction dimension.
            cluster_reduction: Reduction used when collapsing clusters.
        """
        super().__init__()
        self.weight_init = weight_init
        self.cluster_size_decay_factor = cluster_size_decay_factor
        self.target_dim = target_dim
        self.cluster_reduction = cluster_reduction

        # Add linear layers for each diffpool layer
        self.cluster_prediction_layers = nn.ModuleList(
            [
                nn.Linear(diffpool_hidden_channels, target_dim)
                for _ in range(num_diffpool_layers)
            ]
        )

        assert norm in [
            None,
            "batch",
            "instance",
            "layer",
            "graph",
            "pair",
            "mean_subtraction",
        ], "Invalid norm type"

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

        self.output_linear = nn.Linear(diffpool_out_channels, target_dim)

        # initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize module weights according to the configured scheme."""

        def init_func(module: nn.Module) -> None:
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

    def get_norm_layer(self, norm: str, channels: int) -> nn.Module:
        """Return the PyG normalization layer for the given type and channels."""
        if norm is None:
            return nn.Identity()
        elif norm == "batch":
            return cast(nn.Module, BatchNorm(channels))
        elif norm == "instance":
            return cast(nn.Module, InstanceNorm(channels))
        elif norm == "layer":
            return cast(nn.Module, LayerNorm(channels))
        elif norm == "graph":
            return cast(nn.Module, GraphNorm(channels))
        elif norm == "pair":
            return cast(nn.Module, PairNorm())
        elif norm == "mean_subtraction":
            return cast(nn.Module, MeanSubtractionNorm())
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

    def forward(
        self, x: torch.Tensor, edge_indices: list[torch.Tensor], batch: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        list[Any],  # mixed: (edge_index, att_weights) tuples and bare att_weights
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """Encode the input graphs through GAT and DiffPool and return predictions."""
        graph_outputs = []
        # mixed: (edge_index, att_weights) tuples and bare att_weights
        attention_weights: list[Any] = []
        cluster_assignments: list[torch.Tensor] = []
        link_pred_losses = []
        entropy_losses = []
        cluster_predictions = []

        for i in range(self.num_graphs):
            edge_index = edge_indices[i]

            # Initial GAT layers
            x_graph = x
            gat_layer: nn.Module
            x_out: torch.Tensor
            att_weights: torch.Tensor
            initial_gat_seq = cast(nn.ModuleList, self.initial_gat_layers[i])
            for j, gat_layer in enumerate(initial_gat_seq):
                x_out, (edge_index, att_weights) = gat_layer(
                    x_graph, edge_index, return_attention_weights=True
                )
                if self.initial_gat_norm_layers is not None:
                    norm_layer = cast(nn.ModuleList, self.initial_gat_norm_layers[i])[j]
                    if isinstance(
                        norm_layer, (GraphNorm, PairNorm, MeanSubtractionNorm)
                    ):
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
            diffpool_layer: nn.Module
            s: torch.Tensor
            diffpool_seq = cast(nn.ModuleList, self.diffpool_layers[i])
            for k, diffpool_layer in enumerate(diffpool_seq):
                s = diffpool_layer(x_pool)
                x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(
                    x_pool, adj_pool, s, mask
                )
                link_pred_losses.append(link_loss)
                entropy_losses.append(ent_loss)
                cluster_assignments.append(s)

                # Make predictions for each cluster
                cluster_pred = self.cluster_prediction_layers[k](x_pool)
                if k < len(diffpool_seq) - 1:  # Not the last layer
                    if self.cluster_reduction == "mean":
                        cluster_pred = cluster_pred.mean(dim=1)
                    elif self.cluster_reduction == "sum":
                        cluster_pred = cluster_pred.sum(dim=1)
                else:  # Last layer
                    cluster_pred = cluster_pred.squeeze(1)
                cluster_predictions.append(cluster_pred)

                # Update batch information after pooling
                batch_size, num_nodes, _ = x_pool.size()
                new_batch = (
                    torch.arange(batch_size)
                    .repeat_interleave(num_nodes)
                    .to(x_pool.device)
                )

                # Add self-loops by setting diagonal to 1
                adj_pool.diagonal(dim1=-2, dim2=-1).fill_(1)

                # Prune edges after pooling if specified
                if self.pruned_max_average_node_degree is not None:
                    adj_pool = self.prune_edges_dense(
                        adj_pool, self.pruned_max_average_node_degree
                    )

                # Post-pooling GAT layers (for all but the last DiffPool layer)
                if k < len(diffpool_seq) - 1:
                    post_pool_gat_seq = cast(
                        nn.ModuleList,
                        cast(nn.ModuleList, self.post_pool_gat_layers[i])[k],
                    )
                    for layer_idx, gat_layer in enumerate(post_pool_gat_seq):
                        x_pool_flat = x_pool.view(-1, x_pool.size(-1))
                        adj_pool_flat = adj_pool.view(-1, adj_pool.size(-1))
                        edge_index_pool = adj_pool_flat.nonzero().t()
                        x_out, att_weights = gat_layer(
                            x_pool_flat, edge_index_pool, return_attention_weights=True
                        )
                        if self.post_pool_gat_norm_layers is not None:
                            norm_layer = cast(
                                nn.ModuleList,
                                cast(nn.ModuleList, self.post_pool_gat_norm_layers[i])[
                                    k
                                ],
                            )[layer_idx]
                            if isinstance(
                                norm_layer, (GraphNorm, PairNorm, MeanSubtractionNorm)
                            ):
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
        final_linear_output = self.final_linear(
            x_concat.squeeze(1)
        )  # Save this to track contributions
        out = self.activation(final_linear_output)
        out = F.dropout(out, p=self.last_layer_dropout_prob, training=self.training)

        # Output layer
        out = self.output_linear(out)

        # Return contributions as part of the output
        return (
            out,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
            cluster_predictions,
            final_linear_output,  # Track the feature contributions here
        )

    def prune_edges_dense(self, adj: torch.Tensor, k: int) -> torch.Tensor:
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


def load_sample_data_batch() -> tuple[Any, int]:
    """Load and return a sample data batch for testing the model."""
    import os
    import os.path as osp

    from dotenv import load_dotenv
    from tqdm import tqdm

    from torchcell.data import (
        GenotypeAggregator,
        MeanExperimentDeduplicator,
        Neo4jCellDataset,
    )
    from torchcell.data.neo4j_cell import (  # type: ignore[attr-defined]  # dev-only helper; symbol lives in graph_processor
        SubgraphRepresentation,
    )
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    assert DATA_ROOT is not None

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        data_root=osp.join(  # type: ignore[call-arg]  # dev-only helper; SCerevisiaeGraph uses sgd_root
            DATA_ROOT, "data/sgd/genome"
        ),
        genome=genome,
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

    with open("experiments/003-fit-int/queries/001-small-build.cql") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={  # type: ignore[arg-type]  # dev-only helper; runtime accepts dict of graphs
            "physical": graph.G_physical,
            "regulatory": graph.G_regulatory,
        },
        node_embeddings={"codon_frequency": codon_frequency},
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
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


def main() -> None:
    """Build the model on sample data and run a forward pass for testing."""
    from torchcell.losses.multi_dim_nan_tolerant import CombinedRegressionLoss

    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x
    edge_index_physical = batch["gene", "physical_interaction", "gene"].edge_index
    edge_index_regulatory = batch["gene", "regulatory_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Model configuration
    model = GatDiffPool(
        in_channels=x.size(1),
        initial_gat_hidden_channels=8,
        initial_gat_out_channels=8,
        diffpool_hidden_channels=8,
        diffpool_out_channels=2,
        num_initial_gat_layers=2,
        num_diffpool_layers=10,
        num_post_pool_gat_layers=1,
        num_graphs=2,
        max_num_nodes=max_num_nodes,
        gat_dropout_prob=0.0,
        last_layer_dropout_prob=0.2,
        cluster_size_decay_factor=2.0,
        norm="mean_subtraction",
        activation="gelu",
        gat_skip_connection=True,
        pruned_max_average_node_degree=None,
        weight_init="xavier_uniform",
        target_dim=2,
        cluster_reduction="mean",
    )

    # Initialize CombinedRegressionLoss with MSE
    # Using equal weights for both dimensions (fitness and gene interaction)
    criterion = CombinedRegressionLoss(loss_type="mse", weights=torch.ones(2))

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(3):  # Run for 3 epochs to check gradient flow
        optimizer.zero_grad()

        # Forward pass
        (
            out,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
            cluster_predictions,
            final_linear_output,
        ) = model(x, [edge_index_physical, edge_index_regulatory], batch_index)

        # Compute primary loss using CombinedRegressionLoss
        main_loss, dim_losses = criterion(out, y)

        # Compute auxiliary losses
        link_pred_loss = sum(link_pred_losses)
        entropy_loss = sum(entropy_losses)
        cluster_loss = sum(criterion(pred, y)[0] for pred in cluster_predictions)

        # Combine all losses
        total_loss = (
            main_loss + 0.1 * link_pred_loss + 0.1 * entropy_loss + 0.1 * cluster_loss
        )

        # Print losses
        print(f"\nEpoch {epoch + 1}")
        print(f"Main loss: {main_loss.item():.4f}")
        print(
            f"Dimension losses: fitness={dim_losses[0].item():.4f}, "
            f"gene_interaction={dim_losses[1].item():.4f}"
        )
        print(f"Link prediction loss: {link_pred_loss.item():.4f}")
        print(f"Entropy loss: {entropy_loss.item():.4f}")
        print(f"Cluster loss: {cluster_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")

        # Backward pass
        total_loss.backward()

        # Check gradients - Fixed this section
        print("\nGradient norms:")
        for name, param in model.named_parameters():  # Changed from model.parameters()
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: {grad_norm:.4f}")

                # Check for NaN gradients
                if torch.isnan(param.grad).any():
                    print(f"WARNING: NaN gradient detected in {name}")
                    print(
                        f"Parameter stats: min={param.min().item():.4f}, "
                        f"max={param.max().item():.4f}, "
                        f"mean={param.mean().item():.4f}"
                    )

        # Update weights
        optimizer.step()

        # Verify model outputs after update
        with torch.no_grad():
            new_out = model(
                x, [edge_index_physical, edge_index_regulatory], batch_index
            )[0]
            print("\nOutput statistics after update:")
            print(f"Output mean: {new_out.mean().item():.4f}")
            print(f"Output std: {new_out.std().item():.4f}")
            print(
                f"Output range: [{new_out.min().item():.4f}, {new_out.max().item():.4f}]"
            )


if __name__ == "__main__":
    main()
