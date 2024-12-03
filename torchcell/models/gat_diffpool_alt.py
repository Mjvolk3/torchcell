import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    BatchNorm,
    LayerNorm,
    GraphNorm,
    InstanceNorm,
    PairNorm,
    MeanSubtractionNorm,
)
from torch_geometric.utils import to_dense_batch, to_dense_adj
from typing import Optional
from torchcell.models.act import act_register
from torch_geometric.nn import GATv2Conv, dense_diff_pool


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
        pruned_max_average_node_degree: Optional[int] = None,
        weight_init: str = "default",
        target_dim: int = 2,
        cluster_reduction: str = "mean",
    ):
        super().__init__()
        self.pruned_max_average_node_degree = pruned_max_average_node_degree
        self.num_graphs = num_graphs
        self.activation = act_register[activation]
        self.last_layer_dropout_prob = last_layer_dropout_prob
        self.gat_skip_connection = gat_skip_connection
        self.weight_init = weight_init
        self.target_dim = target_dim
        self.cluster_reduction = cluster_reduction

        self.cluster_sizes = []
        for i in range(1, num_diffpool_layers + 1):
            # Ensure cluster sizes are reasonable
            size = max(2, int(max_num_nodes / (cluster_size_decay_factor**i)))
            if size > max_num_nodes:
                size = max_num_nodes // 2
            self.cluster_sizes.append(size)
        self.cluster_sizes[-1] = 1
        print(f"Cluster sizes: {self.cluster_sizes}")

        # Create GNN layers for each graph
        self.gnn_embed_layers = nn.ModuleList()
        self.gnn_pool_layers = nn.ModuleList()
        self.norm_embed_layers = nn.ModuleList()
        self.norm_pool_layers = nn.ModuleList()

        for _ in range(num_graphs):
            # Embedding GNN layers (GCN)
            embed_layers = nn.ModuleList()
            embed_norms = nn.ModuleList()
            curr_dim = in_channels

            for i in range(num_initial_gat_layers):
                out_dim = (
                    initial_gat_out_channels
                    if i == num_initial_gat_layers - 1
                    else initial_gat_hidden_channels
                )
                embed_layers.append(GCNConv(curr_dim, out_dim))
                embed_norms.append(self.get_norm_layer(norm, out_dim))
                curr_dim = out_dim

            self.gnn_embed_layers.append(embed_layers)
            self.norm_embed_layers.append(embed_norms)

            # Pooling GNN layers (GAT) - one set for each DiffPool layer
            pool_layers = nn.ModuleList()
            pool_norms = nn.ModuleList()

            for cluster_size in self.cluster_sizes:
                # GAT layers for assignment
                gat_layers = nn.ModuleList()
                gat_norms = nn.ModuleList()
                curr_dim = initial_gat_out_channels  # Use the output dimension from embedding GNN

                for i in range(num_initial_gat_layers):
                    out_dim = initial_gat_hidden_channels
                    gat_layers.append(
                        GATv2Conv(
                            in_channels=curr_dim,  # Input dimension should match previous layer
                            out_channels=out_dim,  # Output dimension
                            heads=1,
                            dropout=gat_dropout_prob,
                            concat=False,  # Don't concatenate head outputs
                        )
                    )
                    gat_norms.append(self.get_norm_layer(norm, out_dim))
                    curr_dim = out_dim  # Update current dimension

                # Final projection to cluster size
                gat_layers.append(nn.Linear(curr_dim, cluster_size))
                pool_layers.append(gat_layers)
                pool_norms.append(gat_norms)

            self.gnn_pool_layers.append(pool_layers)
            self.norm_pool_layers.append(pool_norms)

        # Prediction layers for each pooling level
        self.cluster_prediction_layers = nn.ModuleList(
            [
                nn.Linear(diffpool_hidden_channels, target_dim)
                for _ in range(num_diffpool_layers)
            ]
        )

        # Final output layers
        self.final_linear = nn.Linear(
            num_graphs * diffpool_hidden_channels, diffpool_out_channels
        )
        self.output_linear = nn.Linear(diffpool_out_channels, target_dim)

        # Initialize weights
        self.init_weights()

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
        elif norm == "pair":
            return PairNorm()
        elif norm == "mean_subtraction":
            return MeanSubtractionNorm()
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

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
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, GCNConv):
                if self.weight_init != "default":
                    if hasattr(module, "lin"):  # GCNConv uses 'lin' instead of 'weight'
                        if self.weight_init == "xavier_uniform":
                            nn.init.xavier_uniform_(module.lin.weight)
                        elif self.weight_init == "xavier_normal":
                            nn.init.xavier_normal_(module.lin.weight)
                        elif self.weight_init == "kaiming_uniform":
                            nn.init.kaiming_uniform_(
                                module.lin.weight, nonlinearity="relu"
                            )
                        elif self.weight_init == "kaiming_normal":
                            nn.init.kaiming_normal_(
                                module.lin.weight, nonlinearity="relu"
                            )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x, edge_indices: list[torch.Tensor], batch):
        graph_outputs = []
        attention_weights = []  # Store GAT attention weights
        cluster_assignments = []  # Store cluster assignments
        link_pred_losses = []
        entropy_losses = []
        cluster_predictions = []  # Store predictions at each level

        # Process each graph type (e.g., physical, regulatory)
        for i in range(self.num_graphs):
            edge_index = edge_indices[i]

            # Initial embedding using GCN
            x_current = x
            for gcn, norm in zip(self.gnn_embed_layers[i], self.norm_embed_layers[i]):
                x_current = gcn(x_current, edge_index)
                if isinstance(norm, (GraphNorm, PairNorm, MeanSubtractionNorm)):
                    x_current = norm(x_current, batch)
                else:
                    x_current = norm(x_current)
                x_current = self.activation(x_current)

            # Convert to dense batch for initial DiffPool input
            x_dense, mask = to_dense_batch(x_current, batch)
            adj_dense = to_dense_adj(edge_index, batch)

            current_x = x_dense
            current_adj = adj_dense
            current_batch = batch
            current_edge_index = edge_index
            current_mask = mask

            # Apply hierarchical pooling
            for layer_idx, cluster_size in enumerate(self.cluster_sizes):
                # Compute cluster assignments using GAT
                s = x_current  # Use current node features
                gat_layers = self.gnn_pool_layers[i][layer_idx]
                gat_norms = self.norm_pool_layers[i][layer_idx]

                # Apply GAT layers for assignment
                for j, (gat, norm) in enumerate(zip(gat_layers[:-1], gat_norms)):
                    # Apply GAT and get attention weights
                    s, (edge_index_out, att_weights) = gat(
                        s,  # Current features
                        current_edge_index,  # Current edge indices
                        edge_attr=None,  # Explicitly set edge_attr to None
                        return_attention_weights=True,
                    )

                    # Prune edges if specified
                    if self.pruned_max_average_node_degree is not None:
                        edge_index_out, att_weights = self.prune_edges_with_attention(
                            edge_index_out,
                            att_weights,
                            self.pruned_max_average_node_degree,
                        )

                    # Apply normalization
                    if isinstance(norm, (GraphNorm, PairNorm, MeanSubtractionNorm)):
                        s = norm(s, current_batch)
                    else:
                        s = norm(s)

                    s = self.activation(s)
                    if (
                        self.gat_skip_connection and j > 0
                    ):  # Skip connection after first layer
                        s = s + x_current  # Skip connection with current node features

                    attention_weights.append(att_weights)
                    current_edge_index = edge_index_out

                # Final projection to get assignment matrix
                s = gat_layers[-1](s)  # Linear layer for cluster assignment

                # Convert assignment scores to dense format
                s_dense, _ = to_dense_batch(s, current_batch)

                # Apply DiffPool
                pooled_x, pooled_adj, link_loss, ent_loss = dense_diff_pool(
                    current_x,  # Current node features [batch_size, num_nodes, channels]
                    current_adj,  # Current adjacency matrix [batch_size, num_nodes, num_nodes]
                    s_dense,  # Assignment matrix [batch_size, num_nodes, num_clusters]
                    current_mask,  # Current mask [batch_size, num_nodes]
                )

                # Store assignments and losses
                cluster_assignments.append(s_dense)
                link_pred_losses.append(link_loss)
                entropy_losses.append(ent_loss)

                # Make predictions at this level
                cluster_pred = self.cluster_prediction_layers[layer_idx](pooled_x)
                if layer_idx < len(self.cluster_sizes) - 1:  # Not the last layer
                    if self.cluster_reduction == "mean":
                        cluster_pred = cluster_pred.mean(dim=1)
                    elif self.cluster_reduction == "sum":
                        cluster_pred = cluster_pred.sum(dim=1)
                else:  # Last layer
                    cluster_pred = cluster_pred.squeeze(1)  # Remove cluster dimension
                cluster_predictions.append(cluster_pred)

                # Update tensors for next layer
                current_x = pooled_x
                current_adj = pooled_adj
                batch_size, num_nodes, _ = pooled_x.size()

                # Create new batch tensor for pooled nodes
                current_batch = torch.arange(
                    batch_size, device=x.device
                ).repeat_interleave(num_nodes)

                # Create new mask for pooled nodes
                current_mask = torch.ones(
                    batch_size, num_nodes, dtype=torch.bool, device=x.device
                )

                # Convert pooled features to sparse format for next GAT layer
                x_current = pooled_x.reshape(-1, pooled_x.size(-1))

                # Get new edge index from pooled adjacency matrix
                current_edge_index = current_adj.squeeze(0).nonzero().t()

            # Store final pooled representation
            graph_outputs.append(current_x)

        # Combine outputs from all graphs
        x_concat = torch.cat(graph_outputs, dim=-1)

        # Final prediction
        final_linear_output = self.final_linear(x_concat.squeeze(1))
        out = self.activation(final_linear_output)
        out = F.dropout(out, p=self.last_layer_dropout_prob, training=self.training)
        out = self.output_linear(out)

        return (
            out,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
            cluster_predictions,
            final_linear_output,
        )

    def prune_edges_with_attention(self, edge_index, attention_weights, k):
        """
        Prune edges using attention weights, keeping only top k edges per node.

        Args:
            edge_index (torch.Tensor): Edge index tensor [2, num_edges]
            attention_weights (torch.Tensor): Attention weights from GAT [num_edges]
            k (int): Number of edges to keep per node
        """
        device = edge_index.device

        # Group edges by source node
        edge_dict = {}
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            if src not in edge_dict:
                edge_dict[src] = []
            edge_dict[src].append((i, attention_weights[i].item()))

        # Keep top k edges per node
        kept_edge_indices = []
        for src in edge_dict:
            edges = sorted(edge_dict[src], key=lambda x: x[1], reverse=True)
            kept_edges = edges[:k]
            kept_edge_indices.extend([e[0] for e in kept_edges])

        # Create new edge_index and attention weights
        kept_edge_indices = torch.tensor(kept_edge_indices, device=device)
        new_edge_index = edge_index[:, kept_edge_indices]
        new_attention_weights = attention_weights[kept_edge_indices]

        return new_edge_index, new_attention_weights


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
    from torchcell.data.neo4j_cell import SubgraphRepresentation
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


def main():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss

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
        num_diffpool_layers=3,
        num_post_pool_gat_layers=1,
        num_graphs=2,
        max_num_nodes=max_num_nodes,
        gat_dropout_prob=0.0,
        last_layer_dropout_prob=0.2,
        cluster_size_decay_factor=2.0,
        norm="pair",
        activation="gelu",
        gat_skip_connection=True,
        pruned_max_average_node_degree=None,
        weight_init="xavier_uniform",
        target_dim=2,
        cluster_reduction="mean",
    )

    # Initialize CombinedLoss with MSE
    # Using equal weights for both dimensions (fitness and gene interaction)
    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))

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

        # Compute primary loss using CombinedLoss
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
