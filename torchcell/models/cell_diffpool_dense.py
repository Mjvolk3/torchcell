import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torch_geometric.models.dense_gat_conv import DenseGATConv
from typing import Optional, Literal
from torchcell.models.act import act_register


class DenseDiffPool(nn.Module):
    def __init__(
        self,
        max_num_nodes: int,
        in_channels: int,
        pool_gat_hidden_channels: int,
        num_pool_gat_layers: int,
        embed_gat_hidden_channels: int,
        num_embed_gat_layers: int,
        num_pooling_layers: int,
        cluster_size_decay_factor: float = 10.0,
        activation: str = "relu",
        norm: str = None,
        target_dim: int = 2,
        cluster_aggregation: Literal["mean", "sum"] = "mean",
        heads: int = 1,
        concat: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Set parameters
        self.max_num_nodes = max_num_nodes
        self.in_channels = in_channels
        self.activation = act_register[activation]
        self.target_dim = target_dim
        self.cluster_aggregation = cluster_aggregation

        # Calculate cluster sizes
        self.cluster_sizes = []
        for i in range(1, num_pooling_layers + 1):
            size = max(1, int(max_num_nodes / (cluster_size_decay_factor**i)))
            self.cluster_sizes.append(size)
        self.cluster_sizes[-1] = 1
        print(f"Cluster sizes: {self.cluster_sizes}")

        # Create cluster prediction heads
        self.cluster_heads = nn.ModuleList()
        for _ in range(num_pooling_layers):
            self.cluster_heads.append(nn.Linear(embed_gat_hidden_channels, target_dim))

        # Create pooling networks (Dense GAT) for each layer
        self.pool_networks = nn.ModuleList()
        for layer in range(num_pooling_layers):
            pool_gnn = nn.ModuleList()
            curr_in_channels = in_channels if layer == 0 else embed_gat_hidden_channels

            # Add Dense GAT layers for pooling
            for i in range(num_pool_gat_layers):
                is_last = i == num_pool_gat_layers - 1
                out_channels = (
                    self.cluster_sizes[layer] if is_last else pool_gat_hidden_channels
                )

                pool_gnn.extend(
                    [
                        DenseGATConv(
                            in_channels=curr_in_channels,
                            out_channels=out_channels,
                            heads=heads,
                            concat=concat,
                            dropout=dropout,
                        ),
                        (
                            self.get_norm_layer(norm, out_channels)
                            if norm
                            else nn.Identity()
                        ),
                    ]
                )
                curr_in_channels = out_channels

            self.pool_networks.append(pool_gnn)

        # Create embedding networks (Dense GAT) for each layer
        self.embed_networks = nn.ModuleList()
        for layer in range(num_pooling_layers):
            embed_gnn = nn.ModuleList()
            curr_in_channels = in_channels if layer == 0 else embed_gat_hidden_channels

            # Add Dense GAT layers for embedding
            for i in range(num_embed_gat_layers):
                embed_gnn.extend(
                    [
                        DenseGATConv(
                            in_channels=curr_in_channels,
                            out_channels=embed_gat_hidden_channels,
                            heads=heads,
                            concat=concat,
                            dropout=dropout,
                        ),
                        (
                            self.get_norm_layer(norm, embed_gat_hidden_channels)
                            if norm
                            else nn.Identity()
                        ),
                    ]
                )
                curr_in_channels = embed_gat_hidden_channels

            self.embed_networks.append(embed_gnn)

        # Final prediction layer
        self.lin = nn.Linear(embed_gat_hidden_channels, target_dim)

    def get_norm_layer(self, norm, channels):
        if norm is None:
            return nn.Identity()
        elif norm == "batch":
            return nn.BatchNorm1d(channels)
        elif norm == "layer":
            # LayerNorm normalizes over the channel dimension
            return nn.LayerNorm(channels)
        elif norm == "instance":
            # InstanceNorm1d computes mean and std across each channel independently
            return nn.InstanceNorm1d(channels, affine=True)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

    def bn(self, x, norm_layer):
        """Apply normalization to the input tensor."""
        batch_size, num_nodes, num_channels = x.size()

        if isinstance(norm_layer, nn.BatchNorm1d):
            # BatchNorm expects [N, C] format
            x = x.view(-1, num_channels)
            x = norm_layer(x)
            x = x.view(batch_size, num_nodes, num_channels)
        elif isinstance(norm_layer, nn.InstanceNorm1d):
            # InstanceNorm1d expects [N, C, L] format
            x = x.permute(0, 2, 1)  # [batch_size, channels, nodes]
            x = norm_layer(x)
            x = x.permute(0, 2, 1)  # [batch_size, nodes, channels]
        elif isinstance(norm_layer, nn.LayerNorm):
            # LayerNorm can handle [B, N, C] format directly
            x = norm_layer(x)

        return x

    def forward(self, x, adj, mask=None):
        """
        Forward pass with dense tensors.

        Args:
            x (Tensor): Node feature tensor [batch_size, num_nodes, in_channels]
            adj (Tensor): Adjacency tensor [batch_size, num_nodes, num_nodes]
            mask (Tensor, optional): Mask tensor [batch_size, num_nodes]
        """
        # Initialize variables to store outputs
        cluster_assignments_list = []
        link_losses = []
        entropy_losses = []
        clusters_out = []

        # Process each pooling layer
        for layer, (pool_gnn, embed_gnn) in enumerate(
            zip(self.pool_networks, self.embed_networks)
        ):
            # Compute cluster assignment matrix (S) using Dense GAT
            s = x
            for i in range(0, len(pool_gnn), 2):  # Process conv and norm pairs
                conv_layer = pool_gnn[i]
                norm_layer = pool_gnn[i + 1]

                s = conv_layer(s, adj, mask=mask, add_loop=True)
                s = self.bn(s, norm_layer)
                s = self.activation(s)

            # Store cluster assignments
            cluster_assignments_list.append(s)

            # Compute node embeddings (Z) using Dense GAT
            z = x
            for i in range(0, len(embed_gnn), 2):  # Process conv and norm pairs
                conv_layer = embed_gnn[i]
                norm_layer = embed_gnn[i + 1]

                z = conv_layer(z, adj, mask=mask, add_loop=True)
                z = self.bn(z, norm_layer)
                z = self.activation(z)

            # Apply diffpool
            x, adj, link_loss, ent_loss = dense_diff_pool(x=z, adj=adj, s=s, mask=mask)

            # After clustering, all nodes are valid (no masking needed)
            mask = None

            # Store losses
            link_losses.append(link_loss)
            entropy_losses.append(ent_loss)

            if self.cluster_aggregation == "mean":
                cluster_x = x.mean(dim=1)
            elif self.cluster_aggregation == "sum":
                cluster_x = x.sum(dim=1)
            cluster_pred = self.cluster_heads[layer](cluster_x)
            clusters_out.append(cluster_pred)

        out = self.lin(x.squeeze(1))

        return (
            out,
            link_losses,
            entropy_losses,
            cluster_assignments_list,
            clusters_out,
        )


class DenseCellDiffPool(nn.Module):
    def __init__(
        self,
        num_graphs: int,
        max_num_nodes: int,
        in_channels: int,
        pool_gat_hidden_channels: int,
        num_pool_gat_layers: int,
        embed_gat_hidden_channels: int,
        num_embed_gat_layers: int,
        num_pooling_layers: int,
        cluster_size_decay_factor: float = 10.0,
        activation: str = "relu",
        norm: str = None,
        target_dim: int = 2,
        cluster_aggregation: Literal["mean", "sum"] = "mean",
        heads: int = 1,
        concat: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_graphs = num_graphs

        # Store initialization parameters
        self.init_params = {
            "max_num_nodes": max_num_nodes,
            "in_channels": in_channels,
            "pool_gat_hidden_channels": pool_gat_hidden_channels,
            "num_pool_gat_layers": num_pool_gat_layers,
            "embed_gat_hidden_channels": embed_gat_hidden_channels,
            "num_embed_gat_layers": num_embed_gat_layers,
            "num_pooling_layers": num_pooling_layers,
            "cluster_size_decay_factor": cluster_size_decay_factor,
            "activation": activation,
            "norm": norm,
            "target_dim": target_dim,
            "cluster_aggregation": cluster_aggregation,
            "heads": heads,
            "concat": concat,
            "dropout": dropout,
        }

        # Initialize models for each graph
        self.graph_models = nn.ModuleDict()

        # Store parameters
        self.target_dim = target_dim
        self.activation = act_register[activation]

        # Final combination layer
        self.final_combination = nn.Sequential(
            nn.Linear(target_dim * num_graphs, target_dim * 2),
            self.activation,
            nn.Linear(target_dim * 2, target_dim),
        )

    def _create_model_for_graph(self, graph_name: str):
        """Helper method to create a new DenseDiffPool model for a graph type"""
        return DenseDiffPool(**self.init_params)

    def forward(self, x, adj_dict: dict[str, torch.Tensor], mask=None):
        """
        Forward pass handling multiple graphs with dense tensors.

        Args:
            x (Tensor): Node features [batch_size, num_nodes, in_channels]
            adj_dict (dict): Dictionary mapping graph names to adjacency tensors
            mask (Tensor, optional): Mask tensor [batch_size, num_nodes]
        """
        if len(adj_dict) != self.num_graphs:
            raise ValueError(
                f"Expected {self.num_graphs} graphs but got {len(adj_dict)}. "
                f"Please provide exactly {self.num_graphs} adjacency matrices."
            )

        # Store outputs for each graph
        graph_outputs = {}
        graph_link_losses = {}
        graph_entropy_losses = {}
        graph_pool_attention = {}
        graph_embed_attention = {}
        graph_cluster_assignments = {}
        graph_cluster_outputs = {}
        individual_predictions = {}

        # Process each graph
        for graph_name, adj in adj_dict.items():
            # Create model if it doesn't exist
            if graph_name not in self.graph_models:
                self.graph_models[graph_name] = self._create_model_for_graph(graph_name)

            # Forward pass through individual graph model
            (
                out,
                link_losses,
                entropy_losses,
                pool_attention_weights,
                embed_attention_weights,
                cluster_assignments_list,
                clusters_out,
            ) = self.graph_models[graph_name](x, adj, mask)

            # Store outputs
            individual_predictions[graph_name] = out
            graph_outputs[graph_name] = out
            graph_link_losses[graph_name] = link_losses
            graph_entropy_losses[graph_name] = entropy_losses
            graph_pool_attention[graph_name] = pool_attention_weights
            graph_embed_attention[graph_name] = embed_attention_weights
            graph_cluster_assignments[graph_name] = cluster_assignments_list
            graph_cluster_outputs[graph_name] = clusters_out

        # Combine predictions from all graphs
        combined_features = []
        for graph_name in sorted(adj_dict.keys()):
            combined_features.append(graph_outputs[graph_name])

        combined_features = torch.cat(combined_features, dim=-1)
        final_output = self.final_combination(combined_features)

        return (
            final_output,
            graph_link_losses,
            graph_entropy_losses,
            graph_pool_attention,
            graph_embed_attention,
            graph_cluster_assignments,
            graph_cluster_outputs,
            individual_predictions,
        )


class CellDiffPool(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x, edge_indices: list[torch.Tensor], batch):
        pass


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
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss
    import numpy as np
    from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x
    edge_index_physical = batch["gene", "physical_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Model configuration
    model = DenseDiffPool(
        max_num_nodes=max_num_nodes,
        in_channels=x.size(1),
        pool_gat_hidden_channels=8,
        num_pool_gat_layers=2,
        embed_gat_hidden_channels=8,
        num_embed_gat_layers=2,
        num_pooling_layers=3,
        cluster_size_decay_factor=10.0,
        activation="relu",
        norm="batch",
        target_dim=2,
    )

    # Count and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print parameter details for each named component
    print("\nParameter shapes by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    # Initialize loss and optimizer
    criterion = CombinedLoss(loss_type="l1", weights=torch.ones(2))
    # criterion = CombinedLoss(loss_type="mse", weights=torch.tensor([1.0, 0.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):  # Increased to 10 epochs
        optimizer.zero_grad()

        # Convert sparse to dense format
        adj = to_dense_adj(
            edge_index=edge_index_physical,
            batch=batch_index,
            max_num_nodes=max_num_nodes,
        )
        x_dense, mask = to_dense_batch(
            x=x, batch=batch_index, max_num_nodes=max_num_nodes
        )

        # Forward pass
        out, link_losses, entropy_losses, cluster_assignments_list, clusters_out = (
            model(x_dense, adj, mask)
        )

        # Compute losses
        main_loss = criterion(out, y)[0]
        cluster_losses = [criterion(pred, y)[0] for pred in clusters_out]

        # Combine all losses
        total_link_loss = sum(link_losses)
        total_entropy_loss = sum(entropy_losses)
        total_loss = (
            main_loss
            + 0.1 * total_link_loss
            + 0.1 * total_entropy_loss
            + 0.1 * sum(cluster_losses)
        )

        # Backward pass
        total_loss.backward()

        # Check for NaN gradients
        has_nan = False
        print(f"\nEpoch {epoch + 1}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    has_nan = True
                grad_norm = param.grad.norm().item()
                if grad_norm > 10:  # Print large gradients
                    print(f"Large gradient in {name}: {grad_norm:.4f}")

        if has_nan:
            print("Training stopped due to NaN gradients")
            break

        # Print losses
        print(f"Main loss: {main_loss.item():.4f}")
        print(f"Total link prediction loss: {total_link_loss.item():.4f}")
        print(f"Total entropy loss: {total_entropy_loss.item():.4f}")
        for i, c_loss in enumerate(cluster_losses):
            print(f"Cluster {i} loss: {c_loss.item():.4f}")
        print(
            f"Total loss: {total_loss.item():.4f}"
        )  # Changed from main_loss to total_loss

        # Optimizer step
        optimizer.step()

        # Print sample predictions vs actual
        with torch.no_grad():
            print("\nPredictions vs Actual (first batch):")
            print(f"Predictions: {out[0].detach().numpy()}")
            print(f"Actual: {y[0].numpy()}")


if __name__ == "__main__":
    main()
