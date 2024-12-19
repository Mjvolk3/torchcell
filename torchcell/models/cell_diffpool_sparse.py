import torch
import torch.nn as nn
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
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from typing import Optional, Literal

from torchcell.models.act import act_register
from torchcell.models.norm import norm_register
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch


def from_dense_batch(dense_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Inverts the `to_dense_batch` operation when the mask is all `True`,
    meaning there are no fake nodes (i.e., all nodes are valid).

    Args:
        dense_x (Tensor): The dense node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}`.

    :rtype: (:class:`Tensor`, :class:`Tensor`)
    """

    # Flatten the dense tensor
    batch_size, max_num_nodes, num_features = dense_x.shape
    flattened_x = dense_x.view(batch_size * max_num_nodes, num_features)

    # Create the batch tensor
    batch = torch.arange(batch_size, device=dense_x.device).repeat_interleave(
        max_num_nodes
    )

    # Remove any padding rows (optional depending on whether they exist)
    valid_mask = flattened_x.abs().sum(dim=-1) != 0
    sparse_x = flattened_x[valid_mask]
    sparse_batch = batch[valid_mask]

    return sparse_x, sparse_batch


class SingleDiffPool(nn.Module):
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

        # Create pooling networks (GAT) for each layer
        self.pool_networks = nn.ModuleList()
        for layer in range(num_pooling_layers):
            pool_gnn = nn.ModuleList()
            curr_in_channels = in_channels if layer == 0 else embed_gat_hidden_channels

            # Add GAT layers for pooling
            for i in range(num_pool_gat_layers):
                is_last = i == num_pool_gat_layers - 1
                out_channels = (
                    self.cluster_sizes[layer] if is_last else pool_gat_hidden_channels
                )

                pool_gnn.append(
                    GATv2Conv(
                        in_channels=curr_in_channels,
                        out_channels=out_channels,
                        heads=1,
                        concat=False,
                        add_self_loops=False,
                    )
                )
                if norm:
                    pool_gnn.append(self.get_norm_layer(norm, out_channels))
                curr_in_channels = out_channels

            self.pool_networks.append(pool_gnn)

        # Create embedding networks (GAT) for each layer
        self.embed_networks = nn.ModuleList()
        for layer in range(num_pooling_layers):
            embed_gnn = nn.ModuleList()
            curr_in_channels = in_channels if layer == 0 else embed_gat_hidden_channels

            # Add GAT layers for embedding
            for i in range(num_embed_gat_layers):
                embed_gnn.append(
                    GATv2Conv(
                        in_channels=curr_in_channels,
                        out_channels=embed_gat_hidden_channels,
                        heads=1,
                        concat=False,
                        add_self_loops=False,
                    )
                )
                if norm:
                    embed_gnn.append(
                        self.get_norm_layer(norm, embed_gat_hidden_channels)
                    )
                curr_in_channels = embed_gat_hidden_channels

            self.embed_networks.append(embed_gnn)

        # Final prediction layer
        self.lin = nn.Linear(embed_gat_hidden_channels, target_dim)

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

    def forward(self, x, edge_index, batch):
        # Initialize variables to store outputs
        pool_attention_weights = []  # Store pooling GAT attention weights
        embed_attention_weights = []  # Store embedding GAT attention weights
        cluster_assignments_list = []
        link_losses = []
        entropy_losses = []
        clusters_out = []

        max_num_nodes = self.max_num_nodes
        # Process each pooling layer
        for layer, (pool_gnn, embed_gnn) in enumerate(
            zip(self.pool_networks, self.embed_networks)
        ):
            # # Compute cluster assignment matrix (S) using GAT
            s = x
            for i, pool_layer in enumerate(pool_gnn):
                pool_layer_attention = []
                # GAT layer
                if isinstance(pool_layer, GATv2Conv):

                    # Get output and attention weights
                    s, pool_att = pool_layer(
                        s, edge_index, return_attention_weights=True
                    )
                    pool_layer_attention.append(pool_att)

                # Normalization layer
                else:
                    if isinstance(
                        pool_layer, (GraphNorm, PairNorm, MeanSubtractionNorm)
                    ):
                        # TODO troubleshoot
                        s = pool_layer(s)
                    else:
                        s = pool_layer(s)
                    # Activation function after normalization
                    s = self.activation(s)

            pool_attention_weights.append(pool_layer_attention)
            # Store cluster assignments
            cluster_assignments_list.append(s)

            # Compute node embeddings (Z) using GAT
            z = x
            for i, embed_layer in enumerate(embed_gnn):
                embed_layer_attention = []
                # GAT layer
                if isinstance(embed_layer, GATv2Conv):
                    z, embed_att = embed_layer(
                        z, edge_index, return_attention_weights=True
                    )
                    embed_layer_attention.append(embed_att)
                # Normalization layer
                else:
                    if isinstance(
                        embed_layer, (GraphNorm, PairNorm, MeanSubtractionNorm)
                    ):
                        z = embed_layer(z)
                    else:
                        z = embed_layer(z)
                    z = self.activation(z)

            embed_attention_weights.append(embed_layer_attention)

            # Dense Conversion
            adj = to_dense_adj(
                edge_index=edge_index, batch=batch, max_num_nodes=max_num_nodes
            )
            z, mask = to_dense_batch(x=z, batch=batch, max_num_nodes=max_num_nodes)
            s, _ = to_dense_batch(x=s, batch=batch, max_num_nodes=max_num_nodes)

            # update max_num_nodes for next layer
            max_num_nodes = self.cluster_sizes[layer]

            # Apply diffpool
            x, adj, link_loss, ent_loss = dense_diff_pool(
                x=z, adj=adj, s=s, mask=mask, normalize=True
            )

            # Sparse conversion
            x, batch = from_dense_batch(x)
            edge_index, _ = dense_to_sparse(adj=adj, mask=None)

            # Store losses
            link_losses.append(link_loss)
            entropy_losses.append(ent_loss)

            # Compute cluster-level predictions
            if self.cluster_aggregation == "mean":
                cluster_x = global_mean_pool(x, batch)
            elif self.cluster_aggregation == "sum":
                cluster_x = global_add_pool(x, batch)
            cluster_pred = self.cluster_heads[layer](cluster_x)

            clusters_out.append(cluster_pred)

        # Final prediction
        out = self.lin(x)

        return (
            out,
            link_losses,
            entropy_losses,
            pool_attention_weights,
            embed_attention_weights,
            cluster_assignments_list,
            clusters_out,
        )


class CellDiffPool(nn.Module):
    def __init__(
        self,
        graph_names: list[str],  # e.g. ["physical", "regulatory"]
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
    ):
        super().__init__()

        self.graph_names = sorted(graph_names)  # Sort for consistent ordering
        self.target_dim = target_dim
        self.activation = act_register[activation]

        # Create models for each graph type using ModuleDict
        self.graph_models = nn.ModuleDict(
            {
                name: SingleDiffPool(
                    max_num_nodes=max_num_nodes,
                    in_channels=in_channels,
                    pool_gat_hidden_channels=pool_gat_hidden_channels,
                    num_pool_gat_layers=num_pool_gat_layers,
                    embed_gat_hidden_channels=embed_gat_hidden_channels,
                    num_embed_gat_layers=num_embed_gat_layers,
                    num_pooling_layers=num_pooling_layers,
                    cluster_size_decay_factor=cluster_size_decay_factor,
                    activation=activation,
                    norm=norm,
                    target_dim=target_dim,
                    cluster_aggregation=cluster_aggregation,
                )
                for name in self.graph_names
            }
        )

        # Final combination layers
        self.final_combination = nn.Sequential(
            nn.Linear(target_dim * len(self.graph_names), target_dim * 2),
            self.activation,
            nn.Linear(target_dim * 2, target_dim),
        )

        # Verify parameter registration
        expected_params_per_model = sum(
            p.numel() for p in next(iter(self.graph_models.values())).parameters()
        )
        total_expected = expected_params_per_model * len(self.graph_names) + sum(
            p.numel() for p in self.final_combination.parameters()
        )
        actual_total = sum(p.numel() for p in self.parameters())

        print(f"Parameters per model: {expected_params_per_model:,}")
        print(f"Expected total parameters: {total_expected:,}")
        print(f"Actual total parameters: {actual_total:,}")

        if total_expected != actual_total:
            raise ValueError(
                f"Parameter registration mismatch! Expected {total_expected:,} but got {actual_total:,}"
            )

    def forward(self, x, edge_indices: dict[str, torch.Tensor], batch):
        """Forward pass through all graph models."""
        if set(edge_indices.keys()) != set(self.graph_names):
            raise ValueError(
                f"Expected edge indices for graphs {self.graph_names}, "
                f"but got {list(edge_indices.keys())}"
            )

        # Process each graph
        graph_outputs = {}
        graph_link_losses = {}
        graph_entropy_losses = {}
        graph_pool_attention = {}
        graph_embed_attention = {}
        graph_cluster_assignments = {}
        graph_cluster_outputs = {}
        individual_predictions = {}

        # Process each graph in consistent order
        for name in self.graph_names:
            # Forward pass through corresponding model
            (
                out,
                link_losses,
                entropy_losses,
                pool_attention,
                embed_attention,
                cluster_assignments,
                clusters_out,
            ) = self.graph_models[name](x, edge_indices[name], batch)

            # Store all outputs
            graph_outputs[name] = out
            graph_link_losses[name] = link_losses
            graph_entropy_losses[name] = entropy_losses
            graph_pool_attention[name] = pool_attention
            graph_embed_attention[name] = embed_attention
            graph_cluster_assignments[name] = cluster_assignments
            graph_cluster_outputs[name] = clusters_out
            individual_predictions[name] = out

        # Combine predictions in consistent order
        combined_features = torch.cat(
            [graph_outputs[name] for name in self.graph_names], dim=-1
        )
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

    @property
    def num_parameters(self):
        """Count parameters in all submodules."""
        model_params = sum(
            sum(p.numel() for p in model.parameters())
            for model in self.graph_models.values()
        )
        combination_params = sum(p.numel() for p in self.final_combination.parameters())
        return {
            "per_model": sum(
                p.numel() for p in next(iter(self.graph_models.values())).parameters()
            ),
            "all_models": model_params,
            "combination_layer": combination_params,
            "total": model_params + combination_params,
        }


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


def main_single():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedRegressionLoss
    import numpy as np

    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x
    edge_index_physical = batch["gene", "physical_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Model configuration
    model = SingleDiffPool(
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
    criterion = CombinedRegressionLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):  # Increased to 10 epochs
        optimizer.zero_grad()

        # Forward pass
        (
            out,
            link_losses,
            entropy_losses,
            pool_attention_weights,
            embed_attention_weights,
            cluster_assignments_list,
            clusters_out,
        ) = model(x, edge_index_physical, batch_index)

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
        print(f"Total loss: {main_loss.item():.4f}")

        # Optimizer step
        optimizer.step()

        # Print sample predictions vs actual
        with torch.no_grad():
            print("\nPredictions vs Actual (first batch):")
            print(f"Predictions: {out[0].detach().numpy()}")
            print(f"Actual: {y[0].numpy()}")


def main():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedRegressionLoss
    import numpy as np

    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x
    edge_index_physical = batch["gene", "physical_interaction", "gene"].edge_index
    edge_index_regulatory = batch["gene", "regulatory_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Create dictionary of edge indices
    edge_indices = {
        "physical": edge_index_physical,
        "regulatory": edge_index_regulatory,
    }


    # Model configuration
    model = CellDiffPool(
        graph_names=["physical", "regulatory"],
        max_num_nodes=max_num_nodes,
        in_channels=x.size(1),
        pool_gat_hidden_channels=8,
        num_pool_gat_layers=2,
        embed_gat_hidden_channels=8,
        num_embed_gat_layers=2,
        num_pooling_layers=3,
        cluster_size_decay_factor=10.0,
        activation="relu",
        norm="layer",
        target_dim=2,
    )
    params = model.num_parameters
    print(f"Parameters per model: {params['per_model']:,}")
    print(f"Total parameters: {params['total']:,}")

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
    criterion = CombinedRegressionLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()

        # Forward pass
        (
            final_output,
            graph_link_losses,
            graph_entropy_losses,
            graph_pool_attention,
            graph_embed_attention,
            graph_cluster_assignments,
            graph_cluster_outputs,
            individual_predictions,
        ) = model(x, edge_indices, batch_index)

        # Compute main loss
        main_loss = criterion(final_output, y)[0]

        # Compute individual graph losses
        individual_losses = {
            graph_name: criterion(pred, y)[0]
            for graph_name, pred in individual_predictions.items()
        }

        # Compute cluster losses for each graph
        cluster_losses = {
            graph_name: [criterion(pred, y)[0] for pred in clusters]
            for graph_name, clusters in graph_cluster_outputs.items()
        }

        # Combine all losses
        total_link_loss = sum(sum(losses) for losses in graph_link_losses.values())
        total_entropy_loss = sum(
            sum(losses) for losses in graph_entropy_losses.values()
        )
        total_cluster_loss = sum(sum(losses) for losses in cluster_losses.values())

        # Total loss with weights
        total_loss = (
            main_loss  # Main prediction loss
            + 0.1 * total_link_loss  # Link prediction losses
            + 0.1 * total_entropy_loss  # Entropy losses
            + 0.1 * total_cluster_loss  # Cluster prediction losses
            + 0.1
            * sum(individual_losses.values())  # Individual graph prediction losses
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
        print("Individual graph losses:")
        for graph_name, loss in individual_losses.items():
            print(f"{graph_name}: {loss.item():.4f}")
        print(f"Total link prediction loss: {total_link_loss.item():.4f}")
        print(f"Total entropy loss: {total_entropy_loss.item():.4f}")
        print("Cluster losses:")
        for graph_name, losses in cluster_losses.items():
            print(f"  {graph_name}:")
            for i, loss in enumerate(losses):
                print(f"Layer {i}: {loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")

        # Optimizer step
        optimizer.step()

        # Print sample predictions vs actual
        with torch.no_grad():
            print("\nPredictions vs Actual (first batch):")
            print("Combined prediction:")
            print(f"Prediction: {final_output[0].detach().numpy()}")
            print(f"Actual: {y[0].numpy()}")
            print("Individual graph predictions:")
            for graph_name, pred in individual_predictions.items():
                print(f"  {graph_name}: {pred[0].detach().numpy()}")


if __name__ == "__main__":
    main()
