# torchcell/models/cell_gin_diffpool_dense
# [[torchcell.models.cell_gin_diffpool_dense]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/cell_gin_diffpool_dense
# Test file: tests/torchcell/models/test_cell_gin_diffpool_dense.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torchcell.models.dense_gat_conv import DenseGATConv
from typing import Optional, Literal
from torchcell.models.act import act_register
import torch_geometric.transforms as T

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torchcell.transforms.hetero_to_dense import HeteroToDense


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn.dense import DenseGINConv
from typing import Optional, Literal
from torchcell.models.act import act_register


class DenseDiffPool(nn.Module):
    def __init__(
        self,
        max_num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_pooling_layers: int,
        cluster_size_decay_factor: float = 10.0,
        activation: str = "relu",
        conv_norm: str = "batch",
        mlp_norm: str = "batch",
        target_dim: int = 2,
        cluster_aggregation: Literal["mean", "sum"] = "mean",
        add_skip_connections: bool = True,
        gin_self_loop: bool = True,
        train_eps: bool = True,
        eps: float = 0.0,
    ):
        super().__init__()

        self.max_num_nodes = max_num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.activation = activation  # Store activation name
        self.act_fn = act_register[activation]  # Get activation function
        self.target_dim = target_dim
        self.cluster_aggregation = cluster_aggregation
        self.add_skip_connections = add_skip_connections
        self.gin_self_loop = gin_self_loop

        # Calculate cluster sizes
        self.cluster_sizes = []
        for i in range(1, num_pooling_layers + 1):
            size = max(2, int(max_num_nodes / (cluster_size_decay_factor**i)))
            self.cluster_sizes.append(size)
        self.cluster_sizes[-1] = 1
        print(f"Cluster sizes: {self.cluster_sizes}")

        # Create pooling networks
        self.pool_networks = nn.ModuleList()
        for layer in range(num_pooling_layers):
            curr_in_channels = in_channels if layer == 0 else hidden_channels

            layers = []
            for i in range(num_layers - 1):
                # Create MLP without internal norm
                mlp = nn.Sequential(
                    nn.Linear(curr_in_channels, hidden_channels),
                    self.act_fn,
                    nn.Linear(hidden_channels, hidden_channels),
                )
                layers.append(DenseGINConv(mlp, eps=eps, train_eps=train_eps))

                # Add separate norm after GIN
                if conv_norm:
                    layers.append(self.get_norm_layer(conv_norm, hidden_channels))
                layers.append(self.act_fn)
                curr_in_channels = hidden_channels

            # Final layer for cluster assignments
            final_mlp = nn.Sequential(
                nn.Linear(curr_in_channels, hidden_channels),
                self.act_fn,
                nn.Linear(hidden_channels, self.cluster_sizes[layer]),
            )
            layers.append(DenseGINConv(final_mlp, eps=eps, train_eps=train_eps))

            if conv_norm:
                layers.append(self.get_norm_layer(conv_norm, self.cluster_sizes[layer]))
            layers.append(self.act_fn)

            self.pool_networks.append(nn.ModuleList(layers))

        # Create embedding networks
        self.embed_networks = nn.ModuleList()
        for layer in range(num_pooling_layers):
            curr_in_channels = in_channels if layer == 0 else hidden_channels

            layers = []
            for _ in range(num_layers):
                mlp = nn.Sequential(
                    nn.Linear(curr_in_channels, hidden_channels),
                    self.act_fn,
                    nn.Linear(hidden_channels, hidden_channels),
                )
                layers.append(DenseGINConv(mlp, eps=eps, train_eps=train_eps))

                if conv_norm:
                    layers.append(self.get_norm_layer(conv_norm, hidden_channels))
                layers.append(self.act_fn)
                curr_in_channels = hidden_channels

            self.embed_networks.append(nn.ModuleList(layers))

        # Create cluster prediction heads
        self.cluster_heads = nn.ModuleList()
        for _ in range(num_pooling_layers):
            head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act_fn,
                nn.Linear(hidden_channels, target_dim),
            )
            self.cluster_heads.append(head)

        # Final prediction layer
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.act_fn,
            nn.Linear(hidden_channels, target_dim),
        )

    def get_norm_layer(self, norm, channels):
        """Create normalization layer."""
        if norm is None:
            return nn.Identity()
        elif norm == "batch":
            return nn.BatchNorm1d(channels)
        elif norm == "layer":
            return nn.LayerNorm(channels)
        elif norm == "instance":
            return nn.InstanceNorm1d(channels, affine=True)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

    def bn(self, x, norm_layer):
        """Apply normalization to the input tensor."""
        if isinstance(norm_layer, nn.Identity):
            return x

        batch_size, num_nodes, num_channels = x.size()

        if isinstance(norm_layer, nn.BatchNorm1d):
            x = x.view(-1, num_channels)
            x = norm_layer(x)
            x = x.view(batch_size, num_nodes, num_channels)
        elif isinstance(norm_layer, nn.InstanceNorm1d):
            x = x.permute(0, 2, 1)
            x = norm_layer(x)
            x = x.permute(0, 2, 1)
        elif isinstance(norm_layer, nn.LayerNorm):
            x = norm_layer(x)

        return x

    def forward(self, x, adj, mask=None):
        cluster_assignments_list = []
        link_losses = []
        entropy_losses = []
        clusters_out = []

        current_x = x
        current_adj = adj
        current_mask = mask

        for layer_idx, (pool_gnn, embed_gnn) in enumerate(
            zip(self.pool_networks, self.embed_networks)
        ):
            # Pool network forward pass
            s = current_x
            for i, layer in enumerate(pool_gnn):
                if isinstance(layer, DenseGINConv):
                    s = layer(
                        s, current_adj, mask=current_mask, add_loop=self.gin_self_loop
                    )
                elif isinstance(
                    layer,
                    (nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d, nn.Identity),
                ):
                    s = self.bn(s, layer)
                else:  # Activation function
                    s = layer(s)

                if self.add_skip_connections and i > 0 and s.shape == current_x.shape:
                    s = s + current_x

            cluster_assignments_list.append(s)

            # Embed network forward pass
            z = current_x
            for i, layer in enumerate(embed_gnn):
                if isinstance(layer, DenseGINConv):
                    z = layer(
                        z, current_adj, mask=current_mask, add_loop=self.gin_self_loop
                    )
                elif isinstance(
                    layer,
                    (nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d, nn.Identity),
                ):
                    z = self.bn(z, layer)
                else:  # Activation function
                    z = layer(z)

                if self.add_skip_connections and i > 0 and z.shape == current_x.shape:
                    z = z + current_x

            # Make cluster prediction
            if self.cluster_aggregation == "mean":
                cluster_x = z.mean(dim=1)
            else:  # sum
                cluster_x = z.sum(dim=1)

            cluster_pred = self.cluster_heads[layer_idx](cluster_x)
            clusters_out.append(cluster_pred)

            # Apply diffpool
            current_x, current_adj, link_loss, ent_loss = dense_diff_pool(
                x=z, adj=current_adj, s=s, mask=current_mask
            )
            link_losses.append(link_loss)
            entropy_losses.append(ent_loss)
            current_mask = None

        # Final prediction
        final_out = self.lin(current_x.squeeze(1))

        return (
            final_out,
            link_losses,
            entropy_losses,
            cluster_assignments_list,
            clusters_out,
        )


class DenseCellDiffPool(nn.Module):
    def __init__(
        self,
        graph_names: list[str],
        max_num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_pooling_layers: int,
        cluster_size_decay_factor: float = 10.0,
        activation: str = "relu",
        conv_norm: str = "batch",  # Changed from norm
        mlp_norm: str = "batch",  # Added mlp_norm
        target_dim: int = 2,
        cluster_aggregation: Literal["mean", "sum"] = "mean",
        add_skip_connections: bool = True,
        gin_self_loop: bool = True,
        train_eps: bool = True,
        eps: float = 0.0,
    ):
        super().__init__()

        self.graph_names = sorted(graph_names)  # Sort for consistent ordering
        self.num_graphs = len(graph_names)
        self.target_dim = target_dim
        self.activation = act_register[activation]

        # Create named models for each graph type
        self.graph_models = nn.ModuleDict(
            {
                name: DenseDiffPool(
                    max_num_nodes=max_num_nodes,
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    num_pooling_layers=num_pooling_layers,
                    cluster_size_decay_factor=cluster_size_decay_factor,
                    activation=activation,
                    conv_norm=conv_norm,  # Changed from norm
                    mlp_norm=mlp_norm,  # Added mlp_norm
                    target_dim=target_dim,
                    cluster_aggregation=cluster_aggregation,
                    add_skip_connections=add_skip_connections,
                    gin_self_loop=gin_self_loop,
                    train_eps=train_eps,
                    eps=eps,
                )
                for name in self.graph_names
            }
        )

        # Final combination layers
        self.final_combination = nn.Sequential(
            nn.Linear(target_dim * self.num_graphs, target_dim * 2),
            act_register[activation],
            nn.Linear(target_dim * 2, target_dim),
        )

        # Verify parameter registration
        expected_params_per_model = sum(
            p.numel() for p in next(iter(self.graph_models.values())).parameters()
        )
        total_expected = expected_params_per_model * self.num_graphs + sum(
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

    def forward(self, x, adj_dict: dict[str, torch.Tensor], mask=None):
        """Forward pass through named graph models."""
        if set(adj_dict.keys()) != set(self.graph_names):
            raise ValueError(
                f"Expected adjacency matrices for graphs {self.graph_names}, "
                f"but got {list(adj_dict.keys())}"
            )

        # Store outputs for each graph
        graph_outputs = {}
        graph_link_losses = {}
        graph_entropy_losses = {}
        graph_cluster_assignments = {}
        graph_cluster_outputs = {}

        # Process each named graph
        for name in self.graph_names:
            # Get model outputs
            (out, link_losses, entropy_losses, cluster_assignments, clusters_out) = (
                self.graph_models[name](x, adj_dict[name], mask)
            )

            # Store all outputs with graph names
            graph_outputs[name] = out
            graph_link_losses[name] = link_losses
            graph_entropy_losses[name] = entropy_losses
            graph_cluster_assignments[name] = cluster_assignments
            graph_cluster_outputs[name] = clusters_out

        # Combine predictions in consistent order
        combined = torch.cat([graph_outputs[name] for name in self.graph_names], dim=-1)
        final_output = self.final_combination(combined)

        return (
            final_output,
            graph_link_losses,
            graph_entropy_losses,
            graph_cluster_assignments,
            graph_cluster_outputs,
            graph_outputs,  # Individual predictions
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
        transform=HeteroToDense({"gene": len(genome.gene_set)}),
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
        dense=True,
    )
    perturbation_subset_data_module.setup()

    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    max_num_nodes = len(dataset.gene_set)
    return batch, max_num_nodes


def main_single():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss
    import numpy as np
    from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x  # [batch_size, num_nodes, features]
    adj = batch[
        "gene", "physical_interaction", "gene"
    ].adj  # [batch_size, num_nodes, num_nodes]
    mask = batch["gene"].mask  # [batch_size, num_nodes]
    y = torch.stack(
        [batch["gene"].fitness.squeeze(-1), batch["gene"].gene_interaction.squeeze(-1)],
        dim=1,
    )

    # Model configuration with new GIN parameters
    # Model configuration
    model = DenseDiffPool(
        max_num_nodes=max_num_nodes,
        in_channels=x.size(-1),
        hidden_channels=4,
        num_layers=3,
        num_pooling_layers=3,
        cluster_size_decay_factor=7.0,
        activation="relu",
        conv_norm="batch",
        mlp_norm="batch",
        target_dim=2,
        cluster_aggregation="mean",
        add_skip_connections=True,
        gin_self_loop=True,
        train_eps=True,
        eps=0.0,
    )

    # Count and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print parameter shapes
    print("\nParameter shapes by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    # Initialize loss and optimizer
    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()

        # Forward pass
        (out, link_losses, entropy_losses, cluster_assignments_list, clusters_out) = (
            model(x, adj, mask)
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
        print(f"Total loss: {total_loss.item():.4f}")

        # Print cluster assignment statistics
        print("\nCluster Assignment Statistics:")
        for layer, cluster_assign in enumerate(cluster_assignments_list):
            print(f"Layer {layer}:")
            print(f"  Mean assignment: {cluster_assign.mean().item():.4f}")
            print(f"  Max assignment: {cluster_assign.max().item():.4f}")
            print(f"  Min assignment: {cluster_assign.min().item():.4f}")
            print(f"  Num clusters: {cluster_assign.size(2)}")

        # Optimizer step
        optimizer.step()

        # Print sample predictions vs actual
        with torch.no_grad():
            print("\nPredictions vs Actual (first batch):")
            print(f"Predictions: {out[0].detach().numpy()}")
            print(f"Actual: {y[0].numpy()}")


def main():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss
    import numpy as np

    # Load the sample data batch
    batch, max_num_nodes = load_sample_data_batch()

    # Extract relevant information from the batch
    x = batch["gene"].x  # [batch_size, num_nodes, features]
    adj_physical = batch["gene", "physical_interaction", "gene"].adj
    adj_regulatory = batch["gene", "regulatory_interaction", "gene"].adj
    mask = batch["gene"].mask  # [batch_size, num_nodes]
    y = torch.stack(
        [batch["gene"].fitness.squeeze(-1), batch["gene"].gene_interaction.squeeze(-1)],
        dim=1,
    )

    # Create dictionary of adjacency matrices
    adj_dict = {
        "physical_interaction": adj_physical,
        "regulatory_interaction": adj_regulatory,
    }

    # Model configuration
    model = DenseCellDiffPool(
        graph_names=["physical_interaction", "regulatory_interaction"],
        max_num_nodes=max_num_nodes,
        in_channels=x.size(-1),
        hidden_channels=16,
        num_layers=2,
        num_pooling_layers=3,
        cluster_size_decay_factor=7.0,
        activation="relu",
        conv_norm="batch",
        mlp_norm="batch",
        target_dim=2,
        cluster_aggregation="mean",
        add_skip_connections=True,
        gin_self_loop=True,
        train_eps=True,
        eps=0.0,
    )

    # Count and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {model.num_parameters}")

    # Print parameter shapes
    print("\nParameter shapes by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    # Initialize loss and optimizer
    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()

        # Forward pass (updated to match new return structure)
        (
            final_output,
            graph_link_losses,
            graph_entropy_losses,
            graph_cluster_assignments,
            graph_cluster_outputs,
            individual_predictions,
        ) = model(x, adj_dict, mask)

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

        # Combine all losses with weights
        total_link_loss = sum(sum(losses) for losses in graph_link_losses.values())
        total_entropy_loss = sum(
            sum(losses) for losses in graph_entropy_losses.values()
        )
        total_cluster_loss = sum(sum(losses) for losses in cluster_losses.values())

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

        # Check for NaN gradients and large gradients
        has_nan = False
        print(f"\nEpoch {epoch + 1}")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN gradient detected in {name}")
        #             has_nan = True
        #         grad_norm = param.grad.norm().item()
        #         if grad_norm > 10:  # Print large gradients
        #             print(f"Large gradient in {name}: {grad_norm:.4f}")

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

        # Print cluster assignment statistics
        # print("\nCluster Assignment Statistics:")
        # for graph_name, cluster_assigns in graph_cluster_assignments.items():
        #     print(f"\n{graph_name} graph:")
        #     for layer, cluster_assign in enumerate(cluster_assigns):
        #         print(f"Layer {layer}:")
        #         print(f"  Mean assignment: {cluster_assign.mean().item():.4f}")
        #         print(f"  Max assignment: {cluster_assign.max().item():.4f}")
        #         print(f"  Min assignment: {cluster_assign.min().item():.4f}")
        #         print(f"  Num clusters: {cluster_assign.size(2)}")

        # Optimizer step
        optimizer.step()

        # Print predictions vs actual
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
