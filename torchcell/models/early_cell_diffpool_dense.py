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
from torchcell.models.dense_gat_conv import DenseGATConv
from typing import Dict, List, Optional, Tuple, Union, Literal
from torchcell.models.act import act_register


class EarlyDenseCellDiffPool(nn.Module):
    def __init__(
        self,
        graph_names: list[str],
        max_num_nodes: int,
        in_channels: int,
        init_gat_hidden_channels: int,
        init_num_gat_layers: int,
        pool_hidden_channels: int,
        num_pooling_layers: int,
        embed_hidden_channels: int,
        num_embed_gat_layers: int,
        cluster_size_decay_factor: float = 10.0,
        activation: str = "relu",
        norm: str = "batch",
        target_dim: int = 2,
        cluster_aggregation: Literal["mean", "sum"] = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.graph_names = sorted(graph_names)
        self.max_num_nodes = max_num_nodes
        self.in_channels = in_channels
        self.activation = act_register[activation]
        self.target_dim = target_dim
        self.cluster_aggregation = cluster_aggregation

        # Calculate cluster sizes for each pooling layer
        self.cluster_sizes = []
        for i in range(1, num_pooling_layers + 1):
            size = max(1, int(max_num_nodes / (cluster_size_decay_factor**i)))
            self.cluster_sizes.append(size)
        self.cluster_sizes[-1] = 1  # Ensure final layer pools to single cluster
        print(f"Cluster sizes: {self.cluster_sizes}")

        # Create GAT layers for each graph type
        self.gat_networks = nn.ModuleDict()
        for graph_name in self.graph_names:
            layers = []
            curr_channels = in_channels

            for i in range(init_num_gat_layers):
                out_channels = (
                    init_gat_hidden_channels
                    if i < init_num_gat_layers - 1
                    else embed_hidden_channels
                )
                gat = DenseGATConv(
                    in_channels=curr_channels,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
                norm_layer = self.get_norm_layer(norm, out_channels)
                layers.extend([gat, norm_layer])
                curr_channels = out_channels

            self.gat_networks[graph_name] = nn.ModuleList(layers)

        # Create pooling networks and cluster heads for each pooling layer
        self.pool_networks = nn.ModuleList()
        self.cluster_heads = nn.ModuleList()
        self.embed_networks = nn.ModuleList()

        for layer in range(num_pooling_layers):
            # Pool network for this layer
            pool_layers = []
            curr_channels = embed_hidden_channels

            for i in range(num_embed_gat_layers):
                is_last = i == num_embed_gat_layers - 1
                out_channels = (
                    self.cluster_sizes[layer] if is_last else pool_hidden_channels
                )

                gat = DenseGATConv(
                    in_channels=curr_channels,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
                norm_layer = self.get_norm_layer(norm, out_channels)
                pool_layers.extend([gat, norm_layer])
                curr_channels = out_channels

            self.pool_networks.append(nn.ModuleList(pool_layers))

            # Embed network for this layer
            embed_layers = []
            curr_channels = embed_hidden_channels

            for _ in range(num_embed_gat_layers):
                gat = DenseGATConv(
                    in_channels=curr_channels,
                    out_channels=embed_hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
                norm_layer = self.get_norm_layer(norm, embed_hidden_channels)
                embed_layers.extend([gat, norm_layer])

            self.embed_networks.append(nn.ModuleList(embed_layers))

            # Cluster prediction head for this layer
            self.cluster_heads.append(nn.Linear(embed_hidden_channels, target_dim))

        # Final prediction layer
        self.lin = nn.Linear(embed_hidden_channels, target_dim)

    def get_norm_layer(self, norm, channels):
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

    def forward(self, x, adj_dict: Dict[str, torch.Tensor], mask=None):
        # Process each graph through its GAT layers
        graph_embeddings = {}
        graph_attention_weights = {}

        for graph_name in self.graph_names:
            h = x
            attention_weights = []

            for i in range(0, len(self.gat_networks[graph_name]), 2):
                gat_layer = self.gat_networks[graph_name][i]
                norm_layer = self.gat_networks[graph_name][i + 1]

                h, attention = gat_layer(
                    h,
                    adj_dict[graph_name],
                    mask=mask,
                    add_loop=True,
                    return_attention_weights=True,
                )
                h = self.bn(h, norm_layer)
                h = self.activation(h)
                attention_weights.append(attention)

            graph_embeddings[graph_name] = h
            graph_attention_weights[graph_name] = attention_weights

        # Combine graph embeddings (mean pooling)
        combined_embedding = torch.stack(list(graph_embeddings.values())).mean(dim=0)
        avg_adj = sum(adj_dict.values()) / len(adj_dict)

        # Initialize lists for storing intermediate outputs
        pool_attention_weights = []
        cluster_assignments = []
        link_losses = []
        entropy_losses = []
        clusters_out = []
        embed_attention_weights = []

        # Current state
        current_x = combined_embedding
        current_adj = avg_adj
        current_mask = mask

        # Process through pooling layers
        for layer_idx, (pool_network, embed_network) in enumerate(
            zip(self.pool_networks, self.embed_networks)
        ):
            # Compute cluster assignment matrix
            s = current_x
            layer_pool_attention = []

            for i in range(0, len(pool_network), 2):
                conv_layer = pool_network[i]
                norm_layer = pool_network[i + 1]

                s, attention = conv_layer(
                    s,
                    current_adj,
                    mask=current_mask,
                    add_loop=True,
                    return_attention_weights=True,
                )
                s = self.bn(s, norm_layer)
                s = self.activation(s)
                layer_pool_attention.append(attention)

            pool_attention_weights.append(layer_pool_attention)
            cluster_assignments.append(s)

            # Compute node embeddings for cluster prediction
            z = current_x
            layer_embed_attention = []

            for i in range(0, len(embed_network), 2):
                embed_layer = embed_network[i]
                norm_layer = embed_network[i + 1]

                z, attention = embed_layer(
                    z,
                    current_adj,
                    mask=current_mask,
                    add_loop=True,
                    return_attention_weights=True,
                )
                z = self.bn(z, norm_layer)
                z = self.activation(z)
                layer_embed_attention.append(attention)

            embed_attention_weights.append(layer_embed_attention)

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
            current_mask = None  # After clustering, all nodes are valid

        # Final prediction
        out = self.lin(current_x.squeeze(1))

        return (
            out,
            link_losses,
            entropy_losses,
            graph_attention_weights,
            pool_attention_weights,
            embed_attention_weights,
            cluster_assignments,
            clusters_out,
            graph_embeddings,
        )


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


def main():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss

    # Load sample data batch
    batch, max_num_nodes = load_sample_data_batch()
    print(f"max_nodes: { max_num_nodes}")

    # Extract data
    x = batch["gene"].x
    adj_physical = batch["gene", "physical_interaction", "gene"].adj
    adj_regulatory = batch["gene", "regulatory_interaction", "gene"].adj
    mask = batch["gene"].mask
    y = torch.stack(
        [batch["gene"].fitness.squeeze(-1), batch["gene"].gene_interaction.squeeze(-1)],
        dim=1,
    )

    # Create adjacency dictionary
    adj_dict = {
        "physical_interaction": adj_physical,
        "regulatory_interaction": adj_regulatory,
    }
    
    # Model configuration
    model = EarlyDenseCellDiffPool(
        graph_names=["physical_interaction", "regulatory_interaction"],
        max_num_nodes=max_num_nodes,
        in_channels=x.size(-1),
        init_gat_hidden_channels=8,
        init_num_gat_layers=2,
        pool_hidden_channels=8,
        num_pooling_layers=5,
        embed_hidden_channels=8,
        num_embed_gat_layers=2,
        cluster_size_decay_factor=6.0,
        activation="relu",
        norm="batch",
        target_dim=2,
        cluster_aggregation="mean",
        dropout=0.2,
    )

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Print initial cluster sizes
    print("\nTarget cluster sizes per layer:")
    for i, size in enumerate(model.cluster_sizes):
        print(f"Layer {i}: {size}")

    # Initialize loss and optimizer
    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        (
            out,
            link_losses,
            entropy_losses,
            graph_attention_weights,
            pool_attention_weights,
            embed_attention_weights,
            cluster_assignments,
            clusters_out,
            graph_embeddings,
        ) = model(x, adj_dict, mask)

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

        # Print statistics
        print(f"\nEpoch {epoch + 1}")
        print(f"Main loss: {main_loss.item():.4f}")
        print(f"Link loss: {total_link_loss.item():.4f}")
        print(f"Entropy loss: {total_entropy_loss.item():.4f}")

        # Print cluster information and losses
        print("\nCluster Statistics:")
        for i, (assignment, c_loss) in enumerate(
            zip(cluster_assignments, cluster_losses)
        ):
            num_clusters = assignment.size(
                -1
            )  # Get number of clusters from assignment matrix
            avg_cluster_size = x.size(1) / num_clusters
            print(f"Layer {i}:")
            print(f"  Target number of clusters: {model.cluster_sizes[i]}")
            print(f"  Average nodes per cluster: {avg_cluster_size:.1f}")
            print(f"  Loss: {c_loss.item():.4f}")

            # Print cluster assignment statistics
            print("  Assignment Statistics:")
            print(f"    Mean: {assignment.mean().item():.4f}")
            print(f"    Max: {assignment.max().item():.4f}")
            print(f"    Min: {assignment.min().item():.4f}")
            print(f"    Sparsity: {(assignment < 0.01).float().mean().item():.4f}")

        # # Print attention statistics
        # print("\nAttention Statistics:")
        # # Initial GAT attention
        # for graph_name, attentions in graph_attention_weights.items():
        #     print(f"\n{graph_name} Initial GAT:")
        #     for layer, attn in enumerate(attentions):
        #         print(f"Layer {layer}:")
        #         print(f"  Mean: {attn.mean().item():.4f}")
        #         print(f"  Max: {attn.max().item():.4f}")
        #         print(f"  Min: {attn.min().item():.4f}")

        # # Pooling attention
        # for layer, attentions in enumerate(pool_attention_weights):
        #     print(f"\nPooling Layer {layer}:")
        #     for idx, attn in enumerate(attentions):
        #         print(f"  Conv {idx}:")
        #         print(f"    Mean: {attn.mean().item():.4f}")
        #         print(f"    Max: {attn.max().item():.4f}")
        #         print(f"    Min: {attn.min().item():.4f}")

        # # Embedding attention
        # for layer, attentions in enumerate(embed_attention_weights):
        #     print(f"\nEmbedding Layer {layer}:")
        #     for idx, attn in enumerate(attentions):
        #         print(f"  Conv {idx}:")
        #         print(f"    Mean: {attn.mean().item():.4f}")
        #         print(f"    Max: {attn.max().item():.4f}")
        #         print(f"    Min: {attn.min().item():.4f}")

        # print(f"\nTotal loss: {total_loss.item():.4f}")

        # # Check for NaN gradients
        # has_nan = False
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaN gradient detected in {name}")
        #             has_nan = True
        #         grad_norm = param.grad.norm().item()
        #         if grad_norm > 10:  # Print large gradients
        #             print(f"Large gradient in {name}: {grad_norm:.4f}")

        # if has_nan:
        #     print("Training stopped due to NaN gradients")
        #     break

        # Optimizer step
        optimizer.step()

        # Print predictions
        with torch.no_grad():
            print("\nPredictions vs Actual (first batch):")
            print(f"Final prediction: {out[0].detach().numpy()}")
            print(f"Cluster predictions:")
            for i, pred in enumerate(clusters_out):
                print(
                    f"  Layer {i} ({model.cluster_sizes[i]} target clusters): {pred[0].detach().numpy()}"
                )
            print(f"Actual: {y[0].numpy()}")


if __name__ == "__main__":
    main()
