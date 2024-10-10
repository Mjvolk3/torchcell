import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from typing import Dict, List, Optional

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
        num_graphs: int,
        max_num_nodes: int,
        gat_dropout_prob: float = 0.0,
        last_layer_dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        gat_skip_connection: bool = True,
    ):
        super().__init__()

        assert norm in ["batch", "instance", "layer"], "Invalid norm type"
        assert activation in act_register.keys(), "Invalid activation type"

        self.num_graphs = num_graphs
        self.gat_skip_connection = gat_skip_connection
        self.activation = act_register[activation]
        self.last_layer_dropout_prob = last_layer_dropout_prob

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
        cluster_sizes = [
            max(1, int(max_num_nodes / (2**i)))
            for i in range(1, num_diffpool_layers + 1)
        ]
        cluster_sizes[-1] = 1  # Ensure the last cluster size is 1
        # print(f"Cluster sizes: {cluster_sizes}")

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
                                for _ in range(2)  # 2 layers of GAT after each pooling
                            ]
                        )
                        for _ in range(
                            num_diffpool_layers - 1
                        )  # One less than DiffPool layers
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
                                for _ in range(2)  # 2 layers of GAT after each pooling
                            ]
                        )
                        for _ in range(
                            num_diffpool_layers - 1
                        )  # One less than DiffPool layers
                    ]
                )
                for _ in range(num_graphs)
            ]
        )

        # Final linear layer
        self.final_linear = nn.Linear(
            num_graphs * diffpool_hidden_channels, diffpool_out_channels
        )

    def get_norm_layer(self, norm, channels):
        if norm == "batch":
            return nn.BatchNorm1d(channels)
        elif norm == "instance":
            return nn.InstanceNorm1d(channels, affine=True)
        elif norm == "layer":
            return nn.LayerNorm(channels)

    def forward(self, x, edge_indices: List[torch.Tensor], batch):
        graph_outputs = []
        attention_weights = []
        cluster_assignments = []
        link_pred_losses = []
        entropy_losses = []

        for i in range(self.num_graphs):
            edge_index = edge_indices[i]

            # Initial GAT layers
            x_graph = x
            for j, (gat_layer, norm_layer) in enumerate(
                zip(self.initial_gat_layers[i], self.initial_gat_norm_layers[i])
            ):
                x_out, att_weights = gat_layer(
                    x_graph, edge_index, return_attention_weights=True
                )
                x_out = norm_layer(x_out)
                x_out = self.activation(x_out)
                if self.gat_skip_connection and x_graph.shape == x_out.shape:
                    x_out = x_out + x_graph
                x_graph = x_out
                attention_weights.append(att_weights)

            # print(f"Graph {i} after initial GAT layers shape: {x_graph.shape}")

            # Convert to dense batch
            x_dense, mask = to_dense_batch(x_graph, batch)
            adj = to_dense_adj(edge_index, batch)

            # print(
            #     f"x_dense shape: {x_dense.shape}, mask shape: {mask.shape}, adj shape: {adj.shape}"
            # )

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
                # print(f"Graph {i}, DiffPool layer {k} output shape: {x_pool.shape}")

                # Post-pooling GAT layers (for all but the last DiffPool layer)
                if k < len(self.diffpool_layers[i]) - 1:
                    for l, (gat_layer, norm_layer) in enumerate(
                        zip(
                            self.post_pool_gat_layers[i][k],
                            self.post_pool_gat_norm_layers[i][k],
                        )
                    ):
                        x_pool_flat = x_pool.view(-1, x_pool.size(-1))
                        adj_pool_flat = adj_pool.view(-1, adj_pool.size(-1))
                        x_out, att_weights = gat_layer(
                            x_pool_flat,
                            adj_pool_flat.nonzero().t(),
                            return_attention_weights=True,
                        )
                        x_out = norm_layer(x_out)
                        x_out = self.activation(x_out)
                        if (
                            self.gat_skip_connection
                            and x_pool_flat.shape == x_out.shape
                        ):
                            x_out = x_out + x_pool_flat
                        x_pool = x_out.view(x_pool.size(0), -1, x_out.size(-1))
                        attention_weights.append(att_weights)
                    # print(
                    #     f"Graph {i}, Post-pooling GAT layer {k} output shape: {x_pool.shape}"
                    # )

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
        # print(f"Concatenated output shape: {x_concat.shape}")

        # Final linear layer with activation and dropout
        out = self.final_linear(x_concat.squeeze(1))
        out = self.activation(out)
        out = F.dropout(out, p=self.last_layer_dropout_prob, training=self.training)
        # print(f"Final output shape: {out.shape}")

        return (
            out,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
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
        genome=genome,
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
    size = 1e4
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

    # Model configuration
    model = GatDiffPool(
        in_channels=x.size(1),
        initial_gat_hidden_channels=64,
        initial_gat_out_channels=32,
        diffpool_hidden_channels=32,
        diffpool_out_channels=16,
        num_initial_gat_layers=3,
        num_diffpool_layers=6,
        num_graphs=2,
        max_num_nodes=max_num_nodes,
        gat_dropout_prob=0.2,
        last_layer_dropout_prob=0.5,
        norm="batch",
        activation="relu",
        gat_skip_connection=True,
    )

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    # print(f"Input x shape: {x.shape}")
    # print(f"Edge index physical shape: {edge_index_physical.shape}")
    # print(f"Edge index regulatory shape: {edge_index_regulatory.shape}")
    # print(f"Batch index shape: {batch_index.shape}")
    # print(f"Batch size: {batch_index.max().item() + 1}")

    # Forward pass
    out, attention_weights, cluster_assignments, link_pred_losses, entropy_losses = (
        model(x, [edge_index_physical, edge_index_regulatory], batch_index)
    )

    # print("Output shape:", out.shape)
    # print("Number of attention weight tensors:", len(attention_weights))
    # print("First attention weight shape:", attention_weights[0][0].shape)
    # print("Last attention weight shape:", attention_weights[-1][0].shape)
    # print("Number of cluster assignment tensors:", len(cluster_assignments))
    # print("First cluster assignment shape:", cluster_assignments[0].shape)
    # print("Last cluster assignment shape:", cluster_assignments[-1].shape)
    # print("Link prediction losses:", link_pred_losses)
    # print("Entropy losses:", entropy_losses)


if __name__ == "__main__":
    main()
