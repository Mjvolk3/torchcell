import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    SAGPooling,
    BatchNorm,
    LayerNorm,
    GraphNorm,
    InstanceNorm,
    PairNorm,
    MeanSubtractionNorm,
    global_mean_pool,
    global_add_pool,
)
import torch.nn.functional as F
from typing import Optional, Literal
from torchcell.models.act import act_register
from torchcell.models.norm import norm_register


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        activation: str = "relu",
        norm: str = "layer",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = act_register[activation]

        # First layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        self.norms.append(self.get_norm_layer(norm, hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(self.get_norm_layer(norm, hidden_channels))

        # Final layer
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = nn.Dropout(dropout)

    def get_norm_layer(self, norm, channels):
        if norm == "batch":
            return nn.BatchNorm1d(channels)
        elif norm == "layer":
            return nn.LayerNorm(channels)
        elif norm == "instance":
            return nn.InstanceNorm1d(channels)
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return self.layers[-1](x)


class SingleSAGPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        pooling_ratio: float = 0.5,
        activation: str = "relu",
        norm: str = None,
        target_dim: int = 2,
        min_score: Optional[float] = None,
        gnn_type: str = "GATv2Conv",
        heads: int = 1,
        dropout: float = 0.0,
        final_pred_agg: str = "mean",
        num_intermediate_layers: int = 2,
        intermediate_hidden_channels: int = 64,
        intermediate_norm: str = "layer",
    ):
        super().__init__()

        # Store initialization parameters as attributes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pooling_ratio = pooling_ratio
        self.activation = act_register[activation]
        self.norm = norm
        self.target_dim = target_dim
        self.min_score = min_score
        self.gnn_type = gnn_type
        self.heads = heads
        self.dropout = dropout
        self.final_pred_agg = final_pred_agg
        self.num_intermediate_layers = num_intermediate_layers
        self.intermediate_hidden_channels = intermediate_hidden_channels
        self.intermediate_norm = intermediate_norm

        # Calculate actual output size considering heads
        self.conv_out_channels = hidden_channels * heads

        # Initialize lists for layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First convolution layer
        self.convs.append(
            GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
            )
        )
        self.norms.append(self.get_norm_layer(norm, self.conv_out_channels))

        # Add pooling layer after first conv
        self.pools.append(
            SAGPooling(
                in_channels=self.conv_out_channels,
                ratio=pooling_ratio,
                GNN=GATv2Conv,
                min_score=min_score,
                dropout=dropout,
            )
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=self.conv_out_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )
            self.norms.append(self.get_norm_layer(norm, self.conv_out_channels))
            self.pools.append(
                SAGPooling(
                    in_channels=self.conv_out_channels,
                    ratio=pooling_ratio,
                    GNN=GATv2Conv,
                    min_score=min_score,
                    dropout=dropout,
                )
            )

        # Final layers
        self.convs.append(
            GATv2Conv(
                in_channels=self.conv_out_channels,
                out_channels=hidden_channels,
                heads=1,  # Final layer has 1 head
                dropout=dropout,
            )
        )
        self.norms.append(self.get_norm_layer(norm, hidden_channels))

        # Final pooling
        self.pools.append(
            SAGPooling(
                in_channels=hidden_channels,
                ratio=pooling_ratio,
                GNN=GATv2Conv,
                min_score=min_score,
                dropout=dropout,
            )
        )

        # Add intermediate prediction MLPs
        self.intermediate_predictors = nn.ModuleList()
        for i in range(num_layers):
            in_features = (
                self.conv_out_channels if i < num_layers - 1 else hidden_channels
            )
            self.intermediate_predictors.append(
                MLP(
                    in_channels=in_features,
                    hidden_channels=intermediate_hidden_channels,
                    out_channels=target_dim,
                    num_layers=num_intermediate_layers,
                    activation=activation,
                    norm=intermediate_norm,
                    dropout=dropout,
                )
            )

        # Final prediction MLP
        self.final_predictor = MLP(
            in_channels=hidden_channels,
            hidden_channels=intermediate_hidden_channels,
            out_channels=target_dim,
            num_layers=num_intermediate_layers,
            activation=activation,
            norm=intermediate_norm,
            dropout=dropout,
        )

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
        attention_weights = []
        pool_scores = []
        intermediate_predictions = []
        pool_sizes = []
        node_selections = []

        # Get initial graph size and node indices
        unique_batch = torch.unique(batch)
        batch_size = len(unique_batch)
        initial_nodes = torch.tensor(
            [torch.sum(batch == b).item() for b in unique_batch]
        )
        pool_sizes.append(initial_nodes)

        # Track initial node indices
        current_node_indices = torch.arange(len(batch), device=batch.device)

        # Track nodes through layers
        for i, (conv, norm, pool, predictor) in enumerate(
            zip(self.convs, self.norms, self.pools, self.intermediate_predictors)
        ):
            # Apply convolution and get attention weights
            x_conv, attention = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(attention)

            # Apply normalization and activation
            if isinstance(norm, (GraphNorm, PairNorm, MeanSubtractionNorm)):
                x_conv = norm(x_conv)
            else:
                x_conv = norm(x_conv)
            x_conv = self.activation(x_conv)

            # Apply pooling
            x, edge_index, _, batch, perm, score = pool(
                x=x_conv, edge_index=edge_index, batch=batch
            )
            pool_scores.append(score)

            # Track selected nodes
            selected_nodes = current_node_indices[perm]
            node_selections.append((current_node_indices, selected_nodes))
            current_node_indices = selected_nodes

            # Calculate pool statistics
            nodes_after_pool = torch.tensor(
                [torch.sum(batch == b).item() for b in unique_batch]
            )
            pool_sizes.append(nodes_after_pool)

            # Make intermediate prediction
            pooled_features = global_mean_pool(x, batch)
            intermediate_pred = predictor(pooled_features)
            intermediate_predictions.append(intermediate_pred)

        # Final prediction
        if self.final_pred_agg == "mean":
            x = global_mean_pool(x, batch)
        elif self.final_pred_agg == "add":
            x = global_add_pool(x, batch)

        out = self.final_predictor(x)

        return (
            out,
            attention_weights,
            pool_scores,
            intermediate_predictions,
            pool_sizes,
            node_selections,
        )


class CellSAGPool(nn.Module):
    def __init__(
        self,
        graph_names: list[str],
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        pooling_ratio: float = 0.5,
        activation: str = "relu",
        norm: str = None,
        target_dim: int = 2,
        min_score: Optional[float] = None,
        heads: int = 1,
        dropout: float = 0.0,
        num_intermediate_layers: int = 2,
        intermediate_hidden_channels: int = 64,
        intermediate_norm: str = "layer",
        final_hidden_channels: int = 64,
        final_num_layers: int = 2,
        final_norm: str = "layer",
    ):
        super().__init__()

        self.graph_names = sorted(graph_names)
        self.num_graphs = len(graph_names)
        self.target_dim = target_dim

        # Create models for each graph type
        self.graph_models = nn.ModuleDict(
            {
                name: SingleSAGPool(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    pooling_ratio=pooling_ratio,
                    activation=activation,
                    norm=norm,
                    target_dim=target_dim,
                    min_score=min_score,
                    heads=heads,
                    dropout=dropout,
                    num_intermediate_layers=num_intermediate_layers,
                    intermediate_hidden_channels=intermediate_hidden_channels,
                    intermediate_norm=intermediate_norm,
                )
                for name in self.graph_names
            }
        )

        # Final combination MLP
        self.final_combination = MLP(
            in_channels=target_dim * self.num_graphs,
            hidden_channels=final_hidden_channels,
            out_channels=target_dim,
            num_layers=final_num_layers,
            activation=activation,
            norm=final_norm,
            dropout=dropout,
        )

    @property
    def num_parameters(self) -> dict:
        """Count parameters in all submodules."""
        sample_model = next(iter(self.graph_models.values()))
        per_model = sum(p.numel() for p in sample_model.parameters())

        model_params = sum(
            sum(p.numel() for p in model.parameters())
            for model in self.graph_models.values()
        )

        combination_params = sum(p.numel() for p in self.final_combination.parameters())

        return {
            "per_model": per_model,
            "all_models": model_params,
            "combination_layer": combination_params,
            "total": model_params + combination_params,
        }

    def forward(self, x, edge_indices: dict[str, torch.Tensor], batch):
        if set(edge_indices.keys()) != set(self.graph_names):
            raise ValueError(
                f"Expected edge indices for graphs {self.graph_names}, "
                f"but got {list(edge_indices.keys())}"
            )

        graph_outputs = {}
        graph_attention_weights = {}
        graph_pool_scores = {}
        graph_intermediate_predictions = {}
        graph_pool_sizes = {}
        graph_node_selections = {}

        for name in self.graph_names:
            (
                out,
                attention_weights,
                pool_scores,
                intermediate_predictions,
                pool_sizes,
                node_selections,
            ) = self.graph_models[name](x, edge_indices[name], batch)

            graph_outputs[name] = out
            graph_attention_weights[name] = attention_weights
            graph_pool_scores[name] = pool_scores
            graph_intermediate_predictions[name] = intermediate_predictions
            graph_pool_sizes[name] = pool_sizes
            graph_node_selections[name] = node_selections

        # Combine predictions
        combined = torch.cat([graph_outputs[name] for name in self.graph_names], dim=-1)
        final_output = self.final_combination(combined)

        return (
            final_output,
            graph_outputs,
            graph_attention_weights,
            graph_pool_scores,
            graph_intermediate_predictions,
            graph_pool_sizes,
            graph_node_selections,
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
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss

    # Load and prepare data
    batch, _ = load_sample_data_batch()
    x = batch["gene"].x
    # Test with just one graph type (physical)
    edge_index = batch["gene", "physical_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Initialize single model with new parameters
    model = SingleSAGPool(
        in_channels=x.size(1),
        hidden_channels=32,
        num_layers=4,  # Reduced for initial testing
        pooling_ratio=0.5,
        activation="relu",
        norm="batch",
        target_dim=2,
        min_score=None,
        heads=2,
        dropout=0.1,
        # New parameters
        num_intermediate_layers=2,
        intermediate_hidden_channels=64,
        intermediate_norm="layer",
    )

    print("\nModel Architecture:")
    print(f"Input channels: {x.size(1)}")
    print(f"Hidden channels: {model.hidden_channels}")
    print(f"Number of layers: {len(model.convs)}")
    print(f"Intermediate MLP layers: {model.num_intermediate_layers}")
    print(f"Intermediate hidden channels: {model.intermediate_hidden_channels}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    print("\nStarting training loop...")

    for epoch in range(3):  # Reduced epochs for testing
        optimizer.zero_grad()

        # Forward pass
        (
            final_output,
            attention_weights,
            pool_scores,
            intermediate_predictions,
            pool_sizes,
            node_selections,
        ) = model(x, edge_index, batch_index)

        # Print pooling information
        print(f"\nEpoch {epoch + 1} Statistics:")
        print("Pooling sizes through layers:")
        for layer, size in enumerate(pool_sizes):
            if layer == 0:
                print(f"Initial nodes: {size}")
            else:
                prev_size = pool_sizes[layer - 1]
                ratio = size.float() / prev_size.float()
                print(
                    f"Layer {layer}: {size} nodes (ratio from previous: {ratio.mean().item():.3f})"
                )

        # Calculate losses
        main_loss = criterion(final_output, y)[0]
        intermediate_loss = sum(
            criterion(pred, y)[0] for pred in intermediate_predictions
        )

        total_loss = main_loss + 0.1 * intermediate_loss
        total_loss.backward()
        optimizer.step()

        # Print losses
        print(f"Main loss: {main_loss.item():.4f}")
        print(f"Intermediate loss: {intermediate_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")

        # Print predictions vs actual for first batch item
        print("\nPredictions vs Actual (first item):")
        print(f"Final prediction: {final_output[0].detach().numpy()}")
        print(f"Actual values: {y[0].numpy()}")

        # Print intermediate predictions for first batch item
        print("\nIntermediate predictions (first item):")
        for i, pred in enumerate(intermediate_predictions):
            print(f"Layer {i+1}: {pred[0].detach().numpy()}")


def analyze_node_selections(node_selections, graph_name):
    """Analyze how nodes were selected through the pooling layers."""
    print(f"\nAnalyzing node selections for {graph_name}:")

    for layer, (original_nodes, selected_nodes) in enumerate(node_selections):
        print(f"\nLayer {layer + 1}:")
        print(f"Original nodes: {len(original_nodes)}")
        print(f"Selected nodes: {len(selected_nodes)}")

        # Calculate selection ratio
        selection_ratio = len(selected_nodes) / len(original_nodes)
        print(f"Selection ratio: {selection_ratio:.3f}")

        # Show which nodes were kept
        print("Selected node indices:", selected_nodes.tolist())


def main():
    from torchcell.losses.multi_dim_nan_tolerant import CombinedLoss

    # Load and prepare data
    batch, _ = load_sample_data_batch()
    x = batch["gene"].x
    edge_index_physical = batch["gene", "physical_interaction", "gene"].edge_index
    edge_index_regulatory = batch["gene", "regulatory_interaction", "gene"].edge_index
    batch_index = batch["gene"].batch
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    edge_indices = {
        "physical": edge_index_physical,
        "regulatory": edge_index_regulatory,
    }

    # Initialize model with all the new parameters
    model = CellSAGPool(
        graph_names=["physical", "regulatory"],
        in_channels=x.size(1),
        hidden_channels=16,
        num_layers=12,  # Reduced for initial testing
        pooling_ratio=0.6,
        activation="relu",
        norm="layer",
        target_dim=2,
        min_score=None,
        heads=4,
        dropout=0.1,
        num_intermediate_layers=4,
        intermediate_hidden_channels=16,
        intermediate_norm="layer",
        final_hidden_channels=4,
        final_num_layers=4,
        final_norm="layer",
    )

    print("\nModel Architecture:")
    print(f"Input channels: {x.size(1)}")
    print(f"Hidden channels: {model.graph_models['physical'].hidden_channels}")
    print(f"Number of graphs: {len(model.graph_names)}")
    print(
        f"Intermediate MLP layers: {model.graph_models['physical'].num_intermediate_layers}"
    )
    print(
        f"Intermediate hidden channels: {model.graph_models['physical'].intermediate_hidden_channels}"
    )
    print(f"Final combination layers: {model.final_combination.layers}")

    # Print parameter counts
    params = model.num_parameters
    print("\nParameter Counts:")
    print(f"Parameters per graph model: {params['per_model']:,}")
    print(f"Total graph model parameters: {params['all_models']:,}")
    print(f"Combination layer parameters: {params['combination_layer']:,}")
    print(f"Total parameters: {params['total']:,}")

    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    print("\nStarting training loop...")

    for epoch in range(500):  # Reduced epochs for testing
        optimizer.zero_grad()

        # Forward pass
        (
            final_output,
            individual_predictions,
            attention_weights,
            pool_scores,
            intermediate_predictions,
            pool_sizes,
            node_selections,
        ) = model(x, edge_indices, batch_index)

        # Print pooling information
        print(f"\nEpoch {epoch + 1} Statistics:")
        for graph_name, sizes in pool_sizes.items():
            print(f"\n{graph_name} graph pooling sizes:")
            for layer, size in enumerate(sizes):
                if layer == 0:
                    print(f"Initial nodes: {size}")
                else:
                    prev_size = sizes[layer - 1]
                    ratio = size.float() / prev_size.float()
                    print(
                        f"Layer {layer}: {size} nodes (ratio from previous: {ratio.mean().item():.3f})"
                    )

        # Calculate losses
        main_loss = criterion(final_output, y)[0]
        graph_losses = sum(
            criterion(pred, y)[0] for pred in individual_predictions.values()
        )
        intermediate_losses = sum(
            criterion(pred, y)[0]
            for graph_preds in intermediate_predictions.values()
            for pred in graph_preds
        )

        total_loss = main_loss + 0.1 * graph_losses + 0.1 * intermediate_losses
        total_loss.backward()
        optimizer.step()

        # Print detailed predictions and losses
        print("\nLoss Information:")
        print(f"Main loss: {main_loss.item():.4f}")
        print(f"Graph losses: {graph_losses.item():.4f}")
        print(f"Intermediate losses: {intermediate_losses.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")

        # Print predictions for first batch item
        print("\nPredictions vs Actual (first item):")
        print(f"Final prediction: {final_output[0].detach().numpy()}")
        print(f"Individual graph predictions:")
        for graph_name, preds in individual_predictions.items():
            print(f"  {graph_name}: {preds[0].detach().numpy()}")
        print(f"Actual values: {y[0].numpy()}")


if __name__ == "__main__":
    main()
