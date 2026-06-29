"""Self-attention graph pooling (SAGPool) models for cell graphs."""

from typing import Any, cast

import torch
import torch.nn as nn
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GraphNorm,
    InstanceNorm,
    LayerNorm,
    MeanSubtractionNorm,
    PairNorm,
    SAGPooling,
    global_add_pool,
    global_mean_pool,
)

from torchcell.models.act import act_register


class SingleSAGPool(nn.Module):
    """Hierarchical SAGPool network with per-layer intermediate predictions."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        pooling_ratio: float = 0.5,
        activation: str = "relu",
        norm: str | None = None,
        target_dim: int = 2,
        min_score: float | None = None,
        gnn_type: str = "GATv2Conv",
        heads: int = 1,
        dropout: float = 0.0,
        final_pred_agg: str = "mean",
    ):
        """Build the GNN conv, pooling, norm, and predictor layers from config."""
        super().__init__()

        self.in_channels = in_channels
        self.activation = act_register[activation]
        self.target_dim = target_dim
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.final_pred_agg = final_pred_agg

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

        # Prediction layers
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, target_dim)

        # Add intermediate prediction heads
        self.intermediate_predictors = nn.ModuleList()
        for i in range(num_layers):
            in_features = (
                self.conv_out_channels if i < num_layers - 1 else hidden_channels
            )
            self.intermediate_predictors.append(nn.Linear(in_features, target_dim))

    def get_norm_layer(self, norm: str | None, channels: int) -> nn.Module:
        """Return the normalization layer matching the given norm name."""
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
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Run hierarchical pooling and return predictions and pooling diagnostics."""
        attention_weights = []
        pool_scores = []
        intermediate_predictions = []
        pool_sizes = []
        node_selections = []  # Track which nodes were selected at each layer

        # Get initial graph size and node indices
        unique_batch = torch.unique(batch)
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

            # Calculate and print pool statistics
            nodes_after_pool = torch.tensor(
                [torch.sum(batch == b).item() for b in unique_batch]
            )
            pool_sizes.append(nodes_after_pool)

            # Make intermediate prediction
            pooled_features = global_mean_pool(x, batch)
            intermediate_pred = predictor(pooled_features)
            intermediate_predictions.append(intermediate_pred)

        # Final prediction layers
        if self.final_pred_agg == "mean":
            x = global_mean_pool(x, batch)
        elif self.final_pred_agg == "add":
            x = global_add_pool(x, batch)
        x = self.activation(self.lin1(x))
        out = self.lin2(x)

        return (
            out,
            attention_weights,
            pool_scores,
            intermediate_predictions,
            pool_sizes,
            node_selections,  # Return node selection history
        )


class CellSAGPool(nn.Module):
    """Multi-graph SAGPool model that fuses per-graph predictions."""

    def __init__(
        self,
        graph_names: list[str],
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        pooling_ratio: float = 0.5,
        activation: str = "relu",
        norm: str | None = None,
        target_dim: int = 2,
        min_score: float | None = None,
        heads: int = 1,
        dropout: float = 0.0,
    ):
        """Create a SingleSAGPool per graph and the final combination layers."""
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

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: dict[str, torch.Tensor],
        batch: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
        dict[str, list[torch.Tensor]],
        dict[str, list[torch.Tensor]],
        dict[str, list[torch.Tensor]],
        dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """Run each graph's SAGPool and combine their predictions."""
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
        graph_node_selections = {}  # Track node selections for each graph

        for name in self.graph_names:
            (
                out,
                attention_weights,
                pool_scores,
                intermediate_predictions,
                pool_sizes,
                node_selections,  # Receive node selections
            ) = self.graph_models[name](x, edge_indices[name], batch)

            graph_outputs[name] = out
            graph_attention_weights[name] = attention_weights
            graph_pool_scores[name] = pool_scores
            graph_intermediate_predictions[name] = intermediate_predictions
            graph_pool_sizes[name] = pool_sizes
            graph_node_selections[name] = node_selections  # Store node selections

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
            graph_node_selections,  # Return node selections
        )

    @property
    def num_parameters(self) -> dict[str, int]:
        """Count parameters in all submodules."""
        # First, get parameters from one of the models to calculate per_model
        sample_model = next(iter(self.graph_models.values()))
        per_model = sum(p.numel() for p in sample_model.parameters())

        # Calculate total parameters for all graph models
        model_params = sum(
            sum(p.numel() for p in model.parameters())
            for model in self.graph_models.values()
        )

        # Calculate parameters in the final combination layer
        combination_params = sum(p.numel() for p in self.final_combination.parameters())

        return {
            "per_model": per_model,
            "all_models": model_params,
            "combination_layer": combination_params,
            "total": model_params + combination_params,
        }


def load_sample_data_batch() -> tuple[Any, int]:
    # batch is a PyG batch from an untyped external dataloader -> Any
    """Load a sample batch of cell graph data for ad-hoc model testing."""
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
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"),  # type: ignore[call-arg]  # dev-only helper; SCerevisiaeGraph uses sgd_root
        genome=genome,
    )
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
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},  # type: ignore[arg-type]  # dev-only helper; runtime accepts dict of graphs
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


def analyze_node_selections(
    node_selections: list[tuple[torch.Tensor, torch.Tensor]], graph_name: str
) -> None:
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


def main() -> None:
    """Run a sample CellSAGPool forward pass and report diagnostics."""
    from torchcell.losses.multi_dim_nan_tolerant import CombinedRegressionLoss

    # Load and prepare data as before...
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

    # Initialize model
    model = CellSAGPool(
        graph_names=["physical", "regulatory"],
        in_channels=x.size(1),
        hidden_channels=16,
        num_layers=12,
        pooling_ratio=0.6,
        activation="relu",
        norm="batch",
        target_dim=2,
        min_score=None,
        heads=4,
        dropout=0.1,
    )

    print(f"\nModel parameters: {model.num_parameters}")
    criterion = CombinedRegressionLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass with node selections
        (
            final_output,
            individual_predictions,
            attention_weights,
            pool_scores,
            intermediate_predictions,
            pool_sizes,
            node_selections,  # Receive node selections
        ) = model(x, edge_indices, batch_index)

        # Analyze node selections for each graph
        for graph_name, selections in node_selections.items():
            # analyze_node_selections(selections, graph_name)
            pass

        # Print pooling information
        print(f"\nEpoch {epoch + 1} Pooling Statistics:")
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

        # Calculate losses and continue training as before...
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

        # Print training information
        print(f"Main loss: {main_loss.item():.4f}")
        print(f"Graph losses: {graph_losses.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    main()
