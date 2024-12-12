import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    GCNConv,
    GATv2Conv,
    TransformerConv,
    GINConv,
    BatchNorm,
    LayerNorm,
    GraphNorm,
    InstanceNorm,
    PairNorm,
    MeanSubtractionNorm,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from typing import Dict, Optional, List, Literal
from torch_geometric.typing import EdgeType
from torchcell.models.act import act_register


class ProjectedGATConv(nn.Module):
    def __init__(self, gat_conv, out_dim):
        super().__init__()
        self.gat = gat_conv
        self.project = nn.Linear(gat_conv.heads * gat_conv.out_channels, out_dim)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)  # Shape: (..., heads * out_channels)
        return self.project(x)  # Shape: (..., out_dim)


class PredictionHead(nn.Module):
    def __init__(self, layers: nn.ModuleList, residual: bool, dims: List[int]):
        super().__init__()
        self.layers = layers
        self.residual = residual
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        current_idx = 0

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply residual connection if dimensions match and it's a linear layer
            if (
                self.residual
                and isinstance(layer, nn.Linear)
                and self.dims[current_idx] == self.dims[current_idx + 1]
            ):
                x = x + input_x
                input_x = x
                current_idx += 1

        return x


class HeteroGnnPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        edge_types: List[EdgeType],
        conv_type: Literal["GCN", "GAT", "Transformer", "GIN"] = "GCN",
        layer_config: Optional[Dict] = None,
        pooling: Literal["sum", "mean", "max"] = "mean",
        activation: str = "relu",
        norm: Optional[str] = None,
        head_num_layers: int = 2,
        head_hidden_channels: Optional[int] = None,
        head_dropout: float = 0.0,
        head_activation: str = "relu",
        head_residual: bool = False,
        head_norm: Optional[Literal["batch", "layer", "instance"]] = None,
        learnable_embedding: bool = False,
        num_nodes: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_types = edge_types
        self.pooling = pooling
        self.activation = act_register[activation]
        self.conv_type = conv_type
        self.norm = norm
        self.learnable_embedding = learnable_embedding

        if learnable_embedding and num_nodes is None:
            raise ValueError(
                "num_nodes must be provided when using learnable_embedding"
            )

        self.layer_config = self._get_layer_config(layer_config)
        self.dims = self._calculate_dimensions(in_channels, hidden_channels)

        # Initialize learnable embedding if specified
        if learnable_embedding:
            self.node_embedding = nn.Embedding(
                num_embeddings=num_nodes,
                embedding_dim=in_channels,
                max_norm=1.0,  # Hardcoded max_norm=1.0
            )
        else:
            self.node_embedding = None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self._build_network()

        # Build prediction head
        self.pred_head = self._build_prediction_head(
            in_channels=self.dims["actual_hidden"],
            hidden_channels=(
                head_hidden_channels
                if head_hidden_channels is not None
                else hidden_channels
            ),
            out_channels=out_channels,
            num_layers=head_num_layers,
            dropout=head_dropout,
            activation=head_activation,
            residual=head_residual,
            norm=head_norm,
        )

    def _get_layer_config(self, layer_config: Optional[Dict]) -> Dict:
        default_configs = {
            "GCN": {
                "bias": True,
                "dropout": 0.0,
                "add_self_loops": True,
                "normalize": False,
                "is_skip_connection": False,
            },
            "GAT": {
                "heads": 1,
                "concat": True,
                "bias": True,
                "dropout": 0.0,
                "add_self_loops": True,
                "share_weights": False,
                "is_skip_connection": False,
            },
            "Transformer": {
                "heads": 1,
                "concat": True,
                "beta": True,
                "dropout": 0.0,
                "edge_dim": None,
                "bias": True,
                "root_weight": True,
                "add_self_loops": True,
            },
            "GIN": {
                "train_eps": True,
                "hidden_multiplier": 2.0,
                "dropout": 0.0,
                "add_self_loops": True,
                "is_skip_connection": False,
                "num_mlp_layers": 2,
                "is_mlp_skip_connection": True,
            },
        }
        if layer_config is None:
            return default_configs[self.conv_type]
        return {**default_configs[self.conv_type], **layer_config}

    def _calculate_dimensions(self, in_channels: int, hidden_channels: int) -> Dict:
        dims = {"in_channels": in_channels, "hidden_channels": hidden_channels}

        if self.conv_type in ["GAT", "Transformer"]:
            heads = self.layer_config.get("heads", 1)
            if self.layer_config.get("concat", True):
                # For concatenation with output projection:
                dims["actual_hidden"] = (
                    hidden_channels  # Final output dim after projection
                )
                dims["conv_hidden"] = hidden_channels // heads  # Per-head dim
                dims["concat_dim"] = (
                    hidden_channels  # Dim after concatenation (before projection)
                )
            else:
                # For averaging case:
                dims["actual_hidden"] = hidden_channels
                dims["conv_hidden"] = hidden_channels
        else:
            dims["actual_hidden"] = hidden_channels
            dims["conv_hidden"] = hidden_channels

        return dims

    def _create_gin_nn(self, in_dim: int) -> nn.Sequential:
        multiplier = self.layer_config.get("hidden_multiplier", 2.0)
        gin_hidden = int(in_dim * multiplier)
        dropout = self.layer_config.get("dropout", 0.0)
        num_layers = self.layer_config.get("num_mlp_layers", 2)
        is_mlp_skip = self.layer_config.get("is_mlp_skip_connection", True)

        layers = []
        # First layer
        layers.extend(
            [
                nn.Linear(in_dim, gin_hidden),
                nn.BatchNorm1d(gin_hidden),
                type(self.activation)(),
                nn.Dropout(dropout),
            ]
        )

        # Middle layers with optional skip connections
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(gin_hidden, gin_hidden),
                    nn.BatchNorm1d(gin_hidden),
                    type(self.activation)(),
                    nn.Dropout(dropout),
                ]
            )

            if is_mlp_skip:
                # Create skip connection
                prev_out = layers[-8].output if hasattr(layers[-8], "output") else None
                layers[-4] = (
                    layers[-4]
                    if prev_out is None
                    else (lambda x, l=layers[-4], p=prev_out: l(x) + p)
                )

        # Final layer
        layers.append(nn.Linear(gin_hidden, self.dims["conv_hidden"]))

        return nn.Sequential(*layers)

    def _create_conv_dict(self, in_dim: int) -> Dict:
        conv_dict = {}

        for edge_type in self.edge_types:
            if self.conv_type == "GCN":
                conv_dict[edge_type] = GCNConv(
                    in_dim,
                    self.dims["conv_hidden"],
                    **{
                        k: v
                        for k, v in self.layer_config.items()
                        if k in ["bias", "add_self_loops", "normalize"]
                    },
                )

            elif self.conv_type == "GAT":
                base_gat = GATv2Conv(
                    in_dim,
                    self.dims["conv_hidden"],
                    **{
                        k: v
                        for k, v in self.layer_config.items()
                        if k
                        in [
                            "heads",
                            "concat",
                            "dropout",
                            "bias",
                            "add_self_loops",
                            "share_weights",
                        ]
                    },
                )
                if self.layer_config.get("concat", True):
                    conv_dict[edge_type] = ProjectedGATConv(
                        base_gat, self.dims["actual_hidden"]
                    )
                else:
                    conv_dict[edge_type] = base_gat

            elif self.conv_type == "Transformer":
                conv_dict[edge_type] = TransformerConv(
                    in_dim,
                    self.dims["conv_hidden"],
                    **{
                        k: v
                        for k, v in self.layer_config.items()
                        if k
                        in [
                            "heads",
                            "concat",
                            "beta",
                            "dropout",
                            "edge_dim",
                            "bias",
                            "root_weight",
                        ]
                    },
                )

            elif self.conv_type == "GIN":
                gin_nn = self._create_gin_nn(in_dim)
                conv_dict[edge_type] = GINConv(
                    nn=gin_nn, train_eps=self.layer_config.get("train_eps", True)
                )

        return conv_dict

    def _build_network(self):
        for i in range(self.num_layers):
            in_dim = self.dims["in_channels"] if i == 0 else self.dims["actual_hidden"]
            conv_dict = self._create_conv_dict(in_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

            if self.norm is not None:
                self.norms.append(self._get_norm_layer(self.dims["actual_hidden"]))
            else:
                self.norms.append(nn.Identity())

    def _get_norm_layer(self, channels: int) -> nn.Module:
        if self.norm is None:
            return nn.Identity()

        norm_layers = {
            "batch": BatchNorm,
            "instance": InstanceNorm,
            "layer": LayerNorm,
            "graph": GraphNorm,
            "pair": PairNorm,
            "mean_subtraction": MeanSubtractionNorm,
        }

        if self.norm not in norm_layers:
            raise ValueError(f"Unsupported normalization type: {self.norm}")

        norm_layer = norm_layers[self.norm]
        if norm_layer in [GraphNorm, PairNorm, MeanSubtractionNorm]:
            return norm_layer()
        return norm_layer(channels)

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Invalid pooling type: {self.pooling}")

    def _get_head_norm(
        self, channels: int, norm_type: Optional[str]
    ) -> Optional[nn.Module]:
        """Get standard PyTorch normalization layer for prediction head."""
        if norm_type is None:
            return None

        if norm_type == "batch":
            return nn.BatchNorm1d(channels, track_running_stats=True)
        elif norm_type == "layer":
            return nn.LayerNorm(channels)
        else:
            raise ValueError(f"Unsupported head normalization type: {norm_type}")

    def _build_prediction_head(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        activation: str,
        residual: bool,
        norm: Optional[str],
    ) -> nn.Module:
        if num_layers < 1:
            raise ValueError("Prediction head must have at least one layer")

        activation_fn = act_register[activation]  # This gives us an instance
        activation_class = type(activation_fn)  # Get the class from the instance
        layers = []
        dims = []

        # Calculate dimensions for each layer
        if num_layers == 1:
            dims = [in_channels, out_channels]
        else:
            dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        # Build layers
        for i in range(num_layers):
            # Add linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add normalization, activation, and dropout (except for last layer)
            if i < num_layers - 1:
                if norm is not None:
                    norm_layer = self._get_head_norm(dims[i + 1], norm)
                    if norm_layer is not None:
                        layers.append(norm_layer)

                layers.append(activation_class())  # Create new instance using the class

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        return PredictionHead(
            layers=nn.ModuleList(layers),
            residual=residual and num_layers > 1,
            dims=dims,
        )

    # def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict=None):
    def forward(self, batch):
        from torch_geometric.utils import add_self_loops

        if self.learnable_embedding:
            device = batch["gene"].batch.device
            batch_size = len(batch["gene"].x_ptr) - 1
            nodes_per_graph = self.node_embedding.num_embeddings

            # Get base embeddings for all graphs
            all_indices = torch.arange(nodes_per_graph, device=device).repeat(
                batch_size
            )
            all_embeddings = self.node_embedding(all_indices)

            # Calculate adjusted perturbation indices for the full batch
            adjusted_pert_indices = []
            for i in range(batch_size):
                start_idx = batch["gene"].x_pert_ptr[i]
                end_idx = batch["gene"].x_pert_ptr[i + 1]
                # Get indices for this graph
                graph_pert_indices = batch["gene"].cell_graph_idx_pert[
                    start_idx:end_idx
                ]
                # Adjust indices by batch position
                adjusted_pert_indices.append(graph_pert_indices + (i * nodes_per_graph))

            adjusted_pert_indices = torch.cat(adjusted_pert_indices)

            # Create mask and remove perturbed rows
            mask = torch.ones(len(all_embeddings), dtype=torch.bool, device=device)
            mask[adjusted_pert_indices] = False
            x_dict = {"gene": all_embeddings[mask]}

            assert x_dict["gene"].shape[0] == batch["gene"].x.shape[0]

        else:
            x_dict = {"gene": batch["gene"].x}

        # Create edge_index_dict from batch
        edge_index_dict = {
            ("gene", "physical_interaction", "gene"): batch[
                "gene", "physical_interaction", "gene"
            ].edge_index,
            ("gene", "regulatory_interaction", "gene"): batch[
                "gene", "regulatory_interaction", "gene"
            ].edge_index,
        }

        # Rest of forward pass logic
        if self.conv_type in ["GIN", "Transformer"] and self.layer_config.get(
            "add_self_loops", True
        ):
            edge_index_dict = {
                k: add_self_loops(v)[0] for k, v in edge_index_dict.items()
            }

        if self.conv_type == "Transformer":
            x_dict = self.convs[0](x_dict, edge_index_dict)
        else:
            x_dict = self.convs[0](x_dict, edge_index_dict)

        x_dict = {key: x for key, x in x_dict.items()}
        x_dict = {key: self.norms[0](x) for key, x in x_dict.items()}
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        for i in range(1, self.num_layers):
            prev_x_dict = x_dict

            if self.conv_type == "Transformer":
                x_dict = self.convs[i](x_dict, edge_index_dict)
            else:
                x_dict = self.convs[i](x_dict, edge_index_dict)

            x_dict = {key: x for key, x in x_dict.items()}
            x_dict = {key: self.norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}

            if self.conv_type != "Transformer" and self.layer_config.get(
                "is_skip_connection", False
            ):
                x_dict = {
                    key: x + prev_x_dict[key] if key in prev_x_dict else x
                    for key, x in x_dict.items()
                }

        x_pool = self._global_pool(x_dict["gene"], batch["gene"].batch)
        x = self.pred_head(x_pool)

        return x, x_pool

    @property
    def num_parameters(self) -> Dict[str, int]:
        conv_params = sum(
            sum(p.numel() for p in conv.parameters()) for conv in self.convs
        )
        norm_params = sum(
            sum(p.numel() for p in norm.parameters()) for norm in self.norms
        )
        pred_head_params = sum(p.numel() for p in self.pred_head.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "conv_layers": conv_params,
            "norm_layers": norm_params,
            "pred_head": pred_head_params,
            "total": total_trainable,
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
        batch_size=16,
        random_seed=seed,
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=16,
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

    # Load sample data
    batch, max_num_nodes = load_sample_data_batch()

    # Prepare input dictionaries
    x_dict = {"gene": batch["gene"].x}
    edge_index_dict = {
        ("gene", "physical_interaction", "gene"): batch[
            "gene", "physical_interaction", "gene"
        ].edge_index,
        ("gene", "regulatory_interaction", "gene"): batch[
            "gene", "regulatory_interaction", "gene"
        ].edge_index,
    }
    batch_dict = {"gene": batch["gene"].batch}

    # Target values
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Define edge types
    edge_types = [
        ("gene", "physical_interaction", "gene"),
        ("gene", "regulatory_interaction", "gene"),
    ]

    # Example configurations for different conv types
    configs = {
        "GCN": {
            "bias": True,
            "dropout": 0.1,
            "add_self_loops": False,
            "normalize": False,
            "is_skip_connection": True,
        },
        "GAT": {
            "heads": 10,
            "concat": True,
            "dropout": 0.1,
            "bias": True,
            "add_self_loops": False,
            "share_weights": False,
            "is_skip_connection": True,
        },
        "Transformer": {
            "heads": 4,
            "concat": True,
            "beta": True,
            "dropout": 0.1,
            "edge_dim": None,
            "bias": True,
            "root_weight": True,
            "add_self_loops": True,
        },
        "GIN": {
            "train_eps": True,
            "hidden_multiplier": 2.0,
            "dropout": 0.1,
            "add_self_loops": True,
            "is_skip_connection": True,
            "num_mlp_layers": 3,
            "is_mlp_skip_connection": True,
        },
    }

    learnable_embedding = True

    # Create model with new prediction head configuration
    model = HeteroGnnPool(
        # in_channels=x_dict["gene"].size(1),
        in_channels=4,
        hidden_channels=32,
        out_channels=2,
        num_layers=3,
        edge_types=edge_types,
        conv_type="GIN",
        layer_config=configs["GIN"],
        pooling="mean",
        activation="relu",
        norm="batch",
        head_num_layers=3,  # 3-layer prediction head
        head_hidden_channels=16,
        head_dropout=0.1,  # 10% dropout
        head_activation="relu",  # ReLU activation
        head_residual=True,  # Use residual connections
        head_norm="batch",  # Use batch normalization
        num_nodes=max_num_nodes,  # required when learnable_embedding=True
        learnable_embedding=learnable_embedding,
    )

    print("\nModel architecture:")
    print(model)
    print("\nParameter counts:")
    print(model.num_parameters)

    # Initialize loss and optimizer

    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass with batch
        out = model(batch)

        # Calculate loss
        loss, _ = criterion(out, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def main_ordinal_reg():
    from torchcell.losses.multi_dim_nan_tolerant import (
        CombinedLoss,
        OrdinalEntropyRegLoss,
    )

    # Load sample data
    batch, max_num_nodes = load_sample_data_batch()

    # Define edge types
    edge_types = [
        ("gene", "physical_interaction", "gene"),
        ("gene", "regulatory_interaction", "gene"),
    ]

    # Create model with same architecture
    model = HeteroGnnPool(
        in_channels=4,
        hidden_channels=32,
        out_channels=2,
        num_layers=3,
        edge_types=edge_types,
        conv_type="GIN",
        layer_config={"train_eps": True, "hidden_multiplier": 2.0},
        pooling="mean",
        activation="relu",
        norm="batch",
        head_num_layers=3,
        head_hidden_channels=16,
        num_nodes=max_num_nodes,
        learnable_embedding=True,
    )

    # Initialize losses and optimizer
    # Based on paper experiments, lambda_d and lambda_t around 0.1-0.5 work well
    mse_criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    # entropy_criterion = OrdinalEntropyRegLoss(lambda_d=0.1, lambda_t=0.5)
    # TODO tighness looks like it's always 0.
    entropy_criterion = OrdinalEntropyRegLoss(lambda_d=1000, lambda_t=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Get targets
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)
    valid_mask = ~torch.isnan(y).any(dim=1)

    # Training loop
    model.train()
    print("\nStarting training with ordinal entropy regularization:")

    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        predictions, pooled_features = model(batch)

        # Calculate losses
        mse_loss, _ = mse_criterion(predictions, y)
        entropy_loss, diversity_loss, tightness_loss = entropy_criterion(
            pooled_features, y, valid_mask
        )

        # Combine losses
        total_loss = mse_loss + entropy_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}:")
        print(f"  MSE Loss: {mse_loss.item():.2e}")
        print(f"  Diversity Loss (Ld): {diversity_loss.item():.2e}")
        print(f"  Tightness Loss (Lt): {tightness_loss.item():.2e}")
        print(f"  Total Loss: {total_loss.item():.2e}")


if __name__ == "__main__":
    # main()
    main_ordinal_reg()
