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
        pooling: Literal["add", "mean", "max"] = "mean",
        activation: str = "relu",
        norm: Optional[str] = None,
        pred_head_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_types = edge_types
        self.pooling = pooling
        self.activation = act_register[activation]
        self.conv_type = conv_type
        self.norm = norm

        self.layer_config = self._get_layer_config(layer_config)
        self.dims = self._calculate_dimensions(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self._build_network()

        # Prediction head
        self.lin1 = nn.Linear(self.dims["actual_hidden"], hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(pred_head_dropout)

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

        if self.conv_type in ["GAT", "Transformer"] and self.layer_config.get(
            "concat", True
        ):
            heads = self.layer_config.get("heads", 1)
            dims["actual_hidden"] = hidden_channels
            dims["conv_hidden"] = hidden_channels // heads
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
                conv_dict[edge_type] = GATv2Conv(
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
        if self.pooling == "add":
            return global_add_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Invalid pooling type: {self.pooling}")

    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict=None):
        from torch_geometric.utils import add_self_loops

        # Handle self-loops for GIN and Transformer only
        if self.conv_type in ["GIN", "Transformer"] and self.layer_config.get(
            "add_self_loops", True
        ):
            edge_index_dict = {
                k: add_self_loops(v)[0] for k, v in edge_index_dict.items()
            }

        # First layer
        if self.conv_type == "Transformer" and edge_attr_dict is not None:
            x_dict = self.convs[0](x_dict, edge_index_dict, edge_attr_dict)
        else:
            x_dict = self.convs[0](x_dict, edge_index_dict)

        x_dict = {key: self.activation(self.norms[0](x)) for key, x in x_dict.items()}

        # Remaining layers with skip connections
        for i in range(1, self.num_layers):
            prev_x_dict = x_dict

            if self.conv_type == "Transformer" and edge_attr_dict is not None:
                x_dict = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)
            else:
                x_dict = self.convs[i](x_dict, edge_index_dict)

            x_dict = {
                key: self.activation(self.norms[i](x)) for key, x in x_dict.items()
            }

            if self.conv_type != "Transformer" and self.layer_config.get(
                "is_skip_connection", False
            ):
                x_dict = {
                    key: x + prev_x_dict[key] if key in prev_x_dict else x
                    for key, x in x_dict.items()
                }

        # Global pooling and prediction
        x = self._global_pool(x_dict["gene"], batch_dict["gene"])
        x = self.activation(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x

    @property
    def num_parameters(self) -> Dict[str, int]:
        conv_params = sum(
            sum(p.numel() for p in conv.parameters()) for conv in self.convs
        )
        norm_params = sum(
            sum(p.numel() for p in norm.parameters()) for norm in self.norms
        )
        final_params = sum(p.numel() for p in self.lin1.parameters()) + sum(
            p.numel() for p in self.lin2.parameters()
        )
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "conv_layers": conv_params,
            "norm_layers": norm_params,
            "final_layers": final_params,
            "breakdown_total": conv_params + norm_params + final_params,
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

    # Load sample data
    batch, _ = load_sample_data_batch()

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
            "add_self_loops": True,  # Built-in parameter
            "normalize": False,
            "is_skip_connection": True,
        },
        "GAT": {
            "heads": 4,
            "concat": True,
            "dropout": 0.1,
            "bias": True,
            "add_self_loops": True,  # Built-in parameter
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
            "add_self_loops": True,  # Handle manually
        },
        "GIN": {
            "train_eps": True,
            "hidden_multiplier": 2.0,
            "dropout": 0.1,
            "add_self_loops": True,  # Handle manually
            "is_skip_connection": True,
            "num_mlp_layers": 3,
            "is_mlp_skip_connection": True,
        },
    }

    model = HeteroGnnPool(
        in_channels=x_dict["gene"].size(1),
        hidden_channels=16,
        out_channels=2,
        num_layers=3,
        edge_types=edge_types,
        conv_type="Transformer",
        layer_config=configs["Transformer"],
        pooling="mean",
        activation="relu",
        norm="batch",
        pred_head_dropout=0.1,
    )

    print("\nModel architecture:")
    print(model)
    print("\nParameter counts:")
    print(model.num_parameters)

    # Initialize loss and optimizer
    criterion = CombinedLoss(loss_type="mse", weights=torch.ones(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        out = model(x_dict, edge_index_dict, batch_dict)

        # Calculate loss
        loss, _ = criterion(out, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
