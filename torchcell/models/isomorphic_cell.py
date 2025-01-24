# torchcell/models/isomorphic_cell
# [[torchcell.models.isomorphic_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/isomorphic_cell
# Test file: tests/torchcell/models/test_isomorphic_cell.py


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
    HypergraphConv,
)
from torchcell.nn.met_hypergraph_conv import StoichiometricHypergraphConv
from typing import Optional, Literal
from torch_geometric.typing import EdgeType
from torchcell.models.act import act_register
from collections import defaultdict
from torchcell.nn.aggr.set_transformer import SetTransformerAggregation

from typing import Any
from torchcell.nn.aggr.set_transformer import SetTransformerAggregation
import torch
import torch.nn as nn
from torch_geometric.typing import EdgeType
from torch_geometric.utils import sort_edge_index


class PreProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        layers.extend(
            [
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Combiner(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []

        # Input layer (concatenated features)
        layers.extend(
            [
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, gene_features: torch.Tensor, metabolism_features: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([gene_features, metabolism_features], dim=-1)
        return self.mlp(combined)


# FLAG Hetero GNN - Start
class ProjectedGATConv(nn.Module):
    def __init__(self, gat_conv, out_dim):
        super().__init__()
        self.gat = gat_conv
        self.project = nn.Linear(gat_conv.heads * gat_conv.out_channels, out_dim)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)  # Shape: (..., heads * out_channels)
        return self.project(x)  # Shape: (..., out_dim)


class PredictionHead(nn.Module):
    def __init__(self, layers: nn.ModuleList, residual: bool, dims: list[int]):
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


class HeteroGnn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        edge_types: list[EdgeType],
        conv_type: Literal["GCN", "GAT", "Transformer", "GIN"] = "GCN",
        layer_config: Optional[dict] = None,
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
        self.activation = act_register[activation]
        self.conv_type = conv_type
        self.norm = norm
        self.learnable_embedding = learnable_embedding
        # for use in scripts
        self.out_channels = out_channels
        if learnable_embedding and num_nodes is None:
            raise ValueError(
                "num_nodes must be provided when using learnable_embedding"
            )

        self.layer_config = self._get_layer_config(layer_config)
        self.dims = self._calculate_dimensions(in_channels, hidden_channels)

        # Initialize  embedding if specified
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

    def _get_layer_config(self, layer_config: Optional[dict]) -> dict:
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

    def _calculate_dimensions(self, in_channels: int, hidden_channels: int) -> dict:
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

    def _create_conv_dict(self, in_dim: int) -> dict:
        conv_dict = {}

        for edge_type in self.edge_types:
            if self.conv_type == "GCN":
                conv_dict[edge_type] = GCNConv(
                    in_dim,
                    self.dims["conv_hidden"],
                    **{
                        k: v
                        for k, v in self.layer_config.items()
                        if k in ["bias", "add_self_loops"]
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

        x = x_dict["gene"]

        return x

    @property
    def num_parameters(self) -> dict[str, int]:
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


# FLAG Hetero GNN - End


class MetabolismProcessor(nn.Module):
    def __init__(
        self,
        metabolite_dim: int,
        hidden_dim: int,
        hyperconv_num_layers: int = 2,
        set_transformer_heads: int = 4,
        use_attention: bool = True,
        use_skip: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.hidden_dim = hidden_dim

        # Constants from the diagram/math
        self.num_metabolites = 2534
        self.num_reactions = 4881

        self.metabolite_embedding = nn.Embedding(
            self.num_metabolites, hidden_dim, max_norm=1.0
        )

        self.hyper_convs = nn.ModuleList(
            [
                StoichiometricHypergraphConv(
                    in_channels=metabolite_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    use_attention=use_attention,
                    dropout=dropout,
                    bias=True,
                )
                for i in range(hyperconv_num_layers)
            ]
        )

        if use_skip:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(hyperconv_num_layers)]
            )

        transformer_config = {
            "channels": hidden_dim,
            "num_encoder_blocks": 2,
            "num_decoder_blocks": 1,
            "heads": set_transformer_heads,
            "concat": False,
            "layer_norm": True,
            "dropout": dropout,
            "use_isab": False,
        }

        self.metabolite_to_reaction = SetTransformerAggregation(**transformer_config)
        self.reaction_to_gene = SetTransformerAggregation(**transformer_config)
        self.reaction_sab = SetTransformerAggregation(**transformer_config)

    def _aggregate_by_index(
        self, features: torch.Tensor, indices: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        """Aggregates features by indices using mean pooling."""
        from torch_scatter import scatter_mean

        return scatter_mean(features, indices, dim=0, dim_size=dim_size)

    def forward(self, batch):
        device = batch[
            "metabolite", "reaction-genes", "metabolite"
        ].hyperedge_index.device
        num_genes = batch["gene"].num_nodes

        # Get graph components
        hyperedge_index = batch[
            "metabolite", "reaction-genes", "metabolite"
        ].hyperedge_index
        stoichiometry = batch[
            "metabolite", "reaction-genes", "metabolite"
        ].stoichiometry
        gene_features = batch["gene"].x
        reaction_to_genes = batch[
            "metabolite", "reaction-genes", "metabolite"
        ].reaction_to_genes_indices

        # Initialize embeddings
        metabolite_indices = torch.arange(self.num_metabolites, device=device)
        metabolite_embeddings = self.metabolite_embedding(metabolite_indices)
        gene_embeddings = torch.zeros(num_genes, self.hidden_dim, device=device)

        # Process genes to reactions (Hr)
        reaction_features = []
        reaction_indices = []
        for rxn_idx, gene_indices in reaction_to_genes.items():
            if isinstance(gene_indices, (list, tuple)):
                for gene_idx in gene_indices:
                    reaction_features.append(gene_features[gene_idx])
                    reaction_indices.append(rxn_idx)

        if reaction_features:
            features_tensor = torch.stack(reaction_features)
            indices_tensor = torch.tensor(reaction_indices, device=device)
            reaction_embeddings = self._aggregate_by_index(
                features_tensor, indices_tensor, self.num_reactions
            )
        else:
            reaction_embeddings = torch.zeros(
                self.num_reactions, self.hidden_dim, device=device
            )

        # Process through hypergraph convolutions (Zm)
        current = metabolite_embeddings
        for i, conv in enumerate(self.hyper_convs):
            next_emb = conv(
                x=current,
                edge_index=hyperedge_index,
                stoich=stoichiometry,
                hyperedge_attr=reaction_embeddings,
            )
            next_emb = torch.tanh(next_emb)
            if self.use_skip and i > 0:
                next_emb = self.layer_norms[i](next_emb + current)
            current = next_emb

        metabolite_embeddings = current

        # Process metabolites to reactions (Zr)
        metabolite_indices = hyperedge_index[0]
        reaction_indices = hyperedge_index[1]
        reaction_embeddings = self._aggregate_by_index(
            metabolite_embeddings[metabolite_indices],
            reaction_indices,
            self.num_reactions,
        )

        # Process reactions to genes (Zmg)
        gene_features = []
        gene_indices = []
        for rxn_idx, gene_indices_list in reaction_to_genes.items():
            if isinstance(gene_indices_list, (list, tuple)):
                for gene_idx in gene_indices_list:
                    gene_features.append(reaction_embeddings[rxn_idx])
                    gene_indices.append(gene_idx)

        if gene_features:
            features_tensor = torch.stack(gene_features)
            indices_tensor = torch.tensor(gene_indices, device=device)
            metabolic_embeddings = self._aggregate_by_index(
                features_tensor, indices_tensor, num_genes
            )
            gene_embeddings = metabolic_embeddings

        return gene_embeddings


class IsomorphicCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_types: list[EdgeType],
        num_layers: dict[str, int] = {
            "preprocessor": 2,
            "gene_encoder": 2,
            "metabolism": 2,
            "combiner": 2,
        },
        dropout: float = 0.1,
        gene_encoder_config: Optional[dict] = None,
        metabolism_config: Optional[dict] = None,
        set_transformer_config: Optional[dict] = None,
        preprocessor_config: Optional[dict] = None,
        combiner_config: Optional[dict] = None,
        prediction_head_config: Optional[dict] = None,
    ):
        super().__init__()

        # Initialize configs
        self.preprocessor_config = {"dropout": dropout}
        if preprocessor_config:
            self.preprocessor_config.update(preprocessor_config)

        self.metabolism_config = {
            "use_attention": True,
            "set_transformer_heads": 4,
            "use_skip": True,
            "dropout": dropout,
        }
        if metabolism_config:
            self.metabolism_config.update(metabolism_config)

        self.set_transformer_config = {
            "num_encoder_blocks": 2,
            "num_decoder_blocks": 1,
            "heads": 4,
            "concat": False,
            "layer_norm": True,
            "dropout": dropout,
            "num_induced_points": 32,
        }
        if set_transformer_config:
            self.set_transformer_config.update(set_transformer_config)

        self.combiner_config = {"dropout": dropout}
        if combiner_config:
            self.combiner_config.update(combiner_config)

        self.prediction_head_config = {
            "hidden_layers": [hidden_channels],
            "dropout": dropout,
            "activation": "relu",
            "use_layer_norm": True,
            "residual": True,
        }
        if prediction_head_config:
            self.prediction_head_config.update(prediction_head_config)

        # Initialize components
        self.preprocessor = PreProcessor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers["preprocessor"],
            dropout=self.preprocessor_config["dropout"],
        )

        self.gene_encoder = HeteroGnn(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers["gene_encoder"],
            edge_types=edge_types,
            **(gene_encoder_config or {}),
        )

        self.metabolism_processor = MetabolismProcessor(
            metabolite_dim=hidden_channels,
            hidden_dim=hidden_channels,
            hyperconv_num_layers=num_layers["metabolism"],
            set_transformer_heads=self.metabolism_config["set_transformer_heads"],
            use_attention=self.metabolism_config["use_attention"],
            use_skip=self.metabolism_config["use_skip"],
            dropout=self.metabolism_config["dropout"],
        )

        self.combiner = Combiner(
            hidden_channels=hidden_channels,
            num_layers=num_layers["combiner"],
            dropout=self.combiner_config["dropout"],
        )

        transformer_config = {
            k: v
            for k, v in self.set_transformer_config.items()
            if k not in ["channels"]
        }

        self.isab = SetTransformerAggregation(
            channels=hidden_channels, use_isab=True, **transformer_config
        )

        self.perturbed_sab = SetTransformerAggregation(
            channels=hidden_channels, use_isab=False, **transformer_config
        )

        self.growth_head = self._build_mlp(
            hidden_channels, 1, self.prediction_head_config
        )

        self.gene_interaction_head = self._build_mlp(
            hidden_channels, 1, self.prediction_head_config
        )

    def _build_mlp(self, in_dim: int, out_dim: int, config: dict) -> nn.Sequential:
        layers = []
        current_dim = in_dim

        for hidden_dim in config["hidden_layers"]:
            # Create a residual block class for skip connections
            if config["residual"] and current_dim == hidden_dim:

                class ResidualBlock(nn.Module):
                    def __init__(self, linear):
                        super().__init__()
                        self.linear = linear

                    def forward(self, x):
                        return self.linear(x) + x

                layers.append(ResidualBlock(nn.Linear(current_dim, hidden_dim)))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))

            if config["use_layer_norm"]:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(act_register[config["activation"]])
            layers.append(nn.Dropout(config["dropout"]))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    # TODO check for other uses of batch["gene"].ids_pert
    def _get_perturbed_indices(self, batch):
        """Returns list of perturbed indices for each batch item"""
        batch_size = batch["gene"].ptr.size(0) - 1
        pert_indices = []

        # Get start/end indices for each batch item
        for i in range(batch_size):
            start_idx = batch["gene"].x_pert_ptr[i]
            end_idx = batch["gene"].x_pert_ptr[i + 1]
            # Get indices for this batch item
            item_indices = batch["gene"].cell_graph_idx_pert[start_idx:end_idx].tolist()
            pert_indices.append(item_indices)

        return pert_indices

    def forward_single(self, batch):
        # Gene path - pass preprocessed features to gene_encoder
        z_g = self.gene_encoder(batch)

        # Metabolism path
        z_mg = self.metabolism_processor(batch)

        # Combine paths
        z = self.combiner(z_g, z_mg)
        return z

    def forward(self, cell_graph, batch):
        z_w = self.forward_single(cell_graph)
        pert_indices = self._get_perturbed_indices(batch)
        batch_size = len(pert_indices)

        # Expand z_w for batch dimension
        # batch_size, nodes, hidden_dim 
        z_w_expanded = z_w.expand(batch_size, -1, -1)

        # Collect perturbed vectors for each batch item
        z_p_list = []
        for i, indices in enumerate(pert_indices):
            z_p_list.append(z_w_expanded[i, indices])

        # Process intact graph
        z_i = self.forward_single(batch)
        z_i = self.isab(z_i, batch_size=batch_size)

        # Process perturbed embeddings
        z_p = torch.cat(z_p_list, dim=0)  # [total_num_perts, hidden_dim]
        z_p = self.perturbed_sab(z_p, batch_size=batch_size)

        # Calculate final outputs
        growth_w = self.growth_head(z_w)
        growth_i = self.growth_head(z_i)
        fitness_ratio = growth_i / (growth_w + 1e-8)
        gene_interaction = self.gene_interaction_head(z_p)

        predictions = torch.cat([fitness_ratio, gene_interaction], dim=1)

        return predictions, {
            "z_w": z_w,
            "z_p": z_p,
            "z_i": z_i,
            "growth_w": growth_w,
            "growth_i": growth_i,
        }


def initialize_model(dataset, device, config=None):
    if config is None:
        config = {}

    default_config = {
        # Model dimensions
        "in_channels": 64,
        "hidden_channels": 64,
        # Layer counts
        "num_layers": {
            "preprocessor": 2,
            "gene_encoder": 2,
            "metabolism": 2,
            "combiner": 2,
        },
        # Gene encoder settings
        "gene_encoder_config": {
            "conv_type": "GIN",
            "layer_config": {
                "train_eps": True,
                "hidden_multiplier": 2.0,
                "dropout": 0.1,
                "add_self_loops": True,
                "is_skip_connection": True,
                "num_mlp_layers": 3,
                "is_mlp_skip_connection": True,
            },
            "activation": "relu",
            "norm": "layer",
            "head_num_layers": 2,
            "head_hidden_channels": None,
            "head_dropout": 0.1,
            "head_activation": "relu",
            "head_residual": True,
            "head_norm": "layer",
        },
        # Metabolism processor settings
        "metabolism_config": {
            "use_attention": True,
            "attention_mode": "node",
            "heads": 4,
            "concat": True,
            "negative_slope": 0.2,
            "set_transformer_heads": 4,
            "use_skip": True,
        },
        # Set transformer settings
        "set_transformer_config": {
            "num_encoder_blocks": 2,
            "num_decoder_blocks": 1,
            "heads": 4,
            "concat": False,
            "layer_norm": True,
            "num_induced_points": 32,
        },
        # General settings
        "dropout": 0.1,
        "edge_types": [
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
        ],
    }

    # Update config with user-provided values
    for k, v in config.items():
        if isinstance(v, dict):
            default_config[k].update(v)
        else:
            default_config[k] = v

    config = default_config

    model = IsomorphicCell(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        edge_types=config["edge_types"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    return model


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
    from torchcell.metabolism.yeast_GEM import YeastGEM

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    # IDEA we are trying to use all gene reprs
    # genome.drop_chrmt()
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
        incidence_graphs={"metabolism": YeastGEM().reaction_map},
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
        num_workers=6,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=6,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()

    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    max_num_nodes = len(dataset.gene_set)
    return dataset, batch, max_num_nodes


def plot_correlations(predictions, true_values, save_path):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    # Convert to numpy and handle NaN values
    predictions_np = predictions.detach().cpu().numpy()
    true_values_np = true_values.detach().cpu().numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Colors for plotting
    color = "#2971A0"
    alpha = 0.6

    # Plot Fitness Correlations (predicted on x, true on y)
    mask_fitness = ~np.isnan(true_values_np[:, 0])
    y_fitness = true_values_np[mask_fitness, 0]
    x_fitness = predictions_np[mask_fitness, 0]

    pearson_fitness, _ = stats.pearsonr(x_fitness, y_fitness)
    spearman_fitness, _ = stats.spearmanr(x_fitness, y_fitness)
    mse_fitness = np.mean((y_fitness - x_fitness) ** 2)

    ax1.scatter(x_fitness, y_fitness, alpha=alpha, color=color)
    ax1.set_xlabel("Predicted Fitness")
    ax1.set_ylabel("True Fitness")
    ax1.set_title(
        f"Fitness\nMSE={mse_fitness:.3e}, n={len(x_fitness)}\n"
        + f"Pearson={pearson_fitness:.3f}, Spearman={spearman_fitness:.3f}"
    )

    # Add diagonal line for fitness
    min_val = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
    max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    # Plot Gene Interactions (predicted on x, true on y)
    mask_gi = ~np.isnan(true_values_np[:, 1])
    y_gi = true_values_np[mask_gi, 1]
    x_gi = predictions_np[mask_gi, 1]

    pearson_gi, _ = stats.pearsonr(x_gi, y_gi)
    spearman_gi, _ = stats.spearmanr(x_gi, y_gi)
    mse_gi = np.mean((y_gi - x_gi) ** 2)

    ax2.scatter(x_gi, y_gi, alpha=alpha, color=color)
    ax2.set_xlabel("Predicted Gene Interaction")
    ax2.set_ylabel("True Gene Interaction")
    ax2.set_title(
        f"Gene Interaction\nMSE={mse_gi:.3e}, n={len(x_gi)}\n"
        + f"Pearson={pearson_gi:.3f}, Spearman={spearman_gi:.3f}"
    )

    # Add diagonal line for gene interactions
    min_val = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(device="gpu"):
    from torchcell.losses.multi_dim_nan_tolerant import CombinedRegressionLoss
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    import os.path as osp
    import os
    from torchcell.timestamp import timestamp
    import torch

    # Check if CUDA is available when device='cuda' is requested
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    device = torch.device(device)
    print(f"\nUsing device: {device}")

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    MPLSTYLE_PATH = os.getenv("MPLSTYLE_PATH")
    plt.style.use(MPLSTYLE_PATH)

    # Load sample data including metabolism
    dataset, batch, max_num_nodes = load_sample_data_batch()

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)  # Base graph for whole
    batch = batch.to(device)

    # Model configuration
    model = initialize_model(dataset, device)

    print("\nModel architecture:")
    print(model)

    # Training setup
    loss_type = "mse"
    criterion = CombinedRegressionLoss(
        loss_type=loss_type, weights=torch.ones(2, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get targets and move to device
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Training loop
    model.train()
    print("\nStarting training:")
    losses = []
    num_epochs = 1000

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass - now using cell_graph and batch
            predictions, representations = model(cell_graph, batch)
            loss, loss_components = criterion(predictions, y)

            # Rest of training loop remains the same
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print("Predictions shape:", predictions.shape)
                print("Targets shape:", y.shape)
                print("Loss components:", loss_components)

                if device.type == "cuda":
                    print(
                        f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB"
                    )
                    print(
                        f"GPU memory cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB"
                    )

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    except RuntimeError as e:
        print(f"\nError during training: {e}")
        if device.type == "cuda":
            print("\nThis might be a GPU memory issue. Try:")
            print("1. Reducing batch size")
            print("2. Reducing model size")
            print("3. Using gradient checkpointing")
            print("4. Using mixed precision training")
        raise

    # Plotting (move tensors to CPU for matplotlib)
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(1, len(losses) + 1), losses, "b-", label=f"{loss_type} Training Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        osp.join(
            ASSET_IMAGES_DIR,
            f"cell_latent_perturbation_tform_training_loss_{loss_type}_{timestamp()}.png",
        )
    )
    plt.close()

    # Move predictions to CPU for correlation plotting
    correlation_save_path = osp.join(
        ASSET_IMAGES_DIR,
        f"cell_latent_perturbation_tform_correlation_plots_{loss_type}_{timestamp()}.png",
    )
    plot_correlations(predictions.cpu(), y.cpu(), correlation_save_path)

    # Final model evaluation
    print("\nFinal Performance:")
    model.eval()
    with torch.no_grad():
        final_predictions, _ = model(batch)
        final_loss, final_components = criterion(final_predictions, y)
        print(f"Final loss: {final_loss.item():.4f}")
        print("Loss components:", final_components)

        if device.type == "cuda":
            print(f"\nFinal GPU memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")

    # Optional: Clear CUDA cache if using GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # You can easily switch between CPU and GPU by changing the device parameter
    main(device="cuda")  # or main(device='cpu') for CPU
