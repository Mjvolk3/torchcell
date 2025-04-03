# torchcell/models/isomorphic_cell_attentional
# [[torchcell.models.isomorphic_cell_attentional]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/isomorphic_cell_attentional
# Test file: tests/torchcell/models/test_isomorphic_cell_attentional.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import os.path as osp
import os
import hydra
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
from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv
from typing import Optional, Literal
from torch_geometric.typing import EdgeType
from torchcell.models.act import act_register
from collections import defaultdict

from typing import Any, Union, Optional
from torch_geometric.nn.aggr.attention import AttentionalAggregation
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.typing import EdgeType
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import HeteroData
from torch_scatter import scatter, scatter_softmax


def get_norm_layer(channels: int, norm: str):
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


class AttentionalGraphAggregation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        # Gate network to compute attention scores
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )

        # Optional transformation network
        self.transform_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Dropout(dropout)
        )

        self.aggregator = AttentionalAggregation(
            gate_nn=self.gate_nn, nn=self.transform_nn
        )

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None
    ) -> torch.Tensor:
        return self.aggregator(x, index=index, dim_size=dim_size)


class PreProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        act = act_register[activation]
        norm_layer = (
            nn.LayerNorm(hidden_channels)
            if norm == "layer"
            else nn.BatchNorm1d(hidden_channels)
        )
        layers = []
        # initial layer
        layers.extend(
            [
                nn.Linear(in_channels, hidden_channels),
                norm_layer,
                act,
                nn.Dropout(dropout),
            ]
        )
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    norm_layer,
                    act,
                    nn.Dropout(dropout),
                ]
            )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Combiner(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        act = act_register[activation]
        layers = []
        # First (input) layer: concatenated features -> hidden_channels
        layers.extend(
            [
                nn.Linear(hidden_channels * 2, hidden_channels),
                get_norm_layer(hidden_channels, norm),
                act,
                nn.Dropout(dropout),
            ]
        )
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    get_norm_layer(hidden_channels, norm),
                    act,
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
        activation: str = "gelu",
        norm: Optional[str] = None,
        head_num_layers: int = 2,
        head_hidden_channels: Optional[int] = None,
        head_dropout: float = 0.0,
        head_activation: str = "gelu",
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
        norm: Optional[str] = None,
    ) -> nn.Module:
        if num_layers == 0:
            return nn.Identity()

        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        # Get the activation class from the register - notice we don't call it
        act_fn = type(act_register[activation])

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:  # Don't apply norm/act/dropout after the last layer
                if norm is not None:
                    layers.append(get_norm_layer(dims[i + 1], norm))
                # Create a new instance of the activation function
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, batch, preprocessed_features=None):
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
            # Use preprocessed features if provided, otherwise use raw features
            x_dict = {
                "gene": (
                    preprocessed_features
                    if preprocessed_features is not None
                    else batch["gene"].x
                )
            }

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
class GeneContextProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
    ):
        super().__init__()
        act = act_register[activation]
        self.hidden_channels = hidden_channels
        layers = []
        current_dim = in_channels
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_channels),
                    get_norm_layer(hidden_channels, norm),
                    act,
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_channels
        self.mlp = nn.Sequential(*layers)
        # SAB remains as before (optionally you could parameterize its norm as well)
        self.sab = SetTransformerAggregation(
            channels=hidden_channels,
            num_encoder_blocks=2,
            heads=4,
            layer_norm=True,
            dropout=dropout,
            use_isab=False,
        )

    def forward(
        self, gene_features: torch.Tensor, reaction_to_genes: dict[int, list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H_g = self.mlp(gene_features)
        rxn_feats = []
        rxn_indices = []
        for rxn_idx, gene_list in reaction_to_genes.items():
            for gidx in gene_list:
                rxn_feats.append(H_g[gidx])
                rxn_indices.append(rxn_idx)
        if rxn_feats:
            rxn_feats = torch.stack(rxn_feats, dim=0)
            rxn_indices = torch.tensor(rxn_indices, device=gene_features.device)
            sorted_idx = torch.argsort(rxn_indices)
            rxn_feats = rxn_feats[sorted_idx]
            rxn_indices = rxn_indices[sorted_idx]
            H_r = self.sab(rxn_feats, rxn_indices)
        else:
            H_r = torch.zeros(
                (max(reaction_to_genes.keys()) + 1, self.hidden_channels),
                device=gene_features.device,
            )
        return H_g, H_r


class ReactionMapper(nn.Module):
    """
    Aggregates metabolite features -> reaction embeddings via attention.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.1):
        super().__init__()
        self.aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

    def forward(
        self, metabolite_features: torch.Tensor, hyperedge_index: torch.Tensor
    ) -> torch.Tensor:
        # Sort edges by reaction index
        rxn_indices = hyperedge_index[1]
        sorted_idx = torch.argsort(rxn_indices)
        feats = metabolite_features[hyperedge_index[0][sorted_idx]]
        rxn_indices = rxn_indices[sorted_idx]

        Z_r = self.aggregator(feats, index=rxn_indices)
        return Z_r


class GeneMapper(nn.Module):
    """
    Aggregates reaction features -> gene embeddings.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.1):
        super().__init__()
        self.aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

    def forward(
        self,
        reaction_features: torch.Tensor,
        reaction_to_genes: dict[int, list[int]],
        num_genes: int,
    ) -> torch.Tensor:
        feats_list = []
        gene_idx_list = []
        for rxn_idx, g_indices in reaction_to_genes.items():
            for gid in g_indices:
                feats_list.append(reaction_features[rxn_idx])
                gene_idx_list.append(gid)

        if feats_list:
            feats = torch.stack(feats_list, dim=0)
            gene_idx = torch.tensor(gene_idx_list, device=reaction_features.device)
            # Sort by gene index
            sorted_idx = torch.argsort(gene_idx)
            feats = feats[sorted_idx]
            gene_idx = gene_idx[sorted_idx]
            Z_mg = self.aggregator(feats, index=gene_idx)
        else:
            Z_mg = torch.zeros(
                (num_genes, reaction_features.size(1)), device=reaction_features.device
            )
        return Z_mg


class MetabolismProcessor(nn.Module):
    def __init__(
        self,
        max_metabolite_nodes: int,
        hidden_dim: int,
        num_layers: dict[str, int] = {"metabolite": 2},
        dropout: float = 0.1,
        use_attention: bool = True,
        heads: int = 1,
    ):
        super().__init__()
        self.max_metabolite_nodes = max_metabolite_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Metabolite embeddings - We immeditalely layernorm
        self.metabolite_embedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=max_metabolite_nodes,
                embedding_dim=hidden_dim,
                max_norm=1.0,
            ),
            nn.LayerNorm(hidden_dim),
        )

        # Gene -> Reaction aggregation
        self.gene_reaction_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_dim, out_channels=hidden_dim, dropout=dropout
        )

        # Metabolite processing with stoichiometry
        self.metabolite_processors = nn.ModuleList(
            [
                StoichHypergraphConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    use_attention=use_attention,
                    attention_mode="node",
                    dropout=dropout,
                    bias=True,
                    heads=heads,
                )
                for _ in range(num_layers["metabolite"])
            ]
        )

        # Layer norms for metabolite processing
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers["metabolite"])]
        )

        # Final aggregators
        self.metabolite_reaction_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_dim, out_channels=hidden_dim, dropout=dropout
        )

        self.reaction_gene_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_dim, out_channels=hidden_dim, dropout=dropout
        )

    def whole_forward(
        self, graph: HeteroData, preprocessed_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process single instance (whole cell graph)"""
        # 1. Gene -> Reaction context
        gpr_edge_index = graph["gene", "gpr", "reaction"].hyperedge_index
        gene_x = (
            preprocessed_features
            if preprocessed_features is not None
            else graph["gene"].x
        )

        H_r = self.gene_reaction_aggregator(
            x=gene_x[gpr_edge_index[0]],
            index=gpr_edge_index[1],
            dim_size=graph["reaction"].num_nodes,
        )

        # 2. Initialize and process metabolites
        num_metabolites = graph["metabolite"].num_nodes
        device = gene_x.device

        # Generate metabolite embeddings
        metabolite_indices = torch.arange(num_metabolites, device=device)
        metabolite_indices = torch.clamp(
            metabolite_indices, 0, self.max_metabolite_nodes - 1
        )
        Z_m = self.metabolite_embedding(metabolite_indices)

        # Get and sort metabolite edges
        met_edge_index = graph["metabolite", "reaction", "metabolite"].hyperedge_index
        stoich = graph["metabolite", "reaction", "metabolite"].stoichiometry

        # Ensure indices are valid
        if met_edge_index[0].max() >= num_metabolites:
            met_edge_index = met_edge_index.clone()
            met_edge_index[0] = met_edge_index[0].clamp(max=num_metabolites - 1)

        # Sort edges and stoichiometry
        met_edge_index, stoich = sort_edge_index(
            met_edge_index, edge_attr=stoich, sort_by_row=False
        )

        # Process metabolites through layers
        for conv, norm in zip(self.metabolite_processors, self.layer_norms):
            out = conv(
                x=Z_m, edge_index=met_edge_index, stoich=stoich, hyperedge_attr=H_r
            )
            out = norm(out)
            Z_m = Z_m + out  # Residual connection
            Z_m = torch.tanh(Z_m)

        # 3. Metabolite -> Reaction mapping
        Z_r = self.metabolite_reaction_aggregator(
            x=Z_m[met_edge_index[0]],
            index=met_edge_index[1],
            dim_size=graph["reaction"].num_nodes,
        )

        # 4. Reaction -> Gene mapping
        Z_mg = self.reaction_gene_aggregator(
            x=Z_r[gpr_edge_index[1]],
            index=gpr_edge_index[0],
            dim_size=graph["gene"].num_nodes,
        )

        return Z_mg

    def intact_perturbed_forward(
        self, batch: HeteroData, preprocessed_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process batched data"""
        # 1. Gene -> Reaction mapping
        gpr_edge_index = batch["gene", "gpr", "reaction"].hyperedge_index
        gene_x = (
            preprocessed_features
            if preprocessed_features is not None
            else batch["gene"].x
        )

        H_r = self.gene_reaction_aggregator(
            x=gene_x[gpr_edge_index[0]],
            index=gpr_edge_index[1],
            dim_size=batch["reaction"].num_nodes,
        )
        # 2. Initialize metabolite features
        device = gene_x.device
        batch_size = len(batch["metabolite"].ptr) - 1
        metabolites_per_graph = self.max_metabolite_nodes  # 2534
        num_metabolites = batch["metabolite"].num_nodes

        # Safety check
        if num_metabolites > batch_size * metabolites_per_graph:
            print(
                f"Warning: Batch has {num_metabolites} metabolites but embedding only supports {metabolites_per_graph} per graph"
            )

        # Generate indices for each graph in batch
        metabolite_indices = torch.arange(metabolites_per_graph, device=device)
        # Repeat indices for each graph in batch
        metabolite_indices = metabolite_indices.repeat(batch_size)
        metabolite_indices = torch.clamp(
            metabolite_indices, 0, self.max_metabolite_nodes - 1
        )
        Z_m = self.metabolite_embedding(metabolite_indices)

        # Get and process metabolite edges
        met_edge_index = batch["metabolite", "reaction", "metabolite"].hyperedge_index
        stoich = batch["metabolite", "reaction", "metabolite"].stoichiometry

        # Ensure edge indices are valid
        if met_edge_index[0].max() >= num_metabolites:
            met_edge_index = met_edge_index.clone()
            met_edge_index[0] = met_edge_index[0].clamp(max=num_metabolites - 1)

        # Process metabolites
        for conv, norm in zip(self.metabolite_processors, self.layer_norms):
            out = conv(
                x=Z_m,
                edge_index=met_edge_index,
                stoich=stoich,
                hyperedge_attr=H_r,
                num_edges=batch["reaction"].num_nodes,
            )
            out = norm(out)
            Z_m = Z_m + out  # Residual connection
            Z_m = torch.tanh(Z_m)

        # 3. Metabolite -> Reaction mapping
        Z_r = self.metabolite_reaction_aggregator(
            x=Z_m[met_edge_index[0]],
            index=met_edge_index[1],
            dim_size=batch["reaction"].num_nodes,
        )

        # 4. Reaction -> Gene mapping
        Z_mg = self.reaction_gene_aggregator(
            x=Z_r[gpr_edge_index[1]],
            index=gpr_edge_index[0],
            dim_size=batch["gene"].num_nodes,
        )

        return Z_mg

    def forward(
        self, data: HeteroData, preprocessed_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Main forward pass"""
        is_batched = hasattr(data["gene"], "batch")
        if is_batched:
            return self.intact_perturbed_forward(data, preprocessed_features)
        else:
            return self.whole_forward(data, preprocessed_features)


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
        attention_config: Optional[dict] = None,
        preprocessor_config: Optional[dict] = None,
        combiner_config: Optional[dict] = None,
        prediction_head_config: Optional[dict] = None,
    ):
        super().__init__()

        # Check if using learnable embeddings from config
        if gene_encoder_config is not None:
            self.use_learned_embedding = "learnable" in gene_encoder_config.get(
                "embedding_type", ""
            )
            if self.use_learned_embedding:
                self.node_embedding = nn.Embedding(
                    num_embeddings=gene_encoder_config["max_num_nodes"],
                    embedding_dim=gene_encoder_config[
                        "learnable_embedding_input_channels"
                    ],
                    max_norm=1.0,
                )
                # Filter params for HeteroGnn
                hetero_gnn_config = {
                    k: v
                    for k, v in gene_encoder_config.items()
                    if k
                    not in [
                        "embedding_type",
                        "max_num_nodes",
                        "learnable_embedding_input_channels",
                    ]
                }
            else:
                hetero_gnn_config = gene_encoder_config
        else:
            self.use_learned_embedding = False
            hetero_gnn_config = {}
            self.node_embedding = None
        # Configurations
        self.preprocessor_config = {"dropout": dropout}
        if preprocessor_config:
            self.preprocessor_config.update(preprocessor_config)

        self.metabolism_config = {
            "use_attention": True,
            "set_transformer_heads": 4,
            "use_skip": True,
            "dropout": dropout,
            "heads": 1,
        }
        if metabolism_config:
            self.metabolism_config.update(metabolism_config)

        self.combiner_config = {"dropout": dropout}
        if combiner_config:
            self.combiner_config.update(combiner_config)

        self.prediction_head_config = {
            "hidden_layers": [hidden_channels],
            "dropout": dropout,
            "activation": "gelu",
            "use_layer_norm": True,
            "residual": True,
        }
        if prediction_head_config:
            self.prediction_head_config.update(prediction_head_config)

        # Initialize components
        # Preprocessor now immediately normalizes raw gene features via an initial LayerNorm
        self.preprocessor = PreProcessor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers["preprocessor"],
            dropout=self.preprocessor_config["dropout"],
        )

        # Gene encoder: a HeteroGnn that operates on preprocessed features
        self.gene_encoder = HeteroGnn(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers["gene_encoder"],
            edge_types=edge_types,
            **(hetero_gnn_config or {}),
        )

        # Metabolism processor uses the same preprocessed gene features
        self.metabolism_processor = MetabolismProcessor(
            max_metabolite_nodes=self.metabolism_config.get(
                "max_metabolite_nodes", 2534
            ),
            hidden_dim=hidden_channels,
            num_layers={"metabolite": num_layers["metabolism"]},
            dropout=self.metabolism_config["dropout"],
            use_attention=self.metabolism_config["use_attention"],
            heads=self.metabolism_config["heads"],
        )

        # Combiner to merge gene and metabolism paths
        self.combiner = Combiner(
            hidden_channels=hidden_channels,
            num_layers=num_layers["combiner"],
            dropout=self.combiner_config["dropout"],
        )

        # Aggregators for whole and perturbed graph representations
        self.whole_intact_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )
        self.perturbed_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

        # Prediction heads for growth (fitness) and gene interaction
        self.growth_head = self._build_mlp(
            hidden_channels, 1, self.prediction_head_config
        )
        self.gene_interaction_head = self._build_mlp(
            hidden_channels, 1, self.prediction_head_config
        )

    def _build_mlp(self, in_dim: int, out_dim: int, config: dict) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = in_dim

        for hidden_dim in config["hidden_layers"]:
            # Create a residual block with integrated dropout if applicable.
            if config.get("residual", False) and current_dim == hidden_dim:

                class ResidualBlock(nn.Module):
                    def __init__(self, linear: nn.Linear, dropout: float) -> None:
                        super().__init__()
                        self.linear = linear
                        self.dropout = nn.Dropout(dropout)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.dropout(self.linear(x)) + x

                layers.append(
                    ResidualBlock(nn.Linear(current_dim, hidden_dim), config["dropout"])
                )
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))

            if config.get("use_layer_norm", False):
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(act_register[config["activation"]])
            # Add dropout after activation.
            layers.append(nn.Dropout(config["dropout"]))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

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

    def forward_single(self, data: HeteroData) -> torch.Tensor:
        gene_data = data["gene"]
        if self.use_learned_embedding:
            # Check if we have a batched (perturbed) graph
            if hasattr(gene_data, "ptr"):
                # Batched perturbed graphs: ptr exists.
                batch_size = gene_data.ptr.numel() - 1
                device = (
                    gene_data.x.device
                    if hasattr(gene_data, "x")
                    else gene_data.batch.device
                )
                base_embeddings = self.node_embedding.weight  # [num_nodes, emb_dim]
                # Expand embeddings to match the batch size:
                replicated = base_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                all_embeddings = replicated.reshape(-1, replicated.size(-1))
                # Expecting pert_mask to be of shape [batch_size * num_nodes]
                mask = getattr(gene_data, "pert_mask", None)
                if mask is None:
                    mask = torch.zeros(
                        all_embeddings.size(0), dtype=torch.bool, device=device
                    )
                x = all_embeddings[~mask]
            else:
                # Wildtype (cell_graph): single graph, no ptr.
                device = (
                    gene_data.x.device
                    if hasattr(gene_data, "x")
                    else self.node_embedding.weight.device
                )
                base_embeddings = self.node_embedding.weight  # [num_nodes, emb_dim]
                mask = getattr(gene_data, "pert_mask", None)
                if mask is None:
                    mask = torch.zeros(
                        base_embeddings.size(0), dtype=torch.bool, device=device
                    )
                x = base_embeddings[~mask]
        else:
            x = self.preprocessor(gene_data.x)

        z_g = self.gene_encoder(data, preprocessed_features=x)
        z_mg = self.metabolism_processor(data, preprocessed_features=x)
        z = self.combiner(z_g, z_mg)
        return z

    def forward(self, cell_graph, batch):
        # Process whole graph first
        z_w = self.forward_single(cell_graph)
        z_w = self.whole_intact_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )  # [1, hidden_channels]

        # Process intact graph
        z_i = self.forward_single(batch)
        z_i = self.whole_intact_aggregator(
            z_i, index=batch["gene"].batch
        )  # [batch_size, hidden_channels]

        # Get perturbed embeddings
        batch_size = len(batch["gene"].ptr) - 1
        pert_indices = []
        for i in range(batch_size):
            start_idx = batch["gene"].x_pert_ptr[i]
            end_idx = batch["gene"].x_pert_ptr[i + 1]
            item_indices = batch["gene"].cell_graph_idx_pert[start_idx:end_idx]
            pert_indices.append(item_indices)

        # Expand z_w for batch dimension
        z_w_expanded = z_w.expand(batch_size, -1)  # [batch_size, hidden_channels]

        # Get perturbed vectors
        z_p_list = []
        batch_indices_list = []
        for i, indices in enumerate(pert_indices):
            z_p_list.append(z_w_expanded[i : i + 1].expand(len(indices), -1))
            batch_indices_list.append(torch.full((len(indices),), i, device=z_w.device))

        z_p = torch.cat(z_p_list, dim=0)
        batch_idx_pert = torch.cat(batch_indices_list, dim=0)
        z_p = self.perturbed_aggregator(z_p, index=batch_idx_pert, dim_size=batch_size)

        # Calculate growths and predictions
        growth_w = self.growth_head(z_w)  # [1, 1] - single reference value
        growth_i = self.growth_head(z_i)  # [batch_size, 1]

        fitness_ratio = growth_i / (
            growth_w + 1e-8
        )  # growth_w broadcasts automatically
        gene_interaction = self.gene_interaction_head(z_p)

        predictions = torch.cat([fitness_ratio, gene_interaction], dim=1)

        return predictions, {
            "z_w": z_w,  # [1, 64]
            "z_p": z_p,  # [batch_size, 64]
            "z_i": z_i,  # [batch_size, 64]
            "growth_w": growth_w,  # [1, 1]
            "growth_i": growth_i,  # [batch_size, 1]
        }

    @property
    def num_parameters(self) -> dict[str, int]:
        preprocessor_params = sum(p.numel() for p in self.preprocessor.parameters())
        gene_encoder_params = sum(p.numel() for p in self.gene_encoder.parameters())
        metabolism_params = sum(
            p.numel() for p in self.metabolism_processor.parameters()
        )
        combiner_params = sum(p.numel() for p in self.combiner.parameters())
        aggregator_params = sum(
            p.numel()
            for aggregator in [self.whole_intact_aggregator, self.perturbed_aggregator]
            for p in aggregator.parameters()
        )
        growth_head_params = sum(p.numel() for p in self.growth_head.parameters())
        interaction_head_params = sum(
            p.numel() for p in self.gene_interaction_head.parameters()
        )
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "preprocessor": preprocessor_params,
            "gene_encoder": gene_encoder_params,
            "metabolism_processor": metabolism_params,
            "combiner": combiner_params,
            "aggregators": aggregator_params,
            "growth_head": growth_head_params,
            "interaction_head": interaction_head_params,
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

    # from torchcell.datasets.fungal_up_down_transformer import (
    #     FungalUpDownTransformerDataset,
    # )
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
    print(f"DATA_ROOT: {DATA_ROOT}")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    # IDEA we are trying to use all gene reprs
    # genome.drop_chrmt()
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    selected_node_embeddings = ["codon_frequency"]
    node_embeddings = {}
    # if "fudt_downstream" in selected_node_embeddings:
    #     node_embeddings["fudt_downstream"] = FungalUpDownTransformerDataset(
    #         root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
    #         genome=genome,
    #         model_name="species_downstream",
    #     )

    # if "fudt_upstream" in selected_node_embeddings:
    #     node_embeddings["fudt_upstream"] = FungalUpDownTransformerDataset(
    #         root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
    #         genome=genome,
    #         model_name="species_upstream",
    #     )
    if "codon_frequency" in selected_node_embeddings:
        node_embeddings["codon_frequency"] = CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    reaction_map = gem.reaction_map

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism": reaction_map},
        node_embeddings=node_embeddings,
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
        batch_size=4,
        random_seed=seed,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    max_num_nodes = len(dataset.gene_set)
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break
    input_channels = dataset.cell_graph["gene"].x.size()[-1]
    return dataset, batch, input_channels, max_num_nodes


def plot_correlations(
    predictions, true_values, save_path, lambda_info="", weight_decay=""
):
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    # Convert to numpy and handle NaN values
    predictions_np = predictions.detach().cpu().numpy()
    true_values_np = true_values.detach().cpu().numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Add a suptitle with lambda and weight decay information
    fig.suptitle(f"{lambda_info}, wd={weight_decay}", fontsize=12)

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
        f"Pearson={pearson_fitness:.3f}, Spearman={spearman_fitness:.3f}"
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
        f"Pearson={pearson_gi:.3f}, Spearman={spearman_gi:.3f}"
    )
    # Add diagonal line for gene interactions
    min_val = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/003-fit-int/conf"),
    config_name="isomorphic_cell_attentional",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.timestamp import timestamp

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    if cfg.trainer.accelerator.lower() == "gpu":
        device = "cuda"
    else:
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    device = torch.device(device)
    print(f"\nUsing device: {device}")

    # Load sample data including metabolism
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch()

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    gene_encoder_config = dict(cfg.model.gene_encoder_config)  # Convert to dict
    # Add learnable embedding params if specified
    if any("learnable" in emb for emb in cfg.cell_dataset.node_embeddings):
        gene_encoder_config.update(
            {
                "embedding_type": "learnable",
                "max_num_nodes": cell_graph["gene"].num_nodes,
                "learnable_embedding_input_channels": cfg.cell_dataset.learnable_embedding_input_channels,
            }
        )

    model = IsomorphicCell(
        in_channels=input_channels,
        hidden_channels=cfg.model.hidden_channels,
        edge_types=[
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
        ],
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        gene_encoder_config=gene_encoder_config,  # Use updated dict
        metabolism_config=cfg.model.metabolism_config,
        combiner_config=cfg.model.combiner_config,
        prediction_head_config=cfg.model.prediction_head_config,
    )

    print("\nModel architecture:")
    print(model)
    print("Parameter counts:", model.num_parameters)

    # Set lambda values and weight decay
    lambda_dist = cfg.regression_task.lambda_dist
    lambda_supcr = cfg.regression_task.lambda_supcr
    lambda_cell = cfg.regression_task.lambda_cell
    weight_decay = cfg.regression_task.optimizer.weight_decay

    # Compute weights (example calculation)
    total_non_nan = (~batch["gene"].fitness.isnan()).sum() + (
        ~batch["gene"].gene_interaction.isnan()
    ).sum()
    minus_fit_count = 1 - (~batch["gene"].fitness.isnan()).sum()
    minus_gi_count = 1 - (~batch["gene"].gene_interaction.isnan()).sum()
    weights = torch.tensor(
        [minus_fit_count / total_non_nan, minus_gi_count / total_non_nan]
    ).to(device)

    # Training setup using ICLoss.
    criterion = ICLoss(
        lambda_dist=lambda_dist,
        lambda_supcr=lambda_supcr,
        lambda_cell=lambda_cell,
        weights=weights,
    )

    lr = cfg.regression_task.optimizer.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Get targets and move to device
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Training loop
    model.train()
    print("\nStarting training:")
    losses = []
    num_epochs = 100

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Forward pass: obtain predictions and representations (including z_w, z_p, z_i)
            predictions, representations = model(cell_graph, batch)
            loss, loss_components = criterion(
                predictions,
                y,
                representations["z_w"],
                representations["z_p"],
                representations["z_i"],
            )

            if epoch % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print("Predictions shape:", predictions.shape)
                print("Targets shape:", y.shape)
                print("Loss components:", loss_components)
                print("MSE dimension losses:", loss_components.get("mse_dim_losses"))
                print("Dist dimension losses:", loss_components.get("dist_dim_losses"))
                print(
                    "SupCR dimension losses:", loss_components.get("supcr_dim_losses")
                )
                print("Cell dimension losses:", loss_components.get("cell_dim_losses"))
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

    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses) + 1), losses, "b-", label="ICLoss Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(
        f"Training Loss Over Time: λ_dist={lambda_dist}, λ_supcr={lambda_supcr}, λ_cell={lambda_cell}, wd={weight_decay}"
    )
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        osp.join(
            ASSET_IMAGES_DIR,
            f"isomorphic_cell_attentional_training_loss_ICLoss_{timestamp()}.png",
        )
    )
    plt.close()

    # Plot correlations with lambda and weight decay info
    from torchcell.models.isomorphic_cell_attentional import plot_correlations

    correlation_save_path = osp.join(
        ASSET_IMAGES_DIR,
        f"isomorphic_cell_attentional_correlation_plots_ICLoss_{timestamp()}.png",
    )
    lambda_info = f"λ_dist={lambda_dist}, λ_supcr={lambda_supcr}, λ_cell={lambda_cell}"
    plot_correlations(
        predictions.cpu(),
        y.cpu(),
        correlation_save_path,
        lambda_info=lambda_info,
        weight_decay=weight_decay,
    )

    # Final model evaluation
    print("\nFinal Performance:")
    model.eval()
    with torch.no_grad():
        final_predictions, final_reps = model(cell_graph, batch)
        final_loss, final_components = criterion(
            final_predictions,
            y,
            final_reps["z_w"],
            final_reps["z_p"],
            final_reps["z_i"],
        )
        print(f"Final loss: {final_loss.item():.4f}")
        print("Final loss components:", final_components)
        print("Final MSE dimension losses:", final_components.get("mse_dim_losses"))
        print("Final Dist dimension losses:", final_components.get("dist_dim_losses"))
        print("Final SupCR dimension losses:", final_components.get("supcr_dim_losses"))
        print("Final Cell dimension losses:", final_components.get("cell_dim_losses"))
        if device.type == "cuda":
            print(f"\nFinal GPU memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
