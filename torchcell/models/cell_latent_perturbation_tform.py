
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
from torchcell.nn.met_hypergraph_conv import StoichHypergraphConv
from typing import Dict, Optional, List, Literal
from torch_geometric.typing import EdgeType
from torchcell.models.act import act_register
from collections import defaultdict


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


class SetTransformer(nn.Module):
    """Set transformer that processes sets of gene embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through transformer layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Project input
        x = self.input_proj(x)

        # Apply transformer layers
        for layer, norm in zip(self.layers, self.norm_layers):
            # Self-attention
            attended, _ = layer(x, x, x)
            attended = self.dropout(attended)
            x = norm(x + attended)  # Add & norm

        x = self.output_norm(x)

        # Global mean pooling
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
            x = x.sum(dim=1) / mask.float().sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)

        return x


class SetNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_node_layers: int = 2,
        num_set_layers: int = 2,
        dropout: float = 0.1,
        pooling: Literal["mean", "sum"] = "mean",
        activation: str = "relu",  # Added activation parameter
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Map activation string to function
        self.activation = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }[activation]

        # Node processing MLP
        node_layers = []
        current_dim = input_dim
        for i in range(num_node_layers):
            node_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self.activation,  # Use chosen activation
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.node_mlp = nn.Sequential(*node_layers)

        # Set processing MLP
        set_layers = []
        current_dim = hidden_dim
        for i in range(num_set_layers - 1):
            set_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self.activation,  # Use chosen activation
                    nn.Dropout(dropout),
                ]
            )
        set_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.set_mlp = nn.Sequential(*set_layers)

        self.pooling = pooling

    def node_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process nodes without pooling.

        Args:
            x: Input tensor of shape (num_nodes, input_dim)

        Returns:
            Tensor of shape (num_nodes, hidden_dim)
        """
        return self.node_mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input with pooling for whole/intact/perturbed sets.

        Args:
            x: Input tensor of shape (num_nodes, input_dim)

        Returns:
            Tensor of shape (hidden_dim,)
        """
        # Process nodes: (num_nodes, input_dim) -> (num_nodes, hidden_dim)
        x = self.node_mlp(x)

        # Global pooling: (num_nodes, hidden_dim) -> (hidden_dim,)
        if self.pooling == "mean":
            x = x.mean(dim=0)
        else:  # sum pooling
            x = x.sum(dim=0)

        # Process pooled representation
        x = self.set_mlp(x)

        return x


class MetabolismProcessor(nn.Module):
    def __init__(
        self,
        metabolite_dim: int,
        hidden_dim: int,
        hyperconv_num_layers: int = 2,
        set_net_num_layers: int = 2,
        use_attention: bool = False,
        use_skip: bool = False,  # New parameter for skip connections
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_skip = use_skip  # Store skip connection flag
        self.hyperconv_num_layers = hyperconv_num_layers

        # Learnable embeddings for hyperedges (reactions)
        self.hyperedge_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        # Create multiple hypergraph convolution layers
        self.hyper_convs = nn.ModuleList(
            [
                StoichHypergraphConv(
                    in_channels=metabolite_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    use_attention=use_attention,
                    dropout=dropout,
                    bias=True,
                )
                for i in range(hyperconv_num_layers)
            ]
        )

        # Add layer norms for skip connections if enabled
        if use_skip:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(hyperconv_num_layers)]
            )

        # SetNets with unified number of layers
        self.reaction_set_net = SetNet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_node_layers=set_net_num_layers,
            num_set_layers=set_net_num_layers,
            dropout=dropout,
            activation="tanh",
        )

        self.gene_set_net = SetNet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_node_layers=set_net_num_layers,
            num_set_layers=set_net_num_layers,
            dropout=dropout,
            activation="tanh",
        )

    def forward(self, batch) -> torch.Tensor:
        device = batch[
            "metabolite", "reaction_genes", "metabolite"
        ].hyperedge_index.device

        # Process metabolites
        hyperedge_index = batch[
            "metabolite", "reaction_genes", "metabolite"
        ].hyperedge_index
        stoichiometry = batch[
            "metabolite", "reaction_genes", "metabolite"
        ].stoichiometry
        num_metabolites = batch["metabolite"].num_nodes

        # Initialize metabolite features
        metabolite_embeddings = torch.randn(
            num_metabolites, self.hyper_convs[0].in_channels, device=device
        )

        if self.use_attention:
            # Get number of unique reactions for expanding edge features
            num_reactions = hyperedge_index[1].max().item() + 1
            # Expand the learnable embedding for each reaction
            hyperedge_features = self.hyperedge_embedding.expand(num_reactions, -1)
        else:
            hyperedge_features = None

        # Apply multiple hypergraph convolution layers with optional skip connections
        prev_embeddings = metabolite_embeddings
        for i in range(self.hyperconv_num_layers):
            # Apply hypergraph convolution
            current_embeddings = self.hyper_convs[i](
                x=prev_embeddings,
                edge_index=hyperedge_index,
                stoich=stoichiometry,
                hyperedge_attr=hyperedge_features,
            )

            # Apply tanh activation
            current_embeddings = torch.tanh(current_embeddings)

            # Apply skip connection if enabled (except for first layer)
            if self.use_skip and i > 0:
                # Apply layer norm before skip connection
                normalized_current = self.layer_norms[i](current_embeddings)
                normalized_prev = self.layer_norms[i](prev_embeddings)
                metabolite_embeddings = normalized_current + normalized_prev
            else:
                metabolite_embeddings = current_embeddings

            prev_embeddings = metabolite_embeddings

        # Create reaction-to-metabolite sparse matrix
        indices = torch.stack([hyperedge_index[1], hyperedge_index[0]])
        values = stoichiometry
        num_reactions = hyperedge_index[1].max().item() + 1
        reaction_to_metabolite = torch.sparse_coo_tensor(
            indices, values, (num_reactions, num_metabolites)
        ).to_dense()

        # Normalize reaction-to-metabolite matrix using absolute values
        reaction_sizes = torch.abs(reaction_to_metabolite).sum(dim=1, keepdim=True)
        reaction_to_metabolite = reaction_to_metabolite / (reaction_sizes + 1e-8)

        # Compute reaction embeddings
        reaction_metabolites = torch.mm(reaction_to_metabolite, metabolite_embeddings)

        # Process through reaction set net
        reaction_representations = self.reaction_set_net.node_forward(
            reaction_metabolites
        )

        # Map reactions to genes
        reaction_to_genes_indices = batch[
            "metabolite", "reaction_genes", "metabolite"
        ].reaction_to_genes_indices

        # Create gene-to-reaction mapping
        gene_indices = []
        reaction_indices = []
        for reaction_idx, gene_idx_list in reaction_to_genes_indices.items():
            if isinstance(gene_idx_list, (list, tuple)):
                for gene_idx in gene_idx_list:
                    if isinstance(gene_idx, int) and gene_idx >= 0:
                        gene_indices.append(gene_idx)
                        reaction_indices.append(reaction_idx)
            elif isinstance(gene_idx_list, int) and gene_idx_list >= 0:
                gene_indices.append(gene_idx_list)
                reaction_indices.append(reaction_idx)

        num_genes = batch["gene"].num_nodes

        # Create gene-to-reaction matrix
        if len(gene_indices) > 0:
            g2r_indices = torch.tensor([gene_indices, reaction_indices], device=device)
            g2r_values = torch.ones(len(gene_indices), device=device)
            gene_to_reaction = torch.sparse_coo_tensor(
                g2r_indices, g2r_values, (num_genes, num_reactions)
            ).to_dense()
        else:
            gene_to_reaction = torch.zeros((num_genes, num_reactions), device=device)

        # Normalize gene-to-reaction matrix
        gene_counts = gene_to_reaction.sum(dim=1, keepdim=True)
        gene_to_reaction = gene_to_reaction / (gene_counts + 1e-8)

        # Compute gene embeddings
        gene_reaction_embeddings = torch.mm(gene_to_reaction, reaction_representations)

        # Final processing through gene set net
        gene_representations = self.gene_set_net.node_forward(gene_reaction_embeddings)

        return gene_representations


class CellLatentPerturbation(nn.Module):
    def __init__(
        self,
        # Base dimensions
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        # Gene encoder params
        gene_encoder_num_layers: int = 3,
        gene_encoder_conv_type: Literal["GCN", "GAT", "Transformer", "GIN"] = "GCN",
        gene_encoder_layer_config: Optional[Dict] = None,
        gene_encoder_head_num_layers: int = 2,
        # Metabolism processor params
        metabolism_num_layers: int = 2,
        metabolism_attention: bool = False,
        metabolism_hyperconv_layers: int = 2,
        metabolism_setnet_layers: int = 2,
        metabolism_use_skip: bool = True,
        # Set processor params
        set_transformer_heads: int = 4,
        set_transformer_layers: int = 2,
        set_net_node_layers: int = 2,
        set_net_set_layers: int = 2,
        # Combiner params
        combiner_num_layers: int = 1,
        combiner_hidden_factor: float = 1.0,  # Multiplier for combiner hidden dim
        # Head params
        head_hidden_factor: float = 1.0,  # Multiplier for head hidden dim
        head_num_layers: int = 2,
        # Global params
        edge_types: List[EdgeType] = None,
        activation: str = "relu",
        norm: Optional[str] = None,
        dropout: float = 0.1,
        learnable_embedding: bool = False,
        num_nodes: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Gene encoder network
        self.gene_encoder = HeteroGnnPool(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=gene_encoder_num_layers,
            edge_types=edge_types,
            conv_type=gene_encoder_conv_type,
            layer_config=gene_encoder_layer_config,
            activation=activation,
            norm=norm,
            head_num_layers=gene_encoder_head_num_layers,
            learnable_embedding=learnable_embedding,
            num_nodes=num_nodes,
        )

        # Metabolism processor
        self.metabolism_processor = MetabolismProcessor(
            metabolite_dim=hidden_channels,
            hidden_dim=hidden_channels,
            hyperconv_num_layers=metabolism_hyperconv_layers,
            set_net_num_layers=metabolism_setnet_layers,
            use_attention=metabolism_attention,
            use_skip=metabolism_use_skip,  # Pass the skip connection parameter
            dropout=dropout,
        )

        # Combiner with configurable depth
        combiner_hidden = int(hidden_channels * combiner_hidden_factor)
        combiner_layers = []
        combiner_layers.append(nn.Linear(hidden_channels * 2, combiner_hidden))
        combiner_layers.append(nn.LayerNorm(combiner_hidden))
        combiner_layers.append(nn.ReLU())
        combiner_layers.append(nn.Dropout(dropout))

        for _ in range(combiner_num_layers - 1):
            combiner_layers.extend(
                [
                    nn.Linear(combiner_hidden, combiner_hidden),
                    nn.LayerNorm(combiner_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        combiner_layers.append(nn.Linear(combiner_hidden, hidden_channels))
        self.combiner = nn.Sequential(*combiner_layers)

        # Set processors
        # With:
        self.whole_intact_processor = SetTransformer(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            num_heads=set_transformer_heads,
            num_layers=set_transformer_layers,
            dropout=dropout,
        )

        self.perturbed_processor = SetTransformer(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            num_heads=set_transformer_heads,
            num_layers=set_transformer_layers,
            dropout=dropout,
        )

        # Prediction heads with configurable depth
        head_hidden = int(hidden_channels * head_hidden_factor)
        self.fitness_head = self._build_mlp(
            in_dim=hidden_channels,
            hidden_dim=head_hidden,
            out_dim=1,
            num_layers=head_num_layers,
            dropout=dropout,
        )

        self.gene_interaction_head = self._build_mlp(
            in_dim=hidden_channels,
            hidden_dim=head_hidden,
            out_dim=1,
            num_layers=head_num_layers,
            dropout=dropout,
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        """Helper method to build MLP with configurable depth."""
        layers = []

        # Input layer
        layers.extend(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        return nn.Sequential(*layers)

    def _split_embeddings(self, embeddings: torch.Tensor, batch) -> List[tuple]:
        """Split embeddings into whole, intact, and perturbed sets using tensor operations.

        Args:
            embeddings: Tensor of shape (total_nodes, hidden_dim)
            batch: HeteroData object containing perturbation information

        Returns:
            List of tuples (whole_emb, intact_emb, pert_emb) for each batch item
        """
        device = embeddings.device
        batch_size = len(batch["gene"].ptr) - 1

        # Pre-compute batch boundaries
        starts = batch["gene"].ptr[:-1]
        ends = batch["gene"].ptr[1:]

        # Create a batch mask tensor (batch_size x total_nodes)
        batch_masks = torch.zeros(
            (batch_size, embeddings.size(0)), dtype=torch.bool, device=device
        )
        for i, (start, end) in enumerate(zip(starts, ends)):
            batch_masks[i, start:end] = True

        # Handle perturbations
        if hasattr(batch["gene"], "perturbed_genes"):
            # Create masks for perturbed genes
            pert_masks = torch.zeros_like(batch_masks)
            node_id_to_idx_list = []

            # Create node_id to index mappings for each batch
            for i in range(batch_size):
                node_ids = batch["gene"].node_ids[i]
                node_id_to_idx = {
                    nid: idx + starts[i] for idx, nid in enumerate(node_ids)
                }
                node_id_to_idx_list.append(node_id_to_idx)

                # Get perturbation indices for this batch
                pert_genes = batch["gene"].perturbed_genes[i]
                pert_indices = [
                    node_id_to_idx[gene]
                    for gene in pert_genes
                    if gene in node_id_to_idx
                ]

                # Set perturbation mask
                if pert_indices:
                    pert_masks[i, pert_indices] = True
        else:
            # If no perturbations, create empty perturbation masks
            pert_masks = torch.zeros_like(batch_masks)

        # Create intact masks (all genes except perturbed ones)
        intact_masks = batch_masks & ~pert_masks

        # Split embeddings using masks
        embeddings_split = []
        for i in range(batch_size):
            whole_emb = embeddings[batch_masks[i]]
            intact_emb = embeddings[intact_masks[i]]
            pert_emb = embeddings[pert_masks[i]]

            # print(f"\nBatch {i} split details:")
            # print(f"  Total nodes: {len(whole_emb)}")
            # print(f"  Intact nodes: {len(intact_emb)}")
            # print(f"  Perturbed nodes: {len(pert_emb)}")

            if len(pert_emb) > 0:
                pert_indices = torch.where(pert_masks[i])[0] - starts[i]
                # print(f"  Perturbation indices: {pert_indices.tolist()}")

            embeddings_split.append((whole_emb, intact_emb, pert_emb))

        return embeddings_split

    def _process_batch_embeddings(
        self, embeddings_split: List[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process batch embeddings efficiently using batch operations where possible.

        Args:
            embeddings_split: List of tuples (whole_emb, intact_emb, pert_emb) for each batch

        Returns:
            tuple of (z_whole, z_intact, z_pert) tensors
        """
        device = embeddings_split[0][0].device
        batch_size = len(embeddings_split)

        # Pre-allocate output tensors
        z_whole = torch.zeros(batch_size, self.hidden_channels, device=device)
        z_intact = torch.zeros_like(z_whole)
        z_pert = torch.zeros_like(z_whole)

        # Process whole and intact embeddings
        for i, (whole_emb, intact_emb, _) in enumerate(embeddings_split):
            # Reshape for SetTransformer: (num_nodes, input_dim) -> (1, num_nodes, input_dim)
            whole_emb = whole_emb.unsqueeze(0)  # Add batch dimension
            intact_emb = intact_emb.unsqueeze(0)  # Add batch dimension
            
            # Process through SetTransformer
            z_whole[i] = self.whole_intact_processor(whole_emb).squeeze(0)  # Remove batch dimension
            z_intact[i] = self.whole_intact_processor(intact_emb).squeeze(0)  # Remove batch dimension

        # Process perturbed embeddings
        for i, (_, _, pert_emb) in enumerate(embeddings_split):
            if len(pert_emb) > 0:
                # Reshape for SetTransformer: (num_nodes, input_dim) -> (1, num_nodes, input_dim)
                pert_emb = pert_emb.unsqueeze(0)  # Add batch dimension
                z_pert[i] = self.perturbed_processor(pert_emb).squeeze(0)  # Remove batch dimension

        return z_whole, z_intact, z_pert

    def forward(self, batch):
        gene_embeddings = self.gene_encoder(batch)
        metabolism_embeddings = self.metabolism_processor(batch)

        # Validate shapes before combining
        assert gene_embeddings.size(0) == metabolism_embeddings.size(
            0
        ), "Gene and metabolism embeddings must have same batch size"
        assert gene_embeddings.size(1) == metabolism_embeddings.size(
            1
        ), "Gene and metabolism embeddings must have same hidden dimension"

        combined_embeddings = self.combiner(
            torch.cat([gene_embeddings, metabolism_embeddings], dim=-1)
        )
        # print("Combined embeddings shape:", combined_embeddings.shape)

        # Split embeddings for each batch item
        embeddings_split = self._split_embeddings(combined_embeddings, batch)
        # print(f"Number of splits: {len(embeddings_split)}")

        # Process embeddings using optimized batch processing
        z_whole, z_intact, z_pert = self._process_batch_embeddings(embeddings_split)

        # Generate predictions
        whole_fitness = self.fitness_head(z_whole)
        intact_fitness = self.fitness_head(z_intact)
        eps = 1e-8
        fitness_ratio = intact_fitness / (whole_fitness + eps)
        gene_interaction = self.gene_interaction_head(z_pert)

        predictions = torch.stack(
            [fitness_ratio.squeeze(-1), gene_interaction.squeeze(-1)], dim=1
        )

        return predictions, z_pert

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for different components of the model."""
        gene_encoder_params = sum(p.numel() for p in self.gene_encoder.parameters())
        metabolism_params = sum(
            p.numel() for p in self.metabolism_processor.parameters()
        )
        combiner_params = sum(p.numel() for p in self.combiner.parameters())
        whole_intact_params = sum(
            p.numel() for p in self.whole_intact_processor.parameters()
        )
        perturbed_params = sum(p.numel() for p in self.perturbed_processor.parameters())
        fitness_head_params = sum(p.numel() for p in self.fitness_head.parameters())
        gene_int_params = sum(
            p.numel() for p in self.gene_interaction_head.parameters()
        )
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "gene_encoder": gene_encoder_params,
            "metabolism_processor": metabolism_params,
            "combiner": combiner_params,
            "whole_intact_processor": whole_intact_params,
            "perturbed_processor": perturbed_params,
            "fitness_head": fitness_head_params,
            "gene_interaction_head": gene_int_params,
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
    from torchcell.data.neo4j_cell import Unperturbed
    from tqdm import tqdm
    from torchcell.metabolism.yeast_GEM import YeastGEM

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
        incidence_graphs={"metabolism": YeastGEM().reaction_map},
        node_embeddings={"codon_frequency": codon_frequency},
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=Unperturbed(),
    )
    seed = 42
    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=6,
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
        batch_size=6,
        num_workers=6,
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()

    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break

    max_num_nodes = len(dataset.gene_set)
    return batch, max_num_nodes


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


def main(device="cuda"):
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
    batch, max_num_nodes = load_sample_data_batch()

    # Move batch to device
    batch = batch.to(device)

    # Define edge types
    edge_types = [
        ("gene", "physical_interaction", "gene"),
        ("gene", "regulatory_interaction", "gene"),
    ]

    # GIN configuration
    layer_config = {
        "train_eps": True,
        "hidden_multiplier": 2.0,
        "dropout": 0.1,
        "add_self_loops": True,
        "is_skip_connection": True,
        "num_mlp_layers": 3,
        "is_mlp_skip_connection": True,
    }

    # Model configuration
    model = CellLatentPerturbation(
        # Base dimensions
        in_channels=64,
        hidden_channels=64,
        out_channels=2,
        
        # Gene encoder config (unchanged)
        gene_encoder_num_layers=2,
        gene_encoder_conv_type="GIN",
        gene_encoder_layer_config=layer_config,
        gene_encoder_head_num_layers=2,
        
        # Metabolism processor config (unchanged)
        metabolism_num_layers=2,
        metabolism_attention=True,
        metabolism_hyperconv_layers=2,
        metabolism_setnet_layers=2,
        metabolism_use_skip=True,
        
        # Set processor params - updated for both SetTransformers
        set_transformer_heads=4,        # Reduced from 8 to 4 for efficiency
        set_transformer_layers=2,       # Reduced from 3 to 2 for efficiency
        set_net_node_layers=2,         # These parameters won't be used anymore but kept for compatibility
        set_net_set_layers=2,          # These parameters won't be used anymore but kept for compatibility
        
        # Combiner config (unchanged)
        combiner_num_layers=2,
        combiner_hidden_factor=1.0,
        
        # Head config (unchanged)
        head_hidden_factor=1.0,
        head_num_layers=3,
        
        # Global config (unchanged)
        edge_types=edge_types,
        activation="relu",
        norm="layer",
        dropout=0.1,
        num_nodes=max_num_nodes,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print("\nModel Parameter Counts:")
    for component, count in model.num_parameters.items():
        print(f"{component}: {count:,} parameters")

    # Training setup
    loss_type="mse"
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

            # Forward pass
            predictions, z_pert = model(batch)
            loss, loss_components = criterion(predictions, y)

            # Print detailed info every 10 epochs
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print("Predictions shape:", predictions.shape)
                print("Targets shape:", y.shape)
                print("Loss components:", loss_components)

                # Print memory usage if using CUDA
                if device.type == "cuda":
                    print(
                        f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB"
                    )
                    print(
                        f"GPU memory cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB"
                    )

            # Store loss value
            losses.append(loss.item())

            # Backward pass
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
    plt.plot(range(1, len(losses) + 1), losses, "b-", label=f"{loss_type} Training Loss")
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
