import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import os.path as osp
import os
import hydra
from torch_geometric.nn import HeteroConv
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv
from torchcell.models.act import act_register
from typing import Optional, Dict, Any, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch_geometric.data import HeteroData


class GraphMaskedAttention(nn.Module):
    """
    Graph-based masked attention layer that restricts attention to existing edges in the graph.
    """

    def __init__(self, hidden_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert (
            self.head_dim * num_heads == hidden_channels
        ), "hidden_channels must be divisible by num_heads"

        # Projection matrices
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.output_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Apply masked attention where nodes can only attend to their neighbors in the graph.

        Args:
            x: Node features [num_nodes, hidden_channels]
            edge_index: Graph edges [2, num_edges]

        Returns:
            Updated node features [num_nodes, hidden_channels]
        """
        batch_size = 1  # For a single graph
        seq_len = x.size(0)

        # Create attention mask from edge_index
        adj_matrix = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=x.device)
        adj_matrix[edge_index[0], edge_index[1]] = True

        # Define mask_mod for FlexAttention
        def adjacency_mask_mod(b, h, q_idx, kv_idx):
            return adj_matrix[q_idx, kv_idx]

        # Create BlockMask for efficient computation
        block_mask = create_block_mask(
            adjacency_mask_mod,
            B=None,  # Broadcast across batch
            H=None,  # Broadcast across heads
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )

        # Project and reshape for multi-head attention
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Apply FlexAttention with the adjacency mask
        output = flex_attention(q, k, v, block_mask=block_mask)

        # Reshape and project the output
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_channels)
        )
        output = self.output_proj(output)

        # Since batch_size is 1, we can squeeze it out
        return output.squeeze(0)


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


###############################################################################
# Attentional Aggregation Wrapper
###############################################################################
class AttentionalGraphAggregation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )
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


###############################################################################
# PreProcessor: applies a simple MLP with global norm, activation, dropout.
###############################################################################
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
        # Get the activation class type
        act_fn = type(act_register[activation])
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers = []

        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(norm_layer)
        layers.append(act_fn())  # Create new instance
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(norm_layer)
            layers.append(act_fn())  # Create new instance
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)


###############################################################################
# Combiner: MLP to combine two representations.
###############################################################################
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
        layers.append(nn.Linear(hidden_channels * 2, hidden_channels))
        layers.append(get_norm_layer(hidden_channels, norm))
        layers.append(act)
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(get_norm_layer(hidden_channels, norm))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x1, x2], dim=-1))


###############################################################################
# New Model: HeteroCell
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv
from torchcell.models.act import act_register
from torch_scatter import scatter, scatter_softmax
from typing import Optional, Dict, Any, Tuple


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


###############################################################################
# Attentional Aggregation Wrapper
###############################################################################
class AttentionalGraphAggregation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )
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


###############################################################################
# PreProcessor: an MLP to “preprocess” gene embeddings.
###############################################################################
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
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(norm_layer)
        layers.append(act)
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(norm_layer)
            layers.append(act)
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


###############################################################################
# New Model: HeteroCell
###############################################################################
class AttentionConvWrapper(nn.Module):
    def __init__(self, conv: nn.Module, target_dim: int):
        super().__init__()
        self.conv = conv
        # For GATv2Conv-like layers:
        if hasattr(conv, "concat"):
            expected_dim = (
                conv.heads * conv.out_channels if conv.concat else conv.out_channels
            )
        else:
            # For other layers, assume out_channels is the output dimension.
            expected_dim = conv.out_channels
        self.proj = (
            nn.Identity()
            if expected_dim == target_dim
            else nn.Linear(expected_dim, target_dim)
        )

    def forward(self, x, edge_index, **kwargs):
        out = self.conv(x, edge_index, **kwargs)
        return self.proj(out)


class HeteroCell(nn.Module):
    def __init__(
        self,
        gene_num: int,
        reaction_num: int,
        metabolite_num: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.1,
        norm: str = "layer",
        activation: str = "relu",
        gene_encoder_config: Optional[Dict[str, Any]] = None,
        metabolism_config: Optional[Dict[str, Any]] = None,
        prediction_head_config: Optional[Dict[str, Any]] = None,
        gpr_conv_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Learnable embeddings.
        self.gene_embedding = nn.Embedding(gene_num, hidden_channels)
        self.reaction_embedding = nn.Embedding(reaction_num, hidden_channels)
        self.metabolite_embedding = nn.Embedding(metabolite_num, hidden_channels)

        self.preprocessor = PreProcessor(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )

        # Inside the HeteroCell class constructor
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict: Dict[Any, nn.Module] = {}

            # Replace GATv2Conv with GraphMaskedAttention for gene-gene interactions
            gene_att = GraphMaskedAttention(
                hidden_channels=hidden_channels,
                num_heads=gene_encoder_config.get("heads", 4),
                dropout=dropout,
            )

            # Create wrapper to make it compatible with HeteroConv
            class GraphAttentionWrapper(nn.Module):
                def __init__(self, att_module):
                    super().__init__()
                    self.att_module = att_module
                    self.out_channels = att_module.hidden_channels

                def forward(self, x, edge_index, **kwargs):
                    return self.att_module(x, edge_index)

            # Use the same attention layer for both physical and regulatory interactions
            wrapped_gene_att = GraphAttentionWrapper(gene_att)
            conv_dict[("gene", "physical_interaction", "gene")] = wrapped_gene_att
            conv_dict[("gene", "regulatory_interaction", "gene")] = wrapped_gene_att

            # The rest of the model remains the same
            gpr_config = gpr_conv_config if gpr_conv_config is not None else {}
            conv_dict[("gene", "gpr", "reaction")] = GATv2Conv(
                hidden_channels,
                hidden_channels,
                heads=gpr_config.get("heads", 1),
                concat=gpr_config.get("concat", False),
                add_self_loops=gpr_config.get("add_self_loops", False),
            )

            conv_dict[("metabolite", "reaction", "metabolite")] = StoichHypergraphConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                is_stoich_gated=metabolism_config.get("is_stoich_gated", False),
                use_attention=metabolism_config.get("use_attention", True),
                heads=metabolism_config.get("heads", 1),
                concat=metabolism_config.get("concat", True),
                dropout=dropout,
                bias=True,
            )

            # Wrap each conv so its output is projected to hidden_channels
            for key, conv in conv_dict.items():
                if key[0] == "gene" and key[2] == "gene":
                    # Don't wrap our GraphMaskedAttention as it already outputs correct dimensions
                    continue
                conv_dict[key] = AttentionConvWrapper(conv, hidden_channels)

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Global aggregator for intact graphs.
        self.global_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )
        # Dedicated aggregator for perturbed nodes.
        self.perturbed_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

        # Build separate prediction heads:
        # BUG - See buggy model
        # pred_config = prediction_head_config or {}
        # self.fitness_head = self._build_prediction_head(
        #     in_channels=hidden_channels,
        #     hidden_channels=pred_config.get("hidden_channels", hidden_channels),
        #     out_channels=1,
        #     num_layers=pred_config.get("head_num_layers", 1),
        #     dropout=pred_config.get("dropout", dropout),
        #     activation=pred_config.get("activation", activation),
        #     residual=pred_config.get("residual", True),
        #     norm=pred_config.get("head_norm", norm),
        # )
        # self.gene_interaction_head = self._build_prediction_head(
        #     in_channels=hidden_channels,
        #     hidden_channels=pred_config.get("hidden_channels", hidden_channels),
        #     out_channels=1,
        #     num_layers=pred_config.get("head_num_layers", 1),
        #     dropout=pred_config.get("dropout", dropout),
        #     activation=pred_config.get("activation", activation),
        #     residual=pred_config.get("residual", True),
        #     norm=pred_config.get("head_norm", norm),
        # )
        pred_config = prediction_head_config or {}
        self.prediction_head = self._build_prediction_head(
            in_channels=hidden_channels,
            hidden_channels=pred_config.get("hidden_channels", hidden_channels),
            out_channels=2,  # two outputs: fitness and gene interaction
            num_layers=pred_config.get("head_num_layers", 1),
            dropout=pred_config.get("dropout", dropout),
            activation=pred_config.get("activation", activation),
            residual=pred_config.get("residual", True),
            norm=pred_config.get("head_norm", norm),
        )

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
        act_fn = type(act_register[activation])
        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                if norm is not None:
                    layers.append(get_norm_layer(dims[i + 1], norm))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward_single(self, data: HeteroData) -> torch.Tensor:
        # Use the model's device for consistency.
        device = self.gene_embedding.weight.device

        gene_bs = data["gene"].num_nodes
        gene_idx = (
            torch.arange(gene_bs, device=device) % self.gene_embedding.num_embeddings
        )

        reaction_bs = data["reaction"].num_nodes
        reaction_idx = (
            torch.arange(reaction_bs, device=device)
            % self.reaction_embedding.num_embeddings
        )

        metabolite_bs = data["metabolite"].num_nodes
        metabolite_idx = (
            torch.arange(metabolite_bs, device=device)
            % self.metabolite_embedding.num_embeddings
        )

        x_dict = {
            "gene": self.preprocessor(self.gene_embedding(gene_idx)),
            "reaction": self.reaction_embedding(reaction_idx),
            "metabolite": self.metabolite_embedding(metabolite_idx),
        }
        # Ensure edge_index_dict tensors are on device.
        edge_index_dict = {}
        for key, edge in data.edge_index_dict.items():
            if isinstance(edge, torch.Tensor):
                edge_index_dict[key] = edge.to(device)
            else:
                edge_index_dict[key] = edge

        extra_kwargs = {}
        met_edge = ("metabolite", "reaction", "metabolite")
        if met_edge in data.edge_index_dict:
            stoich = data[met_edge].stoichiometry.to(device)
            extra_kwargs["stoich_dict"] = {met_edge: stoich}
            if hasattr(data[met_edge], "reaction_ids"):
                reaction_ids = data[met_edge].reaction_ids.to(device)
                extra_kwargs["edge_attr_dict"] = {
                    met_edge: self.reaction_embedding(
                        reaction_ids % self.reaction_embedding.num_embeddings
                    )
                }
            extra_kwargs["num_edges_dict"] = {met_edge: data[met_edge].num_edges}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, **extra_kwargs)
        return x_dict["gene"]

    # BUG All mean or random prediction.
    # def forward(
    #     self, cell_graph: HeteroData, batch: HeteroData
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    #     # Process the reference (wildtype) graph.
    #     z_w = self.forward_single(cell_graph)
    #     z_w = self.global_aggregator(
    #         z_w,
    #         index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
    #         dim_size=1,
    #     )
    #     # Process the intact (perturbed) batch.
    #     z_i = self.forward_single(batch)
    #     z_i = self.global_aggregator(z_i, index=batch["gene"].batch)

    #     # Extract perturbed node indices from batch gene data.
    #     batch_size = len(batch["gene"].ptr) - 1
    #     pert_indices = []
    #     for i in range(batch_size):
    #         start = batch["gene"].x_pert_ptr[i]
    #         end = batch["gene"].x_pert_ptr[i + 1]
    #         pert_indices.append(batch["gene"].cell_graph_idx_pert[start:end])

    #     # Expand reference representation to match batch size.
    #     z_w_exp = z_w.expand(batch_size, -1)
    #     z_p_list = []
    #     batch_idx_list = []
    #     for i, indices in enumerate(pert_indices):
    #         z_p_list.append(z_w_exp[i : i + 1].expand(len(indices), -1))
    #         batch_idx_list.append(torch.full((len(indices),), i, device=z_w.device))
    #     z_p = torch.cat(z_p_list, dim=0)
    #     batch_idx = torch.cat(batch_idx_list, dim=0)
    #     z_p = self.perturbed_aggregator(z_p, index=batch_idx, dim_size=batch_size)

    #     # Predict fitness from intact representation and gene interaction from perturbed nodes.
    #     fitness = self.fitness_head(z_i)  # Shape: [batch_size, 1]
    #     gene_interaction = self.gene_interaction_head(z_p)  # Shape: [batch_size, 1]

    #     predictions = torch.cat([fitness, gene_interaction], dim=1)

    #     return predictions, {
    #         "z_w": z_w,
    #         "z_i": z_i,
    #         "z_p": z_p,
    #         "fitness": fitness,
    #         "gene_interaction": gene_interaction,
    #     }

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Process the reference (wildtype) graph.
        z_w = self.forward_single(cell_graph)
        z_w = self.global_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )
        # Process the intact (perturbed) batch.
        z_i = self.forward_single(batch)
        z_i = self.global_aggregator(z_i, index=batch["gene"].batch)
        # Compute the difference: use broadcasting to match batch size.
        batch_size: int = z_i.size(0)
        z_w_exp: torch.Tensor = z_w.expand(batch_size, -1)
        z_p: torch.Tensor = z_w_exp - z_i
        # Single prediction head that outputs 2 dimensions (fitness and gene interaction)
        predictions: torch.Tensor = self.prediction_head(z_p)  # shape: [batch_size, 2]
        fitness: torch.Tensor = predictions[:, 0:1]
        gene_interaction: torch.Tensor = predictions[:, 1:2]

        return predictions, {
            "z_w": z_w,
            "z_i": z_i,
            "z_p": z_p,
            "fitness": fitness,
            "gene_interaction": gene_interaction,
        }

    @property
    def num_parameters(self) -> Dict[str, int]:
        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "gene_embedding": count_params(self.gene_embedding),
            "reaction_embedding": count_params(self.reaction_embedding),
            "metabolite_embedding": count_params(self.metabolite_embedding),
            "preprocessor": count_params(self.preprocessor),
            "convs": count_params(self.convs),
            "global_aggregator": count_params(self.global_aggregator),
            "perturbed_aggregator": count_params(self.perturbed_aggregator),
            "prediction_head": count_params(self.prediction_head),  # Changed this line
        }
        counts["total"] = sum(counts.values())
        return counts


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
        batch_size=32,
        random_seed=seed,
        num_workers=8,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=32,
        num_workers=8,
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
    config_name="hetero_cell",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.timestamp import timestamp

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch()
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Initialize model (parameters unchanged)
    model = HeteroCell(
        gene_num=cfg.model.gene_num,
        reaction_num=cfg.model.reaction_num,
        metabolite_num=cfg.model.metabolite_num,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=cfg.model.gene_encoder_config,
        metabolism_config=cfg.model.metabolism_config,
        prediction_head_config=cfg.model.prediction_head_config,
        gpr_conv_config=cfg.model.gpr_conv_config,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))

    # Training setup
    total_non_nan = (~batch["gene"].fitness.isnan()).sum() + (
        ~batch["gene"].gene_interaction.isnan()
    ).sum()
    minus_fit_count = 1 - (~batch["gene"].fitness.isnan()).sum()
    minus_gi_count = 1 - (~batch["gene"].gene_interaction.isnan()).sum()
    weights = torch.tensor(
        [minus_fit_count / total_non_nan, minus_gi_count / total_non_nan]
    ).to(device)

    criterion = ICLoss(
        lambda_dist=cfg.regression_task.lambda_dist,
        lambda_supcr=cfg.regression_task.lambda_supcr,
        weights=weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    # Training targets
    y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

    # Training loop
    model.train()
    print("\nStarting training:")
    losses = []
    num_epochs = cfg.trainer.max_epochs

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Forward pass now expects cell_graph and batch
            predictions, representations = model(cell_graph, batch)
            loss, loss_components = criterion(
                predictions, y, representations["z_p"], representations["z_i"]
            )

            # Logging remains the same
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

    # Plotting and evaluation code remains the same, just update the file names
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses) + 1), losses, "b-", label="ICLoss Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(
        f"Training Loss Over Time: λ_dist={cfg.regression_task.lambda_dist}, "
        f"λ_supcr={cfg.regression_task.lambda_supcr}, "
        f"wd={cfg.regression_task.optimizer.weight_decay}"
    )
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        osp.join(ASSET_IMAGES_DIR, f"hetero_cell_training_loss_{timestamp()}.png")
    )
    plt.close()

    # Update correlation plot
    correlation_save_path = osp.join(
        ASSET_IMAGES_DIR, f"hetero_cell_correlation_plots_{timestamp()}.png"
    )
    plot_correlations(
        predictions.cpu(),
        y.cpu(),
        correlation_save_path,
        lambda_info=f"λ_dist={cfg.regression_task.lambda_dist}, "
        f"λ_supcr={cfg.regression_task.lambda_supcr}",
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
