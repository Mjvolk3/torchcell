# torchcell/models/hetero_cell_isab_split
# [[torchcell.models.hetero_cell_isab_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_isab_split
# Test file: tests/torchcell/models/test_hetero_cell_isab_split.py


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
from torch_geometric.utils import sort_edge_index
from torchcell.nn.aggr.set_transformer import SetTransformerAggregation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index
from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv
from torchcell.models.act import act_register
from typing import Optional, Dict, Any, Tuple
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


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


class SortedSetTransformerAggregation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 2,
        num_decoder_blocks: int = 1,
        heads: int = 4,
        layer_norm: bool = True,
        use_isab: bool = True,
        num_induced_points: int = 32,
    ):
        super().__init__()
        self.aggregator = SetTransformerAggregation(
            channels=in_channels,
            num_seed_points=num_seed_points,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            heads=heads,
            concat=True,
            layer_norm=layer_norm,
            dropout=dropout,
            use_isab=use_isab,
            num_induced_points=num_induced_points,
        )

        # Calculate expected output size
        expected_out_channels = (
            in_channels * num_seed_points if num_seed_points > 1 else in_channels
        )
        self.proj = (
            nn.Linear(expected_out_channels, out_channels)
            if expected_out_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None
    ) -> torch.Tensor:
        # Ensure indices are sorted (required by SetTransformerAggregation)
        if not torch.all(index[:-1] <= index[1:]):
            perm = torch.argsort(index)
            x = x[perm]
            index = index[perm]

        # Run through set transformer
        out = self.aggregator(x, index, dim_size=dim_size)

        # Project if needed
        return self.proj(out)


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
    def __init__(
        self,
        conv: nn.Module,
        target_dim: int,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = conv
        # For GATv2Conv-like layers:
        if hasattr(conv, "concat"):
            expected_dim = (
                conv.heads * conv.out_channels if conv.concat else conv.out_channels
            )
        else:
            expected_dim = conv.out_channels
        self.proj = (
            nn.Identity()
            if expected_dim == target_dim
            else nn.Linear(expected_dim, target_dim)
        )

        # Use PyTorch Geometric's graph normalization layers
        if norm is not None:
            if norm == "batch":
                self.norm = BatchNorm(target_dim)
            elif norm == "layer":
                self.norm = LayerNorm(target_dim)
            elif norm == "graph":
                self.norm = GraphNorm(target_dim)
            elif norm == "instance":
                self.norm = InstanceNorm(target_dim)
            elif norm == "pair":
                self.norm = PairNorm(scale=1.0)
            elif norm == "mean":
                self.norm = MeanSubtractionNorm()
            else:
                self.norm = None
        else:
            self.norm = None

        # Get the activation function from the register without instantiating again
        self.act = act_register[activation] if activation is not None else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, edge_index, **kwargs):
        out = self.conv(x, edge_index, **kwargs)
        out = self.proj(out)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


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

        # Learnable embeddings
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

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict: Dict[Any, nn.Module] = {}

            gene_conv = GATv2Conv(
                hidden_channels,
                hidden_channels // gene_encoder_config.get("heads", 1),
                heads=gene_encoder_config.get("heads", 1),
                concat=gene_encoder_config.get("concat", True),
                add_self_loops=True,
            )
            conv_dict[("gene", "physical_interaction", "gene")] = gene_conv
            conv_dict[("gene", "regulatory_interaction", "gene")] = gene_conv

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

            # Wrap each conv so its output is projected to hidden_channels.
            for key, conv in conv_dict.items():
                conv_dict[key] = AttentionConvWrapper(
                    conv,
                    hidden_channels,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # New implementation with Set Transformer:
        self.global_aggregator = SortedSetTransformerAggregation(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=8,
            layer_norm=True,
            use_isab=True,  # Use ISAB for large graphs
            num_induced_points=128,
        )

        # Build separate prediction heads:
        pred_config = prediction_head_config or {}
        self.prediction_head = self._build_prediction_head(
            in_channels=hidden_channels,
            hidden_channels=pred_config.get("hidden_channels", hidden_channels),
            out_channels=2,  # two outputs: fitness and gene interaction
            num_layers=pred_config.get("head_num_layers", 1),
            dropout=pred_config.get("dropout", dropout),
            activation=pred_config.get("activation", activation),
            # residual=pred_config.get("residual", True),
            norm=pred_config.get("head_norm", norm),
        )
        self.pert_gene_sab = SortedSetTransformerAggregation(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=8,
            layer_norm=True,
            use_isab=False,  # Use SAB for smaller sets
            num_induced_points=None,  # Not needed for SAB
        )

        self.gi_projection = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels),
            (
                nn.LayerNorm(hidden_channels)
                if norm == "layer"
                else nn.BatchNorm1d(hidden_channels)
            ),
            act_register[activation],
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def _build_prediction_head(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        activation: str,
        # residual: bool,
        norm: Optional[str] = None,
    ) -> nn.Module:
        if num_layers == 0:
            return nn.Identity()
        act = act_register[activation]
        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                if norm is not None:
                    layers.append(get_norm_layer(dims[i + 1], norm))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward_single(self, data: HeteroData | Batch) -> torch.Tensor:
        device = self.gene_embedding.weight.device

        is_batch = isinstance(data, Batch) or hasattr(data["gene"], "batch")
        if is_batch:
            gene_data = data["gene"]
            reaction_data = data["reaction"]
            metabolite_data = data["metabolite"]

            batch_size = len(data["gene"].ptr) - 1

            x_gene_exp = self.gene_embedding.weight.expand(batch_size, -1, -1)
            x_gene_comb = x_gene_exp.reshape(-1, x_gene_exp.size(-1))
            x_gene = x_gene_comb[~gene_data.pert_mask]
            x_gene = self.preprocessor(x_gene)

            x_reaction_exp = self.reaction_embedding.weight.expand(batch_size, -1, -1)
            x_reaction_comb = x_reaction_exp.reshape(-1, x_reaction_exp.size(-1))
            x_reaction = x_reaction_comb[~reaction_data.pert_mask]

            x_metabolite_exp = self.metabolite_embedding.weight.expand(
                batch_size, -1, -1
            )
            x_metabolite_comb = x_metabolite_exp.reshape(-1, x_metabolite_exp.size(-1))
            x_metabolite = x_metabolite_comb[~metabolite_data.pert_mask]

            x_dict = {
                "gene": x_gene,
                "reaction": x_reaction,
                "metabolite": x_metabolite,
            }
        else:
            # For a single (cell) graph, use all node embeddings directly.
            x_gene = data["gene"]
            reaction_data = data["reaction"]
            metabolite_data = data["metabolite"]

            gene_idx = torch.arange(x_gene.num_nodes, device=device)
            reaction_idx = torch.arange(reaction_data.num_nodes, device=device)
            metabolite_idx = torch.arange(metabolite_data.num_nodes, device=device)

            x_dict = {
                "gene": self.preprocessor(self.gene_embedding(gene_idx)),
                "reaction": self.reaction_embedding(reaction_idx),
                "metabolite": self.metabolite_embedding(metabolite_idx),
            }

        # Move all edge indices to the correct device.
        edge_index_dict = {}
        for key, edge in data.edge_index_dict.items():
            edge_index_dict[key] = edge.to(device) if torch.is_tensor(edge) else edge

        # Prepare extra edge features for the metabolite hyperedge, if present.
        extra_kwargs: dict[str, Any] = {}
        met_edge = ("metabolite", "reaction", "metabolite")
        if met_edge in data.edge_index_dict:
            met_edge_data = data[met_edge]
            if hasattr(met_edge_data, "stoichiometry"):
                extra_kwargs["stoich_dict"] = {
                    met_edge: met_edge_data.stoichiometry.to(device)
                }
            if hasattr(met_edge_data, "num_edges"):
                extra_kwargs["num_edges_dict"] = {met_edge: met_edge_data.num_edges}

        # Apply each convolution layer sequentially.
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, **extra_kwargs)

        return x_dict["gene"]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward method with simplified approach: direct SAB over WT embeddings for gene interactions.
        """
        device = self.gene_embedding.weight.device

        # Process reference graph with GNN and ISAB
        z_w_raw = self.forward_single(cell_graph)  # Raw gene embeddings after GNN
        z_w = self.global_aggregator(
            z_w_raw,
            index=torch.zeros(z_w_raw.size(0), device=device, dtype=torch.long),
            dim_size=1,
        )

        # Process perturbed batch with GNN and ISAB
        z_i_raw = self.forward_single(batch)  # Raw gene embeddings after GNN
        z_i = self.global_aggregator(z_i_raw, index=batch["gene"].batch)

        # Compute the difference embedding for fitness prediction
        batch_size: int = z_i.size(0)
        z_w_exp: torch.Tensor = z_w.expand(batch_size, -1)
        z_p: torch.Tensor = z_w_exp - z_i

        # Get fitness prediction using z_p
        fitness_head = nn.Sequential(
            *list(self.prediction_head.children())[:-1]
            + [nn.Linear(self.hidden_channels, 1)]
        )
        fitness = fitness_head(z_p)

        # Gene Interaction with simplified processing
        gene_interaction = None

        # Process perturbed genes using the pert_mask
        if hasattr(batch["gene"], "pert_mask") and batch["gene"].pert_mask.any():
            # Create collections for perturbed embeddings
            pert_gene_embeds_list = []
            pert_gene_batch_idx_list = []

            # Work with each batch item's mask
            for batch_idx in range(batch_size):
                # Get the start and end indices for this batch item
                start_idx = batch["gene"].ptr[batch_idx]
                end_idx = batch["gene"].ptr[batch_idx + 1]

                # Get the perturbation mask for this batch item
                batch_pert_mask = batch["gene"].pert_mask[start_idx:end_idx]

                # Find perturbed gene indices
                pert_indices = torch.where(batch_pert_mask)[0]

                # Get the processed embeddings for these genes from z_w_raw
                for idx in pert_indices:
                    if idx < z_w_raw.shape[0]:
                        pert_gene_embeds_list.append(z_w_raw[idx])
                        pert_gene_batch_idx_list.append(batch_idx)

            # If we found any perturbed genes
            if len(pert_gene_embeds_list) > 0:
                # Convert lists to tensors
                pert_gene_embeds = torch.stack(pert_gene_embeds_list)
                pert_gene_batch_idx = torch.tensor(
                    pert_gene_batch_idx_list, device=device
                )

                # Process directly with SAB - this output is directly used for gene interaction
                # without concatenating with z_p
                gene_interaction_embeddings = self.pert_gene_sab(
                    pert_gene_embeds, pert_gene_batch_idx
                )
                
                # Direct projection to gene interaction score
                gene_interaction = nn.Linear(self.hidden_channels, 1).to(device)(gene_interaction_embeddings)
            else:
                # No perturbed genes found - fallback to a default prediction
                print("Warning: No perturbed genes identified from mask")
                gene_interaction = torch.zeros(batch_size, 1, device=device)
        else:
            # No perturbation mask available
            print("Warning: No perturbation mask available")
            gene_interaction = torch.zeros(batch_size, 1, device=device)

        # Combine predictions
        predictions = torch.cat([fitness, gene_interaction], dim=1)

        return predictions, {
            "z_w": z_w,
            "z_i": z_i,
            "z_p": z_p,
            "fitness": fitness,
            "gene_interaction": gene_interaction,
            "z_w_raw": z_w_raw,
            "z_i_raw": z_i_raw,
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
    # selected_node_embeddings = ["codon_frequency"]
    selected_node_embeddings = ["empty"]
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
        num_workers=4,
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=32,
        num_workers=4,
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
    predictions,
    true_values,
    save_path,
    lambda_info="",
    weight_decay="",
    fixed_axes=None,
    epoch=None,
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
    suptitle = (
        f"Epoch {epoch}: {lambda_info}, wd={weight_decay}"
        if epoch is not None
        else f"{lambda_info}, wd={weight_decay}"
    )
    fig.suptitle(suptitle, fontsize=12)

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

    # Set fixed axes for fitness plot if provided
    if fixed_axes and "fitness" in fixed_axes:
        ax1.set_xlim(fixed_axes["fitness"][0])
        ax1.set_ylim(fixed_axes["fitness"][1])
    else:
        # Determine axes limits for fitness plot
        min_val = min(min(x_fitness), min(y_fitness))
        max_val = max(max(x_fitness), max(y_fitness))
        # Add 10% padding
        range_val = max_val - min_val
        min_val -= range_val * 0.1
        max_val += range_val * 0.1
        ax1.set_xlim([min_val, max_val])
        ax1.set_ylim([min_val, max_val])

        if fixed_axes is None:
            fixed_axes = {}
        fixed_axes["fitness"] = ([min_val, max_val], [min_val, max_val])

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

    # Set fixed axes for gene interaction plot if provided
    if fixed_axes and "gene_interaction" in fixed_axes:
        ax2.set_xlim(fixed_axes["gene_interaction"][0])
        ax2.set_ylim(fixed_axes["gene_interaction"][1])
    else:
        # Determine axes limits for gene interaction plot
        min_val = min(min(x_gi), min(y_gi))
        max_val = max(max(x_gi), max(y_gi))
        # Add 10% padding
        range_val = max_val - min_val
        min_val -= range_val * 0.1
        max_val += range_val * 0.1
        ax2.set_xlim([min_val, max_val])
        ax2.set_ylim([min_val, max_val])

        if fixed_axes is None:
            fixed_axes = {}
        fixed_axes["gene_interaction"] = ([min_val, max_val], [min_val, max_val])

    # Add diagonal line for gene interactions
    min_val = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

    return fixed_axes


def plot_embeddings(
    z_w,
    z_i,
    z_p,
    batch_size,
    save_dir="./003-fit-int/hetero_cell_isab/embedding_plots",
    epoch=None,
    fixed_axes=None,
):
    """
    Plot embeddings for visualization and debugging with fixed axes for consistent GIF creation.

    Args:
        z_w: Wildtype (reference) embedding tensor [1, hidden_dim]
        z_i: Intact (perturbed) embedding tensor [batch_size, hidden_dim]
        z_p: Perturbed difference embedding tensor [batch_size, hidden_dim]
        batch_size: Number of samples in the batch
        save_dir: Directory to save plots
        epoch: Current epoch number (optional)
        fixed_axes: Dictionary containing fixed min/max values for axes if provided
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from torchcell.timestamp import timestamp

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    current_timestamp = timestamp()

    # Add epoch to filename if provided
    epoch_str = f"_epoch{epoch:03d}" if epoch is not None else ""

    # Determine fixed axes if not provided
    if fixed_axes is None:
        # Calculate different min/max for z_i and z_p
        z_w_np = z_w.detach().cpu().numpy()
        z_i_np = z_i.detach().cpu().numpy()
        z_p_np = z_p.detach().cpu().numpy()

        fixed_axes = {
            "value_min": min(np.min(z_w_np), np.min(z_i_np), np.min(z_p_np)),
            "value_max": max(np.max(z_w_np), np.max(z_i_np), np.max(z_p_np)),
            "dim_max": z_w.shape[1] - 1,
            "z_i_min": np.min(z_i_np),
            "z_i_max": np.max(z_i_np),
            "z_p_min": np.min(z_p_np),
            "z_p_max": np.max(z_p_np),
        }

    # Convert tensors to numpy arrays
    z_w_np = z_w.detach().cpu().numpy()
    z_i_np = z_i.detach().cpu().numpy()
    z_p_np = z_p.detach().cpu().numpy()

    # Plot 1: First sample comparison
    plt.figure(figsize=(10, 6))

    # Important: Plot z_w first so it's behind the other lines
    plt.plot(z_w_np[0], "b-", label="Reference (z_w)", linewidth=2, alpha=0.6)
    # Then plot z_i and z_p on top
    plt.plot(z_i_np[0], "r-", label="Intact (z_i)", linewidth=2, alpha=0.8)
    plt.plot(z_p_np[0], "g-", label="Difference (z_p)", linewidth=2, alpha=0.8)

    # Set fixed axes with 5% padding to keep lines in frame
    pad = 0.05 * (fixed_axes["value_max"] - fixed_axes["value_min"])
    plt.xlim(0, fixed_axes["dim_max"])
    plt.ylim(fixed_axes["value_min"] - pad, fixed_axes["value_max"] + pad)

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.title(
        f"Embedding Comparison - First Sample - Epoch {epoch if epoch is not None else 0}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/embedding_first_sample{epoch_str}.png", dpi=300)
    plt.close()

    # Plot 2: All samples in batch
    plt.figure(figsize=(12, 8))

    # Plot reference embedding first so it's behind
    plt.plot(z_w_np[0], "b-", label="Reference (z_w)", linewidth=2, alpha=0.6)

    # Then plot ALL samples' intact and perturbed embeddings
    for i in range(batch_size):  # No limit - show all samples
        # Use alpha to make multiple lines distinguishable
        alpha = 0.5 if batch_size > 1 else 0.8
        linewidth = 0.8  # Thinner lines for better visualization

        # Plot intact embedding with sample index in the label
        plt.plot(
            z_i_np[i],
            "r-",
            alpha=alpha,
            label=f"Intact (z_i) - Sample {i}" if i == 0 else None,
            linewidth=linewidth,
        )

        # Plot perturbed embedding
        plt.plot(
            z_p_np[i],
            "g-",
            alpha=alpha,
            label=f"Difference (z_p) - Sample {i}" if i == 0 else None,
            linewidth=linewidth,
        )

    # Set fixed axes with padding
    pad = 0.05 * (fixed_axes["value_max"] - fixed_axes["value_min"])
    plt.xlim(0, fixed_axes["dim_max"])
    plt.ylim(fixed_axes["value_min"] - pad, fixed_axes["value_max"] + pad)

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.title(
        f"Embedding Comparison - All Samples - Epoch {epoch if epoch is not None else 0}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/embedding_all_samples{epoch_str}.png", dpi=300)
    plt.close()

    # Plot 3: Heatmap of all samples with separate color scales
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use different fixed colormap scales for each heatmap
    # For intact embeddings (z_i)
    im1 = axes[0].imshow(
        z_i_np,
        aspect="auto",
        cmap="viridis",
        vmin=fixed_axes["z_i_min"],
        vmax=fixed_axes["z_i_max"],
    )
    axes[0].set_title("Intact Embeddings (z_i)")
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_ylabel("Sample")
    fig.colorbar(im1, ax=axes[0])

    # For perturbed difference embeddings (z_p)
    im2 = axes[1].imshow(
        z_p_np,
        aspect="auto",
        cmap="viridis",
        vmin=fixed_axes["z_p_min"],
        vmax=fixed_axes["z_p_max"],
    )
    axes[1].set_title("Difference Embeddings (z_p)")
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].set_ylabel("Sample")
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle(f"Embedding Heatmaps - Epoch {epoch if epoch is not None else 0}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/embedding_heatmap{epoch_str}.png", dpi=300)
    plt.close()

    # Plot 4: Distribution of values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Set fixed bins for histograms
    padding = 0.05 * (fixed_axes["value_max"] - fixed_axes["value_min"])
    bin_range = (fixed_axes["value_min"] - padding, fixed_axes["value_max"] + padding)
    bins = np.linspace(bin_range[0], bin_range[1], 50)

    # Reference embedding distribution
    axes[0].hist(z_w_np.flatten(), bins=bins, alpha=0.7, color="blue")
    axes[0].set_title("Reference (z_w) Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(bin_range)

    # Intact embedding distribution
    axes[1].hist(z_i_np.flatten(), bins=bins, alpha=0.7, color="red")
    axes[1].set_title("Intact (z_i) Distribution")
    axes[1].set_xlabel("Value")
    axes[1].set_xlim(bin_range)

    # Perturbed difference embedding distribution
    axes[2].hist(z_p_np.flatten(), bins=bins, alpha=0.7, color="green")
    axes[2].set_title("Difference (z_p) Distribution")
    axes[2].set_xlabel("Value")
    axes[2].set_xlim(bin_range)

    plt.suptitle(
        f"Embedding Value Distributions - Epoch {epoch if epoch is not None else 0}"
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/embedding_distribution{epoch_str}.png", dpi=300)
    plt.close()

    return fixed_axes  # Return the fixed axes for reuse in subsequent epochs


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/003-fit-int/conf"),
    config_name="hetero_cell_isab_split",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.timestamp import timestamp
    import numpy as np

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
    fit_nan_count = batch["gene"].fitness.isnan().sum()
    gi_nan_count = batch["gene"].gene_interaction.isnan().sum()
    total_samples = len(batch["gene"].fitness) * 2
    weights = torch.tensor(
        [1 - (gi_nan_count / total_samples), 1 - (fit_nan_count / total_samples)]
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

    # Initialize fixed axes variables for consistent plots
    embedding_fixed_axes = None
    correlation_fixed_axes = None

    # Setup directories for plots
    embeddings_dir = osp.join(ASSET_IMAGES_DIR, "heter_cell_isab_split_embedding_plots")
    os.makedirs(embeddings_dir, exist_ok=True)

    correlation_dir = osp.join(
        ASSET_IMAGES_DIR, "heter_cell_isab_split_correlation_plots"
    )
    os.makedirs(correlation_dir, exist_ok=True)

    # Training loop
    model.train()
    print("\nStarting training:")
    losses = []
    num_epochs = cfg.trainer.max_epochs

    # First, compute the fixed axes by doing a forward pass
    with torch.no_grad():
        predictions, representations = model(cell_graph, batch)

        # Extract separate embedding data for different plots
        z_w_np = representations["z_w"].detach().cpu().numpy()
        z_i_np = representations["z_i"].detach().cpu().numpy()
        z_p_np = representations["z_p"].detach().cpu().numpy()

        # Initialize embedding fixed axes with separate color scales for z_i and z_p
        embedding_fixed_axes = {
            "value_min": min(np.min(z_w_np), np.min(z_i_np), np.min(z_p_np)),
            "value_max": max(np.max(z_w_np), np.max(z_i_np), np.max(z_p_np)),
            "dim_max": representations["z_w"].shape[1] - 1,
            "z_i_min": np.min(z_i_np),
            "z_i_max": np.max(z_i_np),
            "z_p_min": np.min(z_p_np),
            "z_p_max": np.max(z_p_np),
        }

        # Initialize correlation fixed axes with epoch 0
        init_epoch = 0  # Add this line
        correlation_save_path = osp.join(
            correlation_dir, f"correlation_plots_epoch{init_epoch:03d}.png"
        )
        correlation_fixed_axes = plot_correlations(
            predictions.cpu(),
            y.cpu(),
            correlation_save_path,
            lambda_info=f"λ_dist={cfg.regression_task.lambda_dist}, "
            f"λ_supcr={cfg.regression_task.lambda_supcr}",
            weight_decay=cfg.regression_task.optimizer.weight_decay,
            fixed_axes=None,  # This will compute and return the axes
            epoch=init_epoch,  # Add this line
        )

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass now expects cell_graph and batch
            predictions, representations = model(cell_graph, batch)
            loss, loss_components = criterion(
                predictions, y, representations["z_p"], representations["z_i"]
            )

            # Logging and visualization every 10 epochs (or whatever interval you prefer)
            if epoch % 10 == 0 or epoch == num_epochs - 1:  # Also plot on last epoch
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print("Loss components:", loss_components)

                # Plot embeddings at this epoch
                try:
                    # Use the fixed axes for embeddings
                    embedding_fixed_axes = plot_embeddings(
                        representations["z_w"].expand(predictions.size(0), -1),
                        representations["z_i"],
                        representations["z_p"],
                        batch_size=predictions.size(0),
                        save_dir=embeddings_dir,  # Use the correct directory
                        epoch=epoch,
                        fixed_axes=embedding_fixed_axes,
                    )

                    # Create correlation plots
                    correlation_save_path = osp.join(
                        correlation_dir, f"correlation_plots_epoch{epoch:03d}.png"
                    )
                    plot_correlations(
                        predictions.cpu(),
                        y.cpu(),
                        correlation_save_path,
                        lambda_info=f"λ_dist={cfg.regression_task.lambda_dist}, "
                        f"λ_supcr={cfg.regression_task.lambda_supcr}",
                        weight_decay=cfg.regression_task.optimizer.weight_decay,
                        fixed_axes=correlation_fixed_axes,
                    )
                except Exception as e:
                    print(f"Warning: Could not generate plots: {e}")
                    import traceback

                    traceback.print_exc()  # Print full traceback for debugging

                if device.type == "cuda":
                    print(
                        f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB"
                    )
                    print(
                        f"GPU memory reserved: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB"
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

    # Final loss plot
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
        osp.join(ASSET_IMAGES_DIR, f"hetero_cell_isab_training_loss_{timestamp()}.png")
    )
    plt.close()

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
