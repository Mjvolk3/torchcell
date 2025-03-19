# torchcell/models/hetero_cell_bipartite
# [[torchcell.models.hetero_cell_bipartite]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/hetero_cell_bipartite
# Test file: tests/torchcell/models/test_hetero_cell_bipartite.py

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
from torch_geometric.data import Batch
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, BatchNorm, LayerNorm
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from typing import Optional, Dict, Any, Tuple


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


def get_norm_layer(channels: int, norm: str) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(channels)
    elif norm == "batch":
        return nn.BatchNorm1d(channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


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
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()
        norm_layer = get_norm_layer(hidden_channels, norm)
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(norm_layer)
        layers.append(self.act)
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(norm_layer)
            layers.append(self.act)
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


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

        if norm is not None:
            if norm == "batch":
                self.norm = BatchNorm(target_dim)
            elif norm == "layer":
                self.norm = LayerNorm(target_dim)
            else:
                self.norm = None
        else:
            self.norm = None

        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()
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


class HeteroCellBipartite(nn.Module):
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

        # Default configs if not provided
        gene_encoder_config = gene_encoder_config or {}
        metabolism_config = metabolism_config or {}
        gpr_conv_config = gpr_conv_config or {}

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict: Dict[Any, nn.Module] = {}

            # Gene-gene interactions with GATv2Conv
            conv_dict[("gene", "physical_interaction", "gene")] = GATv2Conv(
                hidden_channels,
                hidden_channels // gene_encoder_config.get("heads", 1),
                heads=gene_encoder_config.get("heads", 1),
                concat=gene_encoder_config.get("concat", True),
                add_self_loops=gene_encoder_config.get("add_self_loops", False),
            )

            conv_dict[("gene", "regulatory_interaction", "gene")] = GATv2Conv(
                hidden_channels,
                hidden_channels // gene_encoder_config.get("heads", 1),
                heads=gene_encoder_config.get("heads", 1),
                concat=gene_encoder_config.get("concat", True),
                add_self_loops=gene_encoder_config.get("add_self_loops", False),
            )

            # Gene-reaction interactions with GATv2Conv
            conv_dict[("gene", "gpr", "reaction")] = GATv2Conv(
                hidden_channels,
                hidden_channels,
                heads=gpr_conv_config.get("heads", 1),
                concat=gpr_conv_config.get("concat", False),
                add_self_loops=gpr_conv_config.get("add_self_loops", False),
            )

            # Reaction-metabolite bipartite interactions with GATv2Conv
            conv_dict[("reaction", "rmr", "metabolite")] = GATv2Conv(
                hidden_channels,
                hidden_channels // metabolism_config.get("heads", 1),
                heads=metabolism_config.get("heads", 1),
                concat=metabolism_config.get("concat", True),
                add_self_loops=metabolism_config.get("add_self_loops", False),
                edge_dim=1,
            )

            # Wrap each conv
            for key, conv in conv_dict.items():
                conv_dict[key] = AttentionConvWrapper(
                    conv,
                    hidden_channels,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Global aggregator
        self.global_aggregator = AttentionalGraphAggregation(
            in_channels=hidden_channels, out_channels=hidden_channels, dropout=dropout
        )

        # Prediction head
        pred_config = prediction_head_config or {}
        self.prediction_head = self._build_prediction_head(
            in_channels=hidden_channels,
            hidden_channels=pred_config.get("hidden_channels", hidden_channels),
            out_channels=2,
            num_layers=pred_config.get("head_num_layers", 1),
            dropout=pred_config.get("dropout", dropout),
            activation=pred_config.get("activation", activation),
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
        norm: Optional[str] = None,
    ) -> nn.Module:
        if num_layers == 0:
            return nn.Identity()
        act = nn.ReLU() if activation == "relu" else nn.SiLU()
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

            # Print debugging information about masks and node counts
            # print(f"Batch size: {batch_size}")
            # print(
            #     f"Gene pert_mask shape: {gene_data.pert_mask.shape}, sum: {gene_data.pert_mask.sum()}"
            # )
            # print(f"Total gene nodes before masking: {gene_data.pert_mask.shape[0]}")
            # print(f"Total gene nodes after masking: {(~gene_data.pert_mask).sum()}")

            # Apply pert_mask only when selecting embeddings
            x_gene_exp = self.gene_embedding.weight.expand(batch_size, -1, -1)
            x_gene_comb = x_gene_exp.reshape(-1, x_gene_exp.size(-1))
            x_gene = x_gene_comb[~gene_data.pert_mask]
            x_gene = self.preprocessor(x_gene)

            # Similar debugging for reactions and metabolites
            print(f"Reaction nodes after masking: {(~reaction_data.pert_mask).sum()}")
            print(
                f"Metabolite nodes after masking: {(~metabolite_data.pert_mask).sum()}"
            )

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
            gene_data = data["gene"]
            reaction_data = data["reaction"]
            metabolite_data = data["metabolite"]

            gene_idx = torch.arange(gene_data.num_nodes, device=device)
            reaction_idx = torch.arange(reaction_data.num_nodes, device=device)
            metabolite_idx = torch.arange(metabolite_data.num_nodes, device=device)

            x_dict = {
                "gene": self.preprocessor(self.gene_embedding(gene_idx)),
                "reaction": self.reaction_embedding(reaction_idx),
                "metabolite": self.metabolite_embedding(metabolite_idx),
            }

        # Process edge indices directly - they should already be consistent with node selection
        # Process edge indices directly - they should already be consistent with node selection
        edge_index_dict = {}
        edge_attr_dict = {}
        edge_attr_dict = {}

        # Gene-gene interactions with debugging
        gene_phys_edge_index = data[
            ("gene", "physical_interaction", "gene")
        ].edge_index.to(device)
        edge_index_dict[("gene", "physical_interaction", "gene")] = gene_phys_edge_index

        # Add debugging for edge indices
        if is_batch:
            print(f"Physical edge index shape: {gene_phys_edge_index.shape}")
            if gene_phys_edge_index.numel() > 0:
                print(f"Max source index: {gene_phys_edge_index[0].max().item()}")
                print(f"Max target index: {gene_phys_edge_index[1].max().item()}")

        # Check if max index exceeds number of nodes
        if is_batch and gene_phys_edge_index.numel() > 0:
            max_idx = gene_phys_edge_index.max().item()
            num_nodes = x_gene.size(0)
            if max_idx >= num_nodes:
                print(
                    f"WARNING: Max edge index {max_idx} exceeds gene node count {num_nodes}"
                )

        # Repeat for other edge types
        gene_reg_edge_index = data[
            ("gene", "regulatory_interaction", "gene")
        ].edge_index.to(device)
        edge_index_dict[("gene", "regulatory_interaction", "gene")] = (
            gene_reg_edge_index
        )

        if is_batch and gene_reg_edge_index.numel() > 0:
            print(f"Regulatory edge max idx: {gene_reg_edge_index.max().item()}")

        # Gene-reaction interactions
        gpr_edge_index = data[("gene", "gpr", "reaction")].hyperedge_index.to(device)
        edge_index_dict[("gene", "gpr", "reaction")] = gpr_edge_index

        if is_batch and gpr_edge_index.numel() > 0:
            print(f"GPR max gene idx: {gpr_edge_index[0].max().item()}")
            print(f"GPR max reaction idx: {gpr_edge_index[1].max().item()}")
            if gpr_edge_index[0].max().item() >= x_gene.size(0):
                print(
                    f"WARNING: GPR max gene idx {gpr_edge_index[0].max().item()} exceeds gene count {x_gene.size(0)}"
                )
            if gpr_edge_index[1].max().item() >= x_reaction.size(0):
                print(
                    f"WARNING: GPR max reaction idx {gpr_edge_index[1].max().item()} exceeds reaction count {x_reaction.size(0)}"
                )

        # Reaction-metabolite interactions
        rmr_edge_type = ("reaction", "rmr", "metabolite")
        rmr_edge_index = data[rmr_edge_type].hyperedge_index.to(device)
        edge_index_dict[rmr_edge_type] = rmr_edge_index

        if is_batch and rmr_edge_index.numel() > 0:
            print(f"RMR max reaction idx: {rmr_edge_index[0].max().item()}")
            print(f"RMR max metabolite idx: {rmr_edge_index[1].max().item()}")
            if rmr_edge_index[0].max().item() >= x_reaction.size(0):
                print(
                    f"WARNING: RMR max reaction idx {rmr_edge_index[0].max().item()} exceeds reaction count {x_reaction.size(0)}"
                )
            if rmr_edge_index[1].max().item() >= x_metabolite.size(0):
                print(
                    f"WARNING: RMR max metabolite idx {rmr_edge_index[1].max().item()} exceeds metabolite count {x_metabolite.size(0)}"
                )

        # Process stoichiometry for metabolism edges
        stoich = data[rmr_edge_type].stoichiometry.to(device)
        # Process stoichiometry for metabolism edges
        stoich = data[rmr_edge_type].stoichiometry.to(device)
        if stoich.dim() == 1:
            stoich = stoich.unsqueeze(1)  # Make it [num_edges, 1]
        edge_attr_dict[rmr_edge_type] = stoich
        edge_attr_dict[rmr_edge_type] = stoich

        # Apply convolution layers
        # Apply convolution layers
        for conv in self.convs:
            try:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            except IndexError as e:
                print(f"IndexError in conv: {e}")
                # You can add more detailed debugging here
                raise

        return x_dict["gene"]

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Process reference graph
        z_w = self.forward_single(cell_graph)
        z_w = self.global_aggregator(
            z_w,
            index=torch.zeros(z_w.size(0), device=z_w.device, dtype=torch.long),
            dim_size=1,
        )

        # Process perturbed batch
        z_i = self.forward_single(batch)
        z_i = self.global_aggregator(z_i, index=batch["gene"].batch)

        # Compute difference vector
        batch_size: int = z_i.size(0)
        z_w_exp: torch.Tensor = z_w.expand(batch_size, -1)
        z_p: torch.Tensor = z_w_exp - z_i

        # Generate predictions
        predictions: torch.Tensor = self.prediction_head(z_p)
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
            "prediction_head": count_params(self.prediction_head),
        }
        counts["total"] = sum(counts.values())
        return counts


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/003-fit-int/conf"),
    config_name="hetero_cell_bipartite",
)
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.timestamp import timestamp
    import numpy as np
    from torchcell.scratch.load_batch import load_sample_data_batch
    from torchcell.scratch.cell_batch_overfit_visualization import (
        plot_embeddings,
        plot_correlations,
    )

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=32, num_workers=4, metabolism_graph="metabolism_bipartite"
    )
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Initialize model (parameters unchanged)
    model = HeteroCellBipartite(
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
    embeddings_dir = osp.join(ASSET_IMAGES_DIR, "embedding_plots")
    os.makedirs(embeddings_dir, exist_ok=True)

    correlation_dir = osp.join(ASSET_IMAGES_DIR, "correlation_plots")
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
            loss, loss_components = criterion(predictions, y, representations["z_p"])

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
        osp.join(ASSET_IMAGES_DIR, f"hetero_cell_training_loss_{timestamp()}.png")
    )
    plt.close()

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
