import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection, MeanSquaredError, PearsonCorrCoef
import matplotlib.pyplot as plt
from typing import Optional
import logging
from torchcell.viz.visual_graph_degen import VisGraphDegen
from torchcell.viz import genetic_interaction_score
from torchcell.viz.visual_regression import Visualization
from torchcell.timestamp import timestamp
from torch_geometric.data import HeteroData
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.diffusion_loss import DiffusionLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg

log = logging.getLogger(__name__)


class RegressionTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        cell_graph: torch.Tensor,
        optimizer_config: dict,
        lr_scheduler_config: dict,
        batch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        plot_sample_ceiling: int = 1000,
        plot_every_n_epochs: int = 10,
        plot_transformer_diagnostics_every_n_epochs: int = 10,
        plot_edge_recovery_every_n_epochs: int = 10,
        loss_func: nn.Module = None,
        grad_accumulation_schedule: Optional[dict[int, int]] = None,
        device: str = "cuda",
        inverse_transform: Optional[nn.Module] = None,
        execution_mode: str = "training",  # "training" or "dataloader_profiling"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.execution_mode = execution_mode
        # Clone cell_graph to avoid modifying the dataset's original cell_graph
        # This is necessary for pin_memory compatibility in DataLoader
        self.cell_graph = cell_graph.clone()
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func

        # Initialize gradient accumulation
        self.current_accumulation_steps = 1
        if self.hparams.grad_accumulation_schedule is not None:
            # Get the accumulation steps for epoch 0
            self.current_accumulation_steps = (
                self.hparams.grad_accumulation_schedule.get(0, 1)
            )

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
            }
        )

        # Create metrics for each stage
        for stage in ["train", "val", "test"]:
            metrics_dict = reg_metrics.clone(prefix=f"{stage}/gene_interaction/")
            setattr(self, f"{stage}_metrics", metrics_dict)

            # Add metrics operating in transformed space
            transformed_metrics = reg_metrics.clone(
                prefix=f"{stage}/transformed/gene_interaction/"
            )
            setattr(self, f"{stage}_transformed_metrics", transformed_metrics)

        # Separate accumulators for train, validation, and test samples
        self.train_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.val_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.automatic_optimization = False

        # Edge recovery metric accumulators (for validation)
        self.edge_recovery_ks = [8, 32, 128, 320]
        self.reset_edge_recovery_accumulators()

        # Attention diagnostic accumulators (for validation)
        self.attention_stats_accumulators = (
            {}
        )  # {layer_idx: {"entropy_sum": float, "effective_rank_sum": float, "top5_sum": float, "top10_sum": float, "top50_sum": float, "count": int}}
        self.gradient_norms = {}  # {layer_idx: norm_value}

        # Residual update accumulators (Tier 1 - very cheap)
        self.residual_update_accumulators = (
            {}
        )  # {layer_idx: {"sum_ratio": float, "count": int}}

    def _get_batch_size(self, batch):
        """Get batch size from batch, handling different batch structures."""
        if hasattr(batch["gene"], "x"):
            return batch["gene"].x.size(0)
        elif hasattr(batch["gene"], "perturbation_indices"):
            # For Perturbation processor, count unique batch indices
            if hasattr(batch["gene"], "perturbation_indices_batch"):
                return batch["gene"].perturbation_indices_batch.max().item() + 1
            else:
                # Fallback: assume batch size from perturbation_indices
                return batch["gene"].perturbation_indices.size(0)
        elif hasattr(batch["gene"], "phenotype_values"):
            return batch["gene"].phenotype_values.size(0)
        else:
            # Last resort fallback
            return 1

    def reset_edge_recovery_accumulators(self):
        """Reset accumulators for edge recovery metrics."""
        self.edge_recovery_accumulators = {}

    def _plot_edge_recovery_metrics(self):
        """Create and log matplotlib visualizations for edge recovery metrics."""
        from torchcell.viz.graph_recovery import GraphRecoveryVisualization

        # Initialize visualization
        vis = GraphRecoveryVisualization(base_dir=self.trainer.default_root_dir)

        # Prepare recall metrics (keys now include layer/head: "graph_L0_H1")
        recall_metrics = {}
        for metric_key, acc in self.edge_recovery_accumulators.items():
            if acc["count_nodes_deg"] > 0:
                recall_metrics[metric_key] = (
                    acc["sum_recall_deg"] / acc["count_nodes_deg"]
                )

        # Prepare precision metrics
        precision_metrics = {}
        for metric_key, acc in self.edge_recovery_accumulators.items():
            precision_metrics[metric_key] = {}
            for k in self.edge_recovery_ks:
                if acc["count_nodes_prec"][k] > 0:
                    precision_metrics[metric_key][k] = (
                        acc["sum_prec"][k] / acc["count_nodes_prec"][k]
                    )

        # Prepare edge-mass alignment metrics
        edge_mass_metrics = {}
        for metric_key, acc in self.edge_recovery_accumulators.items():
            if acc["count_batches"] > 0:
                edge_mass_metrics[metric_key] = (
                    acc["sum_edge_mass"] / acc["count_batches"]
                )

        # Create aggregate plots
        if recall_metrics:
            vis.plot_edge_recovery_recall(
                recall_metrics, self.current_epoch, None, stage="val"
            )

        if precision_metrics:
            vis.plot_edge_recovery_precision(
                precision_metrics,
                self.edge_recovery_ks,
                self.current_epoch,
                None,
                stage="val",
            )

        if edge_mass_metrics:
            vis.plot_edge_mass_alignment(
                edge_mass_metrics, self.current_epoch, None, stage="val"
            )

        # Create per-graph plots
        if recall_metrics and precision_metrics:
            vis.plot_edge_recovery_per_graph(
                recall_metrics,
                precision_metrics,
                self.edge_recovery_ks,
                self.current_epoch,
                None,
                stage="val",
            )

    def _accumulate_edge_recovery_metrics(self, attention_weights_list, batch_idx):
        """
        Compute edge recovery metrics from attention weights.

        Args:
            attention_weights_list: List of attention tensors per layer [batch, heads, N, N]
            batch_idx: Current batch index
        """
        # Only compute if model has graph regularization enabled
        if not hasattr(self.model, "regularized_head_config") or not hasattr(
            self.model, "adjacency_matrices"
        ):
            if batch_idx == 0:  # Print once per epoch
                print(
                    "Edge recovery: Model missing regularized_head_config or adjacency_matrices attributes"
                )
            return

        if (
            self.model.regularized_head_config is None
            or self.model.adjacency_matrices is None
        ):
            if batch_idx == 0:  # Print once per epoch
                print(
                    "Edge recovery: Model has None for regularized_head_config or adjacency_matrices"
                )
            return

        # For each regularized graph, expand multi-layer configs
        for graph_name, config in self.model.regularized_head_config.items():
            # Handle both single layer (int) and multiple layers (list)
            layer_config = config["layer"]
            layers = [layer_config] if isinstance(layer_config, int) else layer_config
            head_idx = config["head"]

            # Process each layer separately
            for layer_idx in layers:
                # Skip if this layer's attention not available
                if layer_idx >= len(attention_weights_list):
                    continue

                # Get attention for this graph's layer/head: [batch, heads, N, N]
                attn = attention_weights_list[layer_idx]
                attn_for_head = attn[:, head_idx, :, :]  # [batch, N, N]

                # Get true adjacency for this graph from dict
                if graph_name not in self.model.adjacency_matrices:
                    continue
                adj_true = self.model.adjacency_matrices[graph_name].to(attn_for_head.device)  # [N, N]

                # Average attention across batch dimension
                attn_avg = attn_for_head.mean(dim=0)  # [N, N]

                # Create unique key for this (graph, layer, head) combination
                metric_key = f"{graph_name}_L{layer_idx}_H{head_idx}"

                # Initialize accumulators for this combination if needed
                if metric_key not in self.edge_recovery_accumulators:
                    self.edge_recovery_accumulators[metric_key] = {
                        "sum_recall_deg": 0.0,
                        "count_nodes_deg": 0,
                        "sum_prec": {k: 0.0 for k in self.edge_recovery_ks},
                        "count_nodes_prec": {k: 0 for k in self.edge_recovery_ks},
                        "sum_edge_mass": 0.0,  # NEW: Edge-mass alignment
                        "count_batches": 0,  # NEW: Number of batches
                        "graph_name": graph_name,  # Store original graph name
                        "layer": layer_idx,  # Store layer index
                        "head": head_idx,  # Store head index
                        "degree_correlation_sum": 0.0,
                        "degree_corr_count": 0,
                    }
                elif "degree_correlation_sum" not in self.edge_recovery_accumulators[metric_key]:
                    # Add degree fields if accumulator exists but was created by degree bias
                    self.edge_recovery_accumulators[metric_key]["degree_correlation_sum"] = 0.0
                    self.edge_recovery_accumulators[metric_key]["degree_corr_count"] = 0

                acc = self.edge_recovery_accumulators[metric_key]

                # NEW: Compute edge-mass alignment (fraction of attention on true edges)
                edge_mask = (adj_true > 0).float()  # [N, N] binary mask
                total_attn_val = attn_avg.sum().item()
                edge_attn_val = (attn_avg * edge_mask).sum().item()
                edge_mass_fraction = (
                    (edge_attn_val / total_attn_val) if total_attn_val > 0 else 0.0
                )

                acc["sum_edge_mass"] += edge_mass_fraction
                acc["count_batches"] += 1

                # Get number of nodes
                N = attn_avg.size(0)

                # Pre-compute topk for ALL nodes at once (vectorized - prevents memory leak)
                # This replaces N*5 topk calls with 1 topk call per batch
                k_max = max(self.edge_recovery_ks + [N])
                topk_values, topk_indices = torch.topk(
                    attn_avg, min(k_max, N), dim=-1
                )  # [N, k_max]

                # For each node
                for i in range(N):
                    # Get true neighbors from adjacency
                    true_neighbors = (adj_true[i] > 0).nonzero(as_tuple=True)[0]
                    degree_i = len(true_neighbors)

                    # Skip isolated nodes
                    if degree_i == 0:
                        continue

                    # Use pre-computed topk results (no new allocations)
                    top_indices_i = topk_indices[i]  # [k_max]

                    # === Recall@degree ===
                    k_deg = min(degree_i, N)
                    top_deg_indices = top_indices_i[:k_deg]
                    hits_deg = torch.isin(top_deg_indices, true_neighbors).sum().item()
                    recall_deg_i = hits_deg / degree_i

                    acc["sum_recall_deg"] += recall_deg_i
                    acc["count_nodes_deg"] += 1

                    # === Precision@k for each k ===
                    for k in self.edge_recovery_ks:
                        k_eff = min(k, N)
                        top_k_indices = top_indices_i[:k_eff]
                        hits_k = torch.isin(top_k_indices, true_neighbors).sum().item()
                        prec_k_i = hits_k / k_eff

                        acc["sum_prec"][k] += prec_k_i
                        acc["count_nodes_prec"][k] += 1

                # Clean up GPU memory after processing all nodes
                if torch.cuda.is_available():
                    del topk_values, topk_indices
                    torch.cuda.empty_cache()

    def _accumulate_attention_diagnostics(self, attention_weights_list, batch_idx):
        """
        Compute attention diagnostics (entropy, effective rank, top-k concentration) from attention weights.

        Args:
            attention_weights_list: List of attention tensors per layer [batch, heads, N, N]
            batch_idx: Current batch index
        """
        if attention_weights_list is None or len(attention_weights_list) == 0:
            return

        for layer_idx, attn in enumerate(attention_weights_list):
            # attn shape: [batch, heads, N, N]
            # Average across batch and heads for diagnostics
            attn_avg = attn.mean(dim=(0, 1))  # [N, N]

            # Compute entropy: -sum(p * log(p))
            entropy = -(attn_avg * torch.log(attn_avg + 1e-10)).sum(dim=-1).mean()

            # Compute effective rank: exp(entropy)
            effective_rank = torch.exp(entropy)

            # Compute top-k concentration: fraction of attention mass in top-k positions
            top5_vals, _ = torch.topk(attn_avg, k=min(5, attn_avg.shape[-1]), dim=-1)
            top10_vals, _ = torch.topk(attn_avg, k=min(10, attn_avg.shape[-1]), dim=-1)
            top50_vals, _ = torch.topk(attn_avg, k=min(50, attn_avg.shape[-1]), dim=-1)

            concentration_top5 = top5_vals.sum(dim=-1).mean()
            concentration_top10 = top10_vals.sum(dim=-1).mean()
            concentration_top50 = top50_vals.sum(dim=-1).mean()

            # NEW: Max row weight (one-hot detection)
            max_weights = attn_avg.max(dim=-1)[0]  # [N]
            avg_max_weight = max_weights.mean()

            # NEW: Column-sum concentration (sink collapse detection)
            col_sums = attn_avg.sum(dim=0)  # [N] - attention received per gene
            col_sums_normalized = col_sums / (
                col_sums.sum() + 1e-10
            )  # Normalize to prob dist
            col_entropy = -(
                col_sums_normalized * torch.log(col_sums_normalized + 1e-10)
            ).sum()
            max_col_sum = col_sums.max()

            # Initialize accumulator if needed
            if layer_idx not in self.attention_stats_accumulators:
                self.attention_stats_accumulators[layer_idx] = {
                    "entropy_sum": 0.0,
                    "effective_rank_sum": 0.0,
                    "top5_sum": 0.0,
                    "top10_sum": 0.0,
                    "top50_sum": 0.0,
                    "max_row_weight_sum": 0.0,
                    "col_entropy_sum": 0.0,
                    "max_col_sum_sum": 0.0,
                    "count": 0,
                }

            # Accumulate
            acc = self.attention_stats_accumulators[layer_idx]
            acc["entropy_sum"] += entropy.item()
            acc["effective_rank_sum"] += effective_rank.item()
            acc["top5_sum"] += concentration_top5.item()
            acc["top10_sum"] += concentration_top10.item()
            acc["top50_sum"] += concentration_top50.item()
            acc["max_row_weight_sum"] += avg_max_weight.item()
            acc["col_entropy_sum"] += col_entropy.item()
            acc["max_col_sum_sum"] += max_col_sum.item()
            acc["count"] += 1

    def reset_attention_diagnostics(self):
        """Reset attention diagnostic accumulators."""
        self.attention_stats_accumulators = {}
        self.gradient_norms = {}

    def _accumulate_residual_updates(self, x_in, x_out, layer_idx):
        """
        Track layer update magnitude relative to input (Tier 1 - very cheap).

        Args:
            x_in: [batch, N+1, d] input to layer
            x_out: [batch, N+1, d] output from layer
            layer_idx: Current transformer layer index
        """
        # Compute ratio: ||x_out - x_in|| / ||x_in||
        update_norm = (
            (x_out - x_in).norm(dim=-1).mean()
        )  # Average over batch and sequence
        input_norm = x_in.norm(dim=-1).mean()
        ratio = (update_norm / (input_norm + 1e-10)).item()

        # Initialize accumulator if needed
        if layer_idx not in self.residual_update_accumulators:
            self.residual_update_accumulators[layer_idx] = {
                "sum_ratio": 0.0,
                "count": 0,
            }

        # Accumulate
        acc = self.residual_update_accumulators[layer_idx]
        acc["sum_ratio"] += ratio
        acc["count"] += 1

    def _accumulate_degree_bias(
        self, attention_weights, graph_name, layer_idx, head_idx
    ):
        """
        Compute correlation between graph degree and attention received (Tier 2 - medium cost).

        Args:
            attention_weights: [batch, heads, N, N] gene-gene attention weights
            graph_name: Name of the biological graph
            layer_idx: Current transformer layer index
            head_idx: Attention head index
        """
        # Only for graph-regularized layers
        if graph_name not in self.model.regularized_head_config:
            return

        # Get adjacency matrix from dict
        if graph_name not in self.model.adjacency_matrices:
            return
        adj_true = self.model.adjacency_matrices[graph_name].to(attention_weights.device)  # [N, N]
        degrees = adj_true.sum(dim=-1).detach().cpu().numpy()  # [N]

        # Average attention across batch
        attn_avg = attention_weights.mean(dim=0)  # [heads, N, N]

        # Extract specific head
        attn_head = attn_avg[head_idx]  # [N, N]

        # Column sums = attention received per node
        col_sums = attn_head.sum(dim=0).detach().cpu().numpy()  # [N]

        # Spearman correlation
        from scipy.stats import spearmanr

        degree_corr, _ = spearmanr(degrees, col_sums)

        # Store in edge recovery accumulator (same tier)
        # IMPORTANT: Use same key format as edge recovery (_L{}_H{} not _layer{}_head{})
        key = f"{graph_name}_L{layer_idx}_H{head_idx}"
        if key not in self.edge_recovery_accumulators:
            # Initialize with complete structure to avoid KeyError in logging
            self.edge_recovery_accumulators[key] = {
                "sum_recall_deg": 0.0,
                "count_nodes_deg": 0,
                "sum_prec": {k: 0.0 for k in self.edge_recovery_ks},
                "count_nodes_prec": {k: 0 for k in self.edge_recovery_ks},
                "sum_edge_mass": 0.0,
                "count_batches": 0,
                "graph_name": graph_name,
                "layer": layer_idx,
                "head": head_idx,
                "degree_correlation_sum": 0.0,
                "degree_corr_count": 0,
            }
        elif "degree_correlation_sum" not in self.edge_recovery_accumulators[key]:
            # Add degree fields if accumulator exists but was created by edge recovery
            self.edge_recovery_accumulators[key]["degree_correlation_sum"] = 0.0
            self.edge_recovery_accumulators[key]["degree_corr_count"] = 0

        self.edge_recovery_accumulators[key]["degree_correlation_sum"] += degree_corr
        self.edge_recovery_accumulators[key]["degree_corr_count"] += 1

    def _plot_attention_diagnostics(self):
        """Create and log attention diagnostic visualizations."""
        from torchcell.viz.transformer_diagnostics import TransformerDiagnostics

        # Initialize visualization
        vis = TransformerDiagnostics(base_dir=self.trainer.default_root_dir)

        # Prepare attention stats
        attention_stats = {}
        for layer_idx, acc in self.attention_stats_accumulators.items():
            if acc["count"] > 0:
                attention_stats[layer_idx] = {
                    "entropy": acc["entropy_sum"] / acc["count"],
                    "effective_rank": acc["effective_rank_sum"] / acc["count"],
                    "top5": acc["top5_sum"] / acc["count"],
                    "top10": acc["top10_sum"] / acc["count"],
                    "top50": acc["top50_sum"] / acc["count"],
                    "max_row_weight": acc["max_row_weight_sum"] / acc["count"],
                    "col_entropy": acc["col_entropy_sum"] / acc["count"],
                    "max_col_sum": acc["max_col_sum_sum"] / acc["count"],
                }

        # Prepare residual update ratios
        residual_ratios = {}
        for layer_idx, acc in self.residual_update_accumulators.items():
            if acc["count"] > 0:
                residual_ratios[layer_idx] = acc["sum_ratio"] / acc["count"]

        # Plot if we have stats
        if attention_stats:
            vis.plot_attention_diagnostics(
                attention_stats,
                residual_ratios=residual_ratios if residual_ratios else None,
                gradient_norms=self.gradient_norms if self.gradient_norms else None,
                num_epochs=self.current_epoch,
                stage="val",
            )

    def forward(self, batch, return_attention=False):
        # Get device from batch - handle different batch structures
        if hasattr(batch["gene"], "x"):
            batch_device = batch["gene"].x.device
        elif hasattr(batch["gene"], "perturbation_indices"):
            batch_device = batch["gene"].perturbation_indices.device
        elif hasattr(batch["gene"], "phenotype_values"):
            batch_device = batch["gene"].phenotype_values.device
        else:
            # Fallback to model device
            batch_device = next(self.model.parameters()).device

        if (
            not hasattr(self, "_cell_graph_device")
            or self._cell_graph_device != batch_device
        ):
            self.cell_graph = self.cell_graph.to(batch_device)
            self._cell_graph_device = batch_device

        # Return all outputs from the model
        return self.model(self.cell_graph, batch, return_attention=return_attention)

    def _ensure_no_unused_params_loss(self):
        """Add a dummy loss to ensure all parameters are used in backward pass."""
        dummy_loss = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is None:
                dummy_loss = dummy_loss + 0.0 * param.sum()
        return dummy_loss

    def _shared_step(self, batch, batch_idx, stage="train"):
        # DataLoader profiling mode: Skip model forward, create dummy loss
        if self.execution_mode == "dataloader_profiling":
            # Execute all batch preparation (moving to device happens in forward())
            # Get device from batch - handle different batch structures
            if hasattr(batch["gene"], "x"):
                batch_device = batch["gene"].x.device
            elif hasattr(batch["gene"], "perturbation_indices"):
                batch_device = batch["gene"].perturbation_indices.device
            elif hasattr(batch["gene"], "phenotype_values"):
                batch_device = batch["gene"].phenotype_values.device
            else:
                batch_device = next(self.model.parameters()).device

            # Ensure cell_graph is on correct device
            if (
                not hasattr(self, "_cell_graph_device")
                or self._cell_graph_device != batch_device
            ):
                self.cell_graph = self.cell_graph.to(batch_device)
                self._cell_graph_device = batch_device

            # Create trivial loss that touches ALL model parameters (required for DDP)
            # This ensures no "unused parameters" error in DDP mode
            loss = torch.zeros((), device=batch_device, requires_grad=True)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss = loss + (param * 0.0).sum()

            # Log minimal metrics
            batch_size = self._get_batch_size(batch)
            self.log(
                f"{stage}/dataloader_profile_loss",
                loss,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"{stage}/dataloader_profile_batch_size",
                float(batch_size),
                batch_size=batch_size,
                sync_dist=True,
            )

            return loss, None, None

        # Normal training/validation/test execution
        # Get model outputs - only request attention weights during validation on diagnostic epochs
        if stage == "val":
            # Check if ANY diagnostic tier is scheduled for this epoch
            plot_transformer_freq = self.hparams.get("plot_transformer_diagnostics_every_n_epochs", 10)
            plot_edge_freq = self.hparams.get("plot_edge_recovery_every_n_epochs", 10)

            is_diagnostic_epoch = (
                (self.current_epoch + 1) % plot_transformer_freq == 0
                or (self.current_epoch + 1) % plot_edge_freq == 0
            )
            return_attention = is_diagnostic_epoch

            # Debug: Log attention storage decision (only on rank 0, once per epoch)
            if batch_idx == 0 and self.trainer.global_rank == 0:
                if return_attention:
                    print(f"Epoch {self.current_epoch}: Storing attention for diagnostics")
                else:
                    print(f"Epoch {self.current_epoch}: Skipping attention storage")
        else:
            return_attention = False  # Never during training (graph reg doesn't need storage)

        predictions, representations = self(batch, return_attention=return_attention)

        # In validation stage, compute edge recovery metrics and attention diagnostics
        if stage == "val":
            if "attention_weights" in representations and representations["attention_weights"] is not None:
                attention_weights_list = representations["attention_weights"]

                # TIER 1: Cheap transformer diagnostics (attention stats, entropy, residual ratios)
                plot_transformer_freq = self.hparams.get(
                    "plot_transformer_diagnostics_every_n_epochs", 10
                )
                if (self.current_epoch + 1) % plot_transformer_freq == 0:
                    self._accumulate_attention_diagnostics(
                        attention_weights_list, batch_idx
                    )

                # TIER 2: Edge recovery (already has separate frequency control)
                plot_edge_freq = self.hparams.get(
                    "plot_edge_recovery_every_n_epochs", 10
                )
                if (self.current_epoch + 1) % plot_edge_freq == 0:
                    self._accumulate_edge_recovery_metrics(
                        attention_weights_list, batch_idx
                    )

                # TIER 1: Accumulate residual update ratios (from model)
                if "residual_update_ratios" in representations and representations["residual_update_ratios"]:
                    for layer_idx, ratio in enumerate(representations["residual_update_ratios"]):
                        if layer_idx not in self.residual_update_accumulators:
                            self.residual_update_accumulators[layer_idx] = {
                                "sum_ratio": 0.0,
                                "count": 0,
                            }
                        acc = self.residual_update_accumulators[layer_idx]
                        acc["sum_ratio"] += ratio
                        acc["count"] += 1

                # TIER 2: Medium cost - Degree-bias correlation
                # Iterate through regularized heads to compute degree-bias
                if (
                    hasattr(self.model, "regularized_head_config")
                    and self.model.regularized_head_config
                ):
                    for (
                        graph_name,
                        config,
                    ) in self.model.regularized_head_config.items():
                        layer_spec = config["layer"]
                        head_idx = config["head"]

                        # Handle both single int and list of ints for layer
                        layer_indices = (
                            layer_spec if isinstance(layer_spec, list) else [layer_spec]
                        )

                        # Process each layer
                        for layer_idx in layer_indices:
                            if layer_idx < len(attention_weights_list):
                                self._accumulate_degree_bias(
                                    attention_weights_list[layer_idx],
                                    graph_name,
                                    layer_idx,
                                    head_idx,
                                )

            elif batch_idx == 0:  # Print once per validation epoch
                print(
                    "Edge recovery: Model not returning 'attention_weights' in representations"
                )

        # Ensure predictions has correct shape (batch_size, 1)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)  # Make it [batch_size, 1]

        batch_size = predictions.size(0)

        # Get target values - now in COO format
        # For gene interaction dataset, phenotype_values directly contains the values
        gene_interaction_vals = batch["gene"].phenotype_values

        # Handle tensor shape
        if gene_interaction_vals.dim() == 0:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_vals.dim() == 1:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(1)

        # For original values, check if there's a phenotype_values_original
        if hasattr(batch["gene"], "phenotype_values_original"):
            gene_interaction_orig = batch["gene"].phenotype_values_original
        else:
            gene_interaction_orig = gene_interaction_vals

        # Handle tensor shape
        if gene_interaction_orig.dim() == 0:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_orig.dim() == 1:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(1)

        # Get latent representations from model
        # Backward compatibility: new models return H_genes_pert, old models return z_p
        z_p = representations.get("z_p")
        if z_p is None:
            z_p = representations.get("H_genes_pert")
        H_genes = representations.get("H_genes")  # Pre-perturbation gene embeddings
        H_genes_pert = representations.get(
            "H_genes_pert"
        )  # Post-perturbation gene embeddings
        h_CLS = representations.get("h_CLS")  # CLS token

        # Log CLS token norm
        if h_CLS is not None:
            cls_norm = h_CLS.norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/cls_token_norm",
                cls_norm,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Residual Update Ratio: Measure how much the transformer changes the embeddings
        if stage == "val" and H_genes is not None and H_genes_pert is not None:
            # Compute the ratio of update magnitude to input magnitude
            residual_norm = (H_genes_pert - H_genes).norm(p=2)
            input_norm = H_genes.norm(p=2)
            residual_ratio = residual_norm / (input_norm + 1e-8)  # Avoid division by zero

            self.log(
                f"{stage}/residual_update_ratio",
                residual_ratio.item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
            )

        # Calculate loss based on loss function type
        if self.loss_func is None:
            raise ValueError("No loss function provided")

        if isinstance(self.loss_func, LogCoshLoss):
            # For LogCoshLoss, just pass predictions and targets
            loss = self.loss_func(predictions, gene_interaction_vals)
        elif isinstance(self.loss_func, PointDistGraphReg):
            # For PointDistGraphReg, pass predictions, targets, and representations
            # Returns (total_loss, loss_dict) with all components
            total_loss, loss_dict = self.loss_func(
                predictions,
                gene_interaction_vals,
                representations,
                epoch=self.current_epoch,
            )
            loss = total_loss

            # Log all loss components
            if isinstance(loss_dict, dict):
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            self.log(
                                f"{stage}/{key}",
                                value.item(),
                                batch_size=batch_size,
                                sync_dist=True,
                            )
                    elif isinstance(value, (int, float)):
                        self.log(
                            f"{stage}/{key}",
                            value,
                            batch_size=batch_size,
                            sync_dist=True,
                        )
        else:
            # For ICLoss or other custom losses that might use z_p
            # Check if loss function accepts epoch parameter (for MleDistSupCR and MleWassSupCR)
            if z_p is not None:
                if isinstance(self.loss_func, (MleDistSupCR, MleWassSupCR)):
                    loss_output = self.loss_func(
                        predictions,
                        gene_interaction_vals,
                        z_p,
                        epoch=self.current_epoch,
                    )
                else:
                    loss_output = self.loss_func(
                        predictions, gene_interaction_vals, z_p
                    )
            else:
                loss_output = self.loss_func(predictions, gene_interaction_vals)

            # Handle if loss_func returns a tuple (for ICLoss)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # First element is the loss
                loss_dict = loss_output[1] if len(loss_output) > 1 else {}

                # Log additional loss components if available
                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            # Handle multi-dimensional tensors
                            if value.numel() == 1:
                                # Single element tensor - log as scalar
                                self.log(
                                    f"{stage}/{key}",
                                    value.item(),
                                    batch_size=batch_size,
                                    sync_dist=True,
                                )
                            elif value.numel() > 1:
                                # Multi-element tensor - log each element separately
                                for i in range(value.numel()):
                                    self.log(
                                        f"{stage}/{key}_{i}",
                                        value[i].item(),
                                        batch_size=batch_size,
                                        sync_dist=True,
                                    )
                            # Skip empty tensors
                        elif isinstance(value, (int, float)):
                            # Handle scalar values
                            self.log(
                                f"{stage}/{key}",
                                value,
                                batch_size=batch_size,
                                sync_dist=True,
                            )
            else:
                loss = loss_output

        # Add graph regularization loss if present (for transformer models)
        # IMPORTANT: Skip if using PointDistGraphReg, as it already includes graph_reg in total
        if "graph_reg_loss" in representations and not isinstance(
            self.loss_func, PointDistGraphReg
        ):
            graph_reg_loss = representations["graph_reg_loss"]
            loss = loss + graph_reg_loss
            self.log(
                f"{stage}/graph_reg_loss",
                graph_reg_loss,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Add dummy loss for unused parameters
        dummy_loss = self._ensure_no_unused_params_loss()
        loss = loss + dummy_loss

        # Log the loss
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)

        # Log z_p norm if available
        if z_p is not None:
            z_p_norm = z_p.norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/z_p_norm", z_p_norm, batch_size=batch_size, sync_dist=True
            )

        # Log gate weights if available
        if "gate_weights" in representations:
            gate_weights = representations["gate_weights"]
            # Average gate weights across batch
            avg_gate_weights = gate_weights.mean(dim=0)
            self.log(
                f"{stage}/gate_weight_global",
                avg_gate_weights[0],
                batch_size=batch_size,
                sync_dist=True,
            )
            # Only log local weight if it exists (when local predictor is enabled)
            if avg_gate_weights.size(0) > 1:
                self.log(
                    f"{stage}/gate_weight_local",
                    avg_gate_weights[1],
                    batch_size=batch_size,
                    sync_dist=True,
                )

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            transformed_metrics.update(
                predictions[mask].view(-1), gene_interaction_vals[mask].view(-1)
            )

        # Handle inverse transform if available
        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions in COO format
            temp_data = HeteroData()

            # Create COO format data for predictions
            batch_size = predictions.size(0)
            device = predictions.device
            temp_data["gene"].phenotype_values = predictions.squeeze()
            temp_data["gene"].phenotype_type_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            temp_data["gene"].phenotype_sample_indices = torch.arange(
                batch_size, device=device
            )
            temp_data["gene"].phenotype_types = ["gene_interaction"]

            # Apply the inverse transform
            inv_data = self.inverse_transform(temp_data)

            # Extract the inversed predictions
            inv_gene_int = inv_data["gene"]["phenotype_values"]

            # Handle tensor shape
            if isinstance(inv_gene_int, torch.Tensor):
                if inv_gene_int.dim() == 0:
                    inv_predictions = inv_gene_int.unsqueeze(0).unsqueeze(0)
                elif inv_gene_int.dim() == 1:
                    inv_predictions = inv_gene_int.unsqueeze(1)
                else:
                    inv_predictions = inv_gene_int

        # Update metrics with original scale values
        mask = ~torch.isnan(gene_interaction_orig)
        if mask.sum() > 0:
            metrics = getattr(self, f"{stage}_metrics")
            metrics.update(
                inv_predictions[mask].view(-1), gene_interaction_orig[mask].view(-1)
            )

        # Collect samples for visualization
        if (
            stage == "train"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            current_count = sum(t.size(0) for t in self.train_samples["true_values"])
            if current_count < self.hparams.plot_sample_ceiling:
                remaining = self.hparams.plot_sample_ceiling - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.train_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.train_samples["predictions"].append(
                        inv_predictions[idx].detach()
                    )

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.train_samples:
                        self.train_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # [batch, 1, d]
                        H_combined = torch.cat([h_CLS_batched, H_genes_pert], dim=1)  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.train_samples["latents"]["H_pooled"].append(H_pooled[idx].detach().cpu())
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach())

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.train_samples:
                        self.train_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # [batch, 1, d]
                        H_combined = torch.cat([h_CLS_batched, H_genes_pert], dim=1)  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.train_samples["latents"]["H_pooled"].append(H_pooled.detach().cpu())
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot, respecting ceiling
            current_count = sum(t.size(0) for t in self.val_samples["true_values"])
            if current_count < self.hparams.plot_sample_ceiling:
                remaining = self.hparams.plot_sample_ceiling - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.val_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.val_samples["predictions"].append(
                        inv_predictions[idx].detach()
                    )

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.val_samples:
                        self.val_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # [batch, 1, d]
                        H_combined = torch.cat([h_CLS_batched, H_genes_pert], dim=1)  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.val_samples["latents"]["H_pooled"].append(H_pooled[idx].detach().cpu())
                else:
                    self.val_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.val_samples["predictions"].append(inv_predictions.detach())

                    # Collect latent representations for UMAP visualization
                    if "latents" not in self.val_samples:
                        self.val_samples["latents"] = {}

                    # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
                    if h_CLS is not None and H_genes_pert is not None:
                        if "H_pooled" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["H_pooled"] = []
                        # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                        h_CLS_batched = h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # [batch, 1, d]
                        H_combined = torch.cat([h_CLS_batched, H_genes_pert], dim=1)  # [batch, N+1, d]
                        H_pooled = H_combined.mean(dim=1)  # [batch, d]
                        self.val_samples["latents"]["H_pooled"].append(H_pooled.detach().cpu())
        elif stage == "test":
            # For test, always collect samples (no epoch check since test runs once)
            self.test_samples["true_values"].append(gene_interaction_orig.detach())
            self.test_samples["predictions"].append(inv_predictions.detach())

            # Collect latent representations for UMAP visualization
            if "latents" not in self.test_samples:
                self.test_samples["latents"] = {}

            # Collect H_pooled (combines h_CLS with perturbed genes, pooled)
            if h_CLS is not None and H_genes_pert is not None:
                if "H_pooled" not in self.test_samples["latents"]:
                    self.test_samples["latents"]["H_pooled"] = []
                # Pool immediately: mean([h_CLS, H_genes_pert]) -> [batch, d]
                h_CLS_batched = h_CLS.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # [batch, 1, d]
                H_combined = torch.cat([h_CLS_batched, H_genes_pert], dim=1)  # [batch, N+1, d]
                H_pooled = H_combined.mean(dim=1)  # [batch, d]
                self.test_samples["latents"]["H_pooled"].append(H_pooled.detach().cpu())

        return loss, predictions, gene_interaction_orig

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        # Model profiling mode: Skip optimizer step to isolate model compute
        if self.execution_mode == "model_profiling":
            return loss

        # Get batch size using helper method
        batch_size = self._get_batch_size(batch)

        # Normal training: Run optimizer
        if self.hparams.grad_accumulation_schedule is not None:
            loss = loss / self.current_accumulation_steps
        opt = self.optimizers()
        self.manual_backward(loss)
        if (
            self.hparams.grad_accumulation_schedule is None
            or (batch_idx + 1) % self.current_accumulation_steps == 0
        ):
            if self.hparams.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self.hparams.clip_grad_norm_max_norm
                )
            opt.step()
            opt.zero_grad()
        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],
            batch_size=batch_size,
            sync_dist=True,
        )
        # Log effective batch size when using gradient accumulation
        if self.hparams.grad_accumulation_schedule is not None:
            # Get world size for DDP
            world_size = 1
            if hasattr(self.trainer, "strategy") and hasattr(
                self.trainer.strategy, "_strategy_name"
            ):
                if self.trainer.strategy._strategy_name == "ddp":
                    import torch.distributed as dist

                    if dist.is_initialized():
                        world_size = dist.get_world_size()

            effective_batch_size = (
                batch_size * self.current_accumulation_steps * world_size
            )
            self.log(
                "effective_batch_size",
                effective_batch_size,
                batch_size=batch_size,
                sync_dist=True,
            )
        # print(f"Loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "val")

        # Defragment GPU memory every 50 batches to prevent OOM from fragmentation
        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def _compute_metrics_safely(self, metrics_dict):
        results = {}
        for metric_name, metric in metrics_dict.items():
            try:
                results[metric_name] = metric.compute()
            except ValueError as e:
                if any(
                    msg in str(e)
                    for msg in [
                        "Needs at least two samples",
                        "No samples to concatenate",
                    ]
                ):
                    continue
                raise e
        return results

    def _plot_samples(self, samples, stage: str) -> None:
        if not samples["true_values"]:
            return

        true_values = torch.cat(samples["true_values"], dim=0)
        predictions = torch.cat(samples["predictions"], dim=0)

        # Process latents if they exist
        latents = {}
        if "latents" in samples and samples["latents"]:
            for k, v in samples["latents"].items():
                if v:  # Check if the list is not empty
                    # Ensure all tensors are 2D before concatenating (defensive check)
                    tensors_2d = [t.unsqueeze(0) if t.ndim == 1 else t for t in v]
                    latents[k] = torch.cat(tensors_2d, dim=0)

        max_samples = self.hparams.plot_sample_ceiling
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]
            for key in latents:
                latents[key] = latents[key][idx]

        # Use Visualization for enhanced plotting
        vis = Visualization(
            base_dir=self.trainer.default_root_dir, max_points=max_samples
        )

        loss_name = (
            self.loss_func.__class__.__name__ if self.loss_func is not None else "Loss"
        )

        # Ensure data is in the correct format for visualization
        # For gene interactions, we only need a single dimension
        if true_values.dim() == 1:
            true_values = true_values.unsqueeze(1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        # Collect sample-level latent representations for UMAP visualization
        sample_latents = {}

        # Use pre-computed H_pooled (already combines h_CLS with H_genes_pert)
        if "H_pooled" in latents:
            sample_latents["H_pooled"] = latents["H_pooled"]

        # Use our updated visualize_model_outputs method
        vis.visualize_model_outputs(
            predictions,
            true_values,
            sample_latents,
            loss_name,
            self.current_epoch,
            None,
            stage=stage,
        )

        # Log oversmoothing metrics on latent spaces if available
        if "z_p" in latents:
            smoothness = VisGraphDegen.compute_smoothness(latents["z_p"])
            wandb.log({f"{stage}/oversmoothing_z_p": smoothness.item()})

        # Log genetic interaction box plot
        if torch.any(~torch.isnan(true_values)):
            # For gene interactions, values are in the first dimension
            fig_gi = genetic_interaction_score.box_plot(
                true_values[:, 0].cpu(), predictions[:, 0].cpu()
            )
            wandb.log({f"{stage}/gene_interaction_box_plot": wandb.Image(fig_gi)})
            plt.close(fig_gi)

    def on_train_epoch_end(self):
        # Log training metrics
        computed_metrics = self._compute_metrics_safely(self.train_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.train_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_transformed_metrics.reset()

        # Plot training samples
        if (
            self.current_epoch + 1
        ) % self.hparams.plot_every_n_epochs == 0 and self.train_samples["true_values"]:
            self._plot_samples(self.train_samples, "train_sample")
            # Reset the sample containers
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

        # Step the scheduler when using manual optimization
        sch = self.lr_schedulers()
        if sch is not None:
            # Lightning returns a list of schedulers even if there's only one
            if isinstance(sch, list) and len(sch) > 0:
                sch[0].step()
            else:
                sch.step()

        # CRITICAL: Clear GPU memory at end of training epoch
        # This ensures validation starts with maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_start(self):
        # Update gradient accumulation steps based on current epoch
        if self.hparams.grad_accumulation_schedule is not None:
            for epoch_threshold in sorted(
                self.hparams.grad_accumulation_schedule.keys()
            ):
                # Convert epoch_threshold to int if it's a string
                epoch_threshold_int = (
                    int(epoch_threshold)
                    if isinstance(epoch_threshold, str)
                    else epoch_threshold
                )
                if self.current_epoch >= epoch_threshold_int:
                    self.current_accumulation_steps = (
                        self.hparams.grad_accumulation_schedule[epoch_threshold]
                    )
            print(
                f"Epoch {self.current_epoch}: Using gradient accumulation steps = {self.current_accumulation_steps}"
            )

        # ALWAYS clear sample containers at epoch start to prevent memory accumulation
        # (unconditional - fixes OOM from stale samples persisting across epochs)
        self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_start(self):
        # CRITICAL: Aggressively clear GPU memory before validation starts
        # This prevents OOM when transitioning from training to validation
        # Training state (optimizer, gradients, cached activations) can fragment memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all pending operations complete before validation
            torch.cuda.synchronize()

        # Reset edge recovery accumulators and attention diagnostics
        self.reset_edge_recovery_accumulators()
        self.reset_attention_diagnostics()

        # ALWAYS clear sample containers at validation start to prevent memory accumulation
        # (unconditional - fixes OOM from stale samples persisting across epochs)
        self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_start(self):
        # Always clear sample containers for test (test runs only once)
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_end(self):
        # Log validation metrics
        computed_metrics = self._compute_metrics_safely(self.val_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(self.val_transformed_metrics)
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_transformed_metrics.reset()

        # Log edge recovery metrics (now includes layer and head info)
        for metric_key, acc in self.edge_recovery_accumulators.items():
            # Recall@degree
            if acc["count_nodes_deg"] > 0:
                recall_deg = acc["sum_recall_deg"] / acc["count_nodes_deg"]
                self.log(
                    f"val_edge_recovery/{metric_key}/recall_at_deg",
                    recall_deg,
                    sync_dist=True,
                )

            # Precision@k for each k
            for k in self.edge_recovery_ks:
                if acc["count_nodes_prec"][k] > 0:
                    prec_k = acc["sum_prec"][k] / acc["count_nodes_prec"][k]
                    self.log(
                        f"val_edge_recovery/{metric_key}/precision_k{k}",
                        prec_k,
                        sync_dist=True,
                    )

        # TIER 1: Plot basic transformer diagnostics (cheap - controlled by separate frequency)
        plot_transformer_freq = self.hparams.get(
            "plot_transformer_diagnostics_every_n_epochs", 10
        )
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch + 1) % plot_transformer_freq == 0
        ):
            if self.attention_stats_accumulators:
                self._plot_attention_diagnostics()

        # TIER 2: Plot edge recovery + degree-bias (medium cost - less frequent)
        if (
            not self.trainer.sanity_checking
            and hasattr(self.hparams, "plot_edge_recovery_every_n_epochs")
            and self.hparams.plot_edge_recovery_every_n_epochs is not None
            and self.hparams.plot_edge_recovery_every_n_epochs > 0
            and (self.current_epoch + 1)
            % self.hparams.plot_edge_recovery_every_n_epochs
            == 0
        ):
            if self.edge_recovery_accumulators:
                self._plot_edge_recovery_metrics()  # Includes degree-bias

        # Reset all accumulators
        self.reset_edge_recovery_accumulators()
        self.reset_attention_diagnostics()

        # Explicit GPU memory cleanup after validation epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Plot validation samples
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
            and self.val_samples["true_values"]
        ):
            self._plot_samples(self.val_samples, "val_sample")
            # Reset the sample containers
            self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_end(self):
        # Log test metrics
        computed_metrics = self._compute_metrics_safely(self.test_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.test_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_transformed_metrics.reset()

        # Plot test samples
        if self.test_samples["true_values"]:
            self._plot_samples(self.test_samples, "test_sample")
            # Reset the sample containers
            self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self.hparams.optimizer_config.items() if k != "type"
        }
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # If no lr_scheduler_config is provided, return just the optimizer
        if self.hparams.lr_scheduler_config is None:
            return optimizer

        # Handle different scheduler types
        scheduler_type = self.hparams.lr_scheduler_config.get(
            "type", "ReduceLROnPlateau"
        )
        scheduler_params = {
            k: v for k, v in self.hparams.lr_scheduler_config.items() if k != "type"
        }

        if scheduler_type == "CosineAnnealingWarmupRestarts":
            # Import the custom scheduler
            from torchcell.scheduler.cosine_annealing_warmup import (
                CosineAnnealingWarmupRestarts,
            )

            scheduler = CosineAnnealingWarmupRestarts(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # Default to ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/gene_interaction/MSE",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
