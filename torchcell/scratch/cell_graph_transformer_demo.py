# torchcell/scratch/cell_graph_transformer_demo
# [[torchcell.scratch.cell_graph_transformer_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/cell_graph_transformer_demo
"""Main training function for overfitting test.

Moved verbatim from torchcell/models/cell_graph_transformer.py on 2026-10-06 (test
campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/cell_graph_transformer_demo.py

Hydra reads the same experiments/006-kuzmin-tmi/conf directory as before, because
config_path resolves against os.getcwd(). Needs DATA_ROOT, EXPERIMENT_ROOT and
ASSET_IMAGES_DIR in .env, the 006-kuzmin-tmi 001-small-build dataset and the
genome, GO, network and yeast-GEM data under DATA_ROOT (loaded through
torchcell.scratch.load_batch_006_perturbation), and a GPU when the config asks for
one (it falls back to the CPU otherwise).
"""

import os
import os.path as osp
from collections.abc import Sequence
from typing import cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from torchcell.models.cell_graph_transformer import (
    CellGraphTransformer,
    calculate_weight_l2_norm,
    compute_smoothness,
)


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="cell_graph_transformer",
)
def main(cfg: DictConfig) -> None:
    """Main training function for overfitting test."""
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from scipy import stats
    from scipy.stats import gaussian_kde

    from torchcell.losses.logcosh import LogCoshLoss
    from torchcell.scratch.load_batch_006_perturbation import load_perturbation_batch
    from torchcell.timestamp import timestamp

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    dataset, batch, cell_graph, gene_set_size = load_perturbation_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        subset_size=cfg.data_module.perturbation_subset_size,
        device=device,
    )

    cell_graph = cell_graph.to(device)
    batch = batch.to(device)

    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing model...")
    print("=" * 80)

    # Extract gene-gene edge types from cell_graph
    print("\nCell graph edge types:")
    for edge_type in cell_graph.edge_types:
        src, rel, dst = edge_type
        if src == "gene" and dst == "gene":
            print(f"  {edge_type}: {cell_graph[edge_type].num_edges} edges")

    model = CellGraphTransformer(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_transformer_layers=cfg.model.num_transformer_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        cell_graph=cell_graph,
        graph_regularization_config=cfg.model.graph_regularization,
        perturbation_head_config=cfg.model.perturbation_head,
        dropout=cfg.model.dropout,
        adaptive_loss_weighting=cfg.model.get("adaptive_loss_weighting", False),
        graph_reg_scale=cfg.model.get("graph_reg_scale", 0.001),
    ).to(device)

    print("\nModel architecture:")
    print(model)
    param_counts = model.num_parameters
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Setup loss and optimizer
    criterion = LogCoshLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.regression_task.optimizer.lr,
        weight_decay=cfg.regression_task.optimizer.weight_decay,
    )

    # Learning rate scheduler (optional)
    lr_scheduler = None
    if (
        hasattr(cfg.regression_task, "lr_scheduler")
        and cfg.regression_task.lr_scheduler is not None
    ):
        from torchcell.scheduler.cosine_annealing_warmup import (
            CosineAnnealingWarmupRestarts,
        )

        scheduler_config = cfg.regression_task.lr_scheduler
        if scheduler_config.type == "CosineAnnealingWarmupRestarts":
            lr_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=scheduler_config.first_cycle_steps,
                cycle_mult=scheduler_config.get("cycle_mult", 1.0),
                max_lr=scheduler_config.max_lr,
                min_lr=scheduler_config.min_lr,
                warmup_steps=scheduler_config.warmup_steps,
                gamma=scheduler_config.get("gamma", 1.0),
            )
            print("Using CosineAnnealingWarmupRestarts scheduler")
        else:
            print(f"Warning: Unknown scheduler type {scheduler_config.type}")
    else:
        print("Using constant learning rate (no scheduler)")

    # Training target
    y = batch["gene"].phenotype_values.to(device)

    # Setup directory for plots
    plot_dir = osp.join(ASSET_IMAGES_DIR, f"cell_graph_transformer_{timestamp()}")
    os.makedirs(plot_dir, exist_ok=True)

    def save_intermediate_plot(
        epoch: int,
        losses: list[float],
        pred_losses: list[float],
        graph_reg_losses: list[float],
        correlations: list[float],
        spearman_correlations: list[float],
        mses: list[float],
        maes: list[float],
        rmses: list[float],
        learning_rates: list[float],
        weight_l2_norms: list[float],
        smoothness_history: list[float],
        cfg: DictConfig,
        model: CellGraphTransformer,
        cell_graph: HeteroData,
        batch: HeteroData,
        y: torch.Tensor,
    ) -> None:
        """Save intermediate training plot every print interval."""
        plt.figure(figsize=(20, 12))

        # ROW 1: Total Loss, Prediction Loss, Graph Reg Loss
        plt.subplot(3, 4, 1)
        plt.plot(range(1, epoch + 2), losses, "b-", label="Total Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Total Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale("log")

        plt.subplot(3, 4, 2)
        plt.plot(
            range(1, epoch + 2),
            pred_losses,
            "orange",
            label="Prediction Loss",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Prediction Loss (LogCosh)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale("log")

        plt.subplot(3, 4, 3)
        plt.plot(
            range(1, epoch + 2),
            graph_reg_losses,
            "green",
            label="Graph Reg Loss",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Graph Regularization Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale("log")

        # ROW 1 cont: Correlations
        plt.subplot(3, 4, 4)
        plt.plot(range(1, epoch + 2), correlations, "g-", label="Pearson", linewidth=2)
        if spearman_correlations:
            plt.plot(
                range(1, epoch + 2),
                spearman_correlations,
                "b--",
                label="Spearman",
                linewidth=2,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.title("Correlation Evolution")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1)

        # ROW 2: Error Metrics
        plt.subplot(3, 4, 5)
        epochs_range = range(1, epoch + 2)
        ax1 = plt.gca()
        ax1.plot(epochs_range, mses, "r-", label="MSE", linewidth=2)
        ax1.plot(epochs_range, rmses, "b-", label="RMSE", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE / RMSE")
        ax1.set_yscale("log")
        ax1.tick_params(axis="y")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(epochs_range, maes, "orange", label="MAE", linewidth=2)
        ax2.set_ylabel("MAE", color="orange")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="orange")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title("Error Metrics Evolution")

        # Get current predictions for visualization
        model.eval()
        with torch.no_grad():
            current_predictions, _ = model(cell_graph, batch)
            true_np = y.cpu().numpy()
            pred_np = current_predictions.squeeze().cpu().numpy()

            valid_mask = ~np.isnan(true_np)
            if np.sum(valid_mask) > 0:
                pred_std = np.std(pred_np[valid_mask])
                true_std = np.std(true_np[valid_mask])

                if pred_std < 1e-8 or true_std < 1e-8:
                    current_corr = 0.0
                else:
                    try:
                        corr_matrix = np.corrcoef(
                            pred_np[valid_mask], true_np[valid_mask]
                        )
                        current_corr = corr_matrix[0, 1]
                        if np.isnan(current_corr):
                            current_corr = 0.0
                    except Exception:
                        current_corr = 0.0
            else:
                current_corr = 0.0
        model.train()

        # Scatter plot
        plt.subplot(3, 4, 6)
        plt.scatter(pred_np[valid_mask], true_np[valid_mask], alpha=0.7)
        min_val = min(true_np[valid_mask].min(), pred_np[valid_mask].min())
        max_val = max(true_np[valid_mask].max(), pred_np[valid_mask].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Predictions vs Truth (r={current_corr:.4f})")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Distribution comparison with KDE
        plt.subplot(3, 4, 7)
        bins = np.linspace(
            min(true_np[valid_mask].min(), pred_np[valid_mask].min()),
            max(true_np[valid_mask].max(), pred_np[valid_mask].max()),
            30,
        )
        plt.hist(
            true_np[valid_mask],
            bins=cast(Sequence[float], bins),
            alpha=0.5,
            label="True",
            color="blue",
            density=True,
        )
        plt.hist(
            pred_np[valid_mask],
            bins=cast(Sequence[float], bins),
            alpha=0.5,
            label="Predicted",
            color="red",
            density=True,
        )

        # Add KDE
        if len(true_np[valid_mask]) > 1:
            try:
                kde_true = gaussian_kde(true_np[valid_mask])
                kde_pred = gaussian_kde(pred_np[valid_mask])
                x_range = np.linspace(
                    true_np[valid_mask].min(), true_np[valid_mask].max(), 200
                )
                plt.plot(
                    x_range, kde_true(x_range), "b-", linewidth=2, label="True KDE"
                )
                plt.plot(
                    x_range, kde_pred(x_range), "r-", linewidth=2, label="Pred KDE"
                )
            except Exception:
                pass

        plt.xlabel("Gene Interaction Score")
        plt.ylabel("Density")
        plt.title("Value Distributions")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning rate
        plt.subplot(3, 4, 8)
        plt.plot(range(1, epoch + 2), learning_rates, "purple", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # ROW 3: Model Configuration
        plt.subplot(3, 4, 9)
        plt.title("Model Configuration", pad=20)
        param_counts = model.num_parameters
        total_params = param_counts["total"]

        y_pos = 0.92
        params_text = [
            f"Total Parameters: {total_params:,}",
            f"Hidden Channels: {cfg.model.hidden_channels}",
            f"Transformer Layers: {cfg.model.num_transformer_layers}",
            f"Attention Heads: {cfg.model.num_attention_heads}",
            f"Dropout: {cfg.model.dropout}",
            f"Graph Reg Scale: {cfg.model.graph_reg_scale}",
            f"Weight Decay: {cfg.regression_task.optimizer.weight_decay}",
            f"Learning Rate: {cfg.regression_task.optimizer.lr}",
            f"Batch Size: {cfg.data_module.batch_size}",
        ]

        for i, text in enumerate(params_text):
            plt.text(
                0.05,
                y_pos - i * 0.09,
                text,
                transform=plt.gca().transAxes,
                fontsize=9,
                ha="left",
                va="top",
            )

        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # L2 Norm with Weighted Losses
        plt.subplot(3, 4, 10)
        epochs_range = range(1, len(weight_l2_norms) + 1)
        if len(weight_l2_norms) >= len(pred_losses):
            l2_norms = weight_l2_norms[: len(pred_losses)]
            weight_decay = cfg.regression_task.optimizer.weight_decay
            l2_penalty = [norm * weight_decay for norm in l2_norms]

            plt.plot(epochs_range, pred_losses, "b-", label="Pred Loss", linewidth=2)
            plt.plot(
                epochs_range,
                graph_reg_losses,
                "g-",
                label="Graph Reg Loss",
                linewidth=2,
            )
            plt.plot(
                epochs_range,
                l2_penalty,
                "purple",
                label=f"L2 Penalty (wd={weight_decay})",
                linewidth=2,
                linestyle="--",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Loss Component")
            plt.title("Loss Components with L2 Norm")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale("log")

        # Error histogram
        plt.subplot(3, 4, 11)
        errors = pred_np[valid_mask] - true_np[valid_mask]
        plt.hist(errors, bins=30, alpha=0.7, edgecolor="black", color="purple")
        plt.axvline(x=0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(
            f"Error Distribution (μ={np.mean(errors):.4f}, σ={np.std(errors):.4f})"
        )
        plt.grid(True, alpha=0.3)

        # Smoothness evolution (oversmoothing diagnostic)
        plt.subplot(3, 4, 12)
        if smoothness_history:
            epochs_range = range(1, len(smoothness_history) + 1)
            plt.plot(epochs_range, smoothness_history, "darkorange", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Smoothness (Frobenius Norm)")
            plt.title("Gene Embedding Smoothness\n↓ Lower = Oversmoothing")
            plt.grid(True, alpha=0.3)

            # Add current value annotation
            current_smoothness = smoothness_history[-1]
            plt.text(
                0.95,
                0.95,
                f"Current: {current_smoothness:.2f}",
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Add warning zone if smoothness is very low (< 10% of initial)
            if len(smoothness_history) > 1:
                initial_smoothness = smoothness_history[0]
                if current_smoothness < 0.1 * initial_smoothness:
                    plt.axhline(
                        y=0.1 * initial_smoothness,
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.5,
                    )
                    plt.text(
                        0.05,
                        0.15,
                        "⚠ Oversmoothing",
                        transform=plt.gca().transAxes,
                        color="red",
                        fontsize=10,
                        weight="bold",
                    )

        plt.suptitle(
            f"Cell Graph Transformer Training - Epoch {epoch + 1}/{cfg.trainer.max_epochs}",
            fontsize=16,
            y=0.998,
        )

        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"training_epoch_{epoch + 1:04d}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    print(f"Batch size: {y.size(0)}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print(f"Plot directory: {plot_dir}")

    # Training loop - Initialize tracking lists
    losses = []
    pred_losses = []
    graph_reg_losses = []
    correlations = []
    spearman_correlations = []
    mses = []
    maes = []
    rmses = []
    learning_rates = []
    weight_l2_norms = []
    smoothness_history = []

    plot_interval = cfg.regression_task.plot_every_n_epochs

    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions, representations = model(cell_graph, batch)

        # Compute smoothness of gene embeddings (oversmoothing diagnostic)
        with torch.no_grad():
            H_genes = representations["H_genes"]  # [N, d]
            smoothness = compute_smoothness(H_genes)
            smoothness_history.append(smoothness)

        # Compute losses
        pred_loss = criterion(predictions.squeeze(), y)
        graph_reg_loss = representations["graph_reg_loss"]

        # Apply adaptive weighting if enabled
        if model.adaptive_loss_weighting:
            mse_weight = model.log_mse_weight.exp()
            reg_weight = model.log_reg_weight.exp()
            # Normalize weights
            weight_sum = mse_weight + reg_weight
            mse_weight = 2 * mse_weight / weight_sum
            reg_weight = 2 * reg_weight / weight_sum
            total_loss = mse_weight * pred_loss + reg_weight * graph_reg_loss
        else:
            total_loss = pred_loss + graph_reg_loss

        # Compute metrics before backward pass
        with torch.no_grad():
            pred_np = predictions.squeeze().cpu().numpy()
            y_np = y.cpu().numpy()
            valid_mask = ~np.isnan(y_np)

            if np.sum(valid_mask) > 0:
                pred_std = np.std(pred_np[valid_mask])
                y_std = np.std(y_np[valid_mask])

                # Pearson correlation
                if pred_std < 1e-8 or y_std < 1e-8:
                    corr = 0.0
                    spearman_corr = 0.0
                else:
                    try:
                        corr = np.corrcoef(pred_np[valid_mask], y_np[valid_mask])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0

                    try:
                        spearman_corr, _ = stats.spearmanr(
                            pred_np[valid_mask], y_np[valid_mask]
                        )
                        if np.isnan(spearman_corr):
                            spearman_corr = 0.0
                    except Exception:
                        spearman_corr = 0.0

                # Error metrics
                mse = np.mean((pred_np[valid_mask] - y_np[valid_mask]) ** 2)
                mae = np.mean(np.abs(pred_np[valid_mask] - y_np[valid_mask]))
                rmse = np.sqrt(mse)
            else:
                corr = 0.0
                spearman_corr = 0.0
                mse = float("inf")
                mae = float("inf")
                rmse = float("inf")

        # Track metrics
        losses.append(total_loss.item())
        pred_losses.append(pred_loss.item())
        graph_reg_losses.append(graph_reg_loss.item())
        correlations.append(corr)
        spearman_correlations.append(spearman_corr)
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Calculate L2 norm
        l2_norm = calculate_weight_l2_norm(model)
        weight_l2_norms.append(l2_norm)

        # Save intermediate plot
        if epoch % plot_interval == 0 or epoch == cfg.trainer.max_epochs - 1:
            save_intermediate_plot(
                epoch,
                losses,
                pred_losses,
                graph_reg_losses,
                correlations,
                spearman_correlations,
                mses,
                maes,
                rmses,
                learning_rates,
                weight_l2_norms,
                smoothness_history,
                cfg,
                model,
                cell_graph,
                batch,
                y,
            )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if cfg.regression_task.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.regression_task.clip_grad_norm_max_norm
            )

        optimizer.step()

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Print progress
        if epoch % plot_interval == 0:
            print(
                f"Epoch {epoch:4d}: "
                f"Loss={total_loss.item():.4f}, "
                f"Pred={pred_loss.item():.4f}, "
                f"GraphReg={graph_reg_loss.item():.4f}, "
                f"Corr={corr:.4f}, "
                f"Spearman={spearman_corr:.4f}, "
                f"MSE={mse:.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Final Pearson correlation: {correlations[-1]:.4f}")
    print(f"Final Spearman correlation: {spearman_correlations[-1]:.4f}")
    print(f"Final total loss: {losses[-1]:.4f}")
    print(f"Final prediction loss: {pred_losses[-1]:.4f}")
    print(f"Final graph reg loss: {graph_reg_losses[-1]:.4f}")
    print(f"Final MSE: {mses[-1]:.4f}")
    print(f"Final MAE: {maes[-1]:.4f}")
    print(f"Final RMSE: {rmses[-1]:.4f}")
    print(f"Final L2 norm: {weight_l2_norms[-1]:.4f}")
    print(f"Final smoothness: {smoothness_history[-1]:.2f}")
    print(f"\nAll training plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
