# torchcell/scratch/dcell_opt_demo
# [[torchcell.scratch.dcell_opt_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/dcell_opt_demo
"""Train and evaluate the optimized DCell model on a sample batch.

Moved verbatim from torchcell/models/dcell_opt.py on 2026-10-06 (test campaign
Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/dcell_opt_demo.py

It is a hydra app; ``config_path`` resolves against ``os.getcwd()``, so it reads
the same ``experiments/006-kuzmin-tmi/conf`` directory as before
(``dcell_kuzmin2018_tmi``). The ``__main__`` block sets the ``spawn`` start
method. Needs ``DATA_ROOT`` in ``.env`` and the sample batch loaded by
``torchcell.scratch.load_batch_005.load_sample_data_batch``; it trains on a CUDA
GPU when one is available and the config's accelerator is not ``cpu``, and
writes training plots.
"""

import os
import os.path as osp
import time
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from torchcell.models.dcell_opt import DCellOpt
from torchcell.timestamp import timestamp


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="dcell_kuzmin2018_tmi",
)
def main(cfg: DictConfig) -> tuple[nn.Module, tuple[torch.Tensor, dict[str, Any]]]:
    """Train and evaluate the optimized DCell model on a sample batch."""
    import torch.optim as optim
    from tqdm import tqdm

    from torchcell.losses.dcell import DCellLoss
    from torchcell.scratch.load_batch_005 import load_sample_data_batch

    load_dotenv()

    # Set device based on config
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() != "cpu"
        else "cpu"
    )
    print(f"Using device: {device}")

    # Setup directories for plots
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    if ASSET_IMAGES_DIR is None:
        ASSET_IMAGES_DIR = "assets/images"

    plot_dir = osp.join(ASSET_IMAGES_DIR, f"dcell_opt_training_{timestamp()}")
    os.makedirs(plot_dir, exist_ok=True)

    def save_intermediate_plot(
        epoch: int,
        all_losses: list[float],
        primary_losses: list[float],
        auxiliary_losses: list[float],
        weighted_auxiliary_losses: list[float],
        learning_rates: list[float],
        model: nn.Module,
        batch: HeteroData,
    ) -> None:
        """Save intermediate training plot every print interval."""
        plt.figure(figsize=(12, 8))

        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(range(1, epoch + 2), all_losses, "b-", label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Loss components
        plt.subplot(2, 3, 2)
        plt.plot(range(1, epoch + 2), primary_losses, "r-", label="Primary Loss")
        plt.plot(range(1, epoch + 2), auxiliary_losses, "g-", label="Auxiliary Loss")
        plt.plot(
            range(1, epoch + 2),
            weighted_auxiliary_losses,
            "orange",
            label="Weighted Auxiliary Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Loss Components")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")

        # Get current model predictions for correlation plot
        model.eval()
        with torch.no_grad():
            current_predictions, current_outputs = model(cell_graph, batch)
            true_scores = batch["gene"].phenotype_values

            # Convert to numpy for plotting
            true_np = true_scores.cpu().numpy()
            pred_np = current_predictions.cpu().numpy()

            # Calculate correlation
            correlation = (
                np.corrcoef(true_np, pred_np)[0, 1] if len(true_np) > 1 else 0.0
            )
            mse = np.mean((pred_np - true_np) ** 2)
        model.train()  # Back to training mode

        # Correlation
        plt.subplot(2, 3, 3)
        plt.scatter(true_np, pred_np, alpha=0.7)
        min_val = min(true_np.min(), pred_np.min())
        max_val = max(true_np.max(), pred_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("True Phenotype Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Correlation (r={correlation:.4f})")
        plt.grid(True)

        # Error distribution
        plt.subplot(2, 3, 4)
        errors = pred_np - true_np
        plt.hist(errors, bins=10, alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"Error Distribution (MSE={mse:.6f})")
        plt.grid(True)

        # Learning rate evolution
        plt.subplot(2, 3, 5)
        plt.plot(range(1, epoch + 2), learning_rates, "purple")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.yscale("log")

        # Subsystem analysis
        plt.subplot(2, 3, 6)
        linear_outputs = current_outputs.get("linear_outputs", {})
        subsystem_activities = {}
        for k, v in linear_outputs.items():
            if k != "GO:ROOT":
                subsystem_activities[k] = v.abs().mean().item()

        if subsystem_activities:
            subsystem_activations = list(subsystem_activities.values())
            plt.hist(
                subsystem_activations,
                bins=min(20, len(subsystem_activations)),
                alpha=0.7,
            )
            plt.xlabel("Mean Absolute Activation")
            plt.ylabel("Number of Subsystems")
            plt.title("Subsystem Activation Distribution")
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            osp.join(plot_dir, f"dcell_opt_epoch_{epoch + 1:04d}_{timestamp()}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # Load sample data
    print("Loading sample data with GO ontology...")

    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        config="dcell",
        is_dense=False,
    )

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Print batch information
    print(f"Batch size: {batch.num_graphs}")
    print(f"Gene nodes: {batch['gene'].num_nodes}")
    print(f"GO nodes: {batch['gene_ontology'].num_nodes}")
    print(f"Perturbation indices shape: {batch['gene'].perturbation_indices.shape}")
    print(
        f"GO gene strata state shape: {batch['gene_ontology'].go_gene_strata_state.shape}"
    )
    print(f"Phenotype values shape: {batch['gene'].phenotype_values.shape}")

    # Initialize optimized DCell model
    print("\nInitializing Optimized DCell model...")
    model = DCellOpt(
        cell_graph,
        min_subsystem_size=cfg.model.subsystem_output_min,
        subsystem_ratio=cfg.model.subsystem_output_max_mult,
        output_size=cfg.model.output_size,
    ).to(device)

    param_counts = model.num_parameters
    print("\nParameter counts:")
    print(f"  Subsystems: {param_counts['subsystems']:,}")
    print(f"  Linear heads: {param_counts['dcell_linear']:,}")
    print(f"  Total: {param_counts['total']:,}")

    # Test torch.compile if requested
    compile_mode = cfg.get("compile_mode", None)
    if compile_mode is not None:
        print(f"\nAttempting torch.compile with mode='{compile_mode}'...")
        try:
            # torch.compile returns an OptimizedModule callable, not DCellOpt.
            model = torch.compile(  # type: ignore[assignment]  # compiled module replaces DCellOpt
                model, mode=compile_mode, dynamic=True
            )
            print("✓ Successfully compiled model")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
            print("Continuing without compilation")

    # Initialize DCellLoss
    loss_func = DCellLoss(
        alpha=cfg.regression_task.dcell_loss.alpha,
        use_auxiliary_losses=cfg.regression_task.dcell_loss.use_auxiliary_losses,
        aux_reduction=cfg.regression_task.dcell_loss.aux_reduction,
    )

    # Create optimizer
    optimizer: optim.Optimizer
    if cfg.regression_task.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.regression_task.optimizer.lr,
            weight_decay=cfg.regression_task.optimizer.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.regression_task.optimizer.lr,
            weight_decay=cfg.regression_task.optimizer.weight_decay,
        )

    # Setup learning rate scheduler if specified
    scheduler = None
    if cfg.regression_task.lr_scheduler.type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.regression_task.lr_scheduler.mode,
            factor=cfg.regression_task.lr_scheduler.factor,
            patience=cfg.regression_task.lr_scheduler.patience,
            threshold=cfg.regression_task.lr_scheduler.threshold,
            min_lr=cfg.regression_task.lr_scheduler.min_lr,
        )

    # Training parameters - reduce epochs for testing
    epochs = min(cfg.trainer.max_epochs, 100)  # Limit to 100 for testing
    plot_interval = cfg.regression_task.plot_every_n_epochs

    # Lists to track metrics
    all_losses = []
    primary_losses = []
    auxiliary_losses = []
    weighted_auxiliary_losses = []
    learning_rates = []

    # Training loop
    print(f"\nTraining Optimized DCell for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        epoch_start_time = time.time()

        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions, outputs = model(cell_graph, batch)

        # Get targets
        targets = batch["gene"].phenotype_values

        # Compute loss using DCellLoss
        total_loss, loss_components = loss_func(predictions, outputs, targets)

        # Extract individual loss components
        primary_loss = loss_components["primary_loss"]
        auxiliary_loss = loss_components["auxiliary_loss"]
        weighted_auxiliary_loss = loss_components["weighted_auxiliary_loss"]

        # Record losses
        all_losses.append(total_loss.item())
        primary_losses.append(primary_loss.item())
        auxiliary_losses.append(auxiliary_loss.item())
        weighted_auxiliary_losses.append(weighted_auxiliary_loss.item())
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Backward pass and optimization
        total_loss.backward()

        # Gradient clipping if specified
        if cfg.regression_task.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.regression_task.clip_grad_norm_max_norm
            )

        optimizer.step()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(total_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Get GPU memory stats if using CUDA
        gpu_memory_str = ""
        if device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
            gpu_memory_str = f", GPU: {allocated_gb:.2f}/{reserved_gb:.2f}GB"

        # Print stats every epoch
        current_batch_size = batch["gene"].batch.max().item() + 1
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Total Loss: {total_loss.item():.6f}, "
            f"Primary Loss: {primary_loss.item():.6f}, "
            f"Aux Loss: {auxiliary_loss.item():.6f}, "
            f"Weighted Aux Loss: {weighted_auxiliary_loss.item():.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}, "
            f"Time: {epoch_time:.2f}s, "
            f"Time/instance: {epoch_time / current_batch_size:.4f}s"
            f"{gpu_memory_str}"
        )

        # Save intermediate plot at intervals
        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            save_intermediate_plot(
                epoch,
                all_losses,
                primary_losses,
                auxiliary_losses,
                weighted_auxiliary_losses,
                learning_rates,
                model,
                batch,
            )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_predictions, final_outputs = model(cell_graph, batch)

        print("\nFinal Optimized DCell training results:")
        if final_predictions.numel() > 0:
            true_scores = batch["gene"].phenotype_values

            print("True Phenotype Values:", true_scores.cpu().numpy()[:5])
            print("Predicted Values:", final_predictions.cpu().numpy()[:5])

            # Calculate final metrics
            mse = F.mse_loss(final_predictions, true_scores).item()
            mae = F.l1_loss(final_predictions, true_scores).item()

            # Calculate correlation coefficient
            true_np = true_scores.cpu().numpy()
            pred_np = final_predictions.cpu().numpy()
            correlation = np.corrcoef(true_np, pred_np)[0, 1] if len(true_np) > 1 else 0

            print(f"Final Mean Squared Error: {mse:.6f}")
            print(f"Final Mean Absolute Error: {mae:.6f}")
            print(f"Correlation Coefficient: {correlation:.6f}")

            print(f"\nResults plot saved to '{plot_dir}'")
        else:
            print("No predictions were generated. Check model and data setup.")

    print("\nOptimized DCell training demonstration complete!")

    return model, (final_predictions, final_outputs)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
