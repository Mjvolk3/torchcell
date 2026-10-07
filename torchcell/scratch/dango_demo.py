# torchcell/scratch/dango_demo
# [[torchcell.scratch.dango_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/dango_demo
"""Test the DANGO model by overfitting on a single batch.

Moved verbatim from torchcell/models/dango.py on 2026-10-06 (test campaign Phase
23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/dango_demo.py

It is a hydra app; ``config_path`` resolves against ``os.getcwd()``, so it reads
the same ``experiments/005-kuzmin2018-tmi/conf`` directory as before
(``dango_kuzmin2018_tmi``). Needs ``DATA_ROOT`` in ``.env`` and the sample batch
loaded by ``torchcell.scratch.load_batch_005.load_sample_data_batch``; it trains
on a CUDA GPU when one is available and the config's accelerator is not ``cpu``,
and writes loss and result plots.
"""

import os
import os.path as osp
from collections.abc import Callable
from typing import cast

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from torchcell.models.dango import Dango


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/005-kuzmin2018-tmi/conf"),
    config_name="dango_kuzmin2018_tmi",
)
def main(
    cfg: DictConfig,
) -> tuple["Dango", tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Test the DANGO model by overfitting on a single batch."""
    import os
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import torch.optim as optim
    from dotenv import load_dotenv

    # Import all scheduler types for easy toggling
    from torchcell.losses.dango import DangoLoss
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = f"dango_training_plots_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)

    # Create subdirectories for different plot types
    loss_dir = os.path.join(plot_dir, "loss_plots")
    correlation_dir = os.path.join(plot_dir, "correlation_plots")

    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(correlation_dir, exist_ok=True)

    # Load sample data
    print("Loading sample data...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=2, num_workers=2, config="dango_string9_1", is_dense=False
    )

    # Move data to device
    cell_graph = dataset.cell_graph.to(device)
    batch = batch.to(device)

    # Print batch information
    print(f"Batch size: {batch.num_graphs}")
    print(f"Perturbation indices shape: {batch['gene'].perturbation_indices.shape}")
    if hasattr(batch["gene"], "perturbation_indices_batch"):
        print(
            f"Perturbation batch indices shape: {batch['gene'].perturbation_indices_batch.shape}"
        )
    print(f"Phenotype values shape: {batch['gene'].phenotype_values.shape}")

    # Initialize model
    print("Initializing DANGO model...")
    # Define default STRING v9.1 edge types for demo
    edge_types = [
        "string9_1_neighborhood",
        "string9_1_fusion",
        "string9_1_cooccurence",
        "string9_1_coexpression",
        "string9_1_experimental",
        "string9_1_database",
    ]
    model = Dango(
        gene_num=max_num_nodes,
        edge_types=edge_types,
        hidden_channels=cfg.model.hidden_channels,
        num_heads=cfg.model.num_heads,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Num parameters: {model.num_parameters}")
    print(f"Using {model.hyper_sagnn.num_heads} attention heads in HyperSAGNN")

    # Use lambda values directly from the model's pretrain component
    lambda_values = model.pretrain_model.lambda_values.copy()

    # Set up training parameters from config
    epochs = cfg.trainer.max_epochs
    plot_interval = cfg.regression_task.plot_every_n_epochs
    transition_epoch = cfg.regression_task.loss_scheduler.transition_epoch

    # Get scheduler class from scheduler map
    scheduler_type = cfg.regression_task.loss_scheduler.type
    from torchcell.losses.dango import SCHEDULER_MAP, DangoLossSched

    if scheduler_type not in SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Create the scheduler instance with transition_epoch
    # SCHEDULER_MAP values are concrete subclasses; the inferred value type is the
    # abstract base DangoLossSched, so view the class as a concrete factory.
    scheduler_class = cast(Callable[..., DangoLossSched], SCHEDULER_MAP[scheduler_type])
    scheduler_kwargs = {"transition_epoch": transition_epoch}

    # No additional parameters needed for LinearUntilFlipped

    scheduler = scheduler_class(**scheduler_kwargs)
    print(f"Using {scheduler_type} scheduler with transition_epoch={transition_epoch}")

    # Initialize the DangoLoss module with the selected scheduler
    loss_func = DangoLoss(
        edge_types=model.pretrain_model.edge_types,
        lambda_values=lambda_values,
        scheduler=scheduler,
        reduction="mean",
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Lists to track metrics
    all_losses = []
    recon_losses = []
    interaction_losses = []
    weighted_recon_losses = []
    weighted_interaction_losses = []

    # Initialize validation metrics
    best_mse = float("inf")
    best_epoch = 0

    # Training loop
    print("Training to overfit on batch...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass - returns tuple of (interaction_scores, outputs)
        interaction_scores, outputs = model(cell_graph, batch)

        # Compute reconstruction loss for each edge type
        adjacency_matrices = {}
        for edge_type in model.pretrain_model.edge_types:
            edge_key = ("gene", edge_type, "gene")
            if edge_key in cell_graph.edge_types:
                adj_size = outputs["reconstructions"][edge_type].shape
                # Convert edge_index to dense adjacency matrix
                adjacency_matrices[edge_type] = torch.sparse_coo_tensor(
                    cell_graph[edge_key].edge_index,
                    torch.ones(cell_graph[edge_key].edge_index.shape[1], device=device),
                    (adj_size[0], adj_size[1]),
                ).to_dense()

        # Use the DangoLoss to compute the combined loss
        if interaction_scores.numel() > 0:
            total_loss, loss_dict = loss_func(
                predictions=interaction_scores,
                targets=batch["gene"].phenotype_values,
                reconstructions=outputs["reconstructions"],
                adjacency_matrices=adjacency_matrices,
                current_epoch=epoch,
            )

            # Extract individual losses from the loss dictionary
            recon_loss = loss_dict["reconstruction_loss"]
            interaction_loss = loss_dict["interaction_loss"]
            weighted_recon_loss = loss_dict["weighted_reconstruction_loss"]
            weighted_interaction_loss = loss_dict["weighted_interaction_loss"]
            alpha = loss_dict["alpha"]
        else:
            # If no interaction scores, just use reconstruction loss
            recon_loss = loss_func.compute_reconstruction_loss(
                outputs["reconstructions"], adjacency_matrices
            )
            total_loss = recon_loss
            interaction_loss = torch.tensor(0.0, device=device)
            alpha = torch.tensor(1.0, device=device)
            weighted_recon_loss = recon_loss
            weighted_interaction_loss = torch.tensor(0.0, device=device)

        # Record losses
        all_losses.append(total_loss.item())
        recon_losses.append(recon_loss.item())
        interaction_losses.append(interaction_loss.item())

        # Record weighted losses
        weighted_recon_losses.append(weighted_recon_loss.item())
        weighted_interaction_losses.append(weighted_interaction_loss.item())

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Print progress and generate plots at intervals
        if (epoch + 1) % plot_interval == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss.item():.4f}, "
                f"Recon Loss: {recon_loss.item():.4f}, Interaction Loss: {interaction_loss.item():.4f}, "
                f"Alpha: {alpha.item():.2f}, Weighted Recon: {weighted_recon_loss.item():.4f}, "
                f"Weighted Interaction: {weighted_interaction_loss.item():.4f}"
            )

            # Plot loss curves
            plt.figure(figsize=(14, 12))  # Increase height slightly
            plt.subplot(3, 2, 1)  # Change to 3x2 grid to match final plot
            plt.plot(range(1, epoch + 2), all_losses, "b-", label="Total Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Training Loss")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 3)
            plt.plot(
                range(1, epoch + 2), recon_losses, "r-", label="Reconstruction Loss"
            )
            plt.plot(
                range(1, epoch + 2), interaction_losses, "g-", label="Interaction Loss"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Unweighted Component Losses")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 5)
            plt.plot(
                range(1, epoch + 2),
                weighted_recon_losses,
                "r-",
                label="Weighted Reconstruction Loss",
            )
            plt.plot(
                range(1, epoch + 2),
                weighted_interaction_losses,
                "g-",
                label="Weighted Interaction Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title(f"Weighted Component Losses (alpha={alpha.item():.2f})")
            plt.grid(True)
            plt.legend()

            # The plot saving is now handled after adding the correlation plot

            # Evaluate on training data
            model.eval()
            with torch.no_grad():
                # Updated to unpack tuple
                predicted_scores, eval_outputs = model(cell_graph, batch)

                if predicted_scores.numel() > 0:
                    true_scores = batch["gene"].phenotype_values

                    # Calculate metrics
                    mse = F.mse_loss(predicted_scores, true_scores).item()
                    mae = F.l1_loss(predicted_scores, true_scores).item()

                    # Track best model
                    if mse < best_mse:
                        best_mse = mse
                        best_epoch = epoch + 1

                    # Add correlation plot to same figure in the right column
                    plt.subplot(3, 2, 2)
                    plt.scatter(
                        true_scores.cpu().numpy(),
                        predicted_scores.cpu().numpy(),
                        alpha=0.7,
                    )

                    # Get min/max for plot limits
                    min_val = min(
                        true_scores.min().item(), predicted_scores.min().item()
                    )
                    max_val = max(
                        true_scores.max().item(), predicted_scores.max().item()
                    )
                    plt.plot([min_val, max_val], [min_val, max_val], "r--")

                    plt.xlabel("True Interaction Scores")
                    plt.ylabel("Predicted Interaction Scores")
                    plt.title(f"Epoch {epoch + 1}: MSE={mse:.6f}, MAE={mae:.6f}")
                    plt.grid(True)

                    # Add error distribution
                    plt.subplot(3, 2, 4)
                    errors = predicted_scores.cpu().numpy() - true_scores.cpu().numpy()
                    plt.hist(errors, bins=20, alpha=0.7)
                    plt.xlabel("Prediction Error")
                    plt.ylabel("Frequency")
                    plt.title(f"Error Distribution (MSE={mse:.6f})")
                    plt.grid(True)

                    # Save the combined figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(loss_dir, f"loss_epoch_{epoch + 1}.png"))
                    plt.close()

                    # Also save separate correlation plot for specific directory
                    plt.figure(figsize=(10, 8))
                    plt.scatter(
                        true_scores.cpu().numpy(),
                        predicted_scores.cpu().numpy(),
                        alpha=0.7,
                    )
                    plt.plot([min_val, max_val], [min_val, max_val], "r--")
                    plt.xlabel("True Interaction Scores")
                    plt.ylabel("Predicted Interaction Scores")
                    plt.title(f"Epoch {epoch + 1}: MSE={mse:.6f}, MAE={mae:.6f}")
                    plt.grid(True)
                    plt.savefig(
                        os.path.join(
                            correlation_dir, f"correlation_epoch_{epoch + 1}.png"
                        )
                    )
                    plt.close()

                    # Network embedding visualization removed

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Updated to unpack tuple
        predicted_scores, final_outputs = model(cell_graph, batch)

        print("\nTraining results:")
        if predicted_scores.numel() > 0:
            true_scores = batch["gene"].phenotype_values

            print("True Phenotype Values:", true_scores.cpu().numpy())
            print("Predicted Scores:", predicted_scores.cpu().numpy())

            # Calculate final metrics
            mse = F.mse_loss(predicted_scores, true_scores).item()
            mae = F.l1_loss(predicted_scores, true_scores).item()

            # Calculate correlation coefficient
            true_np = true_scores.cpu().numpy()
            pred_np = predicted_scores.cpu().numpy()
            correlation = np.corrcoef(true_np, pred_np)[0, 1]

            print(f"Final Mean Squared Error: {mse:.6f}")
            print(f"Final Mean Absolute Error: {mae:.6f}")
            print(f"Correlation Coefficient: {correlation:.6f}")
            print(f"Best MSE: {best_mse:.6f} at epoch {best_epoch}")

            # Create a comprehensive final results plot
            plt.figure(figsize=(14, 14))

            # Plot loss curves
            plt.subplot(3, 2, 1)
            plt.plot(range(1, epochs + 1), all_losses, "b-", label="Total Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 3)
            plt.plot(
                range(1, epochs + 1), recon_losses, "r-", label="Reconstruction Loss"
            )
            plt.plot(
                range(1, epochs + 1), interaction_losses, "g-", label="Interaction Loss"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Unweighted Component Losses")
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 5)
            plt.plot(
                range(1, epochs + 1),
                weighted_recon_losses,
                "r-",
                label="Weighted Reconstruction Loss",
            )
            plt.plot(
                range(1, epochs + 1),
                weighted_interaction_losses,
                "g-",
                label="Weighted Interaction Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")
            plt.title("Weighted Component Losses")
            plt.grid(True)
            plt.legend()

            # Plot final correlation
            plt.subplot(3, 2, 2)
            plt.scatter(true_np, pred_np, alpha=0.7)
            min_val = min(true_np.min(), pred_np.min())
            max_val = max(true_np.max(), pred_np.max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--")
            plt.xlabel("True Interaction Scores")
            plt.ylabel("Predicted Interaction Scores")
            plt.title(f"Final Correlation (r={correlation:.4f})")
            plt.grid(True)

            # Plot error distribution
            plt.subplot(3, 2, 4)
            errors = pred_np - true_np
            plt.hist(errors, bins=20, alpha=0.7)
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(f"Error Distribution (MSE={mse:.6f})")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "final_results.png"))
            plt.close()

            print(
                f"\nFinal results plot saved to '{os.path.join(plot_dir, 'final_results.png')}'"
            )
        else:
            print("No interaction scores were predicted. Check batch format.")

    print("\nDemonstration complete!")
    print(f"All plots saved to directory: {plot_dir}")

    return model, (predicted_scores, final_outputs)


if __name__ == "__main__":
    main()
