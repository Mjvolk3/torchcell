# torchcell/scratch/hetero_cell_bipartite_dango_demo
# [[torchcell.scratch.hetero_cell_bipartite_dango_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/hetero_cell_bipartite_dango_demo
"""Instantiate the model from config and run a demonstration forward pass.

Moved verbatim from torchcell/models/hetero_cell_bipartite_dango.py on 2026-10-06
(test campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/hetero_cell_bipartite_dango_demo.py

Hydra reads the same experiments/003-fit-int/conf directory as before, because
config_path resolves against os.getcwd(). Needs DATA_ROOT and ASSET_IMAGES_DIR in
.env, the 003-fit-int 001-small-build sample batch and the genome, GO, network and
yeast-GEM data under DATA_ROOT (loaded through torchcell.scratch.load_batch), and a
GPU when the config asks for one (it falls back to the CPU otherwise).
"""

import os
import os.path as osp
from typing import cast

import hydra
import torch
from omegaconf import DictConfig

from torchcell.models.hetero_cell_bipartite_dango import HeteroCellBipartite


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/003-fit-int/conf"),
    config_name="hetero_cell_bipartite_dango",
)
def main(cfg: DictConfig) -> None:
    """Instantiate the model from config and run a demonstration forward pass."""
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv

    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.scratch.cell_batch_overfit_visualization import (
        plot_correlations,
        plot_embeddings,
    )
    from torchcell.scratch.load_batch import load_sample_data_batch
    from torchcell.timestamp import timestamp

    load_dotenv()
    ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
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
                        f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
                    )
                    print(
                        f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB"
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
