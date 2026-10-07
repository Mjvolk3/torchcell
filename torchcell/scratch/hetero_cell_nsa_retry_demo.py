# torchcell/scratch/hetero_cell_nsa_retry_demo
# [[torchcell.scratch.hetero_cell_nsa_retry_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/hetero_cell_nsa_retry_demo
"""Run a manual smoke test of the model from the Hydra config.

Moved verbatim from torchcell/models/hetero_cell_nsa_retry.py on 2026-10-06 (test
campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/hetero_cell_nsa_retry_demo.py

Hydra reads the same experiments/006-kuzmin-tmi/conf directory as before, because
config_path resolves against os.getcwd(). Needs DATA_ROOT and ASSET_IMAGES_DIR in
.env, the 005 sample batch (torchcell.scratch.load_batch_005), and a GPU when
cfg.trainer.accelerator is gpu.
"""

import os
import os.path as osp

import hydra
import torch
from omegaconf import DictConfig

from torchcell.models.hetero_cell_nsa_retry import HeteroCellNSA


@hydra.main(
    version_base=None,
    config_path=osp.join(os.getcwd(), "experiments/006-kuzmin-tmi/conf"),
    config_name="hetero_cell_nsa_retry",
)
def main(cfg: DictConfig) -> None:
    """Run a manual smoke test of the model from the Hydra config."""
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv

    from torchcell.losses.isomorphic_cell_loss import ICLoss
    from torchcell.scratch.load_batch_005 import load_sample_data_batch
    from torchcell.timestamp import timestamp

    # Visualization functions need update for single target
    # from torchcell.scratch.cell_batch_overfit_visualization import (
    #     plot_embeddings,
    #     plot_correlations,
    # )
    from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator.lower() == "gpu"
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Load data with dense mask transformation
    dataset, batch, _, _ = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        config="hetero_cell_bipartite",
        is_dense=True,  # This applies dense transformation to the dataset
    )

    # Get the cell_graph from the dataset

    dense_transform = HeteroToDenseMask(
        {"gene": 6607, "reaction": 7122, "metabolite": 2806}
    )
    dataset.transform = dense_transform
    cell_graph = dense_transform(dataset.cell_graph)

    # Verify cell_graph has the necessary dense mask attributes
    for edge_type in cell_graph.edge_types:
        src, rel, dst = edge_type
        if src != dst:  # Skip self-loops
            if (
                "adj_mask" not in cell_graph[edge_type]
                and "inc_mask" not in cell_graph[edge_type]
            ):
                print(
                    f"Warning: Cell graph edge type {edge_type} missing dense mask attributes."
                )
                # Apply dense transformation explicitly if needed
                dense_transform = HeteroToDenseMask(
                    {
                        "gene": cfg.model.gene_num,
                        "reaction": cfg.model.reaction_num,
                        "metabolite": cfg.model.metabolite_num,
                    }
                )
                cell_graph = dense_transform(cell_graph)
                print("Applied dense mask transformation to cell_graph.")
                break

    # Verify batch also has mask attributes
    for edge_type in batch.edge_types:
        src, rel, dst = edge_type
        if src != dst:  # Skip self-loops
            if (
                "adj_mask" not in batch[edge_type]
                and "inc_mask" not in batch[edge_type]
            ):
                print(
                    f"Warning: Batch edge type {edge_type} missing dense mask attributes!"
                )

    # Move to device
    cell_graph = cell_graph.to(device)
    batch = batch.to(device)

    # Print dimensions for verification
    print("\nVerifying data dimensions:")
    print("Cell graph dimensions:")
    for node_type in cell_graph.node_types:
        print(f"  {node_type}: {cell_graph[node_type].num_nodes} nodes")

    print("Batch dimensions:")
    for node_type in batch.node_types:
        print(f"  {node_type}: {batch[node_type].num_nodes} nodes")

    print("\nModel configuration:")
    print(f"  gene_num: {cfg.model.gene_num}")
    print(f"  reaction_num: {cfg.model.reaction_num}")
    print(f"  metabolite_num: {cfg.model.metabolite_num}")
    print(f"  hidden_channels: {cfg.model.hidden_channels}")
    print(f"  attention_pattern: {cfg.model.attention_pattern}")

    # Initialize model with verified dimensions
    model = HeteroCellNSA(
        gene_num=cfg.model.gene_num,
        reaction_num=cfg.model.reaction_num,
        metabolite_num=cfg.model.metabolite_num,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=1,  # Only gene interaction prediction
        attention_pattern=cfg.model.attention_pattern,
        num_heads=cfg.model.heads,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        prediction_head_config=cfg.model.prediction_head_config,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass before training
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            predictions, representations = model(cell_graph, batch)
            print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return  # Exit if forward pass fails

    # Training target - gene interaction in COO format
    # The phenotype_values contains the gene interaction scores
    y = batch["gene"].phenotype_values

    # Set up loss function - only single weight for gene interaction
    if cfg.regression_task.is_weighted_phenotype_loss:
        weights = torch.ones(1).to(device)
    else:
        weights = None

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

    # Setup directories for plots
    embeddings_dir = osp.join(ASSET_IMAGES_DIR, "embedding_plots")
    os.makedirs(embeddings_dir, exist_ok=True)
    correlation_dir = osp.join(ASSET_IMAGES_DIR, "correlation_plots")
    os.makedirs(correlation_dir, exist_ok=True)

    # Training preparation
    model.train()
    losses = []
    num_epochs = cfg.trainer.max_epochs

    # Initial visualization - skip for now since visualization expects 2D
    # TODO: Update visualization functions for single target

    # Training loop
    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions, representations = model(cell_graph, batch)
            # Reshape y and predictions for loss calculation
            y_reshaped = y.unsqueeze(1) if y.dim() == 1 else y
            loss, loss_components = criterion(
                predictions, y_reshaped, representations["z_p"]
            )

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"Loss: {loss.item():.4f}")
                print("Loss components:", loss_components)
                # Skip visualization for now - needs update for single target
                # TODO: Update visualization functions for single target
                pass
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
            print(
                "GPU memory may be insufficient. Consider reducing batch size or model size."
            )
        raise

    # Plot final training loss
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
        osp.join(ASSET_IMAGES_DIR, f"hetero_cell_nsa_training_loss_{timestamp()}.png")
    )
    plt.close()

    # Clean up GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
