import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import numpy as np
from torchcell.timestamp import timestamp
from torchcell.timestamp import timestamp


def plot_correlations(
    predictions,
    true_values,
    save_path,
    lambda_info="",
    weight_decay="",
    fixed_axes=None,
    epoch=None,
):
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

    # Calculate metrics only if we have enough data points
    if len(x_fitness) >= 2:
        pearson_fitness, _ = stats.pearsonr(x_fitness, y_fitness)
        spearman_fitness, _ = stats.spearmanr(x_fitness, y_fitness)
    else:
        pearson_fitness = float("nan")
        spearman_fitness = float("nan")

    mse_fitness = (
        np.mean((y_fitness - x_fitness) ** 2) if len(x_fitness) > 0 else float("nan")
    )

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
        if len(x_fitness) > 0:
            min_val = min(min(x_fitness), min(y_fitness))
            max_val = max(max(x_fitness), max(y_fitness))
            # Add 10% padding
            range_val = max_val - min_val
            min_val -= range_val * 0.1
            max_val += range_val * 0.1
        else:
            min_val, max_val = -1, 1  # Default if no data

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

    # Calculate metrics only if we have enough data points
    if len(x_gi) >= 2:
        pearson_gi, _ = stats.pearsonr(x_gi, y_gi)
        spearman_gi, _ = stats.spearmanr(x_gi, y_gi)
    else:
        pearson_gi = float("nan")
        spearman_gi = float("nan")

    mse_gi = np.mean((y_gi - x_gi) ** 2) if len(x_gi) > 0 else float("nan")

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
        if len(x_gi) > 0:
            min_val = min(min(x_gi), min(y_gi))
            max_val = max(max(x_gi), max(y_gi))
            # Add 10% padding
            range_val = max_val - min_val
            min_val -= range_val * 0.1
            max_val += range_val * 0.1
        else:
            min_val, max_val = -1, 1  # Default if no data

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
    save_dir="./003-fit-int/hetero_cell/embedding_plots",
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
