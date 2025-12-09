# torchcell/viz/genetic_interaction_score.py
# [[torchcell.viz.genetic_interaction_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/viz/genetic_interaction_score.py
# Test file: torchcell/viz/test_genetic_interaction_score.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def box_plot(true_values: torch.tensor, predictions: torch.tensor) -> plt.Figure:
    # Convert input to numpy arrays (convert to float32 first to handle BFloat16 from mixed precision)
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().float().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().float().numpy()

    # Calculate percentage of NaN predictions
    nan_percentage = np.isnan(predictions).mean() * 100

    mask = ~np.isnan(true_values) & ~np.isnan(predictions)
    true_values = true_values[mask]
    predictions = predictions[mask]

    # Calculate correlations and R2 with error handling
    try:
        pearson_corr, _ = stats.pearsonr(true_values, predictions)
    except ValueError:
        pearson_corr = np.nan

    try:
        spearman_corr, _ = stats.spearmanr(true_values, predictions)
    except ValueError:
        spearman_corr = np.nan

    try:
        r_squared = stats.linregress(predictions, true_values).rvalue ** 2
    except ValueError:
        r_squared = np.nan

    # Define bins for the genetic interaction score
    bins = [
        -float("inf"),
        -0.40,
        -0.32,
        -0.24,
        -0.16,
        -0.08,
        0.00,
        0.08,
        0.16,
        0.24,
        float("inf"),
    ]

    # font name
    font_name = "DejaVu Sans"

    # Bin predictions and collect corresponding true values
    binned_true_values = []

    for i in range(len(bins) - 1):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        binned_values = true_values[mask]
        binned_true_values.append(binned_values)

    # Create a box plot using matplotlib
    aspect_ratio = 1.18
    height = 6
    width = height * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=140)

    # Equally spaced box positions
    box_positions = [i + 0.5 for i in range(len(bins) - 1)]

    # Compute tick values
    xticks = [f"{bin_val:.2f}" for bin_val in bins[:-1]]
    xticks[0] = "-Inf"
    xticks.append("Inf")
    # Tick positions
    tick_positions = [i for i in range(len(bins))]

    # Plot the vertical grey lines
    for pos in tick_positions:
        ax.axvline(x=pos, color="#838383", linewidth=0.8, zorder=0)

    # set the spine - outside box
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    # Create the box plot
    boxplots = ax.boxplot(
        binned_true_values,
        patch_artist=True,
        vert=True,
        widths=0.98,
        positions=box_positions,
        showfliers=False,
        capwidths=0,
        zorder=1,
    )

    # Apply coloring to the boxes, whiskers, and medians
    for patch in boxplots["boxes"]:
        patch.set_facecolor("#F6A9A3")
        patch.set_edgecolor("#D86B2B")
        patch.set_linewidth(2.2)
    for whisker in boxplots["whiskers"]:
        whisker.set_color("#D86B2B")
        whisker.set_linewidth(2.0)
    for median in boxplots["medians"]:
        median.set_color("#D86B2B")
        median.set_linewidth(4.0)
        x = median.get_xdata()
        width_reduction = 0.026
        x[0] += width_reduction
        x[1] -= width_reduction
        median.set_xdata(x)

    # Add a black horizontal line at y=0
    ax.axhline(y=0, color="black", linewidth=1.4, zorder=2)
    # Add a vertical black line at x=6 (0.00 on the tick label)
    ax.axvline(x=6, color="black", linewidth=1.4, zorder=2)

    # Set tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(xticks, ha="center", rotation=45)

    # Adjust x and y label positions
    ax.set_xlabel(
        "Predicted genetic interaction", labelpad=8, size=17.0, fontname=font_name
    )
    ax.set_ylabel(
        "Measured genetic interaction", labelpad=8, size=17.0, fontname=font_name
    )

    # Set y-axis limits and ticks
    y_ticks = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
    y_min, y_max = (min(y_ticks) - 0.05, max(y_ticks) + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    # Set tick size
    ax.tick_params(axis="x", length=4, width=0, labelsize=16.0)
    ax.tick_params(axis="y", length=7, width=1.6, labelsize=16.0)

    # Font adjustments
    for label in ax.get_xticklabels():
        label.set_fontname(font_name)
    for label in ax.get_yticklabels():
        label.set_fontname(font_name)

    # Update the title to include NaN percentage in scientific notation and handle NaN values correctly
    # title = "Pearson: {}, Spearman: {}, R²: {}, NaN: {:.1e}%".format(
    #     f"{pearson_corr:.3f}" if not np.isnan(pearson_corr) else "N/A",
    #     f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "N/A",
    #     f"{r_squared:.3f}" if not np.isnan(r_squared) else "N/A",
    #     nan_percentage,
    # )
    title = "Pearson: {}, Spearman: {}, R²: {}".format(
        f"{pearson_corr:.3f}" if not np.isnan(pearson_corr) else "N/A",
        f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "N/A",
        f"{r_squared:.3f}" if not np.isnan(r_squared) else "N/A",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()

    return fig


def generate_simulated_data(n_samples=10000):
    # Generate true genetic interaction scores
    true_values = np.random.normal(loc=0, scale=0.2, size=n_samples)

    # Clip values to a reasonable range for genetic interactions
    true_values = np.clip(true_values, -0.8, 0.8)

    # Generate predictions with some noise and bias
    predictions = true_values + np.random.normal(loc=0, scale=0.1, size=n_samples)
    predictions = np.clip(predictions, -0.8, 0.8)

    # Add some extreme interactions (both positive and negative)
    extreme_samples = np.random.uniform(low=-0.8, high=0.8, size=n_samples // 50)
    extreme_predictions = extreme_samples + np.random.normal(
        loc=0, scale=0.05, size=n_samples // 50
    )

    true_values = np.concatenate([true_values, extreme_samples])
    predictions = np.concatenate([predictions, extreme_predictions])

    # Add some NaN values (about 1% of the data)
    nan_mask = np.random.choice([True, False], size=true_values.shape, p=[0.01, 0.99])
    true_values[nan_mask] = np.nan
    predictions[nan_mask] = np.nan

    return torch.tensor(true_values), torch.tensor(predictions)


def generate_simulated_data_with_nan(n_samples=10000):
    # Generate true genetic interaction scores
    true_values = np.random.normal(loc=0, scale=0.2, size=n_samples)
    true_values = np.clip(true_values, -0.8, 0.8)  # Clip values to a reasonable range

    # Generate predictions with some noise and bias
    predictions = true_values + np.random.normal(loc=0, scale=0.1, size=n_samples)
    predictions = np.clip(predictions, -0.8, 0.8)

    # Add some extreme interactions (both positive and negative)
    extreme_samples = np.random.uniform(low=-0.8, high=0.8, size=n_samples // 50)
    extreme_predictions = extreme_samples + np.random.normal(
        loc=0, scale=0.05, size=n_samples // 50
    )

    true_values = np.concatenate([true_values, extreme_samples])
    predictions = np.concatenate([predictions, extreme_predictions])

    # Add some NaN values (about 5% of the data)
    nan_mask = np.random.choice([True, False], size=true_values.shape, p=[0.05, 0.95])
    true_values[nan_mask] = np.nan
    predictions[nan_mask] = np.nan

    # Add some infinite values (about 1% of the data)
    inf_mask = np.random.choice([True, False], size=true_values.shape, p=[0.01, 0.99])
    true_values[inf_mask] = np.inf * np.sign(true_values[inf_mask])
    predictions[inf_mask] = np.inf * np.sign(predictions[inf_mask])

    # Add a bin with all identical values to potentially break correlation calculations
    identical_values = np.full(n_samples // 100, 0.1)
    true_values = np.concatenate([true_values, identical_values])
    predictions = np.concatenate([predictions, identical_values])

    # Add a bin with only two distinct values to potentially break certain statistical calculations
    two_value_bin = np.random.choice([0.2, 0.3], size=n_samples // 100)
    true_values = np.concatenate([true_values, two_value_bin])
    predictions = np.concatenate([predictions, two_value_bin])

    # Shuffle the arrays to mix up the special cases
    shuffle_idx = np.random.permutation(len(true_values))
    true_values = true_values[shuffle_idx]
    predictions = predictions[shuffle_idx]

    return torch.tensor(true_values), torch.tensor(predictions)


# def main():
#     true_values, predictions = generate_simulated_data_with_nan()
#     fig = box_plot(true_values, predictions)
#     plt.show()


def main():
    true_values, predictions = generate_simulated_data()
    fig = box_plot(true_values, predictions)
    plt.show()


if __name__ == "__main__":
    main()
