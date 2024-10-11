# torchcell/viz/genetic_interaction_score.py
# [[torchcell.viz.genetic_interaction_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/viz/genetic_interaction_score.py
# Test file: torchcell/viz/test_genetic_interaction_score.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def box_plot(true_values: torch.tensor, predictions: torch.tensor) -> plt.Figure:
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # mask = ~np.isnan(true_values) & ~np.isnan(predictions)
    mask = ~np.isnan(true_values)
    true_values = true_values[mask]
    predictions = predictions[mask]

    # Calculate correlations and R2
    pearson_corr, _ = stats.pearsonr(true_values, predictions)
    spearman_corr, _ = stats.spearmanr(true_values, predictions)
    r_squared = stats.linregress(predictions, true_values).rvalue ** 2

    # Define bins for the second image
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
    # width / height
    aspect_ratio = 1.18
    height = 6
    width = height * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=140)

    # Equally spaced box positions
    box_positions = [i + 0.5 for i in range(len(bins) - 1)]

    # Compute tick values with three decimal places
    xticks = [f"{bin_val:.2f}" for bin_val in bins[:-1]]
    # Add 'inf' as the first and last tick label
    xticks[0] = "-Inf"
    xticks.append("Inf")

    # Tick positions
    tick_positions = [i for i in range(len(bins))]

    # Plot the vertical grey lines
    for pos in tick_positions:
        ax.axvline(x=pos, color="#838383", linewidth=0.8, zorder=0)

    # Set spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    # Boxplots
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

    # Apply coloring
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
    # Add a vertical black line at x=0
    x_position_for_zero_bin = bins.index(0.00)
    ax.axvline(x=x_position_for_zero_bin, color="black", linewidth=1.4, zorder=2)
    # Set tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(xticks, ha="center", rotation=45, fontsize=16.0)
    # After setting your x-ticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    formatted_labels = [
        label.replace("-", "–") if "-" in label else label for label in labels
    ]
    ax.set_xticklabels(formatted_labels)

    # Adjust x and y label positions
    ax.set_xlabel(
        "Predicted genetic interaction", labelpad=8, size=17.0, fontname=font_name
    )
    ax.set_ylabel(
        "Measured genetic interaction", labelpad=8, size=17.0, fontname=font_name
    )

    # Set y-axis limits and ticks
    y_ticks = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
    # Leave half space above max and below min
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

    # Add title with correlation information
    title = f"Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}, R²: {r_squared:.3f}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

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


def main():
    true_values, predictions = generate_simulated_data()
    fig = box_plot(true_values, predictions)
    plt.show()


if __name__ == "__main__":
    main()
