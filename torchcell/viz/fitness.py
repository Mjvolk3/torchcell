# torchcell/viz/fitness.py
# [[torchcell.viz.fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/viz/fitness.py
# Test file: torchcell/viz/test_fitness.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def box_plot(true_values, predictions) -> plt.Figure:
    # Convert input to numpy arrays
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

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

    # Define bins
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, float("inf")]

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

    # Compute tick values
    xticks = [str(round(bin_val, 2)) for bin_val in bins[:-1]]
    # Add 'inf' as the last tick label
    xticks.append("Inf")
    # Tick positions
    tick_positions = [i for i in range(len(bins))]

    # Plot the vertical grey lines
    # Set zorder to 0 to be behind boxes
    for pos in tick_positions:
        ax.axvline(x=pos, color="#838383", linewidth=0.8, zorder=0)

    # set the spine - outside box
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    # Set zorder to 1 to be above the horizontal line
    # Adjust for box width
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
        # Increase outline width
        patch.set_linewidth(2.2)
    for whisker in boxplots["whiskers"]:
        whisker.set_color("#D86B2B")
        # Increase whisker width
        whisker.set_linewidth(2.0)
    for median in boxplots["medians"]:
        median.set_color("#D86B2B")
        # Increase median width
        median.set_linewidth(4.0)
        # Get current x-values (endpoints) of the median line
        x = median.get_xdata()
        # Adjust the endpoints to reduce the width
        width_reduction = 0.026
        x[0] += width_reduction
        x[1] -= width_reduction
        # Set the modified x-values back to the median line
        median.set_xdata(x)

    # Add a black horizontal line at y=1.0
    # Set zorder to 0 to be behind boxes
    ax.axhline(y=1, color="black", linewidth=1.4, zorder=2)

    # Add a black vertical line at x=7.0 (1.0 on the tick label)
    # Set zorder to 2 to be above boxes
    ax.axvline(x=7.0, color="black", linewidth=1.4, zorder=2)

    # Add "(WT)" labels to x and y axes
    ax.text(-0.96, 0.995, "(WT)", fontsize=17.0, va="center", ha="right")
    ax.text(7.48, -0.08, "(WT)", fontsize=17.0, va="center", ha="right")

    # Set tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(xticks, ha="center")

    # Adjust x and y label positions
    ax.set_xlabel("Predicted growth", labelpad=34, size=17.0, fontname=font_name)
    ax.set_ylabel("Measured growth", labelpad=29, size=17.0, fontname=font_name)

    # Set y-axis limits and ticks
    ax.set_ylim(0.05, 1.25)
    # Show only till 1.2 with spacing of 0.1
    ax.set_yticks(np.arange(0.1, 1.3, 0.1))

    # Set the size of xticks and yticks
    ax.tick_params(axis="x", length=4, width=0, labelsize=16.0)
    ax.tick_params(axis="y", length=7, width=1.4, labelsize=16.0)

    # For x-ticks
    for label in ax.get_xticklabels():
        label.set_fontname(font_name)

    # For y-ticks
    for label in ax.get_yticklabels():
        label.set_fontname(font_name)

    # Update the title to include NaN percentage in scientific notation and handle NaN values correctly
    title = "Pearson: {}, Spearman: {}, R²: {}, NaN: {:.1e}%".format(
        f"{pearson_corr:.3f}" if not np.isnan(pearson_corr) else "N/A",
        f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "N/A",
        f"{r_squared:.3f}" if not np.isnan(r_squared) else "N/A",
        nan_percentage,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()

    return fig


def generate_simulated_data(n_samples=10000):
    # Generate true values
    true_values = np.random.normal(loc=0.8, scale=0.2, size=n_samples)
    true_values = np.clip(true_values, 0, 1.2)  # Clip values between 0 and 1.2

    # Generate predictions with some noise and bias
    predictions = true_values + np.random.normal(loc=0, scale=0.1, size=n_samples)
    predictions = np.clip(predictions, 0, 1.2)  # Clip values between 0 and 1.2

    # Add some outliers and wild type (WT) samples
    wt_samples = np.random.normal(loc=1, scale=0.05, size=n_samples // 20)
    wt_samples = np.clip(wt_samples, 0.9, 1.1)

    true_values = np.concatenate([true_values, wt_samples])
    predictions = np.concatenate(
        [
            predictions,
            wt_samples + np.random.normal(loc=0, scale=0.02, size=n_samples // 20),
        ]
    )

    return torch.tensor(true_values), torch.tensor(predictions)


def generate_simulated_data_with_nan(n_samples=10000):
    # Generate true values
    true_values = np.random.normal(loc=0.8, scale=0.2, size=n_samples)
    true_values = np.clip(true_values, 0, 1.2)  # Clip values between 0 and 1.2

    # Generate predictions with some noise and bias
    predictions = true_values + np.random.normal(loc=0, scale=0.1, size=n_samples)
    predictions = np.clip(predictions, 0, 1.2)  # Clip values between 0 and 1.2

    # Add some outliers and wild type (WT) samples
    wt_samples = np.random.normal(loc=1, scale=0.05, size=n_samples // 20)
    wt_samples = np.clip(wt_samples, 0.9, 1.1)

    true_values = np.concatenate([true_values, wt_samples])
    predictions = np.concatenate(
        [
            predictions,
            wt_samples + np.random.normal(loc=0, scale=0.02, size=n_samples // 20),
        ]
    )

    # Add a bin with identical values
    identical_values = np.full(n_samples // 50, 0.5)
    true_values = np.concatenate([true_values, identical_values])
    predictions = np.concatenate([predictions, identical_values])

    # Add some NaN values
    nan_mask = np.random.choice([True, False], size=true_values.shape, p=[0.01, 0.99])
    true_values[nan_mask] = np.nan
    predictions[nan_mask] = np.nan

    # Add some infinite values
    inf_mask = np.random.choice([True, False], size=true_values.shape, p=[0.005, 0.995])
    true_values[inf_mask] = np.inf
    predictions[inf_mask] = np.inf

    return torch.tensor(true_values), torch.tensor(predictions)


# def main():
#     true_values, predictions = generate_simulated_data()
#     fig = box_plot(true_values, predictions)
#     plt.show()


def main():
    true_values, predictions = generate_simulated_data_with_nan()
    fig = box_plot(true_values, predictions)
    plt.show()


if __name__ == "__main__":
    main()
