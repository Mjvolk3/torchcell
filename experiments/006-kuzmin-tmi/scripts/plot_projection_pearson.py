import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from os.path import join as osp_join
from dotenv import load_dotenv
from torchcell.timestamp import timestamp

# Load environment variables
load_dotenv(".env")
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")


def log_model(x, a, b, c):
    """Logarithmic model: y = a * log(b * x + 1) + c"""
    return a * np.log(b * x + 1) + c


# Load data
csv_path = osp_join(
    DATA_ROOT,
    "data",
    "torchcell",
    "experiments",
    "006-kuzmin-tmi",
    "projected_results",
    "wandb_export_2025-10-22T11_16_25.781-05_00.csv",
)
df = pd.read_csv(csv_path)

# Define the three groups
groups = {
    "Model 1 (5-7-1950725)": "Group: compute-5-7-1950725_efc2a9389ffc377f6cc4f29cbd4e3a6569b897198821c9f636a59f5000e4f57f - val/gene_interaction/Pearson",
    "Model 2 (3-3-1949733)": "Group: compute-3-3-1949733_c7fcc3112f121b5e62a25e07d7c1f3bbfbee04bb61b07dc904d3659f88417e4c - val/gene_interaction/Pearson",
    "Model 3 (3-3-1949734)": "Group: compute-3-3-1949734_5a0de3b25a522a8cc9bc7aed8dde903281d7b8351c7b68e77a11d4a67e5e63d7 - val/gene_interaction/Pearson",
}

# Extract data for each group
group_data = {}
for name, col_name in groups.items():
    # Get non-empty values for this group
    mask = df[col_name].notna() & (df[col_name] != "")
    steps = df.loc[mask, "Step"].values
    pearson = df.loc[mask, col_name].astype(float).values

    if len(steps) > 0 and len(pearson) > 0:
        group_data[name] = {"steps": steps, "pearson": pearson}
        print(f"{name}: {len(steps)} data points")
        print(f"  Step range: {steps.min()} - {steps.max()}")
        print(f"  Pearson range: {pearson.min():.4f} - {pearson.max():.4f}")

# Fit logarithmic models and make projections
fits = {}
projections = {}
steps_to_037 = {}

max_projection_step = (
    350000  # Project up to 350k steps to show all models reaching 0.37
)

for name, data in group_data.items():
    steps = data["steps"]
    pearson = data["pearson"]

    # Fit logarithmic model
    try:
        popt, pcov = curve_fit(
            log_model, steps, pearson, p0=[0.05, 0.0001, 0.2], maxfev=5000
        )
        fits[name] = popt

        # Create projection
        projection_steps = np.linspace(steps.min(), max_projection_step, 1000)
        projection_pearson = log_model(projection_steps, *popt)
        projections[name] = {"steps": projection_steps, "pearson": projection_pearson}

        # Find when model reaches 0.37 (if it does within projection range)
        if projection_pearson[-1] >= 0.37:
            # Find the step where pearson crosses 0.37
            idx = np.where(projection_pearson >= 0.37)[0]
            if len(idx) > 0:
                step_037 = projection_steps[idx[0]]
                steps_to_037[name] = step_037
                print(f"\n{name} reaches 0.37 Pearson at step {step_037:.0f}")
        else:
            # Extrapolate further to find when it might reach 0.37
            extended_steps = np.linspace(steps.min(), 1000000, 10000)
            extended_pearson = log_model(extended_steps, *popt)
            idx = np.where(extended_pearson >= 0.37)[0]
            if len(idx) > 0:
                step_037 = extended_steps[idx[0]]
                steps_to_037[name] = step_037
                print(
                    f"\n{name} reaches 0.37 Pearson at step {step_037:.0f} (extrapolated)"
                )
            else:
                print(f"\n{name} may not reach 0.37 within reasonable steps")

        # Print model parameters
        print(
            f"  Model: y = {popt[0]:.4f} * log({popt[1]:.6f} * x + 1) + {popt[2]:.4f}"
        )

    except Exception as e:
        print(f"Could not fit {name}: {e}")

# Create visualization
fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))

# Colors for each model
colors = {
    "Model 1 (5-7-1950725)": "blue",
    "Model 2 (3-3-1949733)": "red",
    "Model 3 (3-3-1949734)": "green",
}

# Plot 1: Full projection
for name in group_data.keys():
    if name in projections:
        # Plot actual data
        ax1.scatter(
            group_data[name]["steps"],
            group_data[name]["pearson"],
            color=colors[name],
            alpha=0.6,
            s=20,
            label=f"{name} (actual)",
        )

        # Plot projection
        ax1.plot(
            projections[name]["steps"],
            projections[name]["pearson"],
            color=colors[name],
            linestyle="--",
            linewidth=2,
            label=f"{name} (projection)",
        )

        # Mark current position (last actual data point)
        current_step = group_data[name]["steps"].max()
        current_pearson = group_data[name]["pearson"].max()

        # Add vertical line for current position
        ax1.axvline(
            x=current_step, color=colors[name], linestyle="-", alpha=0.3, linewidth=1
        )

        # Add large marker for current position
        ax1.scatter(
            [current_step],
            [current_pearson],
            color=colors[name],
            s=200,
            marker="o",
            zorder=6,
            edgecolors="black",
            linewidth=2,
        )

        # Add annotation for current step with better positioning
        if name == "Model 1 (5-7-1950725)":
            text_offset = (20000, -0.015)
            time_label = "13.5 days"
        elif name == "Model 2 (3-3-1949733)":
            text_offset = (20000, 0.008)
            time_label = "15 days"
        else:
            text_offset = (20000, -0.008)
            time_label = "15 days"

        ax1.annotate(
            f"Current: {current_step:,}",
            xy=(current_step, current_pearson),
            xytext=(current_step + text_offset[0], current_pearson + text_offset[1]),
            fontsize=8,
            color=colors[name],
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=colors[name], alpha=0.7),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=colors[name],
                alpha=0.8,
            ),
        )

        # Add time label on the current vertical line
        ax1.text(
            current_step,
            0.25,
            time_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            color=colors[name],
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=colors[name],
                alpha=0.9,
            ),
        )

        # Mark 0.37 crossing point if it exists
        if name in steps_to_037:
            ax1.axvline(
                x=steps_to_037[name], color=colors[name], linestyle=":", alpha=0.5
            )
            ax1.scatter(
                [steps_to_037[name]],
                [0.37],
                color=colors[name],
                s=150,
                marker="*",
                zorder=5,
                edgecolors="black",
                linewidth=1,
            )

            # Add annotation for step count with non-overlapping positions
            if name == "Model 1 (5-7-1950725)":
                label_y = 0.376
            elif name == "Model 2 (3-3-1949733)":
                label_y = 0.381  # Higher for model 2
            else:  # Model 3
                label_y = 0.374  # Lower for model 3

            ax1.annotate(
                f"{steps_to_037[name]:,.0f}",
                xy=(steps_to_037[name], 0.37),
                xytext=(steps_to_037[name], label_y),
                ha="center",
                fontsize=9,
                color=colors[name],
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=colors[name],
                    alpha=0.8,
                ),
            )

# Add 0.37 target line
ax1.axhline(y=0.37, color="black", linestyle="-.", alpha=0.5, label="Target (0.37)")

ax1.set_xlabel("Training Steps", fontsize=14)
ax1.set_ylabel("Pearson Correlation", fontsize=14)
ax1.set_title(
    "Gene Interaction Pearson Correlation:\nCurrent Status & Projection to 0.37 Target",
    fontsize=16,
    pad=15,
)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max_projection_step)
ax1.set_ylim(0.2, 0.4)
# Add shaded region around target
ax1.axhspan(0.365, 0.375, alpha=0.1, color="gray", zorder=0)
plt.tight_layout()

# Save the figure
title = "experiments.006.hetero-cell-bipartite-dango-gi.2025-10-22-projection"
save_path = osp_join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to: {save_path}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY: Current Status and Projections to 0.37 Pearson Correlation")
print("=" * 70)
for name in sorted(steps_to_037.keys()):
    current_step = group_data[name]["steps"].max()
    current_max = group_data[name]["pearson"].max()
    remaining_steps = steps_to_037[name] - current_step
    percent_complete = (current_step / steps_to_037[name]) * 100

    print(f"\n{name}:")
    print(f"  Current Training Step: {current_step:,}")
    print(f"  Current Pearson: {current_max:.4f}")
    print(f"  Distance to 0.37: {0.37 - current_max:.4f}")
    print(f"  Projected steps to 0.37: {steps_to_037[name]:,.0f}")
    print(f"  Additional steps needed: {remaining_steps:,.0f}")
    print(f"  Progress to target: {percent_complete:.1f}%")

# Plot saved - no display needed
