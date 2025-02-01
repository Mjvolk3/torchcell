from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import os.path as osp
from datetime import datetime
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from matplotlib.colors import LinearSegmentedColormap


def create_custom_cmap():
    """Create a custom colormap with white at center (0.0)"""
    # Define colors for our custom map
    colors = ["darkred", "red", "white", "lightgreen", "darkgreen"]
    # Create evenly spaced positions for colors
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    return LinearSegmentedColormap.from_list(
        "custom_RdWtGn", list(zip(positions, colors))
    )


def actual_double_mutant(fi: float, fj: float) -> float:
    """Toy 'actual' double mutant fitness with some 'epistatic' shift."""
    return fi * fj + 0.1 * fi * (1 - fj)


def save_3d_scatter_plot(
    n: int = 50, save_filename_prefix: str = "3d_epistasis_scatter"
) -> None:
    """
    Saves a 3D scatter plot of fi, fj, and fij with epistasis as color using RdYlGn colormap.
    """
    # Load environment variables
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Ensure the output directory exists
    if not os.path.isdir(ASSET_IMAGES_DIR):
        os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    # Generate grid data for fi, fj, and fij
    fi_vals = np.linspace(0, 2, n)
    fj_vals = np.linspace(0, 2, n)
    fij_vals = np.linspace(0, 2, n)

    FI, FJ, Fij = np.meshgrid(fi_vals, fj_vals, fij_vals)
    FI = FI.ravel()
    FJ = FJ.ravel()
    Fij = Fij.ravel()

    # Calculate baseline and epistasis
    baseline = FI * FJ
    epistasis = Fij - baseline

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set fixed limits to match level sets plot
    vmin, vmax = -4, 4

    scatter = ax.scatter(
        FI, FJ, Fij, c=epistasis, cmap="RdYlGn", vmin=vmin, vmax=vmax, s=5
    )

    # Add color bar and axis labels
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(r"$\epsilon = f_{ij} - f_i f_j$", fontsize=18)
    # Set integer ticks
    cbar.set_ticks(np.arange(vmin, vmax + 1))

    ax.set_xlabel(r"$f_i$", fontsize=18)
    ax.set_ylabel(r"$f_j$", fontsize=18)
    ax.set_zlabel(r"$f_{ij}$", fontsize=18)
    ax.set_title("3D Epistasis Scatter Plot", fontsize=14)

    # Set axis limits and smaller ticks
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_xticks(np.linspace(0, 2, 5))
    ax.set_yticks(np.linspace(0, 2, 5))
    ax.set_zticks(np.linspace(0, 2, 5))

    # Save the plot
    out_path = osp.join(ASSET_IMAGES_DIR, f"{save_filename_prefix}_{timestamp()}.png")
    plt.savefig(out_path, dpi=600)
    plt.close()
    print(f"3D scatter plot saved to: {out_path}")


def save_level_sets_plot(
    n: int = 50, save_filename_prefix: str = "level_sets_epistasis"
):
    """
    Saves a 3x3 grid of level sets as fij increases, with integer colorbar levels.
    """
    # Load environment variables
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Ensure the output directory exists
    if not os.path.isdir(ASSET_IMAGES_DIR):
        os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    # Generate grid data for fi, fj, and fij
    fi_vals = np.linspace(0, 2, n)
    fj_vals = np.linspace(0, 2, n)
    FI, FJ = np.meshgrid(fi_vals, fj_vals)
    baseline = FI * FJ

    # Define evenly spaced fij slices
    fij_slices = np.linspace(0, 2, 9)

    # Set fixed limits and specific levels for colorbar
    vmin, vmax = -4, 4
    levels = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    # Create more levels for smooth contour plot
    plot_levels = np.linspace(vmin, vmax, 41)

    # Create the 3x3 grid of plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for idx, fij in enumerate(fij_slices):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        # Calculate epistasis for the current fij
        epistasis = fij - baseline

        # Plot level set with RdYlGn colormap
        contour = ax.contourf(
            FI, FJ, epistasis, cmap="RdYlGn", levels=plot_levels, vmin=vmin, vmax=vmax
        )

        if col == 2:  # Add color bar on rightmost column
            cbar = fig.colorbar(contour, ax=ax, orientation="vertical", pad=0.02)
            cbar.set_ticks(levels)
            cbar.set_ticklabels([f"{x}" for x in levels])

        ax.set_title(rf"$f_{{ij}} = {fij:.2f}$", fontsize=12)
        ax.set_xlabel(r"$f_i$", fontsize=10)
        ax.set_ylabel(r"$f_j$", fontsize=10)

    plt.suptitle("Epistasis Level Sets Across $f_{ij}$ Slices", fontsize=16)
    plt.tight_layout()

    # Save the plot
    out_path = osp.join(ASSET_IMAGES_DIR, f"{save_filename_prefix}_{timestamp()}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Level sets plot saved to: {out_path}")


if __name__ == "__main__":
    save_3d_scatter_plot()
    save_level_sets_plot()
