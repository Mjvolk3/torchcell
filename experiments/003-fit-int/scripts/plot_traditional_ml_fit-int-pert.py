import os
import numpy as np
import matplotlib.pyplot as plt

# Base directory
base_dir = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/003-fit-int/traditional-ml/random_1"

# Subdirectories
# sub_dirs = ["mean_pert_1e04", "mean_pert_5e04", "mean_pert_1e05"]
sub_dirs = ["mean_pert_5e04", "mean_pert_1e05"]


# Function to load data
def load_data(directory):
    y_fitness = np.load(os.path.join(directory, "all", "y_fitness.npy"))
    y_gene_interaction = np.load(
        os.path.join(directory, "all", "y_gene_interaction.npy")
    )
    pert_count = np.load(os.path.join(directory, "all", "pert_count.npy"))
    return y_fitness, y_gene_interaction, pert_count


# Function to plot histograms
def plot_histograms(data, title, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()


# Main plotting loop
for sub_dir in sub_dirs:
    full_dir = os.path.join(base_dir, sub_dir)
    y_fitness, y_gene_interaction, pert_count = load_data(full_dir)

    # Plot fitness
    plot_histograms(
        y_fitness, f"Fitness Distribution - {sub_dir}", f"fitness_{sub_dir}.png"
    )

    # Plot gene interaction
    plot_histograms(
        y_gene_interaction,
        f"Gene Interaction Distribution - {sub_dir}",
        f"gene_interaction_{sub_dir}.png",
    )

    # Plot perturbation count
    plot_histograms(
        pert_count,
        f"Perturbation Count Distribution - {sub_dir}",
        f"pert_count_{sub_dir}.png",
    )

print("All plots have been generated and saved.")
