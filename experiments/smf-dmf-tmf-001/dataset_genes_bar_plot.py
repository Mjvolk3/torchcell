# experiments/smf-dmf-tmf-001/dataset_genes_bar_plot
# [[experiments.smf-dmf-tmf-001.dataset_genes_bar_plot]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf-001/dataset_genes_bar_plot


import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import torchcell

load_dotenv()
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
DATA_ROOT = os.getenv("DATA_ROOT")


def plot_bar(X, y, dataset_size, split):
    X = (X * -1) + 1  # Invert 1s and 0s
    X_gene_count = np.sum(X, axis=0)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(X_gene_count)), X_gene_count)
    plt.xlabel("Gene Index")
    plt.ylabel("Count")
    plt.title(
        f"SMF-DMF-TMF Traditional ML Gene Count {dataset_size} {split.capitalize()}"
    )

    # Add the total number of genes perturbed to the plot
    total_genes_perturbed = np.sum(X_gene_count > 0)
    plt.text(
        0.95,
        0.95,
        f"Number of genes perturbed: {total_genes_perturbed}",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="top",
    )

    plt.savefig(
        osp.join(
            ASSET_IMAGES_DIR,
            f"smf-dmf-tmf-traditional-ml_gene-count_{dataset_size}-{split}-bar.png",
        )
    )
    plt.close()


def main():
    dataset_sizes = ["1e03", "1e04", "1e05"]
    splits = ["all", "train", "val", "test"]

    for dataset_size in dataset_sizes:
        for split in splits:
            X = np.load(
                osp.join(
                    DATA_ROOT,
                    f"data/torchcell/experiments/smf-dmf-tmf-traditional-ml/one_hot_gene/sum_{dataset_size}/{split}/X.npy",
                )
            )
            y = np.load(
                osp.join(
                    DATA_ROOT,
                    f"data/torchcell/experiments/smf-dmf-tmf-traditional-ml/one_hot_gene/sum_{dataset_size}/{split}/y.npy",
                )
            )
            plot_bar(X, y, dataset_size, split)


if __name__ == "__main__":
    main()
