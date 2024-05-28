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


def plot_histogram(X, y, dataset_size, split, bins=20):
    X = (X * -1) + 1  # Invert 1s and 0s
    X_gene_count = np.sum(X, axis=1)
    df = pd.DataFrame({"num_genes_deleted": X_gene_count, "fitness": y})
    df.hist(bins=bins)
    plt.title(
        f"SMF-DMF-TMF Traditional ML {dataset_size} {split.capitalize()}"
    )
    plt.savefig(
        osp.join(
            ASSET_IMAGES_DIR,
            f"smf-dmf-tmf-traditional-ml_{dataset_size}-{split}-histogram.png",
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
            plot_histogram(X, y, dataset_size, split)


if __name__ == "__main__":
    main()
