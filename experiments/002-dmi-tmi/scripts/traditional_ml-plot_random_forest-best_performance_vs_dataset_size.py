import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define constants
RESULTS_DIR = "experiments/002-dmi-tmi/results/random_forest"
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
MAX_SIZES = [1000, 10000, 100000]

# Define color list (reversed order)
color_list = [
    "#D0838E",
    "#FFA257",
    "#ECD078",
    "#53777A",
    "#8F918B",
    "#D1A0A2",
    "#A8BDB5",
    "#B8AD9E",
    "#7B9EAE",
    "#F75C4C",
    "#82B3AE",
    "#FFD3B6",
    "#746D75",
    "#FF8C94",
    "#5E8C61",
    "#B565A7",
    "#955251",
    "#009B77",
    "#DD4124",
    "#D65076",
]

# Define the desired order of node embeddings
node_embedding_order = [
    "random_1",
    "random_10",
    "codon_frequency",
    "random_100",
    "normalized_chrom_pathways",
    "calm",
    "fudt_upstream",
    "fudt_downstream",
    "random_1000",
    "prot_T5_all",
    "prot_T5_no_dubious",
    "esm2_t33_650M_UR50D_all",
    "esm2_t33_650M_UR50D_no_dubious",
    "nt_window_three_prime_300",
    "nt_window_five_prime_1003",
    "nt_window_5979",
    "one_hot_gene",
]


def load_and_process_data(max_size):
    file_path = os.path.join(RESULTS_DIR, f"random_forest_processed_df_{max_size}.csv")
    print(f"Loading file: {file_path}")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df["node_embeddings"] = df["cell_dataset.node_embeddings"].apply(
        lambda x: eval(x)[0] if isinstance(x, str) else x
    )
    return df


def get_best_runs(df, metric="val_r2"):
    if df is None or df.empty:
        return pd.DataFrame()
    return df.loc[df.groupby("node_embeddings")[metric].idxmax()]


def humanize_metric(metric):
    words = metric.split("_")
    if len(words) > 1:
        return f"{words[0].capitalize()} {' '.join(word.capitalize() for word in words[1:])}"
    else:
        return metric.capitalize()


def create_plot(data_dict, metric="val_r2"):
    plt.figure(figsize=(14, 10))

    color_dict = {
        embedding: color for embedding, color in zip(node_embedding_order, color_list)
    }

    for node_embedding in node_embedding_order:
        if node_embedding in data_dict:
            points = data_dict[node_embedding]
            x = [p["max_size"] for p in points]
            y = [p[metric] for p in points]
            color = color_dict[node_embedding]
            plt.plot(
                x, y, "-o", color=color, label=node_embedding, linewidth=3, markersize=6
            )

    plt.xscale("log")
    plt.xticks(MAX_SIZES, ["$10^3$", "$10^4$", "$10^5$"])
    plt.xlabel("Dataset Size", fontsize=16)

    humanized_metric = humanize_metric(metric)
    plt.ylabel(humanized_metric, fontsize=16)

    plt.title(
        f"002-dmi-tmi Best {humanized_metric} for Node Embeddings across Dataset Sizes", fontsize=18
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
    plt.tick_params(axis="both", which="minor", labelsize=12, width=1, length=4)

    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    plt.tight_layout()

    plt.savefig(
        os.path.join(ASSET_IMAGES_DIR, f"002-dmi-tmi_node_embedding_performance_{metric}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    data_dict = {}

    for max_size in MAX_SIZES:
        df = load_and_process_data(max_size)
        if df is not None and not df.empty:
            best_runs = get_best_runs(df)

            for _, row in best_runs.iterrows():
                node_embedding = row["node_embeddings"]
                if node_embedding not in data_dict:
                    data_dict[node_embedding] = []
                data_dict[node_embedding].append(
                    {
                        "max_size": max_size,
                        "val_r2": row["val_r2"],
                        "test_r2": row["test_r2"],
                        "val_pearson": row["val_pearson"],
                        "test_pearson": row["test_pearson"],
                        "val_spearman": row["val_spearman"],
                        "test_spearman": row["test_spearman"],
                        "val_mse": row["val_mse"],
                        "test_mse": row["test_mse"],
                        "val_mae": row["val_mae"],
                        "test_mae": row["test_mae"],
                        "val_rmse": row["val_rmse"],
                        "test_rmse": row["test_rmse"],
                    }
                )

    if not data_dict:
        print("No data was loaded. Check the file paths and data processing.")
        return

    metrics = [
        "val_r2",
        "test_r2",
        "val_pearson",
        "test_pearson",
        "val_spearman",
        "test_spearman",
        "val_mse",
        "test_mse",
        "val_mae",
        "test_mae",
        "val_rmse",
        "test_rmse",
    ]

    for metric in metrics:
        create_plot(data_dict, metric)


if __name__ == "__main__":
    main()
