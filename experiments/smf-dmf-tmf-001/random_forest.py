# experiments/smf-dmf-tmf-001/traditional_ml-plot_random-forest
# [[experiments.smf-dmf-tmf-001.traditional_ml-plot_random-forest]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf-001/traditional_ml-plot_random-forest

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import os.path as osp
import logging
import torchcell
import os
from torchcell.utils import format_scientific_notation
from dotenv import load_dotenv

load_dotenv()
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = "experiments/smf-dmf-tmf-001/results/random-forest"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dataset(api, project_name):
    runs = api.runs(project_name)
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_list.append(run.name)
    return pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )


def deduplicate_dataframe(df, criterion="mse"):
    # Drop 'summary', 'config', and 'name' columns
    df = df.drop(columns=["summary", "config", "name"])

    # Take the first element of 'cell_dataset.node_embeddings' column
    df["cell_dataset.node_embeddings"] = df["cell_dataset.node_embeddings"].apply(
        lambda x: x[0]
    )

    # Deduplicate rows based on specified columns
    dedup_columns = [
        "cell_dataset.is_pert",
        "cell_dataset.max_size",
        "cell_dataset.aggregation",
        "cell_dataset.node_embeddings",
        "random_forest.n_estimators",
        "random_forest.max_depth",
        "random_forest.min_samples_split",
        "num_params",
    ]

    if criterion == "mse":
        # Select the duplicate with the lowest "mse" value
        df = df.sort_values("val_mse").drop_duplicates(
            subset=dedup_columns, keep="first"
        )
    elif criterion == "spearman":
        # Select the duplicate with the highest "spearman" value
        df = df.sort_values("val_spearman", ascending=False).drop_duplicates(
            subset=dedup_columns, keep="first"
        )
    else:
        raise ValueError("Invalid criterion. Choose either 'mse' or 'spearman'.")

    return df


def create_plots(combined_df, max_size, criterion, is_overwrite=False):
    log = logging.getLogger(__name__)
    style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
    plt.style.use(style_file_path)

    filtered_df = combined_df[combined_df["cell_dataset.max_size"] == max_size]
    features = filtered_df["cell_dataset.node_embeddings"].unique()
    rep_types = ["pert_sum", "pert_mean", "intact_sum", "intact_mean"]
    metrics = ["r2", "pearson", "spearman", "mse", "mae"]

    max_size_str = format_scientific_notation(max_size)

    # Define a darker pastel color palette
    color_list = [
        "#746D75",  # Darker purple
        "#D0838E",  # Darker pink
        "#FFA257",  # Darker orange
        "#ECD078",  # Darker yellow
        "#53777A",  # Dark blue-green
        "#8F918B",  # Darker light purple
        "#D1A0A2",  # Darker light red
        "#A8BDB5",  # Darker soft green
        "#B8AD9E",  # Darker muted brown
        "#7B9EAE",  # Darker soft blue
        "#F75C4C",  # Darker blush red
        "#82B3AE",  # Darker sea foam green
        "#FFD3B6",  # Darker peach
    ]
    color_dict = {feature: color for feature, color in zip(features, color_list)}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 8))
        y = 0
        yticks = []
        ytick_positions = []

        for feature in features:
            group_start_y = y  # Start of the group for current feature
            for rep_type in reversed(
                rep_types
            ):  # Reverse to start with Intact Mean, ends with Pert Sum
                val_key = f"val_{metric}"
                test_key = f"test_{metric}"
                df = filtered_df[
                    (filtered_df["cell_dataset.node_embeddings"] == feature)
                    & (
                        filtered_df["cell_dataset.is_pert"]
                        == rep_type.startswith("pert")
                    )
                    & (
                        filtered_df["cell_dataset.aggregation"]
                        == ("sum" if rep_type.endswith("sum") else "mean")
                    )
                ]

                if df.empty:
                    continue  # Skip plotting if the DataFrame is empty

                val_value = df[val_key].values[0]
                test_value = df[test_key].values[0]

                # Define hatch patterns based on the type
                hatch = (
                    "xxx"
                    if rep_type == "intact_mean"
                    else (
                        "+++"
                        if rep_type == "intact_sum"
                        else ".." if rep_type == "pert_sum" else "......"
                    )
                )
                bar_height = 6.0 * 4  # Increased height

                # Plot Validation first, above Test
                ax.barh(
                    y + bar_height,
                    val_value,
                    height=bar_height,
                    align="edge",
                    color=color_dict[feature],
                    alpha=1.0,
                    hatch=hatch,
                )
                # Then plot Test below Validation
                ax.barh(
                    y,
                    test_value,
                    height=bar_height,
                    align="edge",
                    color=color_dict[feature],
                    alpha=0.5,
                    hatch=hatch,
                )
                y += bar_height * 2  # Increase spacing for the next bar group

            yticks.append(feature)
            ytick_positions.append(
                group_start_y + bar_height * 3.5
            )  # Adjust tick position to middle of the set
            y += bar_height  # Additional space between different node embeddings

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(yticks, fontname="Arial", fontsize=20)

        ax.set_xlabel(metric, fontname="Arial", fontsize=20)
        ax_limit = (
            1
            if metric in ["r2", "pearson", "spearman"]
            else max(val_value, test_value) * 1.5
        )
        ax.set_xlim(0, ax_limit)
        ax.grid(color="#838383", linestyle="-", linewidth=0.8, alpha=0.5)

        # Set the title as the image save path with the criterion included
        plot_name = f"Random_Forest_{max_size_str}_{criterion}_{metric}.png"
        ax.set_title(os.path.splitext(plot_name)[0], fontsize=20)

        # Aggregation legend
        representation_legend = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                hatch="..",
                label="Pert Sum",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                hatch="......",
                label="Pert Mean",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                hatch="+++",
                label="Intact Sum",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                hatch="xxx",
                label="Intact Mean",
            ),
        ]
        dataset_legend = [
            plt.Rectangle((0, 0), 1, 1, color="grey", alpha=1.0, label="Validation"),
            plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.5, label="Test"),
        ]

        # Create a legend with grouped titles
        leg1 = ax.legend(
            handles=representation_legend,
            title="Representation",
            loc="upper right",
            bbox_to_anchor=(1, 1),
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=dataset_legend,
            title="Dataset",
            loc="center right",
            bbox_to_anchor=(1, 0.5),
        )
        plt.tick_params(axis="y", which="major", size=5, width=1)
        plt.tick_params(axis="x", which="major", size=5, width=1)
        ax.spines["top"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)

        plt.tight_layout()
        plt.savefig(osp.join(ASSET_IMAGES_DIR, plot_name), bbox_inches="tight")
        plt.close(fig)

    return None


def main(is_overwrite=False):
    criterion_mse = "mse"
    criterion_spearman = "spearman"

    combined_df_mse_path = osp.join(RESULTS_DIR, "combined_df_mse.csv")
    combined_df_spearman_path = osp.join(RESULTS_DIR, "combined_df_spearman.csv")

    if not is_overwrite and not (
        osp.exists(combined_df_mse_path) and osp.exists(combined_df_spearman_path)
    ):
        print("CSV files not found. Fetching data from the API.")
        is_overwrite = True

    if is_overwrite:
        api = wandb.Api()

        # Load datasets for Random Forest 1e03, 1e04, and 1e05
        project_names = [
            "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e03",
            "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e04",
            "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e05",
        ]
        dataframes = [load_dataset(api, project_name) for project_name in project_names]

        # Combine the dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Extract desired columns from 'config'
        config_columns = [
            "cell_dataset.is_pert",
            "cell_dataset.max_size",
            "cell_dataset.aggregation",
            "cell_dataset.node_embeddings",
            "random_forest.n_estimators",
            "random_forest.max_depth",
            "random_forest.min_samples_split",
        ]
        config_df = pd.json_normalize(combined_df["config"])[config_columns]

        # Extract desired columns from 'summary'
        summary_columns = [
            "num_params",
            "val_r2",
            "test_r2",
            "val_pearson",
            "test_pearson",
            "val_spearman",
            "test_spearman",
            "val_mae",
            "test_mae",
            "val_mse",
            "test_mse",
        ]
        summary_df = pd.json_normalize(combined_df["summary"])[summary_columns]

        # Combine the extracted columns with 'combined_df'
        combined_df = pd.concat([combined_df, config_df, summary_df], axis=1)

        # Deduplicate the DataFrame based on the lowest "mse" value
        combined_df_mse = deduplicate_dataframe(combined_df, criterion=criterion_mse)
        combined_df_spearman = deduplicate_dataframe(
            combined_df, criterion=criterion_spearman
        )

        # Save the deduplicated dataframes as CSV files
        combined_df_mse.to_csv(combined_df_mse_path, index=False)
        combined_df_spearman.to_csv(combined_df_spearman_path, index=False)
    else:
        # Load the deduplicated dataframes from CSV files
        combined_df_mse = pd.read_csv(combined_df_mse_path)
        combined_df_spearman = pd.read_csv(combined_df_spearman_path)

    create_plots(
        combined_df_mse,
        max_size=1000,
        criterion=criterion_mse,
        is_overwrite=is_overwrite,
    )
    create_plots(
        combined_df_mse,
        max_size=10000,
        criterion=criterion_mse,
        is_overwrite=is_overwrite,
    )

    create_plots(
        combined_df_spearman,
        max_size=1000,
        criterion=criterion_spearman,
        is_overwrite=is_overwrite,
    )
    create_plots(
        combined_df_spearman,
        max_size=10000,
        criterion=criterion_spearman,
        is_overwrite=is_overwrite,
    )


if __name__ == "__main__":
    is_overwrite = False
    main(is_overwrite)
