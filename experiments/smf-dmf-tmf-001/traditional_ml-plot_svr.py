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
RESULTS_DIR = "experiments/smf-dmf-tmf-001/results/svr"

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


def create_plots(
    combined_df: pd.DataFrame, max_size: int, criterion: str, add_folds: bool = False
):
    log = logging.getLogger(__name__)
    filtered_df = combined_df[combined_df["cell_dataset.max_size"] == max_size].copy()
    features = sorted(filtered_df["cell_dataset.node_embeddings"].unique())
    rep_types = ["pert_sum", "pert_mean", "intact_sum", "intact_mean"]
    metrics = ["r2", "pearson", "spearman", "mse", "mae", "rmse"]

    max_size_str = format_scientific_notation(max_size)
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
    ]
    color_dict = {feature: color for feature, color in zip(features, color_list)}
    default_color = "#808080"

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 12))
        y = 0
        yticks = []
        ytick_positions = []

        for feature in features:
            group_start_y = y
            for rep_type in reversed(rep_types):
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
                    val_value = np.nan
                    test_value = np.nan
                else:
                    val_value = df[val_key].values[0]
                    test_value = df[test_key].values[0]

                if metric in ["r2", "pearson", "spearman"]:
                    if pd.isna(val_value):
                        val_value = 0
                    if pd.isna(test_value):
                        test_value = 0

                hatch = (
                    "xxx"
                    if rep_type == "intact_mean"
                    else (
                        "+++"
                        if rep_type == "intact_sum"
                        else ".." if rep_type == "pert_sum" else "......"
                    )
                )
                bar_height = 6.0 * 4
                color = color_dict.get(
                    feature[0] if isinstance(feature, list) else feature, default_color
                )

                if pd.isna(val_value):
                    ax.scatter(
                        0, y + bar_height, color=color, marker="o", s=100, zorder=3
                    )
                else:
                    ax.barh(
                        y + bar_height,
                        val_value,
                        height=bar_height,
                        align="edge",
                        color=color,
                        alpha=1.0,
                        hatch=hatch,
                    )

                if pd.isna(test_value):
                    ax.scatter(0, y, color=color, marker="o", s=100, zorder=3)
                else:
                    ax.barh(
                        y,
                        test_value,
                        height=bar_height,
                        align="edge",
                        color=color,
                        alpha=0.5,
                        hatch=hatch,
                    )

                if add_folds:
                    fold_mean_col = f"fold_val_{metric}_mean"
                    fold_std_col = f"fold_val_{metric}_std"
                    if fold_mean_col in df.columns and fold_std_col in df.columns:
                        fold_mean = (
                            df[fold_mean_col].values[0] if not df.empty else np.nan
                        )
                        fold_std = (
                            df[fold_std_col].values[0] if not df.empty else np.nan
                        )
                        if not pd.isna(fold_mean) and not pd.isna(fold_std):
                            ax.errorbar(
                                fold_mean,
                                y + bar_height * 1.5,
                                xerr=fold_std,
                                fmt="o",
                                color="white",
                                markerfacecolor=color,
                                markeredgecolor="black",
                                markersize=8,
                                alpha=0.5,
                                ecolor="black",
                                elinewidth=bar_height * 0.15,
                                capsize=0,
                            )

                y += bar_height * 2

            yticks.append(feature[0] if isinstance(feature, list) else feature)
            ytick_positions.append(group_start_y + bar_height * 3.5)
            y += bar_height

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(yticks, fontname="Arial", fontsize=20)
        ax.set_xlabel(metric, fontname="Arial", fontsize=20)
        ax_limit = (
            1.05
            if metric in ["r2", "pearson", "spearman"]
            else max(ax.get_xlim()[1], 0.1) * 1.1
        )
        ax.set_xlim(0, ax_limit)
        ax.grid(color="#838383", linestyle="-", linewidth=0.8, alpha=0.5)
        plot_name = f"SVR_{max_size_str}_{criterion}_{metric}.png"
        ax.set_title(os.path.splitext(plot_name)[0], fontsize=20)

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


def process_raw_dataframe(
    df: pd.DataFrame, config_columns: list, summary_columns: list
) -> pd.DataFrame:
    # Normalize config columns
    config_df = pd.json_normalize(df["config"])[config_columns]

    # Extract fold data from the summary column
    fold_metrics = ["r2", "pearson", "spearman", "mse", "mae"]
    fold_columns = []
    fold_data = []

    for _, row in df.iterrows():
        summary = row["summary"]
        fold_row = {}

        for metric in fold_metrics:
            for i in range(1, 6):
                fold_val_key = f"fold_{i}_val_{metric}"
                fold_train_key = f"fold_{i}_train_{metric}"

                if fold_val_key in summary:
                    fold_row[fold_val_key] = pd.to_numeric(
                        summary[fold_val_key], errors="coerce"
                    )
                    fold_columns.append(fold_val_key)
                else:
                    fold_row[fold_val_key] = np.nan

                if fold_train_key in summary:
                    fold_row[fold_train_key] = pd.to_numeric(
                        summary[fold_train_key], errors="coerce"
                    )
                    fold_columns.append(fold_train_key)
                else:
                    fold_row[fold_train_key] = np.nan

        fold_data.append(fold_row)

    fold_df = pd.DataFrame(fold_data)
    fold_columns = list(set(fold_columns))

    # Normalize other summary columns
    summary_df = pd.json_normalize(df["summary"])[summary_columns]

    # Calculate RMSE for each fold
    for i in range(1, 6):
        fold_val_mse_key = f"fold_{i}_val_mse"
        fold_val_rmse_key = f"fold_{i}_val_rmse"
        if fold_val_mse_key in fold_df.columns:
            fold_df[fold_val_rmse_key] = np.sqrt(fold_df[fold_val_mse_key])
            fold_columns.append(fold_val_rmse_key)

        fold_train_mse_key = f"fold_{i}_train_mse"
        fold_train_rmse_key = f"fold_{i}_train_rmse"
        if fold_train_mse_key in fold_df.columns:
            fold_df[fold_train_rmse_key] = np.sqrt(fold_df[fold_train_mse_key])
            fold_columns.append(fold_train_rmse_key)

    # Calculate means and standard deviations for fold metrics
    for metric in fold_metrics + ["rmse"]:
        fold_val_cols = [col for col in fold_columns if f"_val_{metric}" in col]
        fold_train_cols = [col for col in fold_columns if f"_train_{metric}" in col]

        fold_df[f"fold_val_{metric}_mean"] = fold_df[fold_val_cols].mean(axis=1)
        fold_df[f"fold_val_{metric}_std"] = fold_df[fold_val_cols].std(axis=1)
        fold_df[f"fold_train_{metric}_mean"] = fold_df[fold_train_cols].mean(axis=1)
        fold_df[f"fold_train_{metric}_std"] = fold_df[fold_train_cols].std(axis=1)

    # Calculate val_rmse, train_rmse, and test_rmse if the corresponding mse columns exist
    if "val_mse" in summary_df.columns:
        summary_df["val_rmse"] = np.sqrt(summary_df["val_mse"])
    if "train_mse" in summary_df.columns:
        summary_df["train_rmse"] = np.sqrt(summary_df["train_mse"])
    if "test_mse" in summary_df.columns:
        summary_df["test_rmse"] = np.sqrt(summary_df["test_mse"])

    # Combine config, summary, and fold DataFrames
    processed_df = pd.concat([config_df, summary_df, fold_df], axis=1)

    return processed_df


def deduplicate_dataframe(
    df: pd.DataFrame, criterion: str = "mse", add_folds: bool = False
) -> pd.DataFrame:
    dedup_columns = [
        "cell_dataset.is_pert",
        "cell_dataset.max_size",
        "cell_dataset.aggregation",
        "cell_dataset.node_embeddings",
        "svr.kernel",
        "svr.C",
        "svr.gamma",
        "num_params",
    ]

    if criterion == "mse":
        sort_column = "val_mse"
    elif criterion == "spearman":
        sort_column = "val_spearman"
    else:
        raise ValueError("Invalid criterion. Choose either 'mse' or 'spearman'.")

    if add_folds:
        fold_mean_cols = [col for col in df.columns if col.endswith("_mean")]
        fold_std_cols = [col for col in df.columns if col.endswith("_std")]

        # Filter out rows where all the fold mean columns are NaN
        df = df.dropna(subset=fold_mean_cols, how="all")

        dedup_columns.extend(fold_mean_cols + fold_std_cols)

    # Update the 'cell_dataset.node_embeddings' column using .loc
    df.loc[:, "cell_dataset.node_embeddings"] = df[
        "cell_dataset.node_embeddings"
    ].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Perform deduplication based on the defined columns
    df = df.sort_values(sort_column).drop_duplicates(subset=dedup_columns, keep="first")

    return df


def main(is_overwrite=False):
    criterion_mse = "mse"
    criterion_spearman = "spearman"

    if is_overwrite:
        api = wandb.Api()

        project_names = [
            "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_svr_1e03",
            "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_svr_1e04",
            "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_svr_1e05",
        ]

        dataframes = [load_dataset(api, project_name) for project_name in project_names]

        config_columns = [
            "cell_dataset.is_pert",
            "cell_dataset.max_size",
            "cell_dataset.aggregation",
            "cell_dataset.node_embeddings",
            "svr.kernel",
            "svr.C",
            "svr.gamma",
        ]
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

        for df, name in zip(dataframes, ["1e3", "1e4", "1e5"]):
            if not df.empty:
                processed_df = process_raw_dataframe(
                    df, config_columns, summary_columns
                )
                processed_df.to_csv(
                    osp.join(RESULTS_DIR, f"combined_df_mse_{name}.csv"), index=False
                )
                processed_df.to_csv(
                    osp.join(RESULTS_DIR, f"combined_df_spearman_{name}.csv"),
                    index=False,
                )

    combined_df_1e3_mse = pd.read_csv(osp.join(RESULTS_DIR, "combined_df_mse_1e3.csv"))
    combined_df_1e4_mse = pd.read_csv(osp.join(RESULTS_DIR, "combined_df_mse_1e4.csv"))
    combined_df_1e5_mse = pd.read_csv(osp.join(RESULTS_DIR, "combined_df_mse_1e5.csv"))
    combined_df_1e3_spearman = pd.read_csv(
        osp.join(RESULTS_DIR, "combined_df_spearman_1e3.csv")
    )
    combined_df_1e4_spearman = pd.read_csv(
        osp.join(RESULTS_DIR, "combined_df_spearman_1e4.csv")
    )
    combined_df_1e5_spearman = pd.read_csv(
        osp.join(RESULTS_DIR, "combined_df_spearman_1e5.csv")
    )

    combined_df_1e3_mse = deduplicate_dataframe(
        combined_df_1e3_mse, criterion=criterion_mse, add_folds=True
    )
    combined_df_1e4_mse = deduplicate_dataframe(
        combined_df_1e4_mse, criterion=criterion_mse, add_folds=True
    )
    combined_df_1e5_mse = deduplicate_dataframe(
        combined_df_1e5_mse, criterion=criterion_mse, add_folds=False
    )
    combined_df_1e3_spearman = deduplicate_dataframe(
        combined_df_1e3_spearman, criterion=criterion_spearman, add_folds=True
    )
    combined_df_1e4_spearman = deduplicate_dataframe(
        combined_df_1e4_spearman, criterion=criterion_spearman, add_folds=True
    )
    combined_df_1e5_spearman = deduplicate_dataframe(
        combined_df_1e5_spearman, criterion=criterion_spearman, add_folds=False
    )

    create_plots(
        combined_df_1e3_mse, max_size=1000, criterion=criterion_mse, add_folds=True
    )
    create_plots(
        combined_df_1e4_mse, max_size=10000, criterion=criterion_mse, add_folds=True
    )
    create_plots(
        combined_df_1e5_mse, max_size=100000, criterion=criterion_mse, add_folds=False
    )

    create_plots(
        combined_df_1e3_spearman,
        max_size=1000,
        criterion=criterion_spearman,
        add_folds=True,
    )
    create_plots(
        combined_df_1e4_spearman,
        max_size=10000,
        criterion=criterion_spearman,
        add_folds=True,
    )
    create_plots(
        combined_df_1e5_spearman,
        max_size=100000,
        criterion=criterion_spearman,
        add_folds=False,
    )


if __name__ == "__main__":
    main(is_overwrite=True)
