# experiments/002-dmi-tmi/scripts/traditional_ml-plot_svr
# [[experiments.002-dmi-tmi.scripts.traditional_ml-plot_svr]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/traditional_ml-plot_svr
# Test file: experiments/002-dmi-tmi/scripts/test_traditional_ml-plot_svr.py

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import os.path as osp
import logging
import torchcell
import os
from torchcell.utils import format_scientific_notation, savefig_true_size_svg
from dotenv import load_dotenv

load_dotenv()
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

# --- Nature paper style (WYSIWYG, added for SVG export): Arial, 6 pt min, editable SVG text ---
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "legend.title_fontsize": 6, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.5, "lines.linewidth": 0.7, "patch.linewidth": 0.4,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.01,
})

ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = "experiments/002-dmi-tmi/results/svr"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dataset(api, project_name):
    runs = api.runs(project_name)
    summary_list, config_list, name_list, run_id_list = [], [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_list.append(run.name)
        run_id_list.append(run.id)  # Add this line to store the run ID
    return pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "run_id": run_id_list,  # Add the run ID column
        }
    )


def create_plots(
    combined_df: pd.DataFrame, size: int, criterion: str, add_cv: bool = False
):
    log = logging.getLogger(__name__)
    filtered_df = combined_df[combined_df["cell_dataset.size"] == size].copy()
    FEATURE_ORDER = [
        "['random_1']", "['random_10']", "['codon_frequency']", "['random_100']",
        "['normalized_chrom_pathways']", "['calm']", "['fudt_upstream']", "['fudt_downstream']",
        "['random_1000']", "['prot_T5_all']", "['prot_T5_no_dubious']",
        "['esm2_t33_650M_UR50D_all']", "['esm2_t33_650M_UR50D_no_dubious']",
        "['nt_window_5979']", "['nt_window_three_prime_300']", "['nt_window_five_prime_1003']",
        "['one_hot_gene']",
    ]
    all_features = filtered_df["cell_dataset.node_embeddings"].unique()
    features = [f for f in FEATURE_ORDER if f in all_features]
    features += [f for f in all_features if f not in FEATURE_ORDER]
    features = features[::-1]
    rep_types = ["pert_sum", "pert_mean", "intact_sum", "intact_mean"]
    metrics = ["spearman", "pearson", "mse"]

    size_str = format_scientific_notation(size)
    color_list = [
        "#BD8800",
        "#5F7DA8",
        "#A24A46",
        "#846592",
        "#C7A24C",
        "#46557E",
        "#9E4C86",
        "#6E5479",
        "#C08552",
        "#6E93B8",
        "#8A3B3B",
        "#9C6EA0",
        "#B5943F",
        "#4F688B",
        "#A85878",
        "#5A5A5A",
        "#7C5A86",
    ]
    color_dict = {feature: color for feature, color in zip(FEATURE_ORDER, color_list)}
    default_color = "#808080"

    bar_height = 6.0 * 4
    x_marker_size = 3
    x_marker_linewidth = 0.5
    x_marker_vertical_offset = bar_height * 0.50
    red_dot_size = 6
    red_dot_alpha = 0.6
    red_dot_vertical_offset = 0.08
    nan_bar_color = "black"
    nan_marker_color = "red"
    alpha_light_bar = 0.3
    alpha_frame = 0.5
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(2.276, 6.5))
        y = 0
        yticks = []
        ytick_positions = []
        header_labels = []
        max_bar_value = 0

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

                # Mutually distinct mark types so the four representations stay
                # separable on thin bars (diagonals / dots / plus / cross).
                hatch = {
                    "pert_sum": "///",
                    "pert_mean": "...",
                    "intact_sum": "+++",
                    "intact_mean": "xxx",
                }[rep_type]
                color = color_dict.get(str(feature), default_color)

                max_bar_value = max(max_bar_value, val_value, test_value)

                ax.barh(
                    y + bar_height,
                    val_value,
                    height=bar_height,
                    align="edge",
                    color=color,
                    alpha=1.0,
                    hatch=hatch,
                )
                ax.barh(
                    y,
                    test_value,
                    height=bar_height,
                    align="edge",
                    color=color,
                    alpha=alpha_light_bar,
                    hatch=hatch,
                )

                if add_cv:
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
                                color="black",
                                markerfacecolor="black",
                                markeredgecolor="black",
                                markersize=1.5,
                                alpha=1.0,
                                ecolor="black",
                                elinewidth=0.4,
                                capsize=0,
                            )

                y += bar_height * 2

            yticks.append(str(feature[2:-2]))
            ytick_positions.append(group_start_y + bar_height * 3.5)
            header_labels.append((str(feature[2:-2]), group_start_y + bar_height * 8))
            y += bar_height * 3.6

        ax_limit = (
            1.05
            if metric in ["r2", "pearson", "spearman"]
            else max(max_bar_value, 0.1) * 1.1
        )
        marker_offset = ax_limit * 0.01

        for feature in features:
            group_start_y = ytick_positions[features.index(feature)] - bar_height * 3.5
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

                if pd.isna(val_value) and add_cv:
                    ax.scatter(
                        marker_offset,
                        group_start_y + bar_height + x_marker_vertical_offset,
                        color=nan_bar_color,
                        marker="x",
                        s=x_marker_size,
                        linewidths=x_marker_linewidth,
                        zorder=3,
                    )

                if pd.isna(test_value) and add_cv:
                    ax.scatter(
                        marker_offset,
                        group_start_y + x_marker_vertical_offset,
                        color=nan_bar_color,
                        marker="x",
                        s=x_marker_size,
                        linewidths=x_marker_linewidth,
                        zorder=3,
                    )

                if add_cv:
                    fold_mean_col = f"fold_val_{metric}_mean"
                    fold_std_col = f"fold_val_{metric}_std"
                    if fold_mean_col in df.columns and fold_std_col in df.columns:
                        fold_mean = (
                            df[fold_mean_col].values[0] if not df.empty else np.nan
                        )
                        fold_std = (
                            df[fold_std_col].values[0] if not df.empty else np.nan
                        )
                        if pd.isna(fold_mean) or pd.isna(fold_std):
                            ax.scatter(
                                marker_offset,
                                group_start_y
                                + bar_height * 1.5
                                + red_dot_vertical_offset,
                                color=nan_marker_color,
                                marker="s",
                                s=red_dot_size,
                                alpha=red_dot_alpha,
                                zorder=2,
                            )

                group_start_y += bar_height * 2

        # Encoding names as left-aligned in-plot headers (in the gap above each
        # bar group) instead of long left-margin y-tick labels -- reclaims the
        # full figure width for the bars at Nature column size.
        ax.set_yticks([])
        for name, y_top in header_labels:
            ax.text(
                0,
                y_top + bar_height * 0.3,
                name,
                ha="left",
                va="bottom",
                fontname="Arial",
                fontsize=6,
                clip_on=False,
            )
        # Top headroom holds the legends (drawn below) clear of the bars; the
        # add_cv Dataset legend has two extra rows (NaN markers), so it needs more.
        ax.set_ylim(-bar_height, y + bar_height * (25 if add_cv else 20))
        ax.set_xlabel(metric, fontname="Arial", fontsize=6)
        ax.set_xlim(0, ax_limit)
        ax.grid(color="#838383", linestyle="-", linewidth=0.8, alpha=0.2)

        plot_name = f"002-dmi-tmi_SVR_{size_str}_{criterion}_{metric}_{'add_cv' if add_cv else 'no_cv'}_palette.svg"

        representation_legend = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label="Pert Sum",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                hatch="...",
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
            plt.Rectangle(
                (0, 0), 1, 1, color="grey", alpha=alpha_light_bar, label="Test"
            ),
        ]

        if add_cv:
            dataset_legend.extend(
                [
                    plt.scatter(
                        [],
                        [],
                        color=nan_bar_color,
                        marker="x",
                        s=x_marker_size,
                        linewidths=x_marker_linewidth,
                        label="NaN Bar Value",
                    ),
                    plt.scatter(
                        [],
                        [],
                        color=nan_marker_color,
                        marker="s",
                        s=red_dot_size,
                        alpha=red_dot_alpha,
                        label="NaN CV Mean",
                    ),
                ]
            )

        # Representation: vertical (ncol=1) in the top-right corner, entries top-to-
        # bottom matching the bar stacking order. Dataset: top-left.
        leg1 = ax.legend(
            handles=representation_legend,
            title="Representation",
            loc="upper right",
            bbox_to_anchor=(1, 1),
            ncol=1,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.3,
            framealpha=alpha_frame,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=dataset_legend,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0, 1),
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.3,
            framealpha=alpha_frame,
        )

        # Standard Nature-weight frame + ticks (0.5 pt) instead of the heavy 1 pt,
        # so the frame doesn't compete with the thin bars.
        plt.tick_params(axis="y", which="major", size=2.5, width=0.5)
        plt.tick_params(axis="x", which="major", size=2.5, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        plt.tight_layout()
        savefig_true_size_svg(fig, osp.join(ASSET_IMAGES_DIR, plot_name))  # WYSIWYG + draw.io true-size rescale
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

    for col in summary_df.columns:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").astype(
            "float64"
        )

    # Calculate RMSE for each fold
    for i in range(1, 6):
        fold_val_mse_key = f"fold_{i}_val_mse"
        fold_val_rmse_key = f"fold_{i}_val_rmse"
        if fold_val_mse_key in fold_df.columns:
            fold_df[fold_val_mse_key] = pd.to_numeric(
                fold_df[fold_val_mse_key], errors="coerce"
            ).astype("float64")
            fold_df[fold_val_rmse_key] = np.sqrt(fold_df[fold_val_mse_key])
            fold_columns.append(fold_val_rmse_key)

        fold_train_mse_key = f"fold_{i}_train_mse"
        fold_train_rmse_key = f"fold_{i}_train_rmse"
        if fold_train_mse_key in fold_df.columns:
            fold_df[fold_train_mse_key] = pd.to_numeric(
                fold_df[fold_train_mse_key], errors="coerce"
            ).astype("float64")
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

    # Add run id
    processed_df = pd.concat([df[["run_id"]], config_df, summary_df, fold_df], axis=1)

    return processed_df


def deduplicate_dataframe(
    df: pd.DataFrame, criterion: str = "mse", add_cv: bool = False
) -> pd.DataFrame:
    dedup_columns = [
        "cell_dataset.is_pert",
        "cell_dataset.size",
        "cell_dataset.aggregation",
        "cell_dataset.node_embeddings",
        "svr.kernel",
        "svr.C",
        "svr.gamma",
        "num_params",
    ]

    if criterion == "mse":
        sort_column = "val_mse"
        ascending = True  # For mse, we want to select the lowest values
    elif criterion == "spearman":
        sort_column = "val_spearman"
        ascending = False  # For spearman, we want to select the highest values
    else:
        raise ValueError("Invalid criterion. Choose either 'mse' or 'spearman'.")

    if add_cv:
        fold_mean_cols = [col for col in df.columns if col.endswith("_mean")]
        fold_std_cols = [col for col in df.columns if col.endswith("_std")]

        # Filter out rows where all the fold mean columns are missing
        df = df[
            df[[col for col in df.columns if col.endswith("_mean")]].notna().any(axis=1)
        ]

        # Filter out rows where all the fold mean columns are NaN
        df = df.dropna(subset=fold_mean_cols, how="all")

        dedup_columns.extend(fold_mean_cols + fold_std_cols)

    # Update the 'cell_dataset.node_embeddings' column using .loc
    df.loc[:, "cell_dataset.node_embeddings"] = df[
        "cell_dataset.node_embeddings"
    ].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Perform deduplication based on the defined columns
    df = df.sort_values(by=sort_column, ascending=ascending).drop_duplicates(
        subset=dedup_columns, keep="first"
    )

    return df


def main(is_overwrite=False):
    criterion_mse = "mse"
    criterion_spearman = "spearman"

    if is_overwrite:
        api = wandb.Api()

        project_names = [
            "zhao-group/torchcell_002-dmi-tmi_trad-ml_svr_1e03",
            "zhao-group/torchcell_002-dmi-tmi_trad-ml_svr_1e04",
            "zhao-group/torchcell_002-dmi-tmi_trad-ml_svr_1e05",
        ]

        dataframes = [load_dataset(api, project_name) for project_name in project_names]

        config_columns = [
            "cell_dataset.is_pert",
            "cell_dataset.size",
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

    combined_df_1e3_mse_cv = deduplicate_dataframe(
        combined_df_1e3_mse, criterion=criterion_mse, add_cv=True
    )
    combined_df_1e4_mse_cv = deduplicate_dataframe(
        combined_df_1e4_mse, criterion=criterion_mse, add_cv=True
    )
    combined_df_1e5_mse = deduplicate_dataframe(
        combined_df_1e5_mse, criterion=criterion_mse, add_cv=False
    )
    combined_df_1e3_spearman_cv = deduplicate_dataframe(
        combined_df_1e3_spearman, criterion=criterion_spearman, add_cv=True
    )
    combined_df_1e4_spearman_cv = deduplicate_dataframe(
        combined_df_1e4_spearman, criterion=criterion_spearman, add_cv=True
    )
    combined_df_1e5_spearman = deduplicate_dataframe(
        combined_df_1e5_spearman, criterion=criterion_spearman, add_cv=False
    )

    # Deduplicate DataFrames for 1e03 and 1e04 with add_cv=False
    combined_df_1e3_mse_no_cv = deduplicate_dataframe(
        combined_df_1e3_mse, criterion=criterion_mse, add_cv=False
    )
    combined_df_1e4_mse_no_cv = deduplicate_dataframe(
        combined_df_1e4_mse, criterion=criterion_mse, add_cv=False
    )
    combined_df_1e3_spearman_no_cv = deduplicate_dataframe(
        combined_df_1e3_spearman, criterion=criterion_spearman, add_cv=False
    )
    combined_df_1e4_spearman_no_cv = deduplicate_dataframe(
        combined_df_1e4_spearman, criterion=criterion_spearman, add_cv=False
    )

    # Save the deduplicated DataFrames for manual inspection
    combined_df_1e3_mse_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_mse_1e3_add_cv.csv"),
        index=False,
    )
    combined_df_1e4_mse_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_mse_1e4_add_cv.csv"),
        index=False,
    )
    combined_df_1e5_mse.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_mse_1e5.csv"), index=False
    )
    combined_df_1e3_spearman_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_spearman_1e3_add_cv.csv"),
        index=False,
    )
    combined_df_1e4_spearman_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_spearman_1e4_add_cv.csv"),
        index=False,
    )
    combined_df_1e5_spearman.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_spearman_1e5.csv"), index=False
    )

    # Save the deduplicated DataFrames for 1e03 and 1e04 with add_cv=False
    combined_df_1e3_mse_no_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_mse_1e3.csv"), index=False
    )
    combined_df_1e4_mse_no_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_mse_1e4.csv"), index=False
    )
    combined_df_1e3_spearman_no_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_spearman_1e3.csv"), index=False
    )
    combined_df_1e4_spearman_no_cv.to_csv(
        osp.join(RESULTS_DIR, "deduplicated_combined_df_spearman_1e4.csv"), index=False
    )

    create_plots(
        combined_df_1e3_mse_cv, size=1000, criterion=criterion_mse, add_cv=True
    )
    create_plots(
        combined_df_1e4_mse_cv, size=10000, criterion=criterion_mse, add_cv=True
    )
    create_plots(
        combined_df_1e5_mse, size=100000, criterion=criterion_mse, add_cv=False
    )

    create_plots(
        combined_df_1e3_spearman_cv,
        size=1000,
        criterion=criterion_spearman,
        add_cv=True,
    )
    create_plots(
        combined_df_1e4_spearman_cv,
        size=10000,
        criterion=criterion_spearman,
        add_cv=True,
    )
    create_plots(
        combined_df_1e5_spearman,
        size=100000,
        criterion=criterion_spearman,
        add_cv=False,
    )

    # Create plots for 1e03 and 1e04 with add_cv=False
    create_plots(
        combined_df_1e3_mse_no_cv, size=1000, criterion=criterion_mse, add_cv=False
    )
    create_plots(
        combined_df_1e4_mse_no_cv, size=10000, criterion=criterion_mse, add_cv=False
    )
    create_plots(
        combined_df_1e3_spearman_no_cv,
        size=1000,
        criterion=criterion_spearman,
        add_cv=False,
    )
    create_plots(
        combined_df_1e4_spearman_no_cv,
        size=10000,
        criterion=criterion_spearman,
        add_cv=False,
    )


if __name__ == "__main__":
    main(is_overwrite=False)
