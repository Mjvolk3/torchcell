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
RESULTS_DIR = "experiments/smf-dmf-tmf-001/results/random_forest"

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
        run_id_list.append(run.id)
    return pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "run_id": run_id_list,
        }
    )


def process_raw_dataframe(
    df: pd.DataFrame, config_columns: list, summary_columns: list
) -> pd.DataFrame:
    # Normalize config columns
    config_df = pd.json_normalize(df["config"])[config_columns]

    # Normalize summary columns
    summary_df = pd.json_normalize(df["summary"])
    summary_df = summary_df[summary_df.columns.intersection(summary_columns)]

    # Calculate val_rmse and test_rmse if the corresponding mse columns exist
    if "val_mse" in summary_df.columns:
        summary_df["val_rmse"] = np.sqrt(summary_df["val_mse"])
    if "test_mse" in summary_df.columns:
        summary_df["test_rmse"] = np.sqrt(summary_df["test_mse"])

    # Combine config, summary, and run_id DataFrames
    processed_df = pd.concat([df[["run_id"]], config_df, summary_df], axis=1)

    # Remove rows with missing num_params
    processed_df = processed_df.dropna(subset=["num_params"])

    return processed_df


def create_plots(df: pd.DataFrame, max_size: int):
    metrics = ["r2", "pearson", "spearman", "mse", "mae", "rmse"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    val_color, test_color = colors

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (_, group_df) in enumerate(df.groupby("run_id")):
            val_mask = group_df[f"val_{metric}"].notna()
            test_mask = group_df[f"test_{metric}"].notna()
            if val_mask.any() and test_mask.any():
                ax.scatter(
                    group_df.loc[val_mask, "num_params"],
                    group_df.loc[val_mask, f"val_{metric}"],
                    color=val_color,
                    alpha=0.7,
                    label="Validation" if i == 0 else None,
                )
                ax.scatter(
                    group_df.loc[test_mask, "num_params"],
                    group_df.loc[test_mask, f"test_{metric}"],
                    color=test_color,
                    alpha=0.7,
                    label="Test" if i == 0 else None,
                )

                x_val, y_val = (
                    group_df.loc[val_mask, "num_params"].values[0],
                    group_df.loc[val_mask, f"val_{metric}"].values[0],
                )
                x_test, y_test = (
                    group_df.loc[test_mask, "num_params"].values[0],
                    group_df.loc[test_mask, f"test_{metric}"].values[0],
                )
                curve_x = np.linspace(x_val, x_test, 100)
                curve_y = np.interp(curve_x, [x_val, x_test], [y_val, y_test])
                curve_y += (curve_x - x_val) * (curve_x - x_test) * 0.1
                ax.plot(
                    curve_x, curve_y, linestyle="-", color="gray", alpha=0.5, zorder=-1
                )

        ax.set_xlabel("Number of Parameters", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
        ax.legend(fontsize=12)

        if metric == "pearson":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)

        max_size_str = format_scientific_notation(max_size)
        title = f"Random Forest {max_size_str} - {metric} vs. Number of Parameters"
        ax.set_title(title, fontsize=16)

        plot_name = f"Random_Forest_{max_size_str}_{metric}_vs_num_params.png"
        plt.tight_layout()
        plt.savefig(osp.join(ASSET_IMAGES_DIR, plot_name), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    api = wandb.Api()

    project_names = [
        "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e03",
        "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e04",
        "zhao-group/torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e05",
    ]

    dataframes = [load_dataset(api, project_name) for project_name in project_names]

    config_columns = [
        "cell_dataset.is_pert",
        "cell_dataset.max_size",
        "cell_dataset.aggregation",
        "cell_dataset.node_embeddings",
        "random_forest.max_depth",
        "random_forest.n_estimators",
        "random_forest.min_samples_split",
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
        "val_rmse",
        "test_rmse",
    ]

    for df, max_size in zip(dataframes, [1000, 10000, 100000]):
        if not df.empty:
            processed_df = process_raw_dataframe(df, config_columns, summary_columns)
            print(
                f"Number of rows with missing num_params for max_size {max_size}: {len(df) - len(processed_df)}"
            )
            processed_df.to_csv(
                osp.join(RESULTS_DIR, f"random_forest_processed_df_{max_size}.csv"),
                index=False,
            )
            create_plots(processed_df, max_size)


if __name__ == "__main__":
    main()
