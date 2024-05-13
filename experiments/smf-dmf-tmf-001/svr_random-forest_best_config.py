import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import os.path as osp
import torchcell
from dotenv import load_dotenv
import os

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = "experiments/smf-dmf-tmf-001/results"

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def find_best_config(df, model_type, criterion, max_size):
    filtered_df = df[(df["cell_dataset.max_size"] == max_size)]
    best_configs = {}

    for embedding in filtered_df["cell_dataset.node_embeddings"].unique():
        for is_pert in [True, False]:
            for aggregation in ["sum", "mean"]:
                sub_df = filtered_df[
                    (filtered_df["cell_dataset.node_embeddings"] == embedding)
                    & (filtered_df["cell_dataset.is_pert"] == is_pert)
                    & (filtered_df["cell_dataset.aggregation"] == aggregation)
                ]

                if sub_df.empty:
                    continue

                if criterion == "mse":
                    if sub_df["val_mse"].isna().all():
                        continue
                    best_idx = sub_df["val_mse"].idxmin()
                elif criterion == "spearman":
                    if sub_df["val_spearman"].isna().all():
                        continue
                    best_idx = sub_df["val_spearman"].idxmax()
                else:
                    raise ValueError(
                        "Invalid criterion. Choose either 'mse' or 'spearman'."
                    )

                best_config = sub_df.loc[
                    best_idx,
                    [col for col in sub_df.columns if col.startswith(f"{model_type}.")],
                ]
                config_hash = hashlib.sha256(
                    str(best_config.to_dict()).encode()
                ).hexdigest()

                if config_hash not in best_configs:
                    best_configs[config_hash] = 0
                best_configs[config_hash] += 1

    return best_configs


def plot_histogram(df, best_configs, model_type, criterion, max_size):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(best_configs)), list(best_configs.values()), align="center")

    # Create compressed labels for configurations
    compressed_labels = []
    for config_hash in best_configs.keys():
        config = df[
            df.apply(
                lambda row: hashlib.sha256(
                    str(
                        row[
                            [
                                col
                                for col in df.columns
                                if col.startswith(f"{model_type}.")
                            ]
                        ].to_dict()
                    ).encode()
                ).hexdigest()
                == config_hash,
                axis=1,
            )
        ].iloc[0][[col for col in df.columns if col.startswith(f"{model_type}.")]]
        label = ", ".join(
            [f"{param.split('.')[-1]}={value}" for param, value in config.items()]
        )
        compressed_labels.append(label)

    plt.xticks(range(len(best_configs)), compressed_labels, rotation="vertical")
    plt.xlabel("Configuration")
    plt.ylabel("Frequency")
    plt.title(
        f"{model_type.capitalize()} Best Configs Histogram - {criterion.upper()} - Max Size: {max_size}"
    )
    plt.tight_layout()

    # Save the image with the desired file path format
    image_path = osp.join(
        ASSET_IMAGES_DIR,
        f"experiments_smf-dmf-tmf-001_results_{model_type}_best_configs_histogram_{criterion}_{max_size}.png",
    )
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


def find_overall_best_config(df, model_type, criterion, max_size):
    best_configs = find_best_config(df, model_type, criterion, max_size)
    if not best_configs:
        return None
    best_config_hash = max(best_configs, key=best_configs.get)
    best_config_row = df[
        df.apply(
            lambda row: hashlib.sha256(
                str(
                    row[
                        [col for col in df.columns if col.startswith(f"{model_type}.")]
                    ].to_dict()
                ).encode()
            ).hexdigest()
            == best_config_hash,
            axis=1,
        )
    ]
    return best_config_row.iloc[0][
        [col for col in df.columns if col.startswith(f"{model_type}.")]
    ]


def main():
    for model_type in ["svr", "random_forest"]:
        for criterion in ["mse", "spearman"]:
            csv_path = osp.join(RESULTS_DIR, model_type, f"combined_df_{criterion}.csv")

            if not osp.exists(csv_path):
                print(
                    f"CSV file not found for {model_type.capitalize()} - {criterion.upper()}. Skipping..."
                )
                print()
                continue

            df = pd.read_csv(csv_path)

            for max_size in [1e3, 1e4]:
                best_configs = find_best_config(df, model_type, criterion, max_size)
                plot_histogram(df, best_configs, model_type, criterion, max_size)
                overall_best_config = find_overall_best_config(
                    df, model_type, criterion, max_size
                )
                if overall_best_config is not None:
                    print(
                        f"Overall Best Config for {model_type.capitalize()} - {criterion.upper()} - Max Size: {max_size}:"
                    )
                    print(overall_best_config)
                    print()
                else:
                    print(
                        f"No valid configurations found for {model_type.capitalize()} - {criterion.upper()} - Max Size: {max_size}"
                    )
                    print()


if __name__ == "__main__":
    main()
