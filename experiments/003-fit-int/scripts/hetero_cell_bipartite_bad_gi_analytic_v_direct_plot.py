# experiments/003-fit-int/scripts/hetero_cell_bipartite_bad_gi_analytic_v_direct_plot
# [[experiments.003-fit-int.scripts.hetero_cell_bipartite_bad_gi_analytic_v_direct_plot]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/hetero_cell_bipartite_bad_gi_analytic_v_direct_plot
# Test file: experiments/003-fit-int/scripts/test_hetero_cell_bipartite_bad_gi_analytic_v_direct_plot.py



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import jensenshannon
import os.path as osp
import torchcell

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def calc_metrics(true, pred):
    pearson_corr, p_value_pearson = pearsonr(true, pred)
    spearman_corr, p_value_spearman = spearmanr(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)

    # Calculate Wasserstein distance
    wasserstein = wasserstein_distance(true, pred)

    # For Jensen-Shannon divergence
    min_val = min(np.min(true), np.min(pred))
    max_val = max(np.max(true), np.max(pred))
    bins = np.linspace(min_val, max_val, 20)

    hist_true, _ = np.histogram(true, bins=bins, density=True)
    hist_pred, _ = np.histogram(pred, bins=bins, density=True)

    # Add small constant to avoid zero probabilities
    hist_true = hist_true + 1e-10
    hist_pred = hist_pred + 1e-10

    # Normalize
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)

    # Calculate JS divergence
    js_div = jensenshannon(hist_true, hist_pred)

    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "RMSE": rmse,
        "MAE": mae,
        "Wasserstein": wasserstein,
        "JS_Divergence": js_div,
    }


def plot_comparison(df_subset, gene_count):
    # Get sample size
    sample_size = len(df_subset)

    # Calculate metrics
    metrics_direct = calc_metrics(df_subset["epsilon_true"], df_subset["epsilon_pred"])
    metrics_fitness = calc_metrics(
        df_subset["epsilon_true"], df_subset["epsilon_from_fitness_pred"]
    )

    # Create the plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Comparison of Prediction Methods ({gene_count}-gene deletions, n={sample_size})",
        fontsize=16,
    )

    # Determine common limits for top row plots
    epsilon_min = min(
        df_subset["epsilon_true"].min(),
        df_subset["epsilon_pred"].min(),
        df_subset["epsilon_from_fitness_pred"].min(),
    )

    epsilon_max = max(
        df_subset["epsilon_true"].max(),
        df_subset["epsilon_pred"].max(),
        df_subset["epsilon_from_fitness_pred"].max(),
    )

    # Add a small margin
    margin = 0.05 * (epsilon_max - epsilon_min)
    plot_min = epsilon_min - margin
    plot_max = epsilon_max + margin

    # Plot direct predictions with predictions on x-axis
    axs[0, 0].scatter(df_subset["epsilon_pred"], df_subset["epsilon_true"], s=2)
    axs[0, 0].set_xlabel("Direct Epsilon Prediction")
    axs[0, 0].set_ylabel("True Epsilon")
    axs[0, 0].plot(
        [plot_min, plot_max], [plot_min, plot_max], "k--", alpha=0.5, linewidth=2
    )
    axs[0, 0].set_xlim(plot_min, plot_max)
    axs[0, 0].set_ylim(plot_min, plot_max)
    axs[0, 0].set_title(f'Direct Prediction (Pearson: {metrics_direct["Pearson"]:.3f})')

    # Plot fitness-based predictions with predictions on x-axis
    axs[0, 1].scatter(
        df_subset["epsilon_from_fitness_pred"], df_subset["epsilon_true"], s=2
    )
    axs[0, 1].set_xlabel("Fitness-based Epsilon Prediction")
    axs[0, 1].set_ylabel("True Epsilon")
    axs[0, 1].plot(
        [plot_min, plot_max], [plot_min, plot_max], "k--", alpha=0.5, linewidth=2
    )
    axs[0, 1].set_xlim(plot_min, plot_max)
    axs[0, 1].set_ylim(plot_min, plot_max)
    axs[0, 1].set_title(
        f'Fitness-based Prediction (Pearson: {metrics_fitness["Pearson"]:.3f})'
    )

    # Determine common limits for bottom row (error histograms)
    errors_direct = df_subset["epsilon_true"] - df_subset["epsilon_pred"]
    errors_fitness = df_subset["epsilon_true"] - df_subset["epsilon_from_fitness_pred"]

    error_min = min(errors_direct.min(), errors_fitness.min())
    error_max = max(errors_direct.max(), errors_fitness.max())

    # Add a small margin
    margin = 0.05 * (error_max - error_min)
    error_plot_min = error_min - margin
    error_plot_max = error_max + margin

    # Set common bins for both histograms
    hist_bins = np.linspace(error_plot_min, error_plot_max, 100)

    # Plot error distributions for direct predictions
    axs[1, 0].hist(errors_direct, bins=hist_bins, alpha=0.7)
    axs[1, 0].set_xlabel("Error (True - Predicted)")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_xlim(error_plot_min, error_plot_max)
    axs[1, 0].set_title(
        f'Direct Prediction Errors (RMSE: {metrics_direct["RMSE"]:.3f})'
    )

    # Plot error distributions for fitness-based predictions
    axs[1, 1].hist(errors_fitness, bins=hist_bins, alpha=0.7)
    axs[1, 1].set_xlabel("Error (True - Predicted)")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_xlim(error_plot_min, error_plot_max)
    axs[1, 1].set_title(
        f'Fitness-based Prediction Errors (RMSE: {metrics_fitness["RMSE"]:.3f})'
    )

    # Make sure y-axis limits are the same for error histograms
    max_count = max([axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1]])
    axs[1, 0].set_ylim(0, max_count)
    axs[1, 1].set_ylim(0, max_count)

    plt.tight_layout()
    plt.savefig(f"notes/assets/images/prediction_comparison_{gene_count}gene_n{sample_size}.png", dpi=300)
    plt.show()

    return metrics_direct, metrics_fitness, sample_size


def create_metrics_table(metrics_direct, metrics_fitness, sample_size):
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Pearson",
                "Spearman",
                "RMSE",
                "MAE",
                "Wasserstein",
                "JS_Divergence",
                "Sample Size",
            ],
            "Direct Prediction": [
                f"{metrics_direct['Pearson']:.4f}",
                f"{metrics_direct['Spearman']:.4f}",
                f"{metrics_direct['RMSE']:.4f}",
                f"{metrics_direct['MAE']:.4f}",
                f"{metrics_direct['Wasserstein']:.4f}",
                f"{metrics_direct['JS_Divergence']:.4f}",
                f"{sample_size}",
            ],
            "Fitness-based Prediction": [
                f"{metrics_fitness['Pearson']:.4f}",
                f"{metrics_fitness['Spearman']:.4f}",
                f"{metrics_fitness['RMSE']:.4f}",
                f"{metrics_fitness['MAE']:.4f}",
                f"{metrics_fitness['Wasserstein']:.4f}",
                f"{metrics_fitness['JS_Divergence']:.4f}",
                f"{sample_size}",
            ],
        }
    )
    return metrics_df


def main():
    # Read the data
    df = pd.read_csv("/Users/michaelvolk/Documents/projects/torchcell/experiments/003-fit-int/results/test_results_digenic_only.csv ")

    # Drop rows with NaN values
    df = df.dropna()

    # Let's print out what gene deletion counts are available in the data
    print(f"Available gene deletion counts: {df['num_gene_deletions'].unique()}")
    print(
        f"Count of samples by gene deletion: {df['num_gene_deletions'].value_counts()}"
    )

    # Filter for different gene deletion counts
    df_2gene = df[df["num_gene_deletions"] == 2]
    df_3gene = df[df["num_gene_deletions"] == 3]

    # Print sample sizes to debug
    print(f"Number of 2-gene deletion samples: {len(df_2gene)}")
    print(f"Number of 3-gene deletion samples: {len(df_3gene)}")

    # Generate plots and calculate metrics for 2-gene deletions
    if not df_2gene.empty:
        metrics_direct_2gene, metrics_fitness_2gene, sample_size_2gene = (
            plot_comparison(df_2gene, 2)
        )

        # Create and display metrics tables
        metrics_table_2gene = create_metrics_table(
            metrics_direct_2gene, metrics_fitness_2gene, sample_size_2gene
        )
        print("\nMetrics for 2-gene deletions:")
        print(metrics_table_2gene.to_markdown(index=False))

    # Generate plots and calculate metrics for 3-gene deletions if data exists
    if not df_3gene.empty:
        metrics_direct_3gene, metrics_fitness_3gene, sample_size_3gene = (
            plot_comparison(df_3gene, 3)
        )

        metrics_table_3gene = create_metrics_table(
            metrics_direct_3gene, metrics_fitness_3gene, sample_size_3gene
        )
        print("\nMetrics for 3-gene deletions:")
        print(metrics_table_3gene.to_markdown(index=False))
    else:
        print("\nNo 3-gene deletion data found in the dataset")


if __name__ == "__main__":
    main()
