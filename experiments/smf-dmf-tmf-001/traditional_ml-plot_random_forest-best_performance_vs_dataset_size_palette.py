import pandas as pd
import matplotlib.pyplot as plt

# --- Nature paper style (WYSIWYG, added for SVG export): Arial, 6 pt min, editable SVG text ---
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "legend.title_fontsize": 6, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.5, "lines.linewidth": 0.7, "patch.linewidth": 0.4,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.01,
})
import numpy as np
import os
from dotenv import load_dotenv
from torchcell.utils import savefig_true_size_svg, PLOT_PALETTE

load_dotenv()

# Define constants
RESULTS_DIR = "experiments/smf-dmf-tmf-001/results/random_forest"
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
MAX_SIZES = [1000, 10000, 100000]

# Define color list (reversed order)
color_list = PLOT_PALETTE

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
    "nt_window_5979",
    "nt_window_three_prime_300",
    "nt_window_five_prime_1003",
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


def create_plot(data_dict, metric="val_r2", ylim=None, name_suffix="", show_legend=True):
    # Wider canvas when the legend is drawn inline; tighter plot-only canvas when not
    # (the legend is emitted separately by create_legend() so one shared legend can
    # label side-by-side no-legend panels).
    plt.figure(figsize=(4.6, 2.4) if show_legend else (2.6, 2.2))

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
                x, y, "-o", color=color, label=node_embedding, linewidth=0.9, markersize=2
            )

    plt.xscale("log")
    plt.xticks(MAX_SIZES, ["$10^3$", "$10^4$", "$10^5$"])
    plt.xlabel("Dataset Size", fontsize=6)

    humanized_metric = humanize_metric(metric)
    plt.ylabel(humanized_metric, fontsize=6)

    # Fixed y-scale (shared across experiments) when requested, else autoscale.
    if ylim is not None:
        plt.ylim(ylim)

    if show_legend:
        # Encoding-color legend to the right, single column in node_embedding_order.
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=6,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.3,
            framealpha=0.5,
        )
    # Grid only at the 3 measured dataset sizes. Minor log gridlines (2e3, 3e3, ...)
    # would imply readable intermediate sizes, but each line has only 3 data points;
    # the connecting line is a trend guide, not interpolation.
    plt.grid(True, which="major", ls="--", alpha=0.5)
    plt.minorticks_off()

    # 6 pt ticks (Nature minimum); the previous labelsize=14 override was the
    # "text mis-sized" bug -- do NOT reintroduce a larger tick label size.
    plt.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2)
    plt.tick_params(axis="both", which="minor", labelsize=6, width=0.4, length=1.2)

    if show_legend:
        # Reserve the right ~40% for the 17-entry legend so the plot keeps real width.
        plt.subplots_adjust(left=0.10, right=0.60, top=0.94, bottom=0.17)
    else:
        # No legend: full-width plot, margins trimmed to just the axis labels.
        plt.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.16)

    legend_tag = "" if show_legend else "_no_legend"
    savefig_true_size_svg(
        plt.gcf(),
        os.path.join(
            ASSET_IMAGES_DIR,
            f"smf-dmf-tmf-001_node_embedding_performance_{metric}{name_suffix}{legend_tag}_palette.svg",
        ),
    )
    plt.close()


def create_legend():
    # Standalone encoding-color legend (shared across metrics and experiments) so a
    # single legend can label side-by-side no-legend panels. Tight bounding box.
    color_dict = {
        embedding: color for embedding, color in zip(node_embedding_order, color_list)
    }
    fig = plt.figure(figsize=(2.2, 2.6))
    ax = fig.add_subplot(111)
    handles = [
        plt.Line2D(
            [0], [0], color=color_dict[e], marker="o", markersize=2, linewidth=0.9, label=e
        )
        for e in node_embedding_order
    ]
    leg = ax.legend(
        handles=handles,
        loc="center",
        fontsize=6,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
        framealpha=0.5,
    )
    ax.axis("off")
    # Crop to the legend frame itself. bbox_inches="tight" crops to the (off) axes
    # box, which left a few mm of top/bottom margin; the legend's own window extent
    # is exact.
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    savefig_true_size_svg(
        fig,
        os.path.join(ASSET_IMAGES_DIR, "node_embedding_legend_palette.svg"),
        bbox_inches=bbox,
        pad_inches=0.01,
    )
    plt.close(fig)


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

    metrics = ["test_spearman", "test_pearson", "test_mse"]

    for metric in metrics:
        create_plot(data_dict, metric)
        create_plot(data_dict, metric, show_legend=False)

    # Fixed shared y-scales for test_pearson so the fitness (smf-dmf-tmf-001) and
    # interaction (002-dmi-tmi) plots can sit side-by-side on identical axes and
    # show the greater difficulty of gene interactions. Keep this dict IDENTICAL in
    # both progression scripts. Combined data range across experiments is ~[-0.05, 0.88].
    shared_pearson_ylims = {
        "_shared_0_1": (0.0, 1.0),
    }
    for suffix, ylim in shared_pearson_ylims.items():
        create_plot(data_dict, "test_pearson", ylim=ylim, name_suffix=suffix)
        create_plot(
            data_dict, "test_pearson", ylim=ylim, name_suffix=suffix, show_legend=False
        )

    create_legend()


if __name__ == "__main__":
    main()
