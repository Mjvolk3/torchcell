# experiments/010-kuzmin-tmi/scripts/select_12_k200_tables_hist
# [[experiments.010-kuzmin-tmi.scripts.select_12_k200_tables_hist]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/select_12_k200_tables_hist
# Test file: experiments/010-kuzmin-tmi/scripts/test_select_12_k200_tables_hist.py

"""
Generate tables and histogram for 12-gene panel (k=200).

Outputs:
1. Singles table: 12 genes that need single mutant strains
2. Doubles table: 66 pairs sorted by number of triples they enable
3. Triples table: All panel triples with inferred predictions
4. Overlay histogram: All panel triples vs top-k constructible
"""

import os
import os.path as osp
from glob import glob
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.timestamp import timestamp

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

assert DATA_ROOT is not None, "DATA_ROOT environment variable not set"
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR environment variable not set"
assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT environment variable not set"

# Configuration
PANEL_SIZE = 12
K_VALUE = 200


def load_inference_data() -> pd.DataFrame:
    """Load the inference parquet file for the best model (Pearson=0.4619)."""
    inference_dir = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred"
    )
    pattern = osp.join(inference_dir, "*Pearson=0.4619*.parquet")
    files = glob(pattern)

    if not files:
        raise FileNotFoundError(f"No parquet file matching pattern: {pattern}")

    parquet_path = files[0]
    print(f"Loading inference data from: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} inferred triples")
    return df


def load_gene_selection_results() -> pd.DataFrame:
    """Load gene selection results."""
    results_path = osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi/results/gene_selection_results.csv"
    )
    df = pd.read_csv(results_path)
    print(f"Loaded gene selection results: {len(df)} rows")
    return df


def parse_selected_genes(genes_str: str) -> list[str]:
    """Parse comma-separated gene string into list."""
    return [g.strip() for g in genes_str.split(",")]


def create_triple_lookup(df: pd.DataFrame) -> dict[frozenset, float]:
    """Create lookup from triple frozenset to prediction value."""
    lookup = {}
    for _, row in df.iterrows():
        triple = frozenset([row["gene1"], row["gene2"], row["gene3"]])
        lookup[triple] = row["prediction"]
    return lookup


def create_singles_table(genes: list[str]) -> pd.DataFrame:
    """Create table of single gene mutants needed."""
    df = pd.DataFrame({"gene": sorted(genes), "mutant_type": "single"})
    df["index"] = range(1, len(df) + 1)
    return df[["index", "gene", "mutant_type"]]


def load_top_k_constructible_triples(panel_size: int, k: int) -> set[frozenset]:
    """
    Load top-k constructible triples as a set of frozensets.

    These are triples that are BOTH:
    1. In the global top-k by predicted gene interaction
    2. Constructible from the selected gene panel
    """
    csv_path = osp.join(
        EXPERIMENT_ROOT,
        f"010-kuzmin-tmi/results/top_k_constructible_panel{panel_size}_k{k}.csv",
    )
    if not osp.exists(csv_path):
        print(f"WARNING: No top-k constructible file at {csv_path}")
        return set()

    df = pd.read_csv(csv_path)
    triples = set()
    for _, row in df.iterrows():
        triples.add(frozenset([row["gene1"], row["gene2"], row["gene3"]]))
    print(f"Loaded {len(triples)} top-k constructible triples")
    return triples


def create_doubles_table(
    genes: list[str],
    triple_lookup: dict[frozenset, float],
    top_k_triples: set[frozenset],
) -> pd.DataFrame:
    """
    Create table of double mutants sorted by how many top-k triples they enable.

    For each pair, count:
    - triples_enabled: how many panel triples (with predictions) contain this pair
    - enables_triple_in_top_k: how many TOP-K triples contain this pair (most important!)
    - cumulative_triples_in_top_k: if building in order, how many top-k triples are reachable
    """
    gene_set = set(genes)
    pair_data = []

    for g1, g2 in combinations(sorted(genes), 2):
        pair = frozenset([g1, g2])

        # Count triples containing this pair that have inferred predictions
        triple_count = 0
        for g3 in gene_set - {g1, g2}:
            triple = frozenset([g1, g2, g3])
            if triple in triple_lookup:
                triple_count += 1

        # Count how many TOP-K triples contain this pair
        top_k_count = sum(1 for t in top_k_triples if pair.issubset(t))

        pair_data.append(
            {
                "gene1": g1,
                "gene2": g2,
                "enables_triple_in_top_k": top_k_count,
                "triples_enabled": triple_count,
                "max_possible": len(gene_set) - 2,  # C(n-2, 1) = n-2
            }
        )

    df = pd.DataFrame(pair_data)
    # Sort by top-k first, then by triples_enabled as tiebreaker
    df = df.sort_values(
        ["enables_triple_in_top_k", "triples_enabled"], ascending=[False, False]
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Calculate cumulative triples after building each double in sorted order
    # A triple is "reachable" if ANY of its 3 pairs has been built
    # (since double + 1 transformation = triple)
    cumulative_triples = []
    built_pairs: set[frozenset] = set()

    for _, row in df.iterrows():
        # Add this pair to built set
        pair = frozenset([row["gene1"], row["gene2"]])
        built_pairs.add(pair)

        # Count how many top-k triples have AT LEAST ONE pair built
        reachable = 0
        for triple in top_k_triples:
            triple_genes = list(triple)
            pairs_in_triple = [
                frozenset([triple_genes[0], triple_genes[1]]),
                frozenset([triple_genes[0], triple_genes[2]]),
                frozenset([triple_genes[1], triple_genes[2]]),
            ]
            if any(p in built_pairs for p in pairs_in_triple):
                reachable += 1

        cumulative_triples.append(reachable)

    df["cumulative_triples_in_top_k"] = cumulative_triples

    return df[
        [
            "rank",
            "gene1",
            "gene2",
            "enables_triple_in_top_k",
            "cumulative_triples_in_top_k",
            "triples_enabled",
            "max_possible",
        ]
    ]


def create_triples_table(
    genes: list[str], triple_lookup: dict[frozenset, float]
) -> pd.DataFrame:
    """Create table of all panel triples with their inferred predictions."""
    triple_data = []

    for g1, g2, g3 in combinations(sorted(genes), 3):
        triple = frozenset([g1, g2, g3])
        if triple in triple_lookup:
            prediction = triple_lookup[triple]
            triple_data.append(
                {
                    "gene1": g1,
                    "gene2": g2,
                    "gene3": g3,
                    "inferred_gene_interaction": prediction,
                }
            )

    df = pd.DataFrame(triple_data)
    if len(df) > 0:
        df = df.sort_values("inferred_gene_interaction", ascending=False).reset_index(
            drop=True
        )
        df["rank"] = range(1, len(df) + 1)
        return df[["rank", "gene1", "gene2", "gene3", "inferred_gene_interaction"]]
    return df


def load_constructible_triples(panel_size: int, k: int) -> pd.DataFrame | None:
    """Load constructible triples parquet if it exists."""
    path = osp.join(
        EXPERIMENT_ROOT,
        f"010-kuzmin-tmi/results/constructible_triples_panel{panel_size}_k{k}.parquet",
    )
    if osp.exists(path):
        df = pd.read_parquet(path)
        print(f"Loaded {len(df)} constructible triples from {path}")
        return df
    print(f"No constructible triples file found at {path}")
    return None


def plot_overlay_histogram(
    all_predictions: list[float],
    constructible_predictions: list[float],
    panel_size: int,
    k: int,
    output_dir: str,
) -> None:
    """
    Plot overlay histogram showing all panel triples vs constructible triples.

    Args:
        all_predictions: Predictions for all triples in the panel (that have inferred values)
        constructible_predictions: Predictions for top-k constructible triples
        panel_size: Number of genes in panel
        k: k value for gene selection
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    design_space = len(list(combinations(range(panel_size), 3)))
    bins = np.linspace(-0.05, 0.20, 26)

    # Plot all panel triples (background)
    ax.hist(
        all_predictions,
        bins=bins,
        color="#2166ac",
        edgecolor="black",
        alpha=0.5,
        label=f"all inferred: {len(all_predictions)}/{design_space}",
    )

    # Overlay constructible triples
    if constructible_predictions:
        ax.hist(
            constructible_predictions,
            bins=bins,
            color="#b2182b",
            edgecolor="black",
            alpha=0.7,
            label=f"top-{k} constructible: {len(constructible_predictions)}",
        )

    # Reference lines
    ax.axvline(
        x=0, color="gray", linestyle="--", alpha=0.7, label="gene interaction = 0"
    )

    if all_predictions:
        mean_all = np.mean(all_predictions)
        ax.axvline(
            x=mean_all,
            color="#2166ac",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=f"mean (all): {mean_all:.3f}",
        )

    if constructible_predictions:
        mean_constr = np.mean(constructible_predictions)
        ax.axvline(
            x=mean_constr,
            color="#b2182b",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=f"mean (constructible): {mean_constr:.3f}",
        )

    ax.set_xlabel("Inferred Gene Interaction", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Gene Interaction Distribution: {panel_size}-Gene Panel (k={k})\n"
        f"Design space = {design_space} triples",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    output_path = osp.join(
        output_dir, f"overlay_histogram_panel{panel_size}_k{k}_{timestamp()}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print(f"Tables and Histogram for {PANEL_SIZE}-Gene Panel (k={K_VALUE})")
    print("=" * 60)

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)
    inference_df = load_inference_data()
    selection_df = load_gene_selection_results()

    # Get panel genes for k=200
    panel_row = selection_df[
        (selection_df["panel_size"] == PANEL_SIZE) & (selection_df["k"] == K_VALUE)
    ].iloc[0]

    genes = parse_selected_genes(panel_row["selected_genes"])
    print(f"\n{PANEL_SIZE}-gene panel (k={K_VALUE}):")
    print(f"  Genes: {genes}")

    # Create triple lookup
    triple_lookup = create_triple_lookup(inference_df)

    # Load top-k constructible triples
    top_k_triples = load_top_k_constructible_triples(PANEL_SIZE, K_VALUE)

    # Create output directory
    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results")
    os.makedirs(results_dir, exist_ok=True)

    plots_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
    os.makedirs(plots_dir, exist_ok=True)

    # =================================================================
    # Table 1: Singles
    # =================================================================
    print("\n" + "=" * 60)
    print("Creating Singles Table...")
    print("=" * 60)
    singles_df = create_singles_table(genes)
    singles_path = osp.join(
        results_dir, f"singles_table_panel{PANEL_SIZE}_k{K_VALUE}.csv"
    )
    singles_df.to_csv(singles_path, index=False)
    print(f"Saved: {singles_path}")
    print(singles_df.to_string(index=False))

    # =================================================================
    # Table 2: Doubles (sorted by top-k triples enabled)
    # =================================================================
    print("\n" + "=" * 60)
    print("Creating Doubles Table...")
    print("=" * 60)
    doubles_df = create_doubles_table(genes, triple_lookup, top_k_triples)
    doubles_path = osp.join(
        results_dir, f"doubles_table_panel{PANEL_SIZE}_k{K_VALUE}.csv"
    )
    doubles_df.to_csv(doubles_path, index=False)
    print(f"Saved: {doubles_path}")
    print(f"\nTop 10 doubles by top-k triples enabled:")
    print(doubles_df.head(10).to_string(index=False))
    print(f"\nTotal doubles: {len(doubles_df)}")
    print(f"Doubles enabling 0 top-k triples: {(doubles_df['enables_triple_in_top_k'] == 0).sum()}")

    # =================================================================
    # Table 3: Triples with predictions
    # =================================================================
    print("\n" + "=" * 60)
    print("Creating Triples Table...")
    print("=" * 60)
    triples_df = create_triples_table(genes, triple_lookup)
    triples_path = osp.join(
        results_dir, f"triples_table_panel{PANEL_SIZE}_k{K_VALUE}.csv"
    )
    triples_df.to_csv(triples_path, index=False)
    print(f"Saved: {triples_path}")
    print(f"\nTop 10 triples by inferred gene interaction:")
    print(triples_df.head(10).to_string(index=False))
    print(f"\nTotal triples with predictions: {len(triples_df)}/220")

    # =================================================================
    # Histogram with overlay
    # =================================================================
    print("\n" + "=" * 60)
    print("Creating Overlay Histogram...")
    print("=" * 60)

    # Get all panel triple predictions
    all_predictions = triples_df["inferred_gene_interaction"].tolist()

    # Load constructible triples if available
    constructible_df = load_constructible_triples(PANEL_SIZE, K_VALUE)
    constructible_predictions = []
    if constructible_df is not None:
        constructible_predictions = constructible_df["prediction"].tolist()
    else:
        # If no file, use top-k from the triples table as approximation
        top_k_constructible = int(panel_row["top_k_constructible"])
        print(f"Using top {top_k_constructible} from triples table as constructible")
        constructible_predictions = triples_df.head(top_k_constructible)[
            "inferred_gene_interaction"
        ].tolist()

    plot_overlay_histogram(
        all_predictions, constructible_predictions, PANEL_SIZE, K_VALUE, plots_dir
    )

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Panel: {PANEL_SIZE} genes, k={K_VALUE}")
    print(f"Singles needed: {len(singles_df)}")
    print(f"Doubles needed: {len(doubles_df)}")
    print(f"Triples in design space: 220")
    print(f"Triples with predictions: {len(triples_df)}")
    print(f"Top-k constructible: {len(constructible_predictions)}")

    if all_predictions:
        print(f"\nAll triples - mean: {np.mean(all_predictions):.4f}, "
              f"std: {np.std(all_predictions):.4f}")
    if constructible_predictions:
        print(f"Constructible - mean: {np.mean(constructible_predictions):.4f}, "
              f"std: {np.std(constructible_predictions):.4f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
