# experiments/010-kuzmin-tmi/scripts/select_12_and_24_genes_top_triples.py
# [[experiments.010-kuzmin-tmi.scripts.select_12_and_24_genes_top_triples]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/select_12_and_24_genes_top_triples.py
# Test file: experiments/010-kuzmin-tmi/scripts/test_select_12_and_24_genes_top_triples.py

"""
Select 12 and 24 genes that maximize coverage of top-k extreme triples.

Algorithm:
1. Load inference predictions from best model (Pearson=0.4619)
2. For panel_size in {12, 24}:
   For k in {25, 50, 100, 200}:
   - Extract top-k and bottom-k triples
   - Greedy selection of genes maximizing top-k coverage
   - Local swap refinement
   - Report coverage statistics
3. Generate combined visualizations comparing 12 vs 24 gene panels

Design space:
- 12 genes: C(12,3) = 220 possible triples (constrained regime)
- 24 genes: C(24,3) = 2024 possible triples (expansive regime)
"""

import math
import os
import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.timestamp import timestamp

# Load environment variables
load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

# Use torchcell style
plt.style.use("torchcell/torchcell.mplstyle")


def load_data() -> pd.DataFrame:
    """Load the inference parquet file for the best model (Pearson=0.4619)."""
    inference_dir = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred"
    )

    # Find the Pearson=0.4619 parquet file
    pattern = osp.join(inference_dir, "*Pearson=0.4619*.parquet")
    files = glob(pattern)

    if not files:
        raise FileNotFoundError(f"No parquet file matching pattern: {pattern}")

    parquet_path = files[0]
    print(f"Loading data from: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} triples")

    return df


def load_sameith_doubles() -> set[str]:
    """
    Load genes with existing double mutants from Sameith et al.

    These genes have published gene expression data for their pairwise combinations,
    enabling reproducibility comparisons with literature.

    Returns:
        Set of gene names (systematic names like YBL054W)
    """
    path = osp.join(
        EXPERIMENT_ROOT,
        "006-kuzmin-tmi/results/inference_preprocessing_expansion/sameith_doubles_genes.txt",
    )
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_all_gene_lists() -> dict[str, set[str]]:
    """
    Load all gene lists from inference_preprocessing_expansion directory.

    Returns:
        Dict mapping list name to set of genes
    """
    base_path = osp.join(
        EXPERIMENT_ROOT,
        "006-kuzmin-tmi/results/inference_preprocessing_expansion",
    )

    gene_lists = {}
    list_files = [
        ("betaxanthin", "betaxanthin_genes.txt"),
        ("expanded_inference", "expanded_genes_inference_1.txt"),
        ("metabolic", "expanded_metabolic_genes.txt"),
        ("kemmeren", "kemmeren_responsive_genes.txt"),
        ("ohya", "ohya_morphology_genes.txt"),
        ("sameith", "sameith_doubles_genes.txt"),
    ]

    for name, filename in list_files:
        path = osp.join(base_path, filename)
        if osp.exists(path):
            with open(path) as f:
                genes = {line.strip() for line in f if line.strip()}
                gene_lists[name] = genes
                print(f"  Loaded {name}: {len(genes)} genes")
        else:
            print(f"  Warning: {filename} not found")
            gene_lists[name] = set()

    return gene_lists


def compute_gene_list_overlaps(
    selected_genes: list[str], gene_lists: dict[str, set[str]]
) -> dict[str, int]:
    """
    Compute overlap counts between selected genes and each gene list.

    Args:
        selected_genes: List of selected gene names
        gene_lists: Dict mapping list name to set of genes

    Returns:
        Dict mapping list name to overlap count
    """
    selected_set = set(selected_genes)
    overlaps = {}
    for name, genes in gene_lists.items():
        overlaps[f"overlap_{name}"] = len(selected_set & genes)
    return overlaps


def get_priority_genes(
    top_k_triples: set[frozenset], sameith_genes: set[str]
) -> set[str]:
    """
    Get genes that appear in top-k triples AND have existing Sameith doubles.

    These genes should be prioritized for selection because they enable
    reproducibility comparisons with published gene expression data.

    Args:
        top_k_triples: Set of top-k triples (frozensets of gene names)
        sameith_genes: Set of genes with existing Sameith doubles

    Returns:
        Set of priority genes (intersection)
    """
    top_k_genes = get_all_genes_from_triples(top_k_triples)
    return top_k_genes & sameith_genes


def compute_doubles_metrics(
    selected_genes: list[str], sameith_genes: set[str]
) -> tuple[int, int, int]:
    """
    Compute metrics about double mutants needed for a gene panel.

    A "sameith double" is a pair where BOTH genes are in sameith_genes,
    meaning published gene expression data exists for that pair.

    Args:
        selected_genes: List of selected gene names
        sameith_genes: Set of genes with existing Sameith doubles

    Returns:
        Tuple of (total_doubles, sameith_overlap, new_doubles_needed)
    """
    from itertools import combinations

    S = set(selected_genes)
    total_doubles = math.comb(len(S), 2)

    # Count pairs where both genes are in sameith
    sameith_overlap = 0
    for g1, g2 in combinations(S, 2):
        if g1 in sameith_genes and g2 in sameith_genes:
            sameith_overlap += 1

    new_doubles_needed = total_doubles - sameith_overlap

    return total_doubles, sameith_overlap, new_doubles_needed


def get_top_bottom_k(df: pd.DataFrame, k: int) -> tuple[set[frozenset], set[frozenset]]:
    """
    Extract top-k and bottom-k triples as frozensets of genes.

    Args:
        df: DataFrame with columns gene1, gene2, gene3, prediction
        k: Number of top/bottom triples to extract

    Returns:
        Tuple of (top_k_triples, bottom_k_triples) as sets of frozensets
    """
    # Sort by prediction (highest first)
    df_sorted = df.sort_values("prediction", ascending=False)

    # Top-k (highest predictions)
    top_k_df = df_sorted.head(k)
    top_k_triples = {
        frozenset([row["gene1"], row["gene2"], row["gene3"]])
        for _, row in top_k_df.iterrows()
    }

    # Bottom-k (lowest predictions)
    bottom_k_df = df_sorted.tail(k)
    bottom_k_triples = {
        frozenset([row["gene1"], row["gene2"], row["gene3"]])
        for _, row in bottom_k_df.iterrows()
    }

    return top_k_triples, bottom_k_triples


def compute_constructible(S: set[str], triples: set[frozenset]) -> int:
    """
    Count how many triples are fully constructible from gene set S.

    A triple is constructible iff all 3 genes are in S.
    """
    count = 0
    for triple in triples:
        if triple.issubset(S):
            count += 1
    return count


def get_all_genes_from_triples(triples: set[frozenset]) -> set[str]:
    """Extract all unique genes from a set of triples."""
    genes = set()
    for triple in triples:
        genes.update(triple)
    return genes


def count_triples_containing_gene(gene: str, triples: set[frozenset]) -> int:
    """Count how many triples contain the given gene."""
    return sum(1 for t in triples if gene in t)


def greedy_select_genes(
    top_k_triples: set[frozenset],
    bot_k_triples: set[frozenset],
    n_genes: int = 12,
    sameith_genes: set[str] | None = None,
) -> list[str]:
    """
    Greedy selection of genes to maximize top-k triple coverage.

    Priority system:
    1. First include genes that appear in BOTH top-k triples AND sameith_genes
       (these enable reproducibility with published gene expression data)
    2. Fill remaining slots greedily

    Tie-breaking (in order):
    1. Maximize marginal gain in top-k coverage
    2. Prefer sameith genes (enables literature reproducibility)
    3. Maximize bottom coverage
    4. Maximize distinct top-k triples containing the gene
    5. Lexicographic gene name

    Args:
        top_k_triples: Set of top-k triples to optimize for
        bot_k_triples: Set of bottom-k triples for tie-breaking
        n_genes: Number of genes to select
        sameith_genes: Set of genes with existing Sameith doubles (optional)

    Returns:
        List of selected gene names (in order of selection)
    """
    if sameith_genes is None:
        sameith_genes = set()

    # Get all candidate genes from top-k triples
    all_genes = get_all_genes_from_triples(top_k_triples)
    # Also include genes from bottom-k for better tie-breaking
    all_genes.update(get_all_genes_from_triples(bot_k_triples))

    S = set()
    selected_order = []

    # Step 1: Auto-include priority genes (in top-k AND sameith)
    priority_genes = get_priority_genes(top_k_triples, sameith_genes)
    if priority_genes:
        # Only take up to n_genes priority genes
        priority_to_add = sorted(priority_genes)[:n_genes]
        S.update(priority_to_add)
        selected_order.extend(priority_to_add)
        tqdm.write(
            f"  Auto-included {len(priority_to_add)} priority genes (in top-k AND sameith): "
            f"{priority_to_add}"
        )

    if len(S) >= n_genes:
        return selected_order[:n_genes]

    # Step 2: Greedy fill remaining slots
    remaining = n_genes - len(S)
    pbar = tqdm(total=remaining, desc="Greedy selection", unit="gene")

    while len(S) < n_genes:
        best_gene = None
        # (marginal_gain, is_sameith, bot_coverage, top_count, name)
        best_score = (-1, -1, -1, -1, "")

        # Evaluate all candidate genes
        candidates = all_genes - S
        for g in candidates:
            # Compute marginal gain for top-k
            S_with_g = S | {g}
            current_top = compute_constructible(S, top_k_triples)
            new_top = compute_constructible(S_with_g, top_k_triples)
            marginal_gain = new_top - current_top

            # Tie-breakers
            is_sameith = 1 if g in sameith_genes else 0  # Prefer sameith genes
            bot_coverage = compute_constructible(S_with_g, bot_k_triples)
            top_count = count_triples_containing_gene(g, top_k_triples)

            # Score tuple for comparison (higher is better, except name)
            score = (marginal_gain, is_sameith, bot_coverage, top_count, g)

            if score[:4] > best_score[:4] or (
                score[:4] == best_score[:4] and g < best_score[4]
            ):
                best_score = score
                best_gene = g

        if best_gene is None:
            # No more genes available
            break

        S.add(best_gene)
        selected_order.append(best_gene)
        pbar.update(1)
        sameith_marker = " [sameith]" if best_gene in sameith_genes else ""
        pbar.set_postfix(gene=f"{best_gene}{sameith_marker}", gain=best_score[0])

    pbar.close()
    return selected_order


def local_swap_refinement(
    S: list[str],
    top_k_triples: set[frozenset],
    bot_k_triples: set[frozenset],
    all_genes: set[str],
) -> list[str]:
    """
    Local swap refinement to improve top-k coverage.

    Try swapping genes in S with genes outside S.
    Accept if top coverage improves (prefer if also improves bottom).

    Args:
        S: Current selected gene list
        top_k_triples: Target triples
        bot_k_triples: Bonus triples for tie-breaking
        all_genes: All candidate genes

    Returns:
        Refined gene list
    """
    S_set = set(S)
    improved = True
    iterations = 0

    while improved and iterations < 100:
        improved = False
        iterations += 1

        current_top = compute_constructible(S_set, top_k_triples)
        current_bot = compute_constructible(S_set, bot_k_triples)

        best_swap = None
        best_improvement = (0, 0)  # (top_improvement, bot_improvement)

        # Progress bar for swap candidates
        genes_out = list(S_set)
        genes_in = list(all_genes - S_set)
        total_swaps = len(genes_out) * len(genes_in)

        with tqdm(
            total=total_swaps,
            desc=f"Swap refinement iter {iterations}",
            unit="swap",
            leave=False,
        ) as pbar:
            for gene_out in genes_out:
                for gene_in in genes_in:
                    S_new = (S_set - {gene_out}) | {gene_in}
                    new_top = compute_constructible(S_new, top_k_triples)
                    new_bot = compute_constructible(S_new, bot_k_triples)

                    top_improvement = new_top - current_top
                    bot_improvement = new_bot - current_bot

                    if top_improvement > 0:
                        improvement = (top_improvement, bot_improvement)
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (gene_out, gene_in)

                    pbar.update(1)

        if best_swap:
            gene_out, gene_in = best_swap
            S_set = (S_set - {gene_out}) | {gene_in}
            improved = True
            tqdm.write(
                f"  Swap: {gene_out} -> {gene_in} "
                f"(top +{best_improvement[0]}, bot +{best_improvement[1]})"
            )

    return list(S_set)


def output_constructible_triples(
    S: set[str],
    top_k_triples: set[frozenset],
    bot_k_triples: set[frozenset],
    df: pd.DataFrame,
    panel_size: int,
    k: int,
    results_dir: str,
) -> pd.DataFrame:
    """
    Output constructible triples with their prediction scores.

    Args:
        S: Selected gene set
        top_k_triples: Set of top-k triples (frozensets)
        bot_k_triples: Set of bottom-k triples (frozensets)
        df: Original DataFrame with gene1, gene2, gene3, prediction
        panel_size: Number of genes in panel (12 or 24)
        k: Number of extreme triples
        results_dir: Directory to save output

    Returns:
        DataFrame with constructible triples and predictions
    """
    # Find constructible triples in both top and bottom
    constructible_rows = []

    for _, row in df.iterrows():
        triple = frozenset([row["gene1"], row["gene2"], row["gene3"]])
        if triple.issubset(S):
            # Check if this triple is in top-k or bottom-k
            category = None
            if triple in top_k_triples:
                category = "top"
            elif triple in bot_k_triples:
                category = "bottom"

            if category:
                constructible_rows.append(
                    {
                        "gene1": row["gene1"],
                        "gene2": row["gene2"],
                        "gene3": row["gene3"],
                        "prediction": row["prediction"],
                        "category": category,
                    }
                )

    constructible_df = pd.DataFrame(constructible_rows)

    if len(constructible_df) > 0:
        # Sort by prediction (descending for top, ascending for bottom)
        constructible_df = constructible_df.sort_values(
            "prediction", ascending=False
        ).reset_index(drop=True)

        # Save to parquet (stable filename for caching)
        output_path = osp.join(
            results_dir,
            f"constructible_triples_panel{panel_size}_k{k}.parquet",
        )
        constructible_df.to_parquet(output_path, index=False)
        tqdm.write(f"  Saved constructible triples: {output_path}")
        tqdm.write(
            f"    Top: {len(constructible_df[constructible_df['category'] == 'top'])}, "
            f"Bottom: {len(constructible_df[constructible_df['category'] == 'bottom'])}"
        )

    return constructible_df


def run_analysis_for_panel_size(
    df: pd.DataFrame,
    n_genes: int = 12,
    k_values: list[int] = [25, 50, 100, 200],
    results_dir: str | None = None,
    sameith_genes: set[str] | None = None,
    gene_lists: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """
    Run the full analysis for a specific panel size across all k values.

    Args:
        df: DataFrame with inference predictions
        n_genes: Number of genes to select (12 or 24)
        k_values: List of k values to analyze
        results_dir: Directory to save constructible triples parquet files
        sameith_genes: Set of genes with existing Sameith doubles (for priority)
        gene_lists: Dict mapping list name to set of genes (for overlap counting)

    Returns:
        DataFrame with results for each k, including panel_size column.
    """
    results = []
    design_space = math.comb(n_genes, 3)

    print(f"\n{'#'*60}")
    print(f"# PANEL SIZE: {n_genes} genes (design space = {design_space} triples)")
    print(f"{'#'*60}")

    for k in tqdm(k_values, desc=f"Panel {n_genes} genes", unit="k"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Processing k={k} (panel_size={n_genes})")
        tqdm.write("=" * 60)

        # Get top-k and bottom-k triples
        top_k, bot_k = get_top_bottom_k(df, k)
        tqdm.write(f"Top-{k} triples: {len(top_k)}, Bottom-{k} triples: {len(bot_k)}")

        # Get all candidate genes
        all_genes = get_all_genes_from_triples(top_k)
        all_genes.update(get_all_genes_from_triples(bot_k))
        tqdm.write(f"Candidate genes: {len(all_genes)}")

        # Greedy selection (with sameith priority)
        tqdm.write(f"\nGreedy selection ({n_genes} genes):")
        selected_genes = greedy_select_genes(
            top_k, bot_k, n_genes=n_genes, sameith_genes=sameith_genes
        )

        # Local refinement
        tqdm.write("\nLocal swap refinement:")
        refined_genes = local_swap_refinement(selected_genes, top_k, bot_k, all_genes)

        # Compute final coverage
        S = set(refined_genes)
        top_constructible = compute_constructible(S, top_k)
        bot_constructible = compute_constructible(S, bot_k)

        # Output constructible triples with predictions
        if results_dir:
            output_constructible_triples(
                S=S,
                top_k_triples=top_k,
                bot_k_triples=bot_k,
                df=df,
                panel_size=n_genes,
                k=k,
                results_dir=results_dir,
            )

        # Compute doubles metrics (sameith overlap)
        if sameith_genes:
            total_doubles, sameith_overlap, new_doubles = compute_doubles_metrics(
                refined_genes, sameith_genes
            )
        else:
            total_doubles = math.comb(n_genes, 2)
            sameith_overlap = 0
            new_doubles = total_doubles

        # Compute gene list overlaps
        overlaps = {}
        if gene_lists:
            overlaps = compute_gene_list_overlaps(refined_genes, gene_lists)

        # Store results with panel_size
        result_row = {
            "panel_size": n_genes,
            "k": k,
            "selected_genes": sorted(refined_genes),
            "top_k_constructible": top_constructible,
            "top_k_fraction": top_constructible / k,
            "bot_k_constructible": bot_constructible,
            "bot_k_fraction": bot_constructible / k,
            "design_space": design_space,
            "total_doubles": total_doubles,
            "sameith_doubles": sameith_overlap,
            "new_doubles_needed": new_doubles,
        }
        # Add overlap columns
        result_row.update(overlaps)
        results.append(result_row)

        tqdm.write(f"\nResults for k={k} (panel_size={n_genes}):")
        tqdm.write(f"  Selected genes: {sorted(refined_genes)}")
        tqdm.write(
            f"  Top-{k} constructible: {top_constructible}/{k} "
            f"({top_constructible/k:.1%})"
        )
        tqdm.write(
            f"  Bot-{k} constructible: {bot_constructible}/{k} "
            f"({bot_constructible/k:.1%})"
        )
        tqdm.write(
            f"  Doubles: {sameith_overlap}/{total_doubles} sameith, "
            f"{new_doubles} new needed"
        )

    return pd.DataFrame(results)


def plot_coverage_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot coverage counts vs k comparing 12 vs 24 gene panels.

    Shows 4 lines: top_12, bot_12, top_24, bot_24
    Each panel plotted with its own k values.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract data for each panel size
    df_12 = results_df[results_df["panel_size"] == 12].sort_values("k")
    df_24 = results_df[results_df["panel_size"] == 24].sort_values("k")

    # Get all unique k values for x-axis
    all_k_values = sorted(results_df["k"].unique())

    # Colors from torchcell.mplstyle color cycle
    color_12 = "#34699D"  # blue from torchcell palette
    color_24 = "#D86E2F"  # orange/rust from torchcell palette

    # Plot 12-gene panel (only its k values)
    ax.plot(
        df_12["k"],
        df_12["top_k_constructible"],
        "o-",
        label="12 genes: Top-k",
        linewidth=2,
        markersize=10,
        color=color_12,
    )
    ax.plot(
        df_12["k"],
        df_12["bot_k_constructible"],
        "o--",
        label="12 genes: Bot-k",
        linewidth=2,
        markersize=10,
        color=color_12,
        alpha=0.6,
    )

    # Plot 24-gene panel (all its k values)
    ax.plot(
        df_24["k"],
        df_24["top_k_constructible"],
        "s-",
        label="24 genes: Top-k",
        linewidth=2,
        markersize=10,
        color=color_24,
    )
    ax.plot(
        df_24["k"],
        df_24["bot_k_constructible"],
        "s--",
        label="24 genes: Bot-k",
        linewidth=2,
        markersize=10,
        color=color_24,
        alpha=0.6,
    )

    # Reference line for perfect coverage (use log-spaced points for log x-axis)
    min_k = min(all_k_values)
    max_k = max(all_k_values)
    ref_k = np.logspace(np.log10(min_k), np.log10(max_k), 100)
    ax.plot(ref_k, ref_k, "k:", alpha=0.5, label="Perfect coverage (count = k)")

    ax.set_xlabel("k (number of extreme triples, log scale)")
    ax.set_ylabel("Number of triples constructible")
    ax.set_title("Coverage Comparison: 12 vs 24 Gene Panels")
    ax.legend(loc="upper left")

    # Use log scale on x-axis to prevent label overlap
    ax.set_xscale("log")
    ax.set_xticks(all_k_values)
    ax.set_xticklabels([str(k) for k in all_k_values])
    ax.grid(True, alpha=0.3)

    # Add design space annotations
    ax.annotate(
        "12 genes: 220 max triples",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        alpha=0.7,
    )
    ax.annotate(
        "24 genes: 2024 max triples",
        xy=(0.98, 0.07),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        alpha=0.7,
    )

    output_path = osp.join(
        output_dir, f"coverage_comparison_12_vs_24_{timestamp()}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_counts_bar_chart_combined(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot absolute counts as grouped bar chart for both panel sizes.

    Groups by k, with sub-bars for 12 vs 24 genes, colored by top/bottom.
    Only plots k values that are common to both panels.
    """
    # Get data for each panel size
    df_12 = results_df[results_df["panel_size"] == 12].sort_values("k")
    df_24 = results_df[results_df["panel_size"] == 24].sort_values("k")

    # Find common k values
    k_12 = set(df_12["k"].unique())
    k_24 = set(df_24["k"].unique())
    common_k = sorted(k_12 & k_24)

    # Filter to common k values
    df_12_common = df_12[df_12["k"].isin(common_k)].sort_values("k")
    df_24_common = df_24[df_24["k"].isin(common_k)].sort_values("k")

    fig, ax = plt.subplots(figsize=(14, 7))

    n_k = len(common_k)
    x = np.arange(n_k)
    width = 0.2

    # Colors from torchcell.mplstyle color cycle
    color_12 = "#34699D"  # blue from torchcell palette
    color_24 = "#D86E2F"  # orange/rust from torchcell palette

    # Plot 4 bars per k: 12-top, 12-bot, 24-top, 24-bot
    bars1 = ax.bar(
        x - 1.5 * width,
        df_12_common["top_k_constructible"],
        width,
        label="12 genes: Top-k",
        color=color_12,
    )
    bars2 = ax.bar(
        x - 0.5 * width,
        df_12_common["bot_k_constructible"],
        width,
        label="12 genes: Bot-k",
        color=color_12,
        alpha=0.5,
    )
    bars3 = ax.bar(
        x + 0.5 * width,
        df_24_common["top_k_constructible"],
        width,
        label="24 genes: Top-k",
        color=color_24,
    )
    bars4 = ax.bar(
        x + 1.5 * width,
        df_24_common["bot_k_constructible"],
        width,
        label="24 genes: Bot-k",
        color=color_24,
        alpha=0.5,
    )

    ax.set_xlabel("k (number of extreme triples)")
    ax.set_ylabel("Number of triples constructible")
    ax.set_title("Constructible Triples: 12 vs 24 Gene Panels (Common k values)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in common_k])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    output_path = osp.join(output_dir, f"counts_bar_chart_combined_{timestamp()}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also plot extended 24-gene only chart if there are extra k values
    extra_k = sorted(k_24 - k_12)
    if extra_k:
        plot_counts_bar_chart_extended_24(df_24, extra_k, output_dir)


def plot_counts_bar_chart_extended_24(
    df_24: pd.DataFrame, extra_k: list[int], output_dir: str
) -> None:
    """Plot bar chart for 24-gene panel at extended k values."""
    df_extended = df_24[df_24["k"].isin(extra_k)].sort_values("k")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(extra_k))
    width = 0.35
    color_24 = "#D86E2F"

    bars1 = ax.bar(
        x - width / 2,
        df_extended["top_k_constructible"],
        width,
        label="Top-k",
        color=color_24,
    )
    bars2 = ax.bar(
        x + width / 2,
        df_extended["bot_k_constructible"],
        width,
        label="Bot-k",
        color=color_24,
        alpha=0.5,
    )

    ax.set_xlabel("k (number of extreme triples)")
    ax.set_ylabel("Number of triples constructible")
    ax.set_title("24-Gene Panel: Extended k Values")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in extra_k])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    output_path = osp.join(
        output_dir, f"counts_bar_chart_24_extended_{timestamp()}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gene_stability_heatmap(
    results_df: pd.DataFrame, panel_size: int, output_dir: str
) -> None:
    """
    Plot gene selection stability showing how many k values each gene appears in.

    Two-panel plot:
    - Left: Bar chart showing count of k values each gene is selected for
    - Right: Heatmap showing which k values each gene is selected at

    Args:
        results_df: Full results DataFrame (will be filtered by panel_size)
        panel_size: Number of genes in panel (12 or 24)
        output_dir: Directory to save the plot
    """
    # Filter to specific panel size
    df_panel = results_df[results_df["panel_size"] == panel_size].sort_values("k")
    k_values = df_panel["k"].tolist()
    total_k = len(k_values)

    # Get all genes across all k values for this panel
    all_selected = set()
    for genes in df_panel["selected_genes"]:
        all_selected.update(genes)

    # Count how many k values each gene appears in
    gene_counts = {}
    for gene in all_selected:
        count = sum(1 for genes in df_panel["selected_genes"] if gene in genes)
        gene_counts[gene] = count

    # Sort genes by count (most stable first)
    sorted_genes = sorted(all_selected, key=lambda g: (-gene_counts[g], g))

    # Build presence matrix (rows = genes sorted by stability, cols = k values)
    presence_matrix = []
    for gene in sorted_genes:
        row = [1 if gene in genes else 0 for genes in df_panel["selected_genes"]]
        presence_matrix.append(row)

    presence_df = pd.DataFrame(
        presence_matrix, index=sorted_genes, columns=[f"k={k}" for k in k_values]
    )

    # Create two-panel figure (don't use sharey - heatmap has different cell positions)
    n_genes = len(sorted_genes)
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(12, max(6, n_genes * 0.35)),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    # Left panel: Bar chart of stability counts
    # Use positions 0.5, 1.5, 2.5... to align with heatmap cells
    counts = [gene_counts[g] for g in sorted_genes]
    colors = ["#2166ac" if c == total_k else "#67a9cf" if c > total_k / 2 else "#d1e5f0"
              for c in counts]
    y_positions = [i + 0.5 for i in range(n_genes)]
    bars = ax_bar.barh(y_positions, counts, height=0.8, color=colors)

    # Add count labels inside bars (right-aligned)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        # Put label inside bar if bar is wide enough, else outside
        if count >= total_k * 0.3:
            ax_bar.text(
                bar.get_width() - 0.2, y_positions[i],
                f"{count}/{total_k}", va="center", ha="right", fontsize=8, color="white"
            )
        else:
            ax_bar.text(
                bar.get_width() + 0.1, y_positions[i],
                f"{count}/{total_k}", va="center", ha="left", fontsize=8
            )

    ax_bar.set_xlabel(f"Times Selected (out of {total_k} k values)")
    ax_bar.set_ylabel("Gene")
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(sorted_genes, fontsize=8)
    ax_bar.set_xlim(0, total_k + 1.5)  # Extra room for labels
    ax_bar.set_ylim(n_genes, 0)  # Invert so most stable at top
    ax_bar.set_title("Selection Frequency")

    # Right panel: Heatmap showing which k values
    sns.heatmap(
        presence_df,
        cmap=["#f7f7f7", "#2166ac"],  # White = not selected, Blue = selected
        cbar=False,  # No colorbar needed for binary
        linewidths=0.5,
        ax=ax_heat,
        annot=False,
    )

    ax_heat.set_title(f"Selection Pattern Across k Values ({panel_size}-gene panel)")
    ax_heat.set_xlabel("k value")
    ax_heat.set_ylabel("")  # Labels on left panel
    ax_heat.set_yticks([])  # Hide y ticks (labels on left panel)

    # Add bounding box around the heatmap
    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    plt.tight_layout()
    output_path = osp.join(
        output_dir, f"gene_stability_{panel_size}_{timestamp()}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_prediction_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot histogram of inference predictions.

    Args:
        df: Inference DataFrame with 'prediction' column
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors from torchcell palette
    color_main = "#000000"  # black from torchcell palette

    ax.hist(
        df["prediction"],
        bins=100,
        color=color_main,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Predicted Gene Interaction Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Inference Prediction Distribution (N={len(df):,})")
    ax.grid(True, alpha=0.3, axis="y")

    # Add vertical lines for top/bottom regions
    top_threshold = df["prediction"].quantile(0.999)
    bot_threshold = df["prediction"].quantile(0.001)
    ax.axvline(
        top_threshold,
        color="#B73C39",
        linestyle="--",
        label=f"Top 0.1%: {top_threshold:.3f}",
    )
    ax.axvline(
        bot_threshold,
        color="#34699D",
        linestyle="--",
        label=f"Bottom 0.1%: {bot_threshold:.3f}",
    )
    ax.legend()

    output_path = osp.join(output_dir, f"prediction_distribution_{timestamp()}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def load_training_data() -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    Load training data from Neo4jCellDataset.

    Returns:
        Tuple of (label_df, gene_index_dict)
        - label_df: DataFrame with gene_interaction labels
        - gene_index_dict: Dict mapping gene names to list of indices
    """
    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("\nLoading training dataset...")

    # Load genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))

    # Load query
    query_path = "experiments/010-kuzmin-tmi/queries/001_small_build.cql"
    with open(query_path, "r") as f:
        query = f.read()

    # Embedding datasets
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )

    # Load dataset (should use cache if already created)
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri="bolt://torchcell-database.ncsa.illinois.edu:7687",
        username="readonly",
        password="ReadOnly",
        graphs=None,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(f"Loaded training dataset: {len(dataset):,} samples")

    # Extract label_df and gene index
    label_df = dataset.label_df
    gene_index = dataset.is_any_perturbed_gene_index

    return label_df, gene_index


def load_009_deletion_dataset() -> tuple[pd.DataFrame | None, dict[str, list[int]] | None]:
    """
    Load 009-kuzmin-tmi deletion-only dataset using all data (not just train split).

    This dataset contains only deletion mutations, used to compare selected
    genes against all available triple deletion data.

    Returns:
        Tuple of (label_df, gene_index_dict)
        - label_df: DataFrame with gene_interaction labels for all samples
        - gene_index_dict: Dict mapping gene names to list of indices
    """
    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    print("\nLoading 009 deletion dataset (all data)...")

    # Load genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))

    # Load 009 query (deletion-only)
    query_path = "experiments/009-kuzmin-tmi/queries/001_small_build.cql"
    with open(query_path, "r") as f:
        query = f.read()

    # Embedding datasets
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    # Note: Using 010 dataset path - both 009 and 010 use the same deletion-only query
    # and the data was built at the 010 location
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )

    # Load dataset
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri="bolt://torchcell-database.ncsa.illinois.edu:7687",
        username="readonly",
        password="ReadOnly",
        graphs=None,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(f"Loaded 009 deletion dataset: {len(dataset):,} samples")

    # Handle case where dataset is empty (cache not available or DB issue)
    if len(dataset) == 0:
        print("WARNING: 009 dataset returned 0 samples. Skipping 009 analysis.")
        print("This may be because the dataset cache doesn't exist at the expected path.")
        print(f"Expected path: {dataset_root}")
        return None, None

    # Extract label_df and gene index (for all samples, not just train)
    label_df = dataset.label_df
    gene_index = dataset.is_any_perturbed_gene_index

    return label_df, gene_index


def analyze_gene_representation_009(
    gene_index: dict[str, list[int]],
    selected_genes_by_panel: dict[int, list[str]],
    output_dir: str,
    results_dir: str,
) -> pd.DataFrame | None:
    """
    Analyze gene representation in 009 deletion dataset for selected genes.

    This compares selected genes against ALL deletion data (not just training).

    Args:
        gene_index: Dict mapping gene names to list of sample indices
        selected_genes_by_panel: Dict mapping panel_size to list of selected genes
        output_dir: Directory to save plots
        results_dir: Directory to save CSV results

    Returns:
        DataFrame with gene representation counts, or None if no data
    """
    # Guard against empty gene_index
    if not gene_index:
        print("WARNING: gene_index is empty, skipping 009 analysis")
        return None

    # Count representation for all genes
    gene_counts = {gene: len(indices) for gene, indices in gene_index.items()}
    total_samples = max(max(indices) for indices in gene_index.values()) + 1

    # Create summary DataFrame
    all_selected = set()
    for genes in selected_genes_by_panel.values():
        all_selected.update(genes)

    gene_rep_data = []
    for gene in sorted(all_selected):
        count = gene_counts.get(gene, 0)
        in_12 = gene in selected_genes_by_panel.get(12, [])
        in_24 = gene in selected_genes_by_panel.get(24, [])
        gene_rep_data.append(
            {
                "gene": gene,
                "deletion_count": count,
                "deletion_fraction": count / total_samples if total_samples > 0 else 0,
                "in_12_panel": in_12,
                "in_24_panel": in_24,
            }
        )

    gene_rep_df = pd.DataFrame(gene_rep_data)
    gene_rep_df = gene_rep_df.sort_values("deletion_count", ascending=False)

    # Save to CSV
    csv_path = osp.join(results_dir, "gene_representation_009_deletion.csv")
    gene_rep_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot gene representation bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color bars by panel membership
    # For genes with 0 count, don't draw bar - just show "0" text
    colors = []
    plot_counts = []
    for _, row in gene_rep_df.iterrows():
        count = row["deletion_count"]
        if count == 0:
            colors.append("none")  # No bar for zero count
            plot_counts.append(0)
        elif row["in_12_panel"] and row["in_24_panel"]:
            colors.append("#6B8D3A")  # green - both panels
            plot_counts.append(count)
        elif row["in_24_panel"]:
            colors.append("#D86E2F")  # orange - 24-gene panel only
            plot_counts.append(count)
        else:
            colors.append("#D86E2F")  # orange - default
            plot_counts.append(count)

    bars = ax.bar(range(len(gene_rep_df)), plot_counts, color=colors)

    # Add value labels above bars (horizontal, not rotated)
    for i, (bar, count) in enumerate(zip(bars, gene_rep_df["deletion_count"])):
        if count == 0:
            # For zero count, put "0" at bottom of plot area
            ax.annotate(
                "0",
                xy=(i, 0.5),  # Position near bottom on log scale
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,
            )
        else:
            height = bar.get_height()
            ax.annotate(
                f"{int(count):,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,
            )

    ax.set_xlabel("Gene")
    ax.set_ylabel("Count in 009 Deletion Dataset (log scale)")
    ax.set_title("009 Deletion Dataset Representation of Selected Genes")
    ax.set_xticks(range(len(gene_rep_df)))
    ax.set_xticklabels(gene_rep_df["gene"], rotation=90, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Use log scale
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3, top=ax.get_ylim()[1] * 2)

    # Add legend (no grey - zero count genes have no bar)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#6B8D3A", label="Both panels (12 & 24 genes)"),
        Patch(facecolor="#D86E2F", label="24-gene panel only"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    output_path = osp.join(output_dir, "gene_representation_009_deletion.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("GENE REPRESENTATION IN 009 DELETION DATASET")
    print("=" * 60)
    print(
        f"{'Gene':<12} | {'Count':>8} | {'Fraction':>10} | {'12-panel':>8} | {'24-panel':>8}"
    )
    print("-" * 60)
    for _, row in gene_rep_df.iterrows():
        print(
            f"{row['gene']:<12} | {row['deletion_count']:>8,} | "
            f"{row['deletion_fraction']:>9.2%} | "
            f"{'Yes' if row['in_12_panel'] else 'No':>8} | "
            f"{'Yes' if row['in_24_panel'] else 'No':>8}"
        )
    print("=" * 60)

    return gene_rep_df


def plot_distribution_comparison(
    inference_df: pd.DataFrame, training_label_df: pd.DataFrame, output_dir: str
) -> None:
    """
    Overlay inference predictions and training labels distributions with KDE.

    Args:
        inference_df: Inference DataFrame with 'prediction' column
        training_label_df: Training DataFrame with 'gene_interaction' column
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors from torchcell palette
    color_inference = "#D86E2F"  # orange/rust
    color_training = "#34699D"  # blue

    # Plot training distribution with histogram and KDE
    ax.hist(
        training_label_df["gene_interaction"],
        bins=100,
        color=color_training,
        alpha=0.4,
        density=True,
    )
    sns.kdeplot(
        x=training_label_df["gene_interaction"],
        ax=ax,
        color=color_training,
        linewidth=2,
        label=f"Training (N={len(training_label_df):,})",
    )

    # Plot inference distribution with histogram and KDE
    ax.hist(
        inference_df["prediction"],
        bins=100,
        color=color_inference,
        alpha=0.4,
        density=True,
    )
    sns.kdeplot(
        x=inference_df["prediction"],
        ax=ax,
        color=color_inference,
        linewidth=2,
        label=f"Inference (N={len(inference_df):,})",
    )

    ax.set_xlabel("Gene Interaction Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Comparison: Training vs Inference")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    output_path = osp.join(output_dir, f"distribution_comparison_{timestamp()}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def analyze_gene_representation(
    gene_index: dict[str, list[int]],
    selected_genes_by_panel: dict[int, list[str]],
    output_dir: str,
    results_dir: str,
) -> pd.DataFrame:
    """
    Analyze gene representation in training data for selected genes.

    Args:
        gene_index: Dict mapping gene names to list of training indices
        selected_genes_by_panel: Dict mapping panel_size to list of selected genes
        output_dir: Directory to save plots
        results_dir: Directory to save CSV results

    Returns:
        DataFrame with gene representation counts
    """
    # Count representation for all genes
    gene_counts = {gene: len(indices) for gene, indices in gene_index.items()}
    total_samples = max(max(indices) for indices in gene_index.values()) + 1

    # Create summary DataFrame
    all_selected = set()
    for genes in selected_genes_by_panel.values():
        all_selected.update(genes)

    gene_rep_data = []
    for gene in sorted(all_selected):
        count = gene_counts.get(gene, 0)
        in_12 = gene in selected_genes_by_panel.get(12, [])
        in_24 = gene in selected_genes_by_panel.get(24, [])
        gene_rep_data.append(
            {
                "gene": gene,
                "training_count": count,
                "training_fraction": count / total_samples,
                "in_12_panel": in_12,
                "in_24_panel": in_24,
            }
        )

    gene_rep_df = pd.DataFrame(gene_rep_data)
    gene_rep_df = gene_rep_df.sort_values("training_count", ascending=False)

    # Save to CSV
    # Stable filename for caching
    csv_path = osp.join(results_dir, "gene_representation_training.csv")
    gene_rep_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot gene representation bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color bars by panel membership
    # Note: 12-gene panel is always a subset of 24-gene panel, so "12-only" never occurs
    # Colors: green (both panels), orange (24-only)
    # For genes with 0 count, don't draw bar - just show "0" text
    colors = []
    plot_counts = []
    for _, row in gene_rep_df.iterrows():
        count = row["training_count"]
        if count == 0:
            colors.append("none")  # No bar for zero count
            plot_counts.append(0)
        elif row["in_12_panel"] and row["in_24_panel"]:
            colors.append("#6B8D3A")  # green - both panels (i.e., in 12-gene panel)
            plot_counts.append(count)
        elif row["in_24_panel"]:
            colors.append("#D86E2F")  # orange - 24-gene panel only
            plot_counts.append(count)
        else:
            colors.append("#D86E2F")  # orange - default (shouldn't happen)
            plot_counts.append(count)

    bars = ax.bar(range(len(gene_rep_df)), plot_counts, color=colors)

    # Add value labels above bars (horizontal, not rotated)
    for i, (bar, count) in enumerate(zip(bars, gene_rep_df["training_count"])):
        if count == 0:
            # For zero count, put "0" at bottom of plot area
            ax.annotate(
                "0",
                xy=(i, 0.5),  # Position near bottom on log scale
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,
            )
        else:
            height = bar.get_height()
            ax.annotate(
                f"{int(count):,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,  # Horizontal labels
            )

    ax.set_xlabel("Gene")
    ax.set_ylabel("Count in Training Data (log scale)")
    ax.set_title("Training Representation of Selected Genes")
    ax.set_xticks(range(len(gene_rep_df)))
    ax.set_xticklabels(gene_rep_df["gene"], rotation=90, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Use log scale to show both small and large counts
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.3, top=ax.get_ylim()[1] * 2)  # Room for labels

    # Add legend (no grey - zero count genes have no bar)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#6B8D3A", label="Both panels (12 & 24 genes)"),
        Patch(facecolor="#D86E2F", label="24-gene panel only"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    output_path = osp.join(output_dir, "gene_representation_training.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("GENE REPRESENTATION IN TRAINING DATA")
    print("=" * 60)
    print(
        f"{'Gene':<12} | {'Count':>8} | {'Fraction':>10} | {'12-panel':>8} | {'24-panel':>8}"
    )
    print("-" * 60)
    for _, row in gene_rep_df.iterrows():
        print(
            f"{row['gene']:<12} | {row['training_count']:>8,} | "
            f"{row['training_fraction']:>9.2%} | "
            f"{'Yes' if row['in_12_panel'] else 'No':>8} | "
            f"{'Yes' if row['in_24_panel'] else 'No':>8}"
        )
    print("=" * 60)

    return gene_rep_df


def main():
    """Main execution function for both 12 and 24 gene panels."""
    print("=" * 60)
    print("Select 12 and 24 Genes for Top Triples Coverage")
    print("=" * 60)

    # Create output directories
    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results")
    plots_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data()

    # Load sameith genes (for priority selection)
    print("\nLoading Sameith doubles genes...")
    sameith_genes = load_sameith_doubles()
    print(f"Loaded {len(sameith_genes)} genes with existing Sameith doubles")

    # Load all gene lists for overlap counting
    print("\nLoading gene lists for overlap analysis...")
    gene_lists = load_all_gene_lists()

    # Run analysis for both panel sizes with panel-specific k values
    # 12 genes: C(12,3) = 220 possible triples → k up to 200 makes sense
    # 24 genes: C(24,3) = 2024 possible triples → k up to 2000 covers nearly full design space
    panel_configs = {12: [25, 50, 100, 200], 24: [25, 50, 100, 200, 500, 1000, 2000]}
    all_results = []

    for n_genes, k_values in panel_configs.items():
        results = run_analysis_for_panel_size(
            df,
            n_genes=n_genes,
            k_values=k_values,
            results_dir=results_dir,
            sameith_genes=sameith_genes,
            gene_lists=gene_lists,
        )
        all_results.append(results)

    # Combine results from both panel sizes
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results table
    # Stable filename for caching (delete file to recompute)
    results_path = osp.join(results_dir, "gene_selection_results.csv")

    # Flatten selected_genes for CSV
    results_for_csv = combined_df.copy()
    results_for_csv["selected_genes"] = results_for_csv["selected_genes"].apply(
        lambda x: ", ".join(x)
    )
    results_for_csv.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # Generate gene selection plots (4 total)
    print("\nGenerating gene selection plots...")
    plot_coverage_comparison(combined_df, plots_dir)
    plot_counts_bar_chart_combined(combined_df, plots_dir)
    plot_gene_stability_heatmap(combined_df, panel_size=12, output_dir=plots_dir)
    plot_gene_stability_heatmap(combined_df, panel_size=24, output_dir=plots_dir)

    # Plot inference prediction distribution
    print("\nGenerating distribution plots...")
    plot_prediction_distribution(df, plots_dir)

    # Load training data and compare distributions
    training_label_df, gene_index = load_training_data()
    plot_distribution_comparison(df, training_label_df, plots_dir)

    # Analyze gene representation in training data
    # Extract selected genes for each panel (use largest k for each panel)
    selected_genes_by_panel = {}
    for panel_size in panel_configs.keys():
        df_panel = combined_df[combined_df["panel_size"] == panel_size]
        # Use the largest k's selection (most comprehensive)
        max_k_row = df_panel[df_panel["k"] == df_panel["k"].max()].iloc[0]
        selected_genes_by_panel[panel_size] = max_k_row["selected_genes"]

    analyze_gene_representation(
        gene_index, selected_genes_by_panel, plots_dir, results_dir
    )

    # Load 009 deletion dataset and analyze gene representation
    print("\n" + "=" * 60)
    print("Analyzing 009 deletion dataset coverage...")
    print("=" * 60)
    _, gene_index_009 = load_009_deletion_dataset()
    if gene_index_009 is not None:
        analyze_gene_representation_009(
            gene_index_009, selected_genes_by_panel, plots_dir, results_dir
        )
    else:
        print("Skipping 009 deletion dataset analysis (no data available)")
        print("=" * 60)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(
        f"{'Panel':>6} | {'k':>5} | {'Top':>6} | {'Top%':>6} | "
        f"{'Bot':>6} | {'Bot%':>6} | {'Design':>8} | "
        f"{'Sameith':>8} | {'New Dbl':>8}"
    )
    print("-" * 120)
    for _, row in combined_df.iterrows():
        print(
            f"{row['panel_size']:>6} | {row['k']:>5} | "
            f"{row['top_k_constructible']:>6} | {row['top_k_fraction']:>5.0%} | "
            f"{row['bot_k_constructible']:>6} | {row['bot_k_fraction']:>5.0%} | "
            f"{row['design_space']:>8} | "
            f"{row['sameith_doubles']:>8} | {row['new_doubles_needed']:>8}"
        )
    print("=" * 120)

    # Print genes selected for each (panel_size, k) combination
    for panel_size in panel_configs.keys():
        print(f"\n{'='*60}")
        print(f"SELECTED GENES ({panel_size} genes):")
        print("=" * 60)
        df_panel = combined_df[combined_df["panel_size"] == panel_size].sort_values("k")
        for _, row in df_panel.iterrows():
            print(f"\nk={row['k']}: {row['selected_genes']}")


if __name__ == "__main__":
    main()
