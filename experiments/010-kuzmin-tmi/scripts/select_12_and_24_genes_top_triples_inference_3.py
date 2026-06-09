# experiments/010-kuzmin-tmi/scripts/select_12_and_24_genes_top_triples_inference_3.py
# [[experiments.010-kuzmin-tmi.scripts.select_12_and_24_genes_top_triples_inference_3]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/select_12_and_24_genes_top_triples_inference_3.py

"""
Select 12 and 24 genes that maximize coverage of top-k extreme triples.

INFERENCE_3 VERSION: Uses inference_3 results (relaxed thresholds for JT test)
- Thresholds: max(SMF) > 1.04, max(DMF) > 1.08
- Baselines: all(SMF) > 0.90, all(DMF) > 0.90
- Designed for Jonckheere-Terpstra test validation (0.04 gap for ~96% power at n=8)

Memory-efficient: Uses PyArrow for all heavy data operations (465M+ rows).
Only converts small subsets to pandas for output. Never loads full table as
pandas DataFrame (which would require ~84 GB for string columns alone).

Algorithm:
1. Load inference predictions as Arrow table (~12 GB vs ~90 GB in pandas)
2. For panel_size in {12, 24}:
   For k in {25, 50, 100, 200}:
   - Extract top-k and bottom-k triples via Arrow sort
   - Greedy selection of genes maximizing top-k coverage
   - Local swap refinement
   - Report coverage statistics
3. Generate combined visualizations comparing 12 vs 24 gene panels

Design space:
- 12 genes: C(12,3) = 220 possible triples (constrained regime)
- 24 genes: C(24,3) = 2024 possible triples (expansive regime)
"""

import argparse
import math
import os
import os.path as osp
from glob import glob
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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


# ---------------------------------------------------------------------------
# Arrow-based data loading (memory efficient)
# ---------------------------------------------------------------------------


def find_parquet_path() -> str:
    """Find the inference_3 parquet file for the best model (Pearson=0.4619)."""
    inference_dir = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_3/inferred"
    )
    pattern = osp.join(inference_dir, "*Pearson=0.4619*.parquet")
    files = glob(pattern)
    # Exclude shard files
    files = [f for f in files if not f.endswith((".rank0", ".rank1", ".rank2", ".rank3"))]
    if not files:
        raise FileNotFoundError(f"No parquet file matching pattern: {pattern}")
    return files[0]


def load_arrow_table(parquet_path: str) -> pa.Table:
    """Load parquet as an Arrow table (memory-mapped, ~12 GB vs ~90 GB in pandas)."""
    print(f"Loading data from: {parquet_path}")
    table = pq.read_table(parquet_path)
    print(f"Loaded {table.num_rows:,} triples")
    return table


def arrow_table_to_frozensets(table: pa.Table) -> set[frozenset]:
    """Convert Arrow table rows to a set of gene-name frozensets."""
    g1 = table.column("gene1")
    g2 = table.column("gene2")
    g3 = table.column("gene3")
    return {
        frozenset([g1[i].as_py(), g2[i].as_py(), g3[i].as_py()])
        for i in range(table.num_rows)
    }


def get_top_bottom_k_arrow(
    table: pa.Table, k: int
) -> tuple[set[frozenset], set[frozenset], pa.Table, pa.Table]:
    """
    Extract top-k and bottom-k triples from Arrow table.

    Returns:
        (top_k_frozensets, bot_k_frozensets, top_k_table, bot_k_table)
    """
    sorted_idx = pc.sort_indices(table, sort_keys=[("prediction", "descending")])
    top_idx = sorted_idx[:k]
    bot_idx = sorted_idx[len(sorted_idx) - k :]

    top_table = table.take(top_idx)
    bot_table = table.take(bot_idx)

    return (
        arrow_table_to_frozensets(top_table),
        arrow_table_to_frozensets(bot_table),
        top_table,
        bot_table,
    )


def filter_constructible_arrow(table: pa.Table, gene_set: set[str]) -> pd.DataFrame:
    """
    Filter Arrow table to rows where all 3 genes are in gene_set.

    Uses Arrow's vectorized is_in() — handles 465M rows in seconds.
    Returns a small pandas DataFrame of matching rows.
    """
    gene_array = pa.array(sorted(gene_set))
    g1_mask = pc.is_in(table.column("gene1"), value_set=gene_array)
    g2_mask = pc.is_in(table.column("gene2"), value_set=gene_array)
    g3_mask = pc.is_in(table.column("gene3"), value_set=gene_array)
    mask = pc.and_(pc.and_(g1_mask, g2_mask), g3_mask)
    return table.filter(mask).to_pandas()


# ---------------------------------------------------------------------------
# Gene list helpers
# ---------------------------------------------------------------------------


def load_sameith_doubles() -> set[str]:
    """Load genes with existing double mutants from Sameith et al."""
    path = osp.join(
        EXPERIMENT_ROOT,
        "006-kuzmin-tmi/results/inference_preprocessing_expansion/sameith_doubles_genes.txt",
    )
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_all_gene_lists() -> dict[str, set[str]]:
    """Load all gene lists from inference_preprocessing_expansion directory."""
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
    selected_set = set(selected_genes)
    return {f"overlap_{name}": len(selected_set & genes) for name, genes in gene_lists.items()}


# ---------------------------------------------------------------------------
# Selection algorithm (works on frozensets only — no DataFrame needed)
# ---------------------------------------------------------------------------


def compute_constructible(S: set[str], triples: set[frozenset]) -> int:
    return sum(1 for t in triples if t.issubset(S))


def get_all_genes_from_triples(triples: set[frozenset]) -> set[str]:
    genes = set()
    for t in triples:
        genes.update(t)
    return genes


def count_triples_containing_gene(gene: str, triples: set[frozenset]) -> int:
    return sum(1 for t in triples if gene in t)


def get_priority_genes(top_k_triples: set[frozenset], sameith_genes: set[str]) -> set[str]:
    return get_all_genes_from_triples(top_k_triples) & sameith_genes


def compute_doubles_metrics(
    selected_genes: list[str], sameith_genes: set[str]
) -> tuple[int, int, int]:
    S = set(selected_genes)
    total_doubles = math.comb(len(S), 2)
    sameith_overlap = sum(
        1 for g1, g2 in combinations(S, 2)
        if g1 in sameith_genes and g2 in sameith_genes
    )
    return total_doubles, sameith_overlap, total_doubles - sameith_overlap


def greedy_select_genes(
    top_k_triples: set[frozenset],
    bot_k_triples: set[frozenset],
    n_genes: int = 12,
    sameith_genes: set[str] | None = None,
) -> list[str]:
    """
    Greedy selection of genes to maximize top-k triple coverage.

    Priority: auto-include genes in top-k AND sameith, then greedy fill.
    Tie-breaking: marginal gain > sameith > bottom coverage > top count > name.
    """
    if sameith_genes is None:
        sameith_genes = set()

    all_genes = get_all_genes_from_triples(top_k_triples)
    all_genes.update(get_all_genes_from_triples(bot_k_triples))

    S = set()
    selected_order = []

    # Auto-include priority genes
    priority_genes = get_priority_genes(top_k_triples, sameith_genes)
    if priority_genes:
        priority_to_add = sorted(priority_genes)[:n_genes]
        S.update(priority_to_add)
        selected_order.extend(priority_to_add)
        tqdm.write(
            f"  Auto-included {len(priority_to_add)} priority genes "
            f"(in top-k AND sameith): {priority_to_add}"
        )

    if len(S) >= n_genes:
        return selected_order[:n_genes]

    # Greedy fill
    remaining = n_genes - len(S)
    pbar = tqdm(total=remaining, desc="Greedy selection", unit="gene")

    while len(S) < n_genes:
        best_gene = None
        best_score = (-1, -1, -1, -1, "")

        candidates = all_genes - S
        for g in candidates:
            S_with_g = S | {g}
            current_top = compute_constructible(S, top_k_triples)
            new_top = compute_constructible(S_with_g, top_k_triples)
            marginal_gain = new_top - current_top

            is_sameith = 1 if g in sameith_genes else 0
            bot_coverage = compute_constructible(S_with_g, bot_k_triples)
            top_count = count_triples_containing_gene(g, top_k_triples)

            score = (marginal_gain, is_sameith, bot_coverage, top_count, g)
            if score[:4] > best_score[:4] or (
                score[:4] == best_score[:4] and g < best_score[4]
            ):
                best_score = score
                best_gene = g

        if best_gene is None:
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
    """Local swap refinement: try swapping genes to improve top-k coverage."""
    S_set = set(S)
    improved = True
    iterations = 0

    while improved and iterations < 100:
        improved = False
        iterations += 1

        current_top = compute_constructible(S_set, top_k_triples)
        current_bot = compute_constructible(S_set, bot_k_triples)

        best_swap = None
        best_improvement = (0, 0)

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


# ---------------------------------------------------------------------------
# Output functions (use Arrow for filtering, pandas only for small outputs)
# ---------------------------------------------------------------------------


def output_constructible_triples(
    S: set[str],
    top_k_triples: set[frozenset],
    bot_k_triples: set[frozenset],
    top_k_table: pa.Table,
    bot_k_table: pa.Table,
    panel_size: int,
    k: int,
    results_dir: str,
) -> pd.DataFrame:
    """
    Output constructible triples from top/bottom-k tables.

    Only filters the small top/bottom-k tables (k rows each), not the
    full 465M-row dataset.
    """
    constructible_rows = []

    for label, triples_set, subtable in [
        ("top", top_k_triples, top_k_table),
        ("bottom", bot_k_triples, bot_k_table),
    ]:
        df_sub = subtable.to_pandas()
        for _, row in df_sub.iterrows():
            triple = frozenset([row["gene1"], row["gene2"], row["gene3"]])
            if triple.issubset(S) and triple in triples_set:
                constructible_rows.append(
                    {
                        "gene1": row["gene1"],
                        "gene2": row["gene2"],
                        "gene3": row["gene3"],
                        "prediction": row["prediction"],
                        "category": label,
                    }
                )

    constructible_df = pd.DataFrame(constructible_rows)

    if len(constructible_df) > 0:
        constructible_df = constructible_df.sort_values(
            "prediction", ascending=False
        ).reset_index(drop=True)

        output_path = osp.join(
            results_dir, f"constructible_triples_panel{panel_size}_k{k}.parquet"
        )
        constructible_df.to_parquet(output_path, index=False)
        tqdm.write(f"  Saved constructible triples: {output_path}")
        tqdm.write(
            f"    Top: {len(constructible_df[constructible_df['category'] == 'top'])}, "
            f"Bottom: {len(constructible_df[constructible_df['category'] == 'bottom'])}"
        )

    return constructible_df


def output_top_k_constructible_csv(
    S: set[str],
    top_k_triples: set[frozenset],
    top_k_table: pa.Table,
    panel_size: int,
    k: int,
    results_dir: str,
) -> None:
    """Output top-k constructible triples as CSV."""
    df_top = top_k_table.to_pandas()
    constructible_rows = []

    for _, row in df_top.iterrows():
        triple = frozenset([row["gene1"], row["gene2"], row["gene3"]])
        if triple.issubset(S) and triple in top_k_triples:
            constructible_rows.append(
                {
                    "gene1": row["gene1"],
                    "gene2": row["gene2"],
                    "gene3": row["gene3"],
                    "prediction": row["prediction"],
                }
            )

    if constructible_rows:
        constructible_df = pd.DataFrame(constructible_rows)
        constructible_df = constructible_df.sort_values(
            "prediction", ascending=False
        ).reset_index(drop=True)

        output_path = osp.join(
            results_dir, f"top_k_constructible_panel{panel_size}_k{k}.csv"
        )
        constructible_df.to_csv(output_path, index=False)
        tqdm.write(f"  Saved top-k constructible CSV: {output_path}")


def output_singles_doubles_triples_tables(
    selected_genes: list[str],
    table: pa.Table,
    panel_size: int,
    k: int,
    results_dir: str,
) -> None:
    """
    Output singles, doubles, and triples tables for the selected gene panel.

    Uses Arrow is_in() to filter 465M rows efficiently instead of iterrows().
    """
    S = set(selected_genes)

    # Singles table
    singles_data = [
        {"index": i + 1, "gene": g, "mutant_type": "single"}
        for i, g in enumerate(sorted(S))
    ]
    singles_df = pd.DataFrame(singles_data)
    singles_path = osp.join(results_dir, f"singles_table_panel{panel_size}_k{k}.csv")
    singles_df.to_csv(singles_path, index=False)
    print(f"  Saved singles table: {singles_path}")

    # Doubles table
    doubles_data = [
        {"index": i + 1, "gene1": g1, "gene2": g2, "mutant_type": "double"}
        for i, (g1, g2) in enumerate(combinations(sorted(S), 2))
    ]
    doubles_df = pd.DataFrame(doubles_data)
    doubles_path = osp.join(results_dir, f"doubles_table_panel{panel_size}_k{k}.csv")
    doubles_df.to_csv(doubles_path, index=False)
    print(f"  Saved doubles table: {doubles_path}")

    # Triples table: filter with Arrow (vectorized, handles 465M rows in seconds)
    print(f"  Filtering constructible triples from {table.num_rows:,} rows...")
    triples_df = filter_constructible_arrow(table, S)
    triples_df = triples_df.sort_values("prediction", ascending=False).reset_index(
        drop=True
    )
    triples_df["index"] = range(1, len(triples_df) + 1)
    triples_df["mutant_type"] = "triple"
    triples_path = osp.join(results_dir, f"triples_table_panel{panel_size}_k{k}.csv")
    triples_df.to_csv(triples_path, index=False)
    print(f"  Saved triples table: {triples_path} ({len(triples_df):,} constructible)")


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------


def run_analysis_for_panel_size(
    table: pa.Table,
    n_genes: int = 12,
    k_values: list[int] = [25, 50, 100, 200],
    results_dir: str | None = None,
    sameith_genes: set[str] | None = None,
    gene_lists: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Run the full analysis for a specific panel size across all k values."""
    results = []
    design_space = math.comb(n_genes, 3)

    print(f"\n{'#'*60}")
    print(f"# PANEL SIZE: {n_genes} genes (design space = {design_space} triples)")
    print(f"{'#'*60}")

    for k in tqdm(k_values, desc=f"Panel {n_genes} genes", unit="k"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Processing k={k} (panel_size={n_genes})")
        tqdm.write("=" * 60)

        # Get top-k and bottom-k from Arrow table
        top_k, bot_k, top_k_table, bot_k_table = get_top_bottom_k_arrow(table, k)
        tqdm.write(
            f"Top-{k} triples: {len(top_k)}, Bottom-{k} triples: {len(bot_k)}"
        )

        all_genes = get_all_genes_from_triples(top_k)
        all_genes.update(get_all_genes_from_triples(bot_k))
        tqdm.write(f"Candidate genes: {len(all_genes)}")

        # Greedy selection
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

        # Output constructible triples (from small top/bot tables only)
        if results_dir:
            output_constructible_triples(
                S=S,
                top_k_triples=top_k,
                bot_k_triples=bot_k,
                top_k_table=top_k_table,
                bot_k_table=bot_k_table,
                panel_size=n_genes,
                k=k,
                results_dir=results_dir,
            )
            output_top_k_constructible_csv(
                S=S,
                top_k_triples=top_k,
                top_k_table=top_k_table,
                panel_size=n_genes,
                k=k,
                results_dir=results_dir,
            )

        # Doubles metrics
        if sameith_genes:
            total_doubles, sameith_overlap, new_doubles = compute_doubles_metrics(
                refined_genes, sameith_genes
            )
        else:
            total_doubles = math.comb(n_genes, 2)
            sameith_overlap = 0
            new_doubles = total_doubles

        # Gene list overlaps
        overlaps = compute_gene_list_overlaps(refined_genes, gene_lists) if gene_lists else {}

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


# ---------------------------------------------------------------------------
# Plotting (unchanged from inference_2 version)
# ---------------------------------------------------------------------------


def plot_coverage_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    df_12 = results_df[results_df["panel_size"] == 12].sort_values("k")
    df_24 = results_df[results_df["panel_size"] == 24].sort_values("k")
    all_k_values = sorted(results_df["k"].unique())
    color_12 = "#34699D"
    color_24 = "#D86E2F"

    ax.plot(df_12["k"], df_12["top_k_constructible"], "o-", label="12 genes: Top-k",
            linewidth=2, markersize=10, color=color_12)
    ax.plot(df_12["k"], df_12["bot_k_constructible"], "o--", label="12 genes: Bot-k",
            linewidth=2, markersize=10, color=color_12, alpha=0.6)
    ax.plot(df_24["k"], df_24["top_k_constructible"], "s-", label="24 genes: Top-k",
            linewidth=2, markersize=10, color=color_24)
    ax.plot(df_24["k"], df_24["bot_k_constructible"], "s--", label="24 genes: Bot-k",
            linewidth=2, markersize=10, color=color_24, alpha=0.6)

    min_k, max_k = min(all_k_values), max(all_k_values)
    ref_k = np.logspace(np.log10(min_k), np.log10(max_k), 100)
    ax.plot(ref_k, ref_k, "k:", alpha=0.5, label="Perfect coverage (count = k)")

    ax.set_xlabel("k (number of extreme triples, log scale)")
    ax.set_ylabel("Number of triples constructible")
    # Design ceilings live in the subtitle (not floating over the curves):
    # 220 = C(12,3), 2024 = C(24,3)
    ax.set_title(
        "Coverage Comparison: 12 vs 24 Gene Panels (inference_3)\n"
        "design ceiling: 12 genes ≤ 220 triples, 24 genes ≤ 2024 triples"
    )
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_xticks(all_k_values)
    ax.set_xticklabels([str(k) for k in all_k_values])
    ax.grid(True, alpha=0.3)

    output_path = osp.join(output_dir, f"coverage_comparison_12_vs_24_inference_3.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_counts_bar_chart_combined(results_df: pd.DataFrame, output_dir: str) -> None:
    df_12 = results_df[results_df["panel_size"] == 12].sort_values("k")
    df_24 = results_df[results_df["panel_size"] == 24].sort_values("k")
    common_k = sorted(set(df_12["k"].unique()) & set(df_24["k"].unique()))
    df_12_c = df_12[df_12["k"].isin(common_k)].sort_values("k")
    df_24_c = df_24[df_24["k"].isin(common_k)].sort_values("k")

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(common_k))
    w = 0.2
    color_12, color_24 = "#34699D", "#D86E2F"

    bars1 = ax.bar(x - 1.5*w, df_12_c["top_k_constructible"], w, label="12: Top-k", color=color_12)
    bars2 = ax.bar(x - 0.5*w, df_12_c["bot_k_constructible"], w, label="12: Bot-k", color=color_12, alpha=0.5)
    bars3 = ax.bar(x + 0.5*w, df_24_c["top_k_constructible"], w, label="24: Top-k", color=color_24)
    bars4 = ax.bar(x + 1.5*w, df_24_c["bot_k_constructible"], w, label="24: Bot-k", color=color_24, alpha=0.5)

    ax.set_xlabel("k (number of extreme triples)")
    ax.set_ylabel("Number of triples constructible")
    ax.set_title("Constructible Triples: 12 vs 24 Gene Panels (inference_3)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in common_k])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8)

    output_path = osp.join(output_dir, f"counts_bar_chart_combined_inference_3.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Extended 24-gene chart if extra k values
    extra_k = sorted(set(df_24["k"].unique()) - set(df_12["k"].unique()))
    if extra_k:
        plot_counts_bar_chart_extended_24(df_24, extra_k, output_dir)


def plot_counts_bar_chart_extended_24(
    df_24: pd.DataFrame, extra_k: list[int], output_dir: str
) -> None:
    df_ext = df_24[df_24["k"].isin(extra_k)].sort_values("k")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(extra_k))
    w = 0.35
    color_24 = "#D86E2F"

    bars1 = ax.bar(x - w/2, df_ext["top_k_constructible"], w, label="Top-k", color=color_24)
    bars2 = ax.bar(x + w/2, df_ext["bot_k_constructible"], w, label="Bot-k", color=color_24, alpha=0.5)

    ax.set_xlabel("k (number of extreme triples)")
    ax.set_ylabel("Number of triples constructible")
    ax.set_title("24-Gene Panel: Extended k Values (inference_3)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in extra_k])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=9)

    output_path = osp.join(output_dir, f"counts_bar_chart_24_extended_inference_3.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gene_stability_heatmap(
    results_df: pd.DataFrame, panel_size: int, output_dir: str
) -> None:
    df_panel = results_df[results_df["panel_size"] == panel_size].sort_values("k")
    k_values = df_panel["k"].tolist()
    total_k = len(k_values)

    all_selected = set()
    for genes in df_panel["selected_genes"]:
        all_selected.update(genes)

    gene_counts = {
        gene: sum(1 for genes in df_panel["selected_genes"] if gene in genes)
        for gene in all_selected
    }
    sorted_genes = sorted(all_selected, key=lambda g: (-gene_counts[g], g))

    presence_matrix = [
        [1 if gene in genes else 0 for genes in df_panel["selected_genes"]]
        for gene in sorted_genes
    ]
    presence_df = pd.DataFrame(
        presence_matrix, index=sorted_genes, columns=[f"k={k}" for k in k_values]
    )

    n_genes = len(sorted_genes)
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(12, max(6, n_genes * 0.35)),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    counts = [gene_counts[g] for g in sorted_genes]
    colors = [
        "#2166ac" if c == total_k else "#67a9cf" if c > total_k / 2 else "#d1e5f0"
        for c in counts
    ]
    y_pos = [i + 0.5 for i in range(n_genes)]
    bars = ax_bar.barh(y_pos, counts, height=0.8, color=colors)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count >= total_k * 0.3:
            ax_bar.text(bar.get_width() - 0.2, y_pos[i], f"{count}/{total_k}",
                        va="center", ha="right", fontsize=8, color="white")
        else:
            ax_bar.text(bar.get_width() + 0.1, y_pos[i], f"{count}/{total_k}",
                        va="center", ha="left", fontsize=8)

    ax_bar.set_xlabel(f"Times Selected (out of {total_k} k values)")
    ax_bar.set_ylabel("Gene")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(sorted_genes, fontsize=8)
    ax_bar.set_xlim(0, total_k + 1.5)
    ax_bar.set_ylim(n_genes, 0)
    ax_bar.set_title("Selection Frequency")

    sns.heatmap(presence_df, cmap=["#f7f7f7", "#2166ac"], cbar=False,
                linewidths=0.5, ax=ax_heat, annot=False)
    ax_heat.set_title(f"Selection Pattern ({panel_size}-gene panel, inference_3)")
    ax_heat.set_xlabel("k value")
    ax_heat.set_ylabel("")
    ax_heat.set_yticks([])

    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    plt.tight_layout()
    output_path = osp.join(output_dir, f"gene_stability_{panel_size}_inference_3.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_prediction_distribution(table: pa.Table, output_dir: str) -> None:
    """Plot prediction distribution using Arrow column (no pandas conversion)."""
    predictions = table.column("prediction").to_numpy()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(predictions, bins=100, color="#000000", alpha=0.7,
            edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Predicted Gene Interaction Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Inference_3 Prediction Distribution (N={len(predictions):,})")
    ax.grid(True, alpha=0.3, axis="y")

    top_thr = np.quantile(predictions, 0.999)
    bot_thr = np.quantile(predictions, 0.001)
    ax.axvline(top_thr, color="#B73C39", linestyle="--", label=f"Top 0.1%: {top_thr:.3f}")
    ax.axvline(bot_thr, color="#34699D", linestyle="--", label=f"Bottom 0.1%: {bot_thr:.3f}")
    ax.legend()

    output_path = osp.join(output_dir, f"prediction_distribution_inference_3.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    del predictions  # Free the numpy copy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def replot_from_saved(results_dir: str, plots_dir: str) -> None:
    """Redraw the combined_df-based plots from saved gene_selection_results.csv.

    Skips the ~12 GB Arrow load + greedy selection. The prediction-distribution
    plot is intentionally omitted here because it needs the full predictions table.
    """
    results_path = osp.join(results_dir, "gene_selection_results.csv")
    print(f"[--plot-only] redrawing from {results_path}")
    combined_df = pd.read_csv(results_path)
    # CSV stores selected_genes as a ", "-joined string; restore to list for heatmap
    combined_df["selected_genes"] = combined_df["selected_genes"].apply(
        lambda s: s.split(", ") if isinstance(s, str) else s
    )
    plot_coverage_comparison(combined_df, plots_dir)
    plot_counts_bar_chart_combined(combined_df, plots_dir)
    plot_gene_stability_heatmap(combined_df, panel_size=12, output_dir=plots_dir)
    plot_gene_stability_heatmap(combined_df, panel_size=24, output_dir=plots_dir)
    print("[--plot-only] done (skipped prediction-distribution — needs full table)")


def main(plot_only: bool = False):
    print("=" * 60)
    print("Select 12 and 24 Genes for Top Triples Coverage")
    print("INFERENCE_3 VERSION (relaxed thresholds for JT test)")
    print("Thresholds: max(SMF) > 1.04, max(DMF) > 1.08")
    print("Baselines: all(SMF) > 0.90, all(DMF) > 0.90")
    print("=" * 60)

    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results/inference_3")
    plots_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if plot_only:
        replot_from_saved(results_dir, plots_dir)
        return

    # Load data as Arrow table (~12 GB vs ~90 GB in pandas)
    print("\nLoading data...")
    parquet_path = find_parquet_path()
    table = load_arrow_table(parquet_path)

    # Load gene lists
    print("\nLoading Sameith doubles genes...")
    sameith_genes = load_sameith_doubles()
    print(f"Loaded {len(sameith_genes)} genes with existing Sameith doubles")

    print("\nLoading gene lists for overlap analysis...")
    gene_lists = load_all_gene_lists()

    # Run analysis for both panel sizes
    panel_configs = {12: [25, 50, 100, 200], 24: [25, 50, 100, 200, 500, 1000, 2000]}
    all_results = []

    for n_genes, k_values in panel_configs.items():
        results = run_analysis_for_panel_size(
            table,
            n_genes=n_genes,
            k_values=k_values,
            results_dir=results_dir,
            sameith_genes=sameith_genes,
            gene_lists=gene_lists,
        )
        all_results.append(results)

        # Output singles/doubles/triples tables for k=200
        if 200 in k_values:
            k200_row = results[results["k"] == 200].iloc[0]
            output_singles_doubles_triples_tables(
                selected_genes=k200_row["selected_genes"],
                table=table,
                panel_size=n_genes,
                k=200,
                results_dir=results_dir,
            )

    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_path = osp.join(results_dir, "gene_selection_results.csv")
    results_for_csv = combined_df.copy()
    results_for_csv["selected_genes"] = results_for_csv["selected_genes"].apply(
        lambda x: ", ".join(x)
    )
    results_for_csv.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # Generate plots
    print("\nGenerating gene selection plots...")
    plot_coverage_comparison(combined_df, plots_dir)
    plot_counts_bar_chart_combined(combined_df, plots_dir)
    plot_gene_stability_heatmap(combined_df, panel_size=12, output_dir=plots_dir)
    plot_gene_stability_heatmap(combined_df, panel_size=24, output_dir=plots_dir)

    print("\nGenerating distribution plots...")
    plot_prediction_distribution(table, plots_dir)

    # Summary table
    print("\n" + "=" * 120)
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

    for panel_size in panel_configs.keys():
        print(f"\n{'='*60}")
        print(f"SELECTED GENES ({panel_size} genes):")
        print("=" * 60)
        df_panel = combined_df[combined_df["panel_size"] == panel_size].sort_values("k")
        for _, row in df_panel.iterrows():
            print(f"\nk={row['k']}: {row['selected_genes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select 12/24 gene panels; --plot-only redraws from saved results."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the ~12GB load + selection; redraw plots from gene_selection_results.csv",
    )
    cli_args = parser.parse_args()
    main(plot_only=cli_args.plot_only)
