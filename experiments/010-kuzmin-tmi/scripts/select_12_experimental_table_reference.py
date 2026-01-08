# experiments/010-kuzmin-tmi/scripts/select_12_experimental_table_reference
# [[experiments.010-kuzmin-tmi.scripts.select_12_experimental_table_reference]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/select_12_experimental_table_reference
# Test file: experiments/010-kuzmin-tmi/scripts/test_select_12_experimental_table_reference.py

"""
Create experimental reference table for 12-gene panel (k=200).

This table helps prioritize double mutant construction by showing:
1. All triples sorted by predicted gene interaction (descending)
2. Gene ordering by frequency (most common → gene1)
3. Sameith overlap flags for genes and pairs
4. Pair count columns showing how many triples each pair enables
"""

import os
import os.path as osp
from collections import Counter
from glob import glob
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

assert DATA_ROOT is not None, "DATA_ROOT environment variable not set"
assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT environment variable not set"


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


def load_sameith_genes() -> set[str]:
    """Load Sameith double expression genes (82 genes)."""
    sameith_path = osp.join(
        EXPERIMENT_ROOT,
        "006-kuzmin-tmi/results/inference_preprocessing_expansion/sameith_doubles_genes.txt",
    )
    with open(sameith_path, "r") as f:
        genes = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(genes)} Sameith genes")
    return genes


def load_constructible_triples(panel_size: int, k: int) -> set[frozenset]:
    """
    Load DMF-constructible triples from parquet file.

    These are triples where all 3 gene pairs have DMF fitness >= 1.0.

    Returns:
        Set of frozensets, each containing 3 gene names
    """
    parquet_path = osp.join(
        EXPERIMENT_ROOT,
        f"010-kuzmin-tmi/results/constructible_triples_panel{panel_size}_k{k}.parquet",
    )
    if not osp.exists(parquet_path):
        print(f"WARNING: No constructible triples file at {parquet_path}")
        return set()

    df = pd.read_parquet(parquet_path)
    triples = set()
    for _, row in df.iterrows():
        triples.add(frozenset([row["gene1"], row["gene2"], row["gene3"]]))
    print(f"Loaded {len(triples)} DMF-constructible triples from parquet")
    return triples


def parse_selected_genes(genes_str: str) -> list[str]:
    """Parse comma-separated gene string into list."""
    return [g.strip() for g in genes_str.split(",")]


def filter_to_panel(df: pd.DataFrame, panel_genes: set[str]) -> pd.DataFrame:
    """Filter inference data to only triples within the gene panel."""
    mask = (
        df["gene1"].isin(panel_genes)
        & df["gene2"].isin(panel_genes)
        & df["gene3"].isin(panel_genes)
    )
    filtered = df[mask].copy()
    print(f"Filtered to {len(filtered)} triples within panel")
    return filtered


def count_gene_frequencies(df: pd.DataFrame) -> Counter:
    """Count how many triples each gene appears in."""
    gene_counts = Counter()
    for _, row in df.iterrows():
        gene_counts[row["gene1"]] += 1
        gene_counts[row["gene2"]] += 1
        gene_counts[row["gene3"]] += 1
    return gene_counts


def reorder_triple_by_frequency(
    g1: str, g2: str, g3: str, gene_rank: dict[str, int]
) -> tuple[str, str, str]:
    """
    Reorder genes in a triple by frequency rank.

    Most frequent gene (lowest rank number) → gene1
    Least frequent gene (highest rank number) → gene3
    """
    genes = [(g1, gene_rank[g1]), (g2, gene_rank[g2]), (g3, gene_rank[g3])]
    # Sort by rank (ascending = most frequent first)
    genes_sorted = sorted(genes, key=lambda x: x[1])
    return genes_sorted[0][0], genes_sorted[1][0], genes_sorted[2][0]


def build_pair_count_lookup(df: pd.DataFrame) -> dict[frozenset, int]:
    """
    Build a lookup of how many triples contain each pair.

    Args:
        df: DataFrame with gene1, gene2, gene3 columns

    Returns:
        Dict mapping frozenset({geneA, geneB}) -> count of triples containing this pair
    """
    pair_counts: dict[frozenset, int] = {}

    for _, row in df.iterrows():
        genes = [row["gene1"], row["gene2"], row["gene3"]]
        # Each triple contributes to 3 pairs
        for g1, g2 in combinations(genes, 2):
            pair = frozenset([g1, g2])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts


def create_experimental_table(
    panel_genes: list[str],
    inference_df: pd.DataFrame,
    sameith_genes: set[str],
    constructible_triples: set[frozenset],
) -> pd.DataFrame:
    """
    Create the experimental reference table.

    Args:
        panel_genes: List of genes in the 12-gene panel
        inference_df: Full inference DataFrame
        sameith_genes: Set of genes with Sameith double expression data
        constructible_triples: Set of DMF-constructible triple frozensets

    Returns:
        DataFrame with experimental reference table
    """
    panel_set = set(panel_genes)

    # 1. Filter to panel triples
    panel_df = filter_to_panel(inference_df, panel_set)

    # 2. Sort by prediction (descending)
    panel_df = panel_df.sort_values("prediction", ascending=False).reset_index(
        drop=True
    )

    # 3. Count gene frequencies and create rank mapping
    gene_freq = count_gene_frequencies(panel_df)
    print("\nGene frequencies in panel triples:")
    for gene, count in gene_freq.most_common():
        print(f"  {gene}: {count}")

    # Create rank mapping (most frequent = rank 0)
    gene_rank = {
        gene: rank
        for rank, (gene, _) in enumerate(gene_freq.most_common())
    }

    # 4. Reorder genes in each triple by frequency
    print("\nReordering genes by frequency...")
    new_gene1 = []
    new_gene2 = []
    new_gene3 = []

    for _, row in panel_df.iterrows():
        g1, g2, g3 = reorder_triple_by_frequency(
            row["gene1"], row["gene2"], row["gene3"], gene_rank
        )
        new_gene1.append(g1)
        new_gene2.append(g2)
        new_gene3.append(g3)

    panel_df["gene1"] = new_gene1
    panel_df["gene2"] = new_gene2
    panel_df["gene3"] = new_gene3

    # 5. Add Sameith flags
    print("Adding Sameith flags...")
    panel_df["gene1_in_sameith"] = panel_df["gene1"].isin(sameith_genes)
    panel_df["gene2_in_sameith"] = panel_df["gene2"].isin(sameith_genes)
    panel_df["gene3_in_sameith"] = panel_df["gene3"].isin(sameith_genes)

    # Pair is "in Sameith" if BOTH genes are in Sameith list
    panel_df["pair_12_in_sameith"] = (
        panel_df["gene1_in_sameith"] & panel_df["gene2_in_sameith"]
    )
    panel_df["pair_13_in_sameith"] = (
        panel_df["gene1_in_sameith"] & panel_df["gene3_in_sameith"]
    )
    panel_df["pair_23_in_sameith"] = (
        panel_df["gene2_in_sameith"] & panel_df["gene3_in_sameith"]
    )

    # 6. Build pair count lookup and add per-row pair counts
    print("Computing pair counts...")
    pair_count_lookup = build_pair_count_lookup(panel_df)

    # Add dynamic pair count columns (value depends on which genes are in this row)
    gene1_gene2_counts = []
    gene1_gene3_counts = []
    gene2_gene3_counts = []

    for _, row in panel_df.iterrows():
        g1, g2, g3 = row["gene1"], row["gene2"], row["gene3"]
        gene1_gene2_counts.append(pair_count_lookup[frozenset([g1, g2])])
        gene1_gene3_counts.append(pair_count_lookup[frozenset([g1, g3])])
        gene2_gene3_counts.append(pair_count_lookup[frozenset([g2, g3])])

    panel_df["gene1_gene2_count"] = gene1_gene2_counts
    panel_df["gene1_gene3_count"] = gene1_gene3_counts
    panel_df["gene2_gene3_count"] = gene2_gene3_counts

    # 7. Add rank column and rename prediction column
    panel_df["rank"] = range(1, len(panel_df) + 1)
    panel_df = panel_df.rename(columns={"prediction": "inferred_gene_interaction"})

    # 8. Add constructible flag (all 3 genes must be in the panel)
    panel_df["constructible"] = (
        panel_df["gene1"].isin(panel_set)
        & panel_df["gene2"].isin(panel_set)
        & panel_df["gene3"].isin(panel_set)
    )

    # 9. Reorder columns
    final_cols = [
        "rank",
        "gene1",
        "gene2",
        "gene3",
        "inferred_gene_interaction",
        "constructible",
        "gene1_in_sameith",
        "gene2_in_sameith",
        "gene3_in_sameith",
        "pair_12_in_sameith",
        "pair_13_in_sameith",
        "pair_23_in_sameith",
        "gene1_gene2_count",
        "gene1_gene3_count",
        "gene2_gene3_count",
    ]
    panel_df = panel_df[final_cols]

    return panel_df


def main():
    """Main execution function."""
    print("=" * 60)
    print("Creating Experimental Reference Table for 12-Gene Panel")
    print("=" * 60)

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)
    inference_df = load_inference_data()
    selection_df = load_gene_selection_results()
    sameith_genes = load_sameith_genes()

    # Get 12-gene panel for k=200
    panel_row = selection_df[
        (selection_df["panel_size"] == 12) & (selection_df["k"] == 200)
    ].iloc[0]

    panel_genes = parse_selected_genes(panel_row["selected_genes"])
    print(f"\n12-gene panel (k=200): {panel_genes}")

    # Create experimental table
    print("\n" + "=" * 60)
    print("Creating experimental table...")
    print("=" * 60)
    result_df = create_experimental_table(panel_genes, inference_df, sameith_genes)

    # Save results - full table with constructible flag
    output_path = osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi/results/experimental_table_12_genes_k200.csv"
    )
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Save filtered table - only constructible triples
    constructible_df = result_df[result_df["constructible"]].copy()
    constructible_df["rank"] = range(1, len(constructible_df) + 1)  # Re-rank
    output_path_filtered = osp.join(
        EXPERIMENT_ROOT,
        "010-kuzmin-tmi/results/experimental_table_12_genes_k200_constructible.csv",
    )
    constructible_df.to_csv(output_path_filtered, index=False)
    print(f"Saved: {output_path_filtered}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total triples: {len(result_df)}")
    print(f"Constructible triples: {len(constructible_df)}/{len(result_df)}")
    print(f"Columns: {len(result_df.columns)}")
    print(f"\nTop 5 triples:")
    print(
        result_df[
            ["rank", "gene1", "gene2", "gene3", "inferred_gene_interaction"]
        ].head()
    )

    # Summarize Sameith coverage
    n_gene1_sameith = result_df["gene1_in_sameith"].sum()
    n_gene2_sameith = result_df["gene2_in_sameith"].sum()
    n_gene3_sameith = result_df["gene3_in_sameith"].sum()
    n_pair12_sameith = result_df["pair_12_in_sameith"].sum()
    n_pair13_sameith = result_df["pair_13_in_sameith"].sum()
    n_pair23_sameith = result_df["pair_23_in_sameith"].sum()

    print(f"\nSameith coverage:")
    print(f"  gene1 in Sameith: {n_gene1_sameith}/{len(result_df)}")
    print(f"  gene2 in Sameith: {n_gene2_sameith}/{len(result_df)}")
    print(f"  gene3 in Sameith: {n_gene3_sameith}/{len(result_df)}")
    print(f"  pair_12 in Sameith: {n_pair12_sameith}/{len(result_df)}")
    print(f"  pair_13 in Sameith: {n_pair13_sameith}/{len(result_df)}")
    print(f"  pair_23 in Sameith: {n_pair23_sameith}/{len(result_df)}")

    print(f"\nPair count statistics:")
    print(f"  gene1_gene2_count: mean={result_df['gene1_gene2_count'].mean():.1f}, "
          f"max={result_df['gene1_gene2_count'].max()}")
    print(f"  gene1_gene3_count: mean={result_df['gene1_gene3_count'].mean():.1f}, "
          f"max={result_df['gene1_gene3_count'].max()}")
    print(f"  gene2_gene3_count: mean={result_df['gene2_gene3_count'].mean():.1f}, "
          f"max={result_df['gene2_gene3_count'].max()}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
