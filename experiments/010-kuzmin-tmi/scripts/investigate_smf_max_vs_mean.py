# experiments/010-kuzmin-tmi/scripts/investigate_smf_max_vs_mean.py
# [[experiments.010-kuzmin-tmi.scripts.investigate_smf_max_vs_mean]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/investigate_smf_max_vs_mean.py
"""
Investigate how YJR060W (mean SMF=0.590) passed the SMF_BASELINE=0.90 filter
in generate_triple_combinations_inference_3.py.

Hypothesis: load_smf_from_dataset() uses max(replicates) per gene instead of
mean(replicates). With multiple strains (KanMX/NatMX) × 2 temperatures (26°C/30°C),
the max replicate for YJR060W may exceed 0.90 even though the mean is 0.59.

This script:
1. Loads SmfCostanzo2016Dataset raw DataFrame
2. Shows ALL rows for YJR060W and all 12 panel genes
3. Compares max-replicate vs mean-replicate SMF
4. Does the same for DMF of problematic doubles
5. Checks the triples parquet for YJR060W count
"""

import os
import os.path as osp

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.costanzo2016 import (
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

# These are the thresholds from generate_triple_combinations_inference_3.py
SMF_THRESHOLD = 1.04
SMF_BASELINE = 0.90
DMF_THRESHOLD = 1.08
DMF_BASELINE = 0.90

# The 12-gene panel from inference_3
PANEL_12_GENES = [
    "YBR203W",
    "YDR057W",
    "YER079W",
    "YGL087C",
    "YIL174W",
    "YJR060W",
    "YKL033W-A",
    "YLL012W",
    "YLR104W",
    "YLR312C-B",
    "YPL046C",
    "YPL081W",
]

# Problematic doubles (mean DMF < 0.90 from queried CSV)
PROBLEMATIC_DOUBLES = [
    ("YGL087C", "YJR060W"),  # mean=0.5129
    ("YDR057W", "YJR060W"),  # mean=0.5914
    ("YJR060W", "YLL012W"),  # mean=0.6049
    ("YBR203W", "YJR060W"),  # mean=0.6091
    ("YER079W", "YJR060W"),  # mean=0.6100
    ("YER079W", "YPL081W"),  # mean=0.8625
    ("YJR060W", "YKL033W-A"),  # mean=0.8712
    ("YKL033W-A", "YPL081W"),  # mean=0.8924
]


def investigate_smf():
    """Step 1-3: Investigate SMF values for all 12 panel genes."""
    print("=" * 80)
    print("STEP 1-3: SMF Investigation (SmfCostanzo2016)")
    print("=" * 80)

    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    )
    df = smf_dataset.df
    print(f"Total rows in DataFrame: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Filter to deletions (same as generation script)
    deletion_mask = df["perturbation_type"].isin(["KanMX_deletion", "NatMX_deletion"])
    df_del = df[deletion_mask].copy()
    print(f"Rows after deletion filter: {len(df_del)}")

    gene_col = "Systematic gene name"
    fitness_col = "Single mutant fitness"

    # Show detailed analysis for each panel gene
    print("\n" + "-" * 80)
    print("Per-gene analysis: ALL rows in raw DataFrame (deletion mutants only)")
    print("-" * 80)

    summary_rows = []

    for gene in PANEL_12_GENES:
        gene_rows = df_del[df_del[gene_col] == gene]
        print(f"\n{'='*60}")
        print(f"Gene: {gene} ({len(gene_rows)} rows)")
        print(f"{'='*60}")

        if len(gene_rows) == 0:
            print("  *** NO ROWS FOUND ***")
            summary_rows.append(
                {
                    "gene": gene,
                    "n_rows": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "max_passes_baseline": None,
                }
            )
            continue

        # Show all rows
        for _, row in gene_rows.iterrows():
            strain_id = row.get("Strain ID", "?")
            fitness = row[fitness_col]
            fitness_std = row.get("Single mutant fitness stddev", "?")
            temp = row.get("Temperature", "?")
            ptype = row.get("perturbation_type", "?")
            print(
                f"  Strain={strain_id:<25s} "
                f"Temp={temp}°C  "
                f"Type={ptype:<18s} "
                f"Fitness={fitness:.4f}  "
                f"Std={fitness_std}"
            )

        fitness_values = gene_rows[fitness_col].dropna()
        gene_mean = fitness_values.mean()
        gene_std = fitness_values.std()
        gene_min = fitness_values.min()
        gene_max = fitness_values.max()

        print(f"\n  Summary: mean={gene_mean:.4f}, std={gene_std:.4f}, "
              f"min={gene_min:.4f}, max={gene_max:.4f}")
        print(f"  Max passes SMF_BASELINE (>{SMF_BASELINE})? "
              f"{'YES' if gene_max > SMF_BASELINE else '*** NO ***'}")
        print(f"  Mean passes SMF_BASELINE (>{SMF_BASELINE})? "
              f"{'YES' if gene_mean > SMF_BASELINE else '*** NO ***'}")

        summary_rows.append(
            {
                "gene": gene,
                "n_rows": len(gene_rows),
                "mean": gene_mean,
                "std": gene_std,
                "min": gene_min,
                "max": gene_max,
                "max_passes_baseline": gene_max > SMF_BASELINE,
            }
        )

    # Summary table
    print("\n\n" + "=" * 80)
    print("SMF SUMMARY TABLE: max-replicate vs mean-replicate")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_rows)
    print(
        f"{'Gene':<14s} {'N':>4s} {'Mean':>8s} {'Std':>8s} "
        f"{'Min':>8s} {'Max':>8s} {'Max>0.90':>10s} {'Mean>0.90':>10s} "
        f"{'Delta':>8s}"
    )
    print("-" * 90)
    for _, row in summary_df.iterrows():
        if row["mean"] is not None:
            delta = row["max"] - row["mean"]
            max_pass = "YES" if row["max"] > SMF_BASELINE else "NO"
            mean_pass = "YES" if row["mean"] > SMF_BASELINE else "NO"
            print(
                f"{row['gene']:<14s} {row['n_rows']:>4d} "
                f"{row['mean']:>8.4f} {row['std']:>8.4f} "
                f"{row['min']:>8.4f} {row['max']:>8.4f} "
                f"{max_pass:>10s} {mean_pass:>10s} "
                f"{delta:>8.4f}"
            )

    # Save summary
    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results/inference_3")
    summary_df.to_csv(
        osp.join(results_dir, "smf_max_vs_mean_panel12.csv"), index=False
    )
    print(f"\nSaved: {osp.join(results_dir, 'smf_max_vs_mean_panel12.csv')}")

    return summary_df


def investigate_dmf():
    """Step 4: Investigate DMF values for problematic doubles."""
    print("\n\n" + "=" * 80)
    print("STEP 4: DMF Investigation (All Datasets)")
    print("=" * 80)

    # Load all DMF datasets
    datasets = {
        "Costanzo2016": DmfCostanzo2016Dataset(
            root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016")
        ),
        "Kuzmin2018": DmfKuzmin2018Dataset(
            root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018")
        ),
        "Kuzmin2020": DmfKuzmin2020Dataset(
            root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020")
        ),
    }

    # Column mappings per dataset (same as in generation script)
    col_maps = {
        "Costanzo2016": {
            "gene1": "Query Systematic Name",
            "gene2": "Array Systematic Name",
            "fitness": "Double mutant fitness",
        },
        "Kuzmin2018": {
            "gene1": "Query systematic name no ho",
            "gene2": "Array systematic name",
            "fitness": "Combined mutant fitness",
        },
        "Kuzmin2020": {
            "gene1": "Query systematic name no ho",
            "gene2": "Array systematic name",
            "fitness": "fitness",
        },
    }

    for pair in PROBLEMATIC_DOUBLES:
        g1, g2 = pair
        pair_sorted = tuple(sorted(pair))
        print(f"\n{'='*60}")
        print(f"Pair: {g1} - {g2}")
        print(f"{'='*60}")

        max_across_datasets = None

        for ds_name, dataset in datasets.items():
            cols = col_maps[ds_name]
            df = dataset.df

            # Find rows matching this pair (in either order)
            mask = (
                (df[cols["gene1"]] == pair_sorted[0])
                & (df[cols["gene2"]] == pair_sorted[1])
            ) | (
                (df[cols["gene1"]] == pair_sorted[1])
                & (df[cols["gene2"]] == pair_sorted[0])
            )
            pair_rows = df[mask]

            if len(pair_rows) == 0:
                print(f"  {ds_name}: No rows found")
                continue

            fitness_vals = pair_rows[cols["fitness"]].dropna()
            if len(fitness_vals) == 0:
                print(f"  {ds_name}: {len(pair_rows)} rows but no valid fitness values")
                continue

            ds_mean = fitness_vals.mean()
            ds_max = fitness_vals.max()
            ds_min = fitness_vals.min()

            print(f"  {ds_name}: {len(fitness_vals)} rows, "
                  f"mean={ds_mean:.4f}, min={ds_min:.4f}, max={ds_max:.4f}")

            # Show all rows
            for _, row in pair_rows.iterrows():
                fitness = row[cols["fitness"]]
                print(f"    fitness={fitness:.4f}" if pd.notna(fitness) else "    fitness=NaN")

            if max_across_datasets is None or ds_max > max_across_datasets:
                max_across_datasets = ds_max

        if max_across_datasets is not None:
            print(f"\n  Max across all datasets: {max_across_datasets:.4f}")
            print(f"  Passes DMF_BASELINE (>{DMF_BASELINE})? "
                  f"{'YES' if max_across_datasets > DMF_BASELINE else '*** NO ***'}")
        else:
            print(f"\n  *** No DMF data found for this pair ***")


def investigate_parquet():
    """Step 5: Check the triples parquet for YJR060W."""
    print("\n\n" + "=" * 80)
    print("STEP 5: Check Triples Parquet for YJR060W")
    print("=" * 80)

    parquet_path = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/010-kuzmin-tmi/inference_3/raw/"
        "triple_combinations_list.parquet",
    )

    if not osp.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        return

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"Total triples in parquet: {len(df):,}")

    # Check for YJR060W
    mask = (
        (df["gene1"] == "YJR060W")
        | (df["gene2"] == "YJR060W")
        | (df["gene3"] == "YJR060W")
    )
    yjr_triples = df[mask]
    print(f"Triples containing YJR060W: {yjr_triples.shape[0]:,}")

    if len(yjr_triples) > 0:
        print(f"\nFirst 20 triples containing YJR060W:")
        print(yjr_triples.head(20).to_string())


def main():
    print("=" * 80)
    print("INVESTIGATION: How did low-SMF genes pass filters in inference_3?")
    print("=" * 80)
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"SMF_BASELINE: {SMF_BASELINE}")
    print(f"SMF_THRESHOLD: {SMF_THRESHOLD}")
    print(f"DMF_BASELINE: {DMF_BASELINE}")
    print(f"DMF_THRESHOLD: {DMF_THRESHOLD}")

    investigate_smf()
    investigate_dmf()
    investigate_parquet()

    print("\n\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print(
        "\nIf max-replicate SMF for YJR060W > 0.90:"
        "\n  → The max() aggregation in load_smf_from_dataset() is the root cause."
        "\n  → Fix: use mean() instead of max() and re-run generation."
        "\n"
        "\nIf max-replicate SMF for YJR060W <= 0.90:"
        "\n  → There's a pipeline mismatch — the predictions were generated from"
        "\n    a different triples list than what the current script would produce."
    )


if __name__ == "__main__":
    main()
