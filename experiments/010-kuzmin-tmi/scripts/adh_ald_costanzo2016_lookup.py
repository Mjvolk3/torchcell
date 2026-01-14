# experiments/010-kuzmin-tmi/scripts/adh_ald_costanzo2016_lookup.py
# [[experiments.010-kuzmin-tmi.scripts.adh_ald_costanzo2016_lookup]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/adh_ald_costanzo2016_lookup.py
"""
Look up single mutant fitness (SMF) and digenic interaction (ε) values
from Costanzo2016 dataset for ADH/ALD genes relevant to 2,3-butanediol production.

This helps us understand:
1. Individual fitness effects of each gene deletion
2. Pairwise epistatic interactions between genes
3. What training data is available for predicting trigenic interactions

Genes of interest (Ng et al. 2012):
- ADH1 (YOL086C): Primary alcohol dehydrogenase
- ADH3 (YMR083W): Mitochondrial alcohol dehydrogenase
- ADH5 (YBR145W): Alcohol dehydrogenase isoenzyme V
- ALD6 (YPL061W): Cytosolic aldehyde dehydrogenase
"""

import os
import os.path as osp
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.costanzo2016 import (
    DmiCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)

# ============================================================================
# Gene Configuration
# ============================================================================
GENE_MAP = {
    "a1": {"name": "ADH1", "systematic": "YOL086C", "function": "Primary alcohol dehydrogenase"},
    "a3": {"name": "ADH3", "systematic": "YMR083W", "function": "Mitochondrial alcohol dehydrogenase"},
    "a5": {"name": "ADH5", "systematic": "YBR145W", "function": "Alcohol dehydrogenase isoenzyme V"},
    "a6": {"name": "ALD6", "systematic": "YPL061W", "function": "Cytosolic aldehyde dehydrogenase"},
}

# All unique pairs
GENE_PAIRS = list(combinations(GENE_MAP.keys(), 2))


def lookup_smf(df: pd.DataFrame, systematic_name: str, temperature: int = 30) -> pd.DataFrame:
    """
    Look up single mutant fitness for a gene.

    Args:
        df: SmfCostanzo2016Dataset.df DataFrame
        systematic_name: Systematic gene name (e.g., "YOL086C")
        temperature: Temperature (26 or 30)

    Returns:
        DataFrame with matching rows (may have multiple alleles)
    """
    mask = (df["Systematic gene name"] == systematic_name) & (df["Temperature"] == temperature)
    return df[mask].copy()


def lookup_dmi(df: pd.DataFrame, gene1: str, gene2: str, temperature: int = 30) -> pd.DataFrame:
    """
    Look up digenic interaction (epsilon) for a gene pair.

    Searches both orderings: (gene1, gene2) and (gene2, gene1)

    Args:
        df: DmiCostanzo2016Dataset.df DataFrame
        gene1: Systematic gene name
        gene2: Systematic gene name
        temperature: Temperature (26 or 30)

    Returns:
        DataFrame with matching rows
    """
    # Check both orderings
    mask_forward = (
        (df["Query Systematic Name"] == gene1) &
        (df["Array Systematic Name"] == gene2) &
        (df["Temperature"] == temperature)
    )
    mask_reverse = (
        (df["Query Systematic Name"] == gene2) &
        (df["Array Systematic Name"] == gene1) &
        (df["Temperature"] == temperature)
    )

    return df[mask_forward | mask_reverse].copy()


def lookup_dmf(df: pd.DataFrame, gene1: str, gene2: str, temperature: int = 30) -> pd.DataFrame:
    """
    Look up double mutant fitness for a gene pair.

    Args:
        df: DmfCostanzo2016Dataset.df DataFrame
        gene1: Systematic gene name
        gene2: Systematic gene name
        temperature: Temperature (26 or 30)

    Returns:
        DataFrame with matching rows
    """
    # Check both orderings
    mask_forward = (
        (df["Query Systematic Name"] == gene1) &
        (df["Array Systematic Name"] == gene2) &
        (df["Temperature"] == temperature)
    )
    mask_reverse = (
        (df["Query Systematic Name"] == gene2) &
        (df["Array Systematic Name"] == gene1) &
        (df["Temperature"] == temperature)
    )

    return df[mask_forward | mask_reverse].copy()


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    print("=" * 80)
    print("Costanzo2016 Dataset Lookup for ADH/ALD Genes")
    print("=" * 80)

    print("\nGene Reference:")
    for alias, info in GENE_MAP.items():
        print(f"  {alias}: {info['name']} ({info['systematic']}) - {info['function']}")

    # ========================================================================
    # Load Datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading Costanzo2016 Datasets...")
    print("=" * 80)

    print("\nLoading SmfCostanzo2016Dataset (single mutant fitness)...")
    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016"),
        io_workers=10,
    )
    smf_df = smf_dataset.df
    print(f"  → {len(smf_df):,} rows")

    print("\nLoading DmiCostanzo2016Dataset (digenic interactions)...")
    dmi_dataset = DmiCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmi_costanzo2016"),
        io_workers=10,
    )
    dmi_df = dmi_dataset.df
    print(f"  → {len(dmi_df):,} rows")

    print("\nLoading DmfCostanzo2016Dataset (double mutant fitness)...")
    dmf_dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016"),
        io_workers=10,
    )
    dmf_df = dmf_dataset.df
    print(f"  → {len(dmf_df):,} rows")

    # ========================================================================
    # Single Mutant Fitness (SMF) Lookup
    # ========================================================================
    print("\n" + "=" * 80)
    print("Single Mutant Fitness (SMF) at 30°C")
    print("=" * 80)

    smf_results = []

    for alias, info in GENE_MAP.items():
        systematic = info["systematic"]
        name = info["name"]

        matches = lookup_smf(smf_df, systematic, temperature=30)

        if len(matches) == 0:
            print(f"\n  {name} ({systematic}): NOT FOUND in dataset")
            smf_results.append({
                "Alias": alias,
                "Gene": name,
                "Systematic": systematic,
                "Allele": "N/A",
                "SMF": None,
                "SMF_std": None,
                "Perturbation": "N/A",
            })
        else:
            print(f"\n  {name} ({systematic}): {len(matches)} allele(s) found")
            for _, row in matches.iterrows():
                smf = row["Single mutant fitness"]
                smf_std = row["Single mutant fitness stddev"]
                allele = row["Allele/Gene name"]
                pert_type = row["perturbation_type"]
                strain_id = row["Strain ID"]

                print(f"    • {allele} ({pert_type}): SMF = {smf:.4f} ± {smf_std:.4f}")

                smf_results.append({
                    "Alias": alias,
                    "Gene": name,
                    "Systematic": systematic,
                    "Allele": allele,
                    "Strain_ID": strain_id,
                    "SMF": smf,
                    "SMF_std": smf_std,
                    "Perturbation": pert_type,
                })

    smf_results_df = pd.DataFrame(smf_results)

    # ========================================================================
    # Digenic Interaction (DMI/ε) Lookup
    # ========================================================================
    print("\n" + "=" * 80)
    print("Digenic Interactions (ε) at 30°C")
    print("=" * 80)

    dmi_results = []

    for alias1, alias2 in GENE_PAIRS:
        gene1 = GENE_MAP[alias1]["systematic"]
        gene2 = GENE_MAP[alias2]["systematic"]
        name1 = GENE_MAP[alias1]["name"]
        name2 = GENE_MAP[alias2]["name"]

        matches = lookup_dmi(dmi_df, gene1, gene2, temperature=30)

        pair_label = f"{name1}-{name2}"

        if len(matches) == 0:
            print(f"\n  {pair_label} ({gene1}, {gene2}): NOT FOUND in dataset")
            dmi_results.append({
                "Alias_pair": f"{alias1}-{alias2}",
                "Gene_pair": pair_label,
                "Systematic_1": gene1,
                "Systematic_2": gene2,
                "Query_allele": "N/A",
                "Array_allele": "N/A",
                "Epsilon": None,
                "P_value": None,
                "Query_SMF": None,
                "Array_SMF": None,
                "DMF": None,
            })
        else:
            print(f"\n  {pair_label} ({gene1}, {gene2}): {len(matches)} measurement(s) found")
            for _, row in matches.iterrows():
                epsilon = row["Genetic interaction score (ε)"]
                p_value = row["P-value"]
                query_smf = row["Query single mutant fitness (SMF)"]
                array_smf = row["Array SMF"]
                dmf = row["Double mutant fitness"]
                query_allele = row["Query allele name"]
                array_allele = row["Array allele name"]

                # Interpretation
                if epsilon < -0.08:
                    interp = "Strong NEG"
                elif epsilon > 0.08:
                    interp = "Strong POS"
                else:
                    interp = "Neutral"

                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                print(f"    • {query_allele} × {array_allele}: ε = {epsilon:+.4f} (p={p_value:.4f}{sig}) [{interp}]")
                print(f"      SMF: {query_smf:.3f} × {array_smf:.3f} → DMF: {dmf:.3f}")

                dmi_results.append({
                    "Alias_pair": f"{alias1}-{alias2}",
                    "Gene_pair": pair_label,
                    "Systematic_1": gene1,
                    "Systematic_2": gene2,
                    "Query_allele": query_allele,
                    "Array_allele": array_allele,
                    "Epsilon": epsilon,
                    "P_value": p_value,
                    "Query_SMF": query_smf,
                    "Array_SMF": array_smf,
                    "DMF": dmf,
                })

    dmi_results_df = pd.DataFrame(dmi_results)

    # ========================================================================
    # Double Mutant Fitness (DMF) Lookup
    # ========================================================================
    print("\n" + "=" * 80)
    print("Double Mutant Fitness (DMF) at 30°C")
    print("=" * 80)

    dmf_results = []

    for alias1, alias2 in GENE_PAIRS:
        gene1 = GENE_MAP[alias1]["systematic"]
        gene2 = GENE_MAP[alias2]["systematic"]
        name1 = GENE_MAP[alias1]["name"]
        name2 = GENE_MAP[alias2]["name"]

        matches = lookup_dmf(dmf_df, gene1, gene2, temperature=30)

        pair_label = f"{name1}-{name2}"

        if len(matches) == 0:
            print(f"\n  {pair_label} ({gene1}, {gene2}): NOT FOUND in dataset")
        else:
            print(f"\n  {pair_label} ({gene1}, {gene2}): {len(matches)} measurement(s) found")
            for _, row in matches.iterrows():
                dmf = row["Double mutant fitness"]
                dmf_std = row["Double mutant fitness standard deviation"]
                query_allele = row["Query allele name"]
                array_allele = row["Array allele name"]

                print(f"    • {query_allele} × {array_allele}: DMF = {dmf:.4f} ± {dmf_std:.4f}")

                dmf_results.append({
                    "Alias_pair": f"{alias1}-{alias2}",
                    "Gene_pair": pair_label,
                    "Systematic_1": gene1,
                    "Systematic_2": gene2,
                    "Query_allele": query_allele,
                    "Array_allele": array_allele,
                    "DMF": dmf,
                    "DMF_std": dmf_std,
                })

    dmf_results_df = pd.DataFrame(dmf_results)

    # ========================================================================
    # Summary Tables
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Single Mutant Fitness")
    print("=" * 80)

    # Get best deletion allele for each gene (prefer KanMX_deletion)
    print("\n┌" + "─" * 70 + "┐")
    print(f"│ {'Gene':<8} │ {'Systematic':<10} │ {'Allele':<20} │ {'SMF':<12} │ {'Type':<15} │")
    print("├" + "─" * 70 + "┤")

    for alias in GENE_MAP.keys():
        gene_rows = smf_results_df[smf_results_df["Alias"] == alias]
        if len(gene_rows) == 0 or gene_rows["SMF"].isna().all():
            print(f"│ {GENE_MAP[alias]['name']:<8} │ {GENE_MAP[alias]['systematic']:<10} │ {'NOT FOUND':<20} │ {'N/A':<12} │ {'N/A':<15} │")
        else:
            # Prefer KanMX deletion if available
            kanmx = gene_rows[gene_rows["Perturbation"] == "KanMX_deletion"]
            if len(kanmx) > 0:
                row = kanmx.iloc[0]
            else:
                row = gene_rows.iloc[0]

            smf_str = f"{row['SMF']:.4f} ± {row['SMF_std']:.2f}"
            print(f"│ {row['Gene']:<8} │ {row['Systematic']:<10} │ {row['Allele']:<20} │ {smf_str:<12} │ {row['Perturbation']:<15} │")

    print("└" + "─" * 70 + "┘")

    print("\n" + "=" * 80)
    print("SUMMARY: Digenic Interactions")
    print("=" * 80)

    print("\n┌" + "─" * 90 + "┐")
    print(f"│ {'Pair':<15} │ {'ε (epsilon)':<12} │ {'P-value':<10} │ {'DMF':<10} │ {'Interpretation':<15} │ {'Status':<12} │")
    print("├" + "─" * 90 + "┤")

    for alias1, alias2 in GENE_PAIRS:
        pair_label = f"{GENE_MAP[alias1]['name']}-{GENE_MAP[alias2]['name']}"
        pair_rows = dmi_results_df[dmi_results_df["Alias_pair"] == f"{alias1}-{alias2}"]

        if len(pair_rows) == 0 or pair_rows["Epsilon"].isna().all():
            print(f"│ {pair_label:<15} │ {'N/A':<12} │ {'N/A':<10} │ {'N/A':<10} │ {'N/A':<15} │ {'NOT FOUND':<12} │")
        else:
            for _, row in pair_rows.iterrows():
                eps = row["Epsilon"]
                p = row["P_value"]
                dmf = row["DMF"]

                if eps < -0.08:
                    interp = "Strong NEG"
                elif eps > 0.08:
                    interp = "Strong POS"
                else:
                    interp = "Neutral"

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

                print(f"│ {pair_label:<15} │ {eps:>+12.4f} │ {p:<10.4f} │ {dmf:<10.4f} │ {interp:<15} │ {sig:<12} │")

    print("└" + "─" * 90 + "┘")

    # ========================================================================
    # Context for Triple Interactions
    # ========================================================================
    print("\n" + "=" * 80)
    print("Implications for Triple Knockout Predictions")
    print("=" * 80)

    print("""
The Costanzo2016 dataset provides training data for:
- Single mutant fitness effects
- Pairwise (digenic) genetic interactions

For triple knockouts, the model must GENERALIZE from:
1. Single gene effects (SMF values above)
2. Pairwise interactions (ε values above)
3. Graph structure (STRING, regulatory networks, etc.)

Key observations:
- If a gene has NO single mutant data (like ADH3), the model relies on:
  • Learnable gene embeddings trained on other genes
  • Graph connections to genes with known phenotypes

- If a pair has NO digenic interaction data, the model must infer from:
  • Pathway relationships (ADH genes are co-functional)
  • Physical/genetic interaction networks

This is the fundamental challenge: predicting HIGHER-ORDER interactions
from LOWER-ORDER data through learned representations.
""")

    print("=" * 80)
    print("Lookup Complete!")
    print("=" * 80)

    return {
        "smf": smf_results_df,
        "dmi": dmi_results_df,
        "dmf": dmf_results_df,
    }


if __name__ == "__main__":
    results = main()
