# experiments/010-kuzmin-tmi/scripts/12_panel_inference_2_queried_data_tables
# [[experiments.010-kuzmin-tmi.scripts.12_panel_inference_2_queried_data_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/12_panel_inference_2_queried_data_tables
# Test file: experiments/010-kuzmin-tmi/scripts/test_12_panel_inference_2_queried_data_tables.py

"""
Query experimental fitness/interaction data for Panel 12 Inference 2.

This script takes singles, doubles, and triples CSV files from inference_2 results
and enriches them with corresponding experimental values from Costanzo2016,
Kuzmin2018, and Kuzmin2020 datasets.

Phase 1: Singles only (SmfCostanzo2016, SmfKuzmin2018, SmfKuzmin2020)
"""

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    SmfKuzmin2018Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]


def build_gene_index(dataset, dataset_name: str) -> dict[frozenset, dict]:
    """
    Build index: frozenset({gene1, gene2, ...}) -> phenotype data dict.

    For fitness datasets: {'fitness': float, 'fitness_std': float, 'strain_id': str}
    For interaction datasets: {'gene_interaction': float, 'gene_interaction_p_value': float, 'strain_id': str}
    """
    index = {}
    for i in tqdm(range(len(dataset)), desc=f"Indexing {dataset_name}"):
        item = dataset[i]
        genes = frozenset(
            p["systematic_gene_name"]
            for p in item["experiment"]["genotype"]["perturbations"]
        )
        phenotype = item["experiment"]["phenotype"]
        strain_id = item["experiment"]["genotype"]["perturbations"][0]["strain_id"]

        if "fitness" in phenotype:
            index[genes] = {
                "fitness": phenotype["fitness"],
                "fitness_std": phenotype["fitness_std"],
                "strain_id": strain_id,
            }
        else:
            index[genes] = {
                "gene_interaction": phenotype["gene_interaction"],
                "gene_interaction_p_value": phenotype["gene_interaction_p_value"],
                "strain_id": strain_id,
            }
    return index


def query_singles(
    singles_df: pd.DataFrame, datasets_config: list[tuple]
) -> pd.DataFrame:
    """
    Query fitness data for singles from all configured datasets.

    Args:
        singles_df: DataFrame with 'gene' column containing systematic gene names
        datasets_config: List of (name, cls, subdir, data_type) tuples

    Returns:
        DataFrame enriched with fitness columns from each dataset
    """
    result_df = singles_df.copy()

    for name, cls, subdir, data_type in datasets_config:
        print(f"\nProcessing {name}...")

        # Initialize dataset
        dataset = cls(
            root=osp.join(DATA_ROOT, f"data/torchcell/{subdir}"),
            io_workers=4,
        )
        print(f"  Dataset size: {len(dataset)}")

        # Build gene index
        gene_index = build_gene_index(dataset, name)
        print(f"  Index size: {len(gene_index)}")

        # Add columns for this dataset
        if data_type == "fitness":
            result_df[f"{name}_fitness"] = None
            result_df[f"{name}_std"] = None
            result_df[f"{name}_strain_id"] = None

            # Query each gene
            for idx, row in result_df.iterrows():
                gene_set = frozenset([row["gene"]])
                if gene_set in gene_index:
                    data = gene_index[gene_set]
                    result_df.at[idx, f"{name}_fitness"] = data["fitness"]
                    result_df.at[idx, f"{name}_std"] = data["fitness_std"]
                    result_df.at[idx, f"{name}_strain_id"] = data["strain_id"]

        elif data_type == "interaction":
            result_df[f"{name}_gene_interaction"] = None
            result_df[f"{name}_gene_interaction_p_value"] = None
            result_df[f"{name}_strain_id"] = None

            for idx, row in result_df.iterrows():
                gene_set = frozenset([row["gene"]])
                if gene_set in gene_index:
                    data = gene_index[gene_set]
                    result_df.at[idx, f"{name}_gene_interaction"] = data[
                        "gene_interaction"
                    ]
                    result_df.at[idx, f"{name}_gene_interaction_p_value"] = data[
                        "gene_interaction_p_value"
                    ]
                    result_df.at[idx, f"{name}_strain_id"] = data["strain_id"]

        # Report matches
        if data_type == "fitness":
            matches = result_df[f"{name}_fitness"].notna().sum()
        else:
            matches = result_df[f"{name}_gene_interaction"].notna().sum()
        print(f"  Matches found: {matches}/{len(result_df)}")

    return result_df


def main():
    # Paths
    exp_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi")
    results_dir = osp.join(exp_dir, "results/inference_2")

    singles_input = osp.join(results_dir, "singles_table_panel12_k200.csv")
    singles_output = osp.join(results_dir, "singles_table_panel12_k200_queried.csv")

    print("=" * 60)
    print("Panel 12 Inference 2: Query Experimental Data Tables")
    print("=" * 60)

    # Phase 1: Singles
    print("\n[Phase 1] Processing singles...")

    singles_df = pd.read_csv(singles_input)
    print(f"Loaded {len(singles_df)} genes from singles table")

    # Configure datasets for singles (Smf = single mutant fitness)
    singles_datasets = [
        ("SmfCostanzo2016Dataset", SmfCostanzo2016Dataset, "smf_costanzo2016", "fitness"),
        ("SmfKuzmin2018Dataset", SmfKuzmin2018Dataset, "smf_kuzmin2018", "fitness"),
        ("SmfKuzmin2020Dataset", SmfKuzmin2020Dataset, "smf_kuzmin2020", "fitness"),
    ]

    # Query all datasets
    singles_result = query_singles(singles_df, singles_datasets)

    # Save output
    singles_result.to_csv(singles_output, index=False)
    print(f"\nSaved: {singles_output}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(singles_result.to_string())


if __name__ == "__main__":
    main()
