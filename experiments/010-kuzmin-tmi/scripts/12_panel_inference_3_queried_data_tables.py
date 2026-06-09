# experiments/010-kuzmin-tmi/scripts/12_panel_inference_3_queried_data_tables.py
# [[experiments.010-kuzmin-tmi.scripts.12_panel_inference_3_queried_data_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/12_panel_inference_3_queried_data_tables

"""
Query experimental fitness/interaction data for Panel 12 Inference 3.

This script takes singles and doubles CSV files from inference_3 results
and enriches them with corresponding experimental values from Costanzo2016,
Kuzmin2018, and Kuzmin2020 datasets.

Phase 1: Singles (SmfCostanzo2016, SmfKuzmin2018, SmfKuzmin2020)
Phase 2: Doubles (DmfCostanzo2016, DmfKuzmin2018, DmfKuzmin2020,
                   DmiCostanzo2016, DmiKuzmin2018, DmiKuzmin2020)
"""

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    DmfKuzmin2018Dataset,
    SmfCostanzo2016Dataset,
    SmfKuzmin2018Dataset,
)
from torchcell.datasets.scerevisiae.costanzo2016 import DmiCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2018 import DmiKuzmin2018Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    DmfKuzmin2020Dataset,
    DmiKuzmin2020Dataset,
    SmfKuzmin2020Dataset,
)

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

        dataset = cls(
            root=osp.join(DATA_ROOT, f"data/torchcell/{subdir}"),
            io_workers=4,
        )
        print(f"  Dataset size: {len(dataset)}")

        gene_index = build_gene_index(dataset, name)
        print(f"  Index size: {len(gene_index)}")

        if data_type == "fitness":
            result_df[f"{name}_fitness"] = None
            result_df[f"{name}_std"] = None
            result_df[f"{name}_strain_id"] = None

            for idx, row in result_df.iterrows():
                gene_set = frozenset([row["gene"]])
                if gene_set in gene_index:
                    data = gene_index[gene_set]
                    result_df.at[idx, f"{name}_fitness"] = data["fitness"]
                    result_df.at[idx, f"{name}_std"] = data["fitness_std"]
                    result_df.at[idx, f"{name}_strain_id"] = data["strain_id"]

        matches = result_df[f"{name}_fitness"].notna().sum()
        print(f"  Matches found: {matches}/{len(result_df)}")

    return result_df


def query_doubles(
    doubles_df: pd.DataFrame, datasets_config: list[tuple]
) -> pd.DataFrame:
    """
    Query fitness/interaction data for doubles from all configured datasets.

    Args:
        doubles_df: DataFrame with 'gene1', 'gene2' columns
        datasets_config: List of (name, cls, subdir, data_type) tuples

    Returns:
        DataFrame enriched with fitness or interaction columns from each dataset
    """
    result_df = doubles_df.copy()

    for name, cls, subdir, data_type in datasets_config:
        print(f"\nProcessing {name}...")

        dataset = cls(
            root=osp.join(DATA_ROOT, f"data/torchcell/{subdir}"),
            io_workers=4,
        )
        print(f"  Dataset size: {len(dataset)}")

        gene_index = build_gene_index(dataset, name)
        print(f"  Index size: {len(gene_index)}")

        if data_type == "fitness":
            result_df[f"{name}_fitness"] = None
            result_df[f"{name}_std"] = None
            result_df[f"{name}_strain_id"] = None

            for idx, row in result_df.iterrows():
                gene_set = frozenset([row["gene1"], row["gene2"]])
                if gene_set in gene_index:
                    data = gene_index[gene_set]
                    result_df.at[idx, f"{name}_fitness"] = data["fitness"]
                    result_df.at[idx, f"{name}_std"] = data["fitness_std"]
                    result_df.at[idx, f"{name}_strain_id"] = data["strain_id"]

            matches = result_df[f"{name}_fitness"].notna().sum()

        elif data_type == "interaction":
            result_df[f"{name}_gene_interaction"] = None
            result_df[f"{name}_gene_interaction_p_value"] = None
            result_df[f"{name}_strain_id"] = None

            for idx, row in result_df.iterrows():
                gene_set = frozenset([row["gene1"], row["gene2"]])
                if gene_set in gene_index:
                    data = gene_index[gene_set]
                    result_df.at[idx, f"{name}_gene_interaction"] = data[
                        "gene_interaction"
                    ]
                    result_df.at[idx, f"{name}_gene_interaction_p_value"] = data[
                        "gene_interaction_p_value"
                    ]
                    result_df.at[idx, f"{name}_strain_id"] = data["strain_id"]

            matches = result_df[f"{name}_gene_interaction"].notna().sum()

        print(f"  Matches found: {matches}/{len(result_df)}")

    return result_df


def main():
    exp_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi")
    results_dir = osp.join(exp_dir, "results/inference_3")

    singles_input = osp.join(results_dir, "singles_table_panel12_k200.csv")
    singles_output = osp.join(results_dir, "singles_table_panel12_k200_queried.csv")
    doubles_input = osp.join(results_dir, "doubles_table_panel12_k200.csv")
    doubles_output = osp.join(results_dir, "doubles_table_panel12_k200_queried.csv")

    print("=" * 60)
    print("Panel 12 Inference 3: Query Experimental Data Tables")
    print("=" * 60)

    # ── Phase 1: Singles ──────────────────────────────────────
    print("\n[Phase 1] Processing singles...")

    singles_df = pd.read_csv(singles_input)
    print(f"Loaded {len(singles_df)} genes from singles table")

    singles_datasets = [
        ("SmfCostanzo2016", SmfCostanzo2016Dataset, "smf_costanzo2016", "fitness"),
        ("SmfKuzmin2018", SmfKuzmin2018Dataset, "smf_kuzmin2018", "fitness"),
        ("SmfKuzmin2020", SmfKuzmin2020Dataset, "smf_kuzmin2020", "fitness"),
    ]

    singles_result = query_singles(singles_df, singles_datasets)
    singles_result.to_csv(singles_output, index=False)
    print(f"\nSaved: {singles_output}")

    print("\n" + "-" * 60)
    print("Singles Summary:")
    print("-" * 60)
    print(singles_result.to_string())

    # ── Phase 2: Doubles ──────────────────────────────────────
    print("\n\n[Phase 2] Processing doubles...")

    doubles_df = pd.read_csv(doubles_input)
    print(f"Loaded {len(doubles_df)} gene pairs from doubles table")

    doubles_datasets = [
        ("DmfCostanzo2016", DmfCostanzo2016Dataset, "dmf_costanzo2016", "fitness"),
        ("DmiCostanzo2016", DmiCostanzo2016Dataset, "dmi_costanzo2016", "interaction"),
        ("DmfKuzmin2018", DmfKuzmin2018Dataset, "dmf_kuzmin2018", "fitness"),
        ("DmiKuzmin2018", DmiKuzmin2018Dataset, "dmi_kuzmin2018", "interaction"),
        ("DmfKuzmin2020", DmfKuzmin2020Dataset, "dmf_kuzmin2020", "fitness"),
        ("DmiKuzmin2020", DmiKuzmin2020Dataset, "dmi_kuzmin2020", "interaction"),
    ]

    doubles_result = query_doubles(doubles_df, doubles_datasets)
    doubles_result.to_csv(doubles_output, index=False)
    print(f"\nSaved: {doubles_output}")

    print("\n" + "-" * 60)
    print("Doubles Summary:")
    print("-" * 60)
    print(doubles_result.to_string())

    # ── Overall Summary ───────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Singles: {len(singles_result)} genes")
    for name, _, _, _ in singles_datasets:
        col = f"{name}_fitness"
        n = singles_result[col].notna().sum()
        print(f"  {name}: {n}/{len(singles_result)} matched")

    print(f"\nDoubles: {len(doubles_result)} gene pairs")
    for name, _, _, data_type in doubles_datasets:
        if data_type == "fitness":
            col = f"{name}_fitness"
        else:
            col = f"{name}_gene_interaction"
        n = doubles_result[col].notna().sum()
        print(f"  {name}: {n}/{len(doubles_result)} matched")


if __name__ == "__main__":
    main()
