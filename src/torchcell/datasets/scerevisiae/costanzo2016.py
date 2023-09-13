# src/torchcell/datasets/scerevisiae/costanzo2016.py
# [[src.torchcell.datasets.scerevisiae.costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/scerevisiae/costanzo2016.py
# Test file: src/torchcell/datasets/scerevisiae/test_costanzo2016.py
import json
import logging
import os
import os.path as osp
import pickle
import random
import re
import shutil
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import h5py
import lmdb
import numpy as np
import pandas as pd
import polars as pl
import torch
from polars import DataFrame, col
from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.prof import prof, prof_input

log = logging.getLogger(__name__)


class SMFCostanzo2016Dataset(InMemoryDataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["strain_ids_and_single_mutant_fitness.xlsx"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_smf.pt"]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = os.path.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)

    def process(self):
        # Process the Excel file for mutant fitness
        excel_file_path = os.path.join(
            self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx"
        )
        df_excel = pd.read_excel(excel_file_path)

        print("Processing SMF data...")

        # Extract genotypes
        df_excel["genotype_id"] = df_excel["Strain ID"].str.split("_").str[0]
        genotypes = df_excel[["genotype_id", "Strain ID"]].to_dict("records")

        # Create 26° observations
        observations_26 = (
            df_excel[
                ["Single mutant fitness (26°)", "Single mutant fitness (26°) stddev"]
            ]
            .rename(
                columns={
                    "Single mutant fitness (26°)": "smf",
                    "Single mutant fitness (26°) stddev": "smf_std",
                }
            )
            .to_dict("records")
        )
        environments_26 = [
            {"media": "YPD", "temperature": 26} for _ in range(len(df_excel))
        ]

        # Create 30° observations
        observations_30 = (
            df_excel[
                ["Single mutant fitness (30°)", "Single mutant fitness (30°) stddev"]
            ]
            .rename(
                columns={
                    "Single mutant fitness (30°)": "smf",
                    "Single mutant fitness (30°) stddev": "smf_std",
                }
            )
            .to_dict("records")
        )
        environments_30 = [
            {"media": "YPD", "temperature": 30} for _ in range(len(df_excel))
        ]

        data_list = []
        for genotype, obs_26, env_26, obs_30, env_30 in zip(
            genotypes,
            observations_26,
            environments_26,
            observations_30,
            environments_30,
        ):
            # For 26°
            data = Data()
            data.genotype = [
                {
                    "id": genotype["genotype_id"],
                    "intervention": "deletion",
                    "id_full": genotype["Strain ID"],
                }
            ]
            data.phenotype = {"observation": obs_26, "environment": env_26}
            data_list.append(data)

            # For 30°
            data = Data()
            data.genotype = [
                {
                    "id": genotype["genotype_id"],
                    "intervention": "deletion",
                    "id_full": genotype["Strain ID"],
                }
            ]
            data.phenotype = {"observation": obs_30, "environment": env_30}
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def gene_set(self):
        gene_ids = set()
        for data in self:
            for genotype in data.genotype:
                gene_ids.add(genotype["id"])
        return gene_ids

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


class DMFCostanzo2016Dataset(Dataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016",
        subset_n: int = None,
        preprocess: str = "low_dmf_std",
        skip_process_file_exist: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.subset_n = subset_n
        self._skip_process_file_exist = skip_process_file_exist
        # TODO consider moving to Dataset
        self.preprocess = preprocess
        # TODO consider moving to Dataset
        self.preprocess_dir = osp.join(root, "preprocess")
        self._length = None
        self._gene_set = None
        self._df = None
        # Check for existing preprocess config
        existing_config = self.load_preprocess_config()
        if existing_config is not None:
            if existing_config["preprocess"] != self.preprocess:
                raise ValueError(
                    "New preprocess does not match existing config."
                    "Delete the processed and process dir for a new Dataset."
                    "Or define a new root."
                )
        super().__init__(root, transform, pre_transform)
        self.env = None

    @property
    def skip_process_file_exist(self):
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> list[str]:
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        # return [f"data_dmf_{i}.pt" for i in range(self.len())]
        return "data.lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = os.path.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)

    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    @property
    def df(self):
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    def process(self):
        self._length = None
        # Initialize an empty DataFrame to hold all raw data
        all_data_df = pd.DataFrame()

        # Read and concatenate all raw files
        print("Reading and Concatenating Raw Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)

            # Reading data using Pandas; limit rows for demonstration
            df = pd.read_csv(file_path, sep="\t")

            # Concatenating data frames
            all_data_df = pd.concat([all_data_df, df], ignore_index=True)
        # Functions for data filtering... duplicates selection,
        all_data_df = self.preprocess_raw(all_data_df, self.preprocess)
        self.save_preprocess_config(self.preprocess)

        # Subset
        if self.subset_n is not None:
            all_data_df = all_data_df.sample(
                n=self.subset_n, random_state=42
            ).reset_index(drop=True)

        # Save preprocssed df - mainly for quick stats
        all_data_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        print("Processing DMF Files...")

        # Extract genotype information using Polars syntax
        query_genotype = [
            {"id": id_val, "intervention": "deletion", "id_full": full_id}
            for id_val, full_id in zip(
                all_data_df["Query Gene"], all_data_df["Query Strain ID"]
            )
        ]
        array_genotype = [
            {"id": id_val, "intervention": "deletion", "id_full": full_id}
            for id_val, full_id in zip(
                all_data_df["Array Gene"], all_data_df["Array Strain ID"]
            )
        ]

        # Combine the genotypes
        combined_genotypes = list(zip(query_genotype, array_genotype))

        # Extract observation information
        # This part is still a loop due to the complexity of the data structure
        observations = [
            {
                "smf": [row["Query single mutant fitness (SMF)"], row["Array SMF"]],
                "dmf": row["Double mutant fitness"],
                "dmf_std": row["Double mutant fitness standard deviation"],
                "genetic_interaction_score": row["Genetic interaction score (ε)"],
                "genetic_interaction_p-value": row["P-value"],
            }
            for index, row in all_data_df.iterrows()
        ]

        # Create environment dict
        environment = {"media": "YPD", "temperature": 30}

        # Combine everything
        combined_data = [
            {
                "genotype": genotype,
                "phenotype": {"observation": observation, "environment": environment},
            }
            for genotype, observation in zip(combined_genotypes, observations)
        ]

        # Initialize LMDB environment
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

        data_list = []
        for idx, item in tqdm(enumerate(combined_data)):
            data = Data()
            data.genotype = item["genotype"]
            data.phenotype = item["phenotype"]
            data_list.append(data)  # fill the data_list

        log.info("lmdb begin")
        # Open a new write transaction
        # TODO make map_size size of disk partition, only virtual address space.
        with env.begin(write=True) as txn:
            # Iterate through each data item
            for idx, item in tqdm(enumerate(combined_data)):
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]

                # Serialize the data object using pickle
                serialized_data = pickle.dumps(data)

                # Save the serialized data in the LMDB environment
                txn.put(f"{idx}".encode(), serialized_data)

        # cache gene property
        self.gene_set = self.compute_gene_set(data_list)

    def preprocess_raw(
        self, all_data_df: pd.DataFrame, preprocess: str = "low_dmf_std"
    ):
        # Function to extract gene name
        def extract_gene_name(x):
            return x.apply(lambda y: y.split("_")[0])

        # Extract gene names
        query_gene = extract_gene_name(all_data_df["Query Strain ID"]).rename(
            "Query Gene"
        )
        array_gene = extract_gene_name(all_data_df["Array Strain ID"]).rename(
            "Array Gene"
        )

        # Create DataFrame with extracted gene names
        new_df = pd.concat([all_data_df, query_gene, array_gene], axis=1)

        # Function to create and sort genotype
        def create_and_sort_genotype(row):
            query, array = row["Query Gene"], row["Array Gene"]
            return "_".join(sorted([query, array]))

        # Add the genotype column
        new_df["genotype"] = new_df.apply(create_and_sort_genotype, axis=1)

        # Find duplicate genotypes
        duplicate_genotypes = new_df["genotype"].duplicated(keep=False)
        duplicates_df = new_df[duplicate_genotypes].copy()

        # Select which duplicate to keep based on preprocess
        if preprocess == "low_dmf_std":
            # Keep the row with the lowest 'Double mutant fitness standard deviation'
            idx_to_keep = duplicates_df.groupby("genotype")[
                "Double mutant fitness standard deviation"
            ].idxmin()
        elif preprocess == "high_dmf":
            # Keep the row with the highest 'Double mutant fitness'
            idx_to_keep = duplicates_df.groupby("genotype")[
                "Double mutant fitness"
            ].idxmax()
        elif preprocess == "low_dmf":
            # Keep the row with the lowest 'Double mutant fitness'
            idx_to_keep = duplicates_df.groupby("genotype")[
                "Double mutant fitness"
            ].idxmin()
        else:
            raise ValueError("Unknown preprocess")

        # Drop duplicates, keeping only the selected rows
        duplicates_df = duplicates_df.loc[idx_to_keep]

        # Combine the non-duplicate and selected duplicate rows
        non_duplicates_df = new_df[~duplicate_genotypes]
        final_df = pd.concat([non_duplicates_df, duplicates_df], ignore_index=True)

        return final_df

    # New method to save preprocess configuration to a JSON file
    def save_preprocess_config(self, preprocess):
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)

        config = {"preprocess": preprocess}

        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(config, f)

    # TODO implement key merge
    # criterion for merge is defined as key, value is the data object itself.

    # New method to load existing preprocess configuration
    def load_preprocess_config(self):
        config_path = osp.join(self.preprocess_dir, "preprocess_config.json")

        if osp.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config
        else:
            return None

    def len(self) -> int:
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def get(self, idx):
        """Initialize LMDB if it hasn't been initialized yet."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            data = pickle.loads(serialized_data)
            if self.transform:
                data = self.transform(data)
            return data

    @staticmethod
    def compute_gene_set(data_list):
        gene_ids = set()
        for data in data_list:
            for genotype in data.genotype:
                gene_ids.add(genotype["id"])
        return gene_ids

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        try:
            if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
                with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                    self._gene_set = set(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return self._gene_set
        except json.JSONDecodeError:
            raise ValueError("Invalid or empty JSON file found.")

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


# @prof_input
def main():
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    os.makedirs(
        osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_init"), exist_ok=True
    )
    # dmf_dataset = DMFCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
    #     subset_n=100000,
    #     preprocess="low_dmf_std",
    # )
    dmf_dataset = DMFCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e4"),
        preprocess="low_dmf_std",
        subset_n=10000,
    )
    print(dmf_dataset)
    print(dmf_dataset.df["Double mutant fitness"].describe())
    print(dmf_dataset.df["Double mutant fitness standard deviation"].describe())
    print(dmf_dataset.gene_set)
    for i in range(10):
        print(dmf_dataset[i])


if __name__ == "__main__":
    # Load workspace
    main()
    # from dotenv import load_dotenv

    # load_dotenv()
    # DATA_ROOT = os.getenv("DATA_ROOT")
    # os.makedirs(osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016"), exist_ok=True)
    # os.makedirs(
    #     osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016/large"), exist_ok=True
    # )
    # Process data

    # smf_dataset = SMFCostanzo2016Dataset()
    # print(smf_dataset)
    # print(smf_dataset[0])
    # print(len(smf_dataset.gene_set))

    # DMF Small
    # dmf_dataset = DMFCostanzo2016SmallDataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016")
    # )
    # print(dmf_dataset)
    # print(dmf_dataset[0])

    # dmf_dataset = DMFCostanzo2016Dataset()
    # print(dmf_dataset)
    # print(dmf_dataset[0])
    # print()

    # dmf_dataset_large = DMFCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_large_nothread")
    # )
    # dmf_dataset_large = DMFCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016"),
    #     preprocess="low_dmf_std",
    # )
    # print(dmf_dataset_large)
    # print(dmf_dataset_large[0])
    # print(dmf_dataset_large[1]
