# src/torchcell/datasets/scerevisiae/costanzo2016.py
# [[src.torchcell.datasets.scerevisiae.costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/scerevisiae/costanzo2016.py
# Test file: src/torchcell/datasets/scerevisiae/test_costanzo2016.py
import json
import os
import os.path as osp
import random
import re
import shutil
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import polars as pl
import torch
from polars import DataFrame, col
from torch_geometric.data import (
    Data,
    DataLoader,
    Dataset,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm

from torchcell.prof import prof


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


class DMFCostanzo2016Dataset(InMemoryDataset):
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
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_dmf.pt"]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = osp.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets:"
            " Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(osp.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)

    def process(self):
        data_list = []

        # Process the DMF Files
        print("Processing DMF Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = osp.join(self.raw_dir, file_name)
            df = pd.read_csv(file_path, sep="\t", header=0)

            # Extract genotype information
            query_id = df["Query Strain ID"].str.split("_").str[0].tolist()
            array_id = df["Array Strain ID"].str.split("_").str[0].tolist()

            query_genotype = [
                {"id": id_val, "intervention": "deletion", "id_full": full_id}
                for id_val, full_id in zip(query_id, df["Query Strain ID"])
            ]
            array_genotype = [
                {"id": id_val, "intervention": "deletion", "id_full": full_id}
                for id_val, full_id in zip(array_id, df["Array Strain ID"])
            ]

            # Combine the genotypes
            combined_genotypes = list(zip(query_genotype, array_genotype))

            # Extract observation information
            # still a loop (no vectorization) due to the data structure complexity
            observations = [
                {
                    "smf": [row["Query single mutant fitness (SMF)"], row["Array SMF"]],
                    "dmf": row["Double mutant fitness"],
                    "dmf_std": row["Double mutant fitness standard deviation"],
                    "genetic_interaction_score": row["Genetic interaction score (ε)"],
                    "genetic_interaction_p-value": row["P-value"],
                }
                for _, row in df.iterrows()
            ]

            # Create environment dict
            environment = {"media": "YPD", "temperature": 30}

            # Combine everything
            combined_data = [
                {
                    "genotype": genotype,
                    "phenotype": {
                        "observation": observation,
                        "environment": environment,
                    },
                }
                for genotype, observation in zip(combined_genotypes, observations)
            ]

            # Convert to Data objects
            for item in combined_data:
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]
                data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    # in DMFCostanzo2016Dataset
    @property
    def gene_set(self):
        gene_ids = set()
        for data in self:
            for genotype in data.genotype:
                gene_ids.add(genotype["id"])
        return gene_ids

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


# TODO there is probably a more efficient way to do this
# Fine for now.
class DMFCostanzo2016SmallDataset(InMemoryDataset):
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
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_dmf_small.pt"]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = osp.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets:"
            " Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(osp.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)

    def process(self):
        data_list = []

        # Process the DMF Files
        print("Processing DMF Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = osp.join(self.raw_dir, file_name)
            df = pd.read_csv(file_path, sep="\t", header=0)

            # Extract genotype information
            query_id = df["Query Strain ID"].str.split("_").str[0].tolist()
            array_id = df["Array Strain ID"].str.split("_").str[0].tolist()

            query_genotype = [
                {"id": id_val, "intervention": "deletion", "id_full": full_id}
                for id_val, full_id in zip(query_id, df["Query Strain ID"])
            ]
            array_genotype = [
                {"id": id_val, "intervention": "deletion", "id_full": full_id}
                for id_val, full_id in zip(array_id, df["Array Strain ID"])
            ]

            # Combine the genotypes
            combined_genotypes = list(zip(query_genotype, array_genotype))

            # Extract observation information
            # still a loop (no vectorization) due to the data structure complexity
            observations = [
                {
                    "smf": [row["Query single mutant fitness (SMF)"], row["Array SMF"]],
                    "dmf": row["Double mutant fitness"],
                    "dmf_std": row["Double mutant fitness standard deviation"],
                    "genetic_interaction_score": row["Genetic interaction score (ε)"],
                    "genetic_interaction_p-value": row["P-value"],
                }
                for _, row in df.iterrows()
            ]

            # Create environment dict
            environment = {"media": "YPD", "temperature": 30}

            # Combine everything
            combined_data = [
                {
                    "genotype": genotype,
                    "phenotype": {
                        "observation": observation,
                        "environment": environment,
                    },
                }
                for genotype, observation in zip(combined_genotypes, observations)
            ]

            # Convert to Data objects
            for item in combined_data:
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]
                data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        # select 1000 random samples from data_list
        random.shuffle(data_list)
        # TODO this is hack
        data_list = data_list[:100000]
        torch.save(self.collate(data_list), self.processed_paths[0])

    # in DMFCostanzo2016Dataset
    @property
    def gene_set(self):
        gene_ids = set()
        for data in self:
            for genotype in data.genotype:
                gene_ids.add(genotype["id"])
        return gene_ids

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


# class DMFCostanzo2016LargeDataset(Dataset):
#     url = (
#         "https://thecellmap.org/costanzo2016/data_files/"
#         "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
#     )

#     def __init__(
#         self,
#         root: str = "data/scerevisiae/costanzo2016/large",
#         transform: Callable | None = None,
#         pre_transform: Callable | None = None,
#     ):
#         # self.data_list = []  # set here for len
#         super().__init__(root, transform, pre_transform) # breakpoint
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self) -> list[str]:
#         return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

#     @property
#     def processed_file_names(self) -> list[str]:
#         return [f"data_dmf_{i}.pt" for i in range(self.len())]

#     def download(self):
#         path = download_url(self.url, self.raw_dir)
#         with zipfile.ZipFile(path, "r") as zip_ref:
#             zip_ref.extractall(self.raw_dir)
#         os.remove(path)

#         # Move the contents of the subdirectory to the parent raw directory
#         sub_dir = os.path.join(
#             self.raw_dir,
#             "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
#         )
#         for filename in os.listdir(sub_dir):
#             shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
#         os.rmdir(sub_dir)

#     @prof
#     def process(self):
#         data_list = []

#         # Process the DMF Files
#         print("Processing DMF Files...")
#         for file_name in tqdm(self.raw_file_names):
#             file_path = os.path.join(self.raw_dir, file_name)
#             df = pd.read_csv(file_path, sep="\t", header=0)

#             # Extract genotype information
#             query_id = df["Query Strain ID"].str.split("_").str[0].tolist()
#             array_id = df["Array Strain ID"].str.split("_").str[0].tolist()

#             query_genotype = [
#                 {"id": id_val, "intervention": "deletion", "id_full": full_id}
#                 for id_val, full_id in zip(query_id, df["Query Strain ID"])
#             ]
#             array_genotype = [
#                 {"id": id_val, "intervention": "deletion", "id_full": full_id}
#                 for id_val, full_id in zip(array_id, df["Array Strain ID"])
#             ]

#             # Combine the genotypes
#             combined_genotypes = list(zip(query_genotype, array_genotype))

#             # Extract observation information
#             observations = [
#                 {
#                     "smf": [
#                         row["Query single mutant fitness (SMF)"],
#                         row["Array SMF"],
#                     ],
#                     "dmf": row["Double mutant fitness"],
#                     "dmf_std": row["Double mutant fitness standard deviation"],
#                     "genetic_interaction_score": row["Genetic interaction score (ε)"],
#                     "genetic_interaction_p-value": row["P-value"],
#                 }
#                 for _, row in df.iterrows()  # This part is still a loop due to the complexity of the data structure
#             ]

#             # Create environment dict
#             environment = {"media": "YPD", "temperature": 30}

#             # Combine everything
#             combined_data = [
#                 {
#                     "genotype": genotype,
#                     "phenotype": {
#                         "observation": observation,
#                         "environment": environment,
#                     },
#                 }
#                 for genotype, observation in zip(combined_genotypes, observations)
#             ]

#             with ThreadPoolExecutor(1) as executor:
#                 futures = []
#                 for idx, item in enumerate(combined_data):
#                     data = Data()
#                     data.genotype = item["genotype"]
#                     data.phenotype = item["phenotype"]
#                     data_list.append(data)  # fill the data_list

#                     # Submit each save operation to the thread pool
#                     future = executor.submit(
#                         self.save_data, data, idx, self.processed_dir
#                     )
#                     futures.append(future)

#             # Optionally, wait for all futures to complete
#             for future in futures:
#                 future.result()

#         self.data_list = data_list  # if you intend to use data_list later
#         self.save_metadata_to_json()  # save metadata.
#         return data_list  # if you want to return it

#     # make it a static method as it doesn't use any instance attributes
#     @staticmethod
#     def save_data(data, idx, processed_dir):
#         file_name = f"data_dmf_{idx}.pt"
#         torch.save(data, os.path.join(processed_dir, file_name))

#     def save_metadata_to_json(self):
#         metadata_list = []
#         for data in self.data_list:
#             metadata_entry = {
#                 "genotype": data.genotype,  # Assuming this is serializable
#                 "phenotype": data.phenotype,  # Assuming this is serializable
#                 # Add other attributes that are stored in your 'Data' objects
#             }
#             metadata_list.append(metadata_entry)

#         # Convert list of metadata entries to JSON and save to file
#         metadata_json = json.dumps(metadata_list, indent=4)
#         metadata_path = os.path.join(self.raw_dir, "metadata.json")
#         with open(metadata_path, "w") as f:
#             f.write(metadata_json)

#             print(f"Metadata saved to {metadata_path}")

#     def len(self):
#         return len(self.data_list) if self.data_list else 0

#     def get(self, idx):
#         sample = self.data_list[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

#     @property
#     def gene_set(self):
#         gene_ids = set()
#         for data in self.data_list:
#             for genotype in data.genotype:
#                 gene_ids.add(genotype["id"])
#         return gene_ids

#     def __repr__(self):
#         return f"{self.__class__.__name__}({len(self)})"


class DMFCostanzo2016LargeDataset(Dataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016/large",
        preprocess: str = "low_dmf_std",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.data_list = []
        self.preprocess = preprocess
        self.preprocess_dir = osp.join(root, "preprocess")
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
        self.data_list = self.load_processed_data()
        self.data, self.slices = torch.load(self.processed_paths[0])

    def load_processed_data(self):
        if self.data_list != []:
            return self.data_list
        data_list = []
        if osp.exists(self.processed_dir):
            for i in range(self.len()):
                file_path = osp.join(self.processed_dir, f"data_dmf_{i}.pt")
                data = torch.load(file_path)
                data_list.append(data)
        return data_list

    @property
    def raw_file_names(self) -> list[str]:
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_dmf_{i}.pt" for i in range(self.len())]

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

    # @prof
    def process(self):
        data_list = []

        # Initialize an empty DataFrame to hold all raw data
        all_data_df = pd.DataFrame()

        # Read and concatenate all raw files
        print("Reading and Concatenating Raw Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)

            # Reading data using Pandas; limit rows for demonstration
            df = pd.read_csv(file_path, sep="\t").head(1000)

            # Concatenating data frames
            all_data_df = pd.concat([all_data_df, df], ignore_index=True)

        # Functions for data filtering... duplicates selection,
        all_data_df = self.preprocess_raw(all_data_df, self.preprocess)
        self.save_preprocess_config(self.preprocess)
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

        with ThreadPoolExecutor(1) as executor:
            futures = []
            for idx, item in tqdm(enumerate(combined_data)):
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]
                data_list.append(data)  # fill the data_list

                # Submit each save operation to the thread pool
                future = executor.submit(self.save_data, data, idx, self.processed_dir)
                futures.append(future)

        # Optionally, wait for all futures to complete
        for future in futures:
            future.result()

        self.data_list = data_list  # if you intend to use data_list later
        return data_list  # if you want to return it

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

        # Select which duplicate to keep based on criteria
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
            raise ValueError("Unknown criteria")

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

    # New method to load existing preprocess configuration
    def load_preprocess_config(self):
        config_path = osp.join(self.preprocess_dir, "preprocess_config.json")

        if osp.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config
        else:
            return None

    # make it a static method as it doesn't use any instance attributes
    @staticmethod
    def save_data(data, idx, processed_dir):
        file_name = f"data_dmf_{idx}.pt"
        torch.save(data, os.path.join(processed_dir, file_name))

    def len(self):
        if osp.exists(self.processed_dir):
            num_files = len(
                [
                    f
                    for f in os.listdir(self.processed_dir)
                    if re.match(r"data_dmf_\d+\.pt", f)
                ]
            )
            return num_files
        else:
            return 0

    def get(self, idx):
        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    @property
    def gene_set(self):
        gene_ids = set()
        for data in self.data_list:
            for genotype in data.genotype:
                gene_ids.add(genotype["id"])
        return gene_ids

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


if __name__ == "__main__":
    # Load workspace
    import os
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    # HACH ... needs to be done in dir.
    os.makedirs(osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016"), exist_ok=True)
    os.makedirs(
        osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016/large"), exist_ok=True
    )
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

    # dmf_dataset_large = DMFCostanzo2016LargeDataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_large_nothread")
    # )
    dmf_dataset_large = DMFCostanzo2016LargeDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_meta_small"),
        preprocess="low_dmf",
    )
    print(dmf_dataset_large)
    print(dmf_dataset_large[0])
    print(dmf_dataset_large[1])
