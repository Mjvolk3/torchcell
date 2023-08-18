# src/torchcell/datasets/scerevisiae/costanzo2016.py
# [[src.torchcell.datasets.scerevisiae.costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/scerevisiae/costanzo2016.py
# Test file: src/torchcell/datasets/scerevisiae/test_costanzo2016.py

import os
import os.path as osp
import random
import shutil
import zipfile
from collections.abc import Callable
from typing import Optional
from torch_geometric.data import Dataset
import pandas as pd
import torch
from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm

# HACH ... needs to be done in dir.
os.makedirs("data/scerevisiae/costanzo2016", exist_ok=True)
os.makedirs("data/scerevisiae/costanzo2016/large", exist_ok=True)


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
                    "Single mutant fitness (26°)": "smf_fitness",
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
                    "Single mutant fitness (30°)": "smf_fitness",
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
                    "smf_fitness": [
                        row["Query single mutant fitness (SMF)"],
                        row["Array SMF"],
                    ],
                    "dmf_fitness": row["Double mutant fitness"],
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
                    "smf_fitness": [
                        row["Query single mutant fitness (SMF)"],
                        row["Array SMF"],
                    ],
                    "dmf_fitness": row["Double mutant fitness"],
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
        # HACK
        data_list = data_list[:10000]
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


class DMFCostanzo2016LargeDataset(Dataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016/large",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.data_list = self.process()

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

    def process(self):
        data_list = []

        # Process the DMF Files
        print("Processing DMF Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)
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
            observations = [
                {
                    "smf_fitness": [
                        row["Query single mutant fitness (SMF)"],
                        row["Array SMF"],
                    ],
                    "dmf_fitness": row["Double mutant fitness"],
                    "dmf_std": row["Double mutant fitness standard deviation"],
                    "genetic_interaction_score": row["Genetic interaction score (ε)"],
                    "genetic_interaction_p-value": row["P-value"],
                }
                for _, row in df.iterrows()  # This part is still a loop due to the complexity of the data structure
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

            # Convert to Data objects and save each instance to a separate file
            for idx, item in enumerate(combined_data):
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]
                data_list.append(data)

                # Save each Data object to its own file
                file_name = f"data_dmf_{idx}.pt"
                torch.save(data, os.path.join(self.processed_dir, file_name))

        return data_list

    def len(self):
        return len(self.data_list)

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
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Process data

    # smf_dataset = SMFCostanzo2016Dataset()
    # print(smf_dataset)
    # print(smf_dataset[0])
    # print(len(smf_dataset.gene_set))

    # DMF Small
    dmf_dataset = DMFCostanzo2016SmallDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016")
    )
    print(dmf_dataset)
    print(dmf_dataset[0])

    # dmf_dataset = DMFCostanzo2016Dataset()
    # print(dmf_dataset)
    # print(dmf_dataset[0])
    # print()

    # dmf_dataset_large = DMFCostanzo2016LargeDataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016/large")
    # )
    # print(dmf_dataset_large)
    # print(dmf_dataset_large[0])
