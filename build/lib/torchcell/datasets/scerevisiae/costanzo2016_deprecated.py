"""Deprecated Costanzo 2016 single/double mutant fitness datasets (legacy)."""

# torchcell/datasets/scerevisiae/costanzo2016.py
# [[torchcell.datasets.scerevisiae.costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/costanzo2016.py
# Test file: torchcell/datasets/scerevisiae/test_costanzo2016.py
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import zipfile
from abc import ABC, abstractproperty
from collections.abc import Callable
from typing import Literal

import lmdb
import pandas as pd

# import polars as pl
# from polars import DataFrame, col
from torch_geometric.data import Data, download_url
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datamodels import ModelStrict
from torchcell.sequence import GeneSet

log = logging.getLogger(__name__)


class SmfCostanzo2016Dataset(Dataset):
    """Single-mutant fitness dataset from Costanzo 2016 (deprecated)."""

    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016",
        subset_n: int = None,
        preprocess: dict | None = None,
        skip_process_file_exist_check: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        """Set up preprocessing config and paths, then initialize the dataset."""
        self.subset_n = subset_n
        self._skip_process_file_exist = skip_process_file_exist_check
        self.preprocess = preprocess
        self.preprocess_dir = osp.join(root, "preprocess")
        self._length = None
        self._gene_set = None
        self._df = None
        # Check for existing preprocess config
        existing_config = self.load_preprocess_config()
        if existing_config is not None:
            if existing_config != self.preprocess:
                raise ValueError(
                    "New preprocess does not match existing config."
                    "Delete the processed and process dir for a new Dataset."
                    "Or define a new root."
                )
        super().__init__(root, transform, pre_transform)
        self.env = None

    @property
    def skip_process_file_exist(self):
        """Return whether the processed-file existence check is skipped."""
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw single-mutant fitness Excel file name."""
        return ["strain_ids_and_single_mutant_fitness.xlsx"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed LMDB file name."""
        return "data.lmdb"

    def download(self):
        """Download and unzip the raw interaction data into the raw directory."""
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

    def close_lmdb(self):
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    # TODO implement
    @property
    def df(self):
        """Return the preprocessed data as a DataFrame if the CSV exists."""
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    @property
    def wt(self):
        """Return a wild-type reference Data object with fitness 1.0."""
        wt = {}
        wt["genotype"] = ({"id": None, "intervention": None, "id_full": None},)
        wt["phenotype"] = {
            "observation": {"fitness": 1.0},
            "environment": {"media": "YEPD", "temperature": 30},
        }
        data = Data()
        data.genotype = wt["genotype"]
        data.phenotype = wt["phenotype"]
        return data

    def process(self):
        """Read the Excel file, preprocess fitness rows, and write to LMDB."""
        os.makedirs(self.preprocess_dir, exist_ok=True)
        # Process the Excel file for mutant fitness
        excel_file_path = os.path.join(
            self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx"
        )
        df_excel = pd.read_excel(excel_file_path)

        # Preproecess
        all_data_df = self.preprocess_raw(df_excel, self.preprocess)
        self.save_preprocess_config(self.preprocess)

        # Subset
        if self.subset_n is not None:
            all_data_df = all_data_df.sample(
                n=self.subset_n, random_state=42
            ).reset_index(drop=True)

        # Save preprocssed df - mainly for quick stats
        all_data_df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        print("Processing SMF data...")

        # Extract genotypes
        all_data_df["genotype_id"] = all_data_df["Strain ID"].str.split("_").str[0]
        genotypes = all_data_df[["genotype_id", "Strain ID"]].to_dict("records")

        # Create 26° observations
        observations_26 = (
            all_data_df[
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
            {"media": "YEPD", "temperature": 26} for _ in range(len(all_data_df))
        ]

        # Create 30° observations
        observations_30 = (
            all_data_df[
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
            {"media": "YEPD", "temperature": 30} for _ in range(len(all_data_df))
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

        # Initialize LMDB environment
        log.info("lmdb begin")
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            # Iterate through each data item
            for idx, item in tqdm(enumerate(data_list)):
                serialized_data = pickle.dumps(item)
                txn.put(f"{idx}".encode(), serialized_data)

        # cache gene property
        self.gene_set = self.compute_gene_set(data_list)

    def preprocess_raw(self, all_data_df: pd.DataFrame, preprocess: dict | None = None):
        """Clean and deduplicate the raw single-mutant fitness DataFrame."""
        # We use the 'Systematic gene name' column as 'genotype' directly
        all_data_df["genotype"] = all_data_df["Systematic gene name"]

        # Find duplicate genotypes
        duplicate_genotypes = all_data_df["genotype"].duplicated(keep=False)
        duplicates_df = all_data_df[duplicate_genotypes].copy()

        # Select which duplicate to keep based on preprocess
        if preprocess is None:
            idx_to_keep = duplicates_df.index
        elif preprocess.get("duplicate_resolution") == "low_std_30":
            idx_to_keep = (
                duplicates_df.dropna(subset=["Single mutant fitness (30°) stddev"])
                .groupby("genotype")["Single mutant fitness (30°) stddev"]
                .idxmin()
            )
        elif preprocess.get("duplicate_resolution") == "high_fitness_30":
            idx_to_keep = (
                duplicates_df.dropna(subset=["Single mutant fitness (30°)"])
                .groupby("genotype")["Single mutant fitness (30°)"]
                .idxmax()
            )
        elif preprocess.get("duplicate_resolution") == "low_fitness_30":
            idx_to_keep = (
                duplicates_df.dropna(subset=["Single mutant fitness (30°)"])
                .groupby("genotype")["Single mutant fitness (30°)"]
                .idxmin()
            )
        elif preprocess.get("duplicate_resolution") == "low_std_26":
            idx_to_keep = (
                duplicates_df.dropna(subset=["Single mutant fitness (26°) stddev"])
                .groupby("genotype")["Single mutant fitness (26°) stddev"]
                .idxmin()
            )
        elif preprocess.get("duplicate_resolution") == "high_fitness_26":
            idx_to_keep = (
                duplicates_df.dropna(subset=["Single mutant fitness (26°)"])
                .groupby("genotype")["Single mutant fitness (26°)"]
                .idxmax()
            )
        elif preprocess.get("duplicate_resolution") == "low_fitness_26":
            idx_to_keep = (
                duplicates_df.dropna(subset=["Single mutant fitness (26°)"])
                .groupby("genotype")["Single mutant fitness (26°)"]
                .idxmin()
            )
        # HACK both abuses the idea of duplicates
        elif preprocess.get("duplicate_resolution") == "low_std_both":
            idx26 = (
                duplicates_df.dropna(subset=["Single mutant fitness (26°) stddev"])
                .groupby("genotype")["Single mutant fitness (26°) stddev"]
                .idxmin()
            )
            idx30 = (
                duplicates_df.dropna(subset=["Single mutant fitness (30°) stddev"])
                .groupby("genotype")["Single mutant fitness (30°) stddev"]
                .idxmin()
            )
            idx_to_keep = list(set(idx26).union(set(idx30)))
        # HACK both abuses the idea of duplicates
        elif preprocess.get("duplicate_resolution") == "low_fitness_both":
            idx26 = (
                duplicates_df.dropna(subset=["Single mutant fitness (26°)"])
                .groupby("genotype")["Single mutant fitness (26°)"]
                .idxmin()
            )
            idx30 = (
                duplicates_df.dropna(subset=["Single mutant fitness (30°)"])
                .groupby("genotype")["Single mutant fitness (30°)"]
                .idxmin()
            )
            idx_to_keep = idx26.append(idx30).unique()
        # HACK both abuses the idea of duplicates
        elif preprocess.get("duplicate_resolution") == "high_fitness_both":
            idx26 = (
                duplicates_df.dropna(subset=["Single mutant fitness (26°)"])
                .groupby("genotype")["Single mutant fitness (26°)"]
                .idxmax()
            )
            idx30 = (
                duplicates_df.dropna(subset=["Single mutant fitness (30°)"])
                .groupby("genotype")["Single mutant fitness (30°)"]
                .idxmax()
            )
            idx_to_keep = idx26.append(idx30).unique()
        else:
            raise ValueError("Unknown preprocess")

        # Drop duplicates, keeping only the selected rows
        duplicates_df = duplicates_df.loc[idx_to_keep]

        # Combine the non-duplicate and selected duplicate rows
        non_duplicates_df = all_data_df[~duplicate_genotypes]
        final_df = pd.concat([non_duplicates_df, duplicates_df], ignore_index=True)

        return final_df

    # New method to save preprocess configuration to a JSON file
    def save_preprocess_config(self, preprocess):
        """Write the preprocess configuration dict to JSON."""
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(preprocess, f)

    # New method to load existing preprocess configuration
    def load_preprocess_config(self):
        """Load the saved preprocess configuration, or None if absent."""
        config_path = osp.join(self.preprocess_dir, "preprocess_config.json")

        if osp.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config
        else:
            return None

    def len(self) -> int:
        """Return the number of records stored in the LMDB."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

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
        """Return the set of all gene ids appearing in the data list genotypes."""
        computed_gene_set = GeneSet()
        for data in data_list:
            for genotype in data.genotype:
                computed_gene_set.add(genotype["id"])
        return computed_gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        """Return the gene set, loading it from JSON if available."""
        try:
            if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
                with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                    self._gene_set = GeneSet(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return self._gene_set
        # CHECK can probably remove this
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
        """Return a string with the class name and record count."""
        return f"{self.__class__.__name__}({len(self)})"


class DatasetConfig(ABC):
    """Abstract base defining a dataset's duplicate-resolution order."""

    @abstractproperty
    def resolution_order(self) -> list[str]:
        """Return the ordered list of duplicate-resolution strategies."""
        pass


class DmfCostanzo2016Config(ModelStrict, DatasetConfig):
    """Config for the double-mutant dataset selecting how duplicates resolve."""

    duplicate_resolution: Literal["low_dmf_std", "high_dmf", "low_dmf"] | None = None

    @property
    def resolution_order(self):
        """Return the resolution order derived from the duplicate resolution setting."""
        return [str(self.duplicate_resolution)]


class DmfCostanzo2016Dataset(Dataset):
    """Double-mutant fitness dataset from Costanzo 2016 (deprecated)."""

    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016",
        subset_n: int = None,
        preprocess: dict | None = None,
        skip_process_file_exist: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        """Set up preprocessing config and paths, then initialize the dataset."""
        self.subset_n = subset_n
        self._skip_process_file_exist = skip_process_file_exist
        # TODO consider moving to a well defined Dataset class
        self.preprocess = preprocess
        # TODO consider moving to Dataset
        self.preprocess_dir = osp.join(root, "preprocess")
        self._length = None
        self._gene_set = None
        self._df = None
        # Check for existing preprocess config
        existing_config = self.load_preprocess_config()
        if existing_config is not None:
            if existing_config != self.preprocess:
                raise ValueError(
                    "New preprocess does not match existing config."
                    "Delete the processed and process dir for a new Dataset."
                    "Or define a new root."
                )
        super().__init__(root, transform, pre_transform)
        self.env = None

    @property
    def skip_process_file_exist(self):
        """Return whether the processed-file existence check is skipped."""
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw SGA interaction text file names."""
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed LMDB file name."""
        return "data.lmdb"

    def download(self):
        """Download and unzip the raw interaction data into the raw directory."""
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

    def close_lmdb(self):
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def df(self):
        """Return the preprocessed data as a DataFrame if the CSV exists."""
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    @property
    def wt(self):
        """Return a wild-type reference Data object with fitness 1.0 and zero GI."""
        wt = {}
        wt["genotype"] = ({"id": None, "intervention": None, "id_full": None},)
        wt["phenotype"] = {
            "observation": {"fitness": 1.0, "genetic_interaction_score": 0},
            "environment": {"media": "YEPD", "temperature": 30},
        }
        data = Data()
        data.genotype = wt["genotype"]
        data.phenotype = wt["phenotype"]
        return data

    def process(self):
        """Read all raw SGA files, preprocess, and write records to LMDB."""
        os.makedirs(self.preprocess_dir, exist_ok=True)
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
        environment = {"media": "YEPD", "temperature": 30}

        # Combine everything
        combined_data = [
            {
                "genotype": genotype,
                "phenotype": {"observation": observation, "environment": environment},
            }
            for genotype, observation in zip(combined_genotypes, observations)
        ]

        data_list = []
        for idx, item in tqdm(enumerate(combined_data)):
            data = Data()
            data.genotype = item["genotype"]
            data.phenotype = item["phenotype"]
            data_list.append(data)  # fill the data_list

        # Initialize LMDB environment
        log.info("lmdb begin")
        # TODO make map_size size of disk partition, only virtual address space.
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

        # Open a new write transaction
        with env.begin(write=True) as txn:
            # Iterate through each data item
            # TODO loop data_list instead...
            for idx, item in tqdm(enumerate(data_list)):
                serialized_data = pickle.dumps(item)
                txn.put(f"{idx}".encode(), serialized_data)

        # cache gene property
        self.gene_set = self.compute_gene_set(data_list)

    def preprocess_raw(self, all_data_df: pd.DataFrame, preprocess: dict | None = None):
        """Clean the raw DataFrame and resolve duplicate genotypes by strategy."""

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
        if preprocess.get("duplicate_resolution") == "low_dmf_std":
            # Keep the row with the lowest 'Double mutant fitness standard deviation'
            idx_to_keep = duplicates_df.groupby("genotype")[
                "Double mutant fitness standard deviation"
            ].idxmin()
        elif preprocess.get("duplicate_resolution") == "high_dmf":
            # Keep the row with the highest 'Double mutant fitness'
            idx_to_keep = duplicates_df.groupby("genotype")[
                "Double mutant fitness"
            ].idxmax()
        elif preprocess.get("duplicate_resolution") == "low_dmf":
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
        """Write the preprocess configuration dict to JSON."""
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(preprocess, f)

    # TODO implement key merge
    # criterion for merge is defined as key, value is the data object itself.

    # New method to load existing preprocess configuration
    def load_preprocess_config(self):
        """Load the saved preprocess configuration, or None if absent."""
        config_path = osp.join(self.preprocess_dir, "preprocess_config.json")

        if osp.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config
        else:
            return None

    def len(self) -> int:
        """Return the number of records stored in the LMDB."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

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
        """Return the set of all gene ids appearing in the data list genotypes."""
        computed_gene_set = GeneSet()
        for data in data_list:
            for genotype in data.genotype:
                computed_gene_set.add(genotype["id"])
        return computed_gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        """Return the gene set, loading it from JSON if available."""
        try:
            if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
                with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                    self._gene_set = GeneSet(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return self._gene_set
        # CHECK can probably remove this
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
        """Return a string with the class name and record count."""
        return f"{self.__class__.__name__}({len(self)})"


# @prof_input
def main():
    """Run a small example building and inspecting the double-mutant dataset."""
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    dmf_dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
        preprocess={"duplicate_resolution": "low_dmf_std"},
        # subset_n=100,
    )
    print(dmf_dataset[0])
    print(len(dmf_dataset.gene_set))

    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf"),
        preprocess={"duplicate_resolution": "low_std_both"},
        skip_process_file_exist_check=True,
        # subset_n=100,
    )

    print(smf_dataset)
    print(smf_dataset[0])
    print(len(smf_dataset.gene_set))


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
