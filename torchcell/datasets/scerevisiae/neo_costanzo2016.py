# torchcell/datasets/scerevisiae/neo_costanzo2016.py
# [[torchcell.datasets.scerevisiae.neo_costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/neo_costanzo2016.py
# Test file: tests/torchcell/datasets/scerevisiae/test_neo_costanzo2016.py
import functools
import json
import logging
import os
import os.path as osp
import pickle
import random
import re
import shutil
import zipfile
from abc import ABC, abstractproperty
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Union
import multiprocessing as mp

# import lmdb
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

# import polars as pl
import torch
from attrs import define, field

# from polars import DataFrame, col
from pydantic import Field, field_validator
from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm
import json
import logging
import os
import os.path as osp
import pickle
import random
import re
import shutil
import zipfile
from abc import ABC, abstractproperty
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import h5py
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

# from polars import DataFrame, col
from torch_geometric.data import (
    Data,
    DataLoader,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datamodels import ModelStrict
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet

from torchcell.data import Dataset
from torchcell.datamodels import (
    BaseEnvironment,
    BaseGenotype,
    BasePhenotype,
    BaseExperiment,
    GenePerturbation,
    Media,
    ModelStrict,
    ReferenceGenome,
    Temperature,
    DeletionGenotype,
    DeletionPerturbation,
    FitnessPhenotype,
    FitnessExperimentReference,
    ExperimentReference,
    FitnessExperiment,
    ExperimentReference,
    DampPerturbation,
    TsAllelePerturbation,
    InterferenceGenotype,
    KanMxDeletionPerturbation,
    NatMxDeletionPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgdTsAllelePerturbation,
    SgdDampPerturbation,
    SuppressorAllelePerturbation,
    SgdSuppressorAllelePerturbation,
    SuppressorGenotype,
)
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ExperimentReferenceIndex(ModelStrict):
    reference: ExperimentReference
    index: List[bool]

    def __repr__(self):
        if len(self.index) > 5:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index[:5]}...)"
        else:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index})"


class ReferenceIndex(ModelStrict):
    data: List[ExperimentReferenceIndex]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @field_validator("data")
    def validate_data(cls, v):
        summed_indices = sum(
            [
                boolean_value
                for exp_ref_index in v
                for boolean_value in exp_ref_index.index
            ]
        )

        if summed_indices != len(v[0].index):
            raise ValueError("Sum of indices must equal the number of experiments")
        return v


# These are the alleles that show in both query and array
# They have very different fitness values depending on query or array
# Since they cannot be swapped, order matters, and so we remove them
TS_ALLELE_PROBLEMATIC = {
    "srv2-ts",
    "apc2-8",
    "frq1-1",
    "act1-3",
    "sgv1-23",
    "dam1-9",
    "dad1-5005",
    "cdc11-2",
    "msl5-5001",
    "sup35-td",
    "emg1-1",
    "cdc20-1",
    "gus1-5001",
    "nse4-ts2",
    "rpg1-1",
    "mvd1-1296",
    "qri1-5001",
    "prp18-ts",
    "tfc8-5001",
    "taf12-9",
    "rpt2-rf",
    "ipl1-1",
    "duo1-2",
    "med6-ts",
    "rna14-5001",
    "cab5-1",
    "prp4-1",
    "nus1-5001",
    "yju2-5001",
    "tbf1-5001",
    "sec12-4",
    "cet1-15",
    "cdc47-ts",
    "ame1-4",
    "rnt1-ts",
    "sld3-5001",
    "lcb2-16",
    "ret2-1",
    "phs1-1",
    "cdc60-ts",
    "sec39-5001",
    "emg1-5001",
    "sec39-1",
}


class SmfCostanzo2016Dataset(Dataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/torchcell/smf_costanzo2016",
        subset_n: int = None,
        preprocess: dict | None = None,
        skip_process_file_exist: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
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
        # TODO remove preprocess config
        existing_config = self.load_preprocess_config()
        if existing_config is not None:
            if existing_config != self.preprocess:
                raise ValueError(
                    "New preprocess does not match existing config."
                    "Delete the processed and process dir for a new Dataset."
                    "Or define a new root."
                )
        self.env = None
        super().__init__(root, transform, pre_transform)

    @property
    def skip_process_file_exist(self):
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> str:
        return "strain_ids_and_single_mutant_fitness.xlsx"

    @property
    def processed_file_names(self) -> list[str]:
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
        # remove any excess files not needed
        for file_name in os.listdir(self.raw_dir):
            # if the file name ends in .txt remove it
            if file_name.endswith(".txt"):
                os.remove(osp.join(self.raw_dir, file_name))

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
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def df(self):
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    def process(self):
        xlsx_path = osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx")
        df = pd.read_excel(xlsx_path)
        df = self.preprocess_raw(df, self.preprocess)
        (
            reference_phenotype_std_26,
            reference_phenotype_std_30,
        ) = self.compute_reference_phenotype_std(df)

        # Save preprocssed df - mainly for quick stats
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        print("Processing SMF Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference = self.create_experiment(
                    row,
                    reference_phenotype_std_26=reference_phenotype_std_26,
                    reference_phenotype_std_30=reference_phenotype_std_30,
                )

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {"experiment": experiment, "reference": reference}
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()
        self.gene_set = self.compute_gene_set()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        df["Strain_ID_suffix"] = df["Strain ID"].str.split("_", expand=True)[1]

        # Determine perturbation type based on Strain_ID_suffix
        df["perturbation_type"] = df["Strain_ID_suffix"].apply(
            lambda x: "damp"
            if "damp" in x
            else "temperature_sensitive"
            if "tsa" in x or "tsq" in x
            else "KanMX_deletion"
            if "dma" in x
            else "NatMX_deletion"
            if "sn" in x  # or "S" in x or "A_S" in x
            else "suppression_allele"
            if "S" in x
            else "unknown"
        )

        # Create separate dataframes for the two temperatures
        df_26 = df[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (26°)",
                "Single mutant fitness (26°) stddev",
                "perturbation_type",
            ]
        ].copy()
        df_26["Temperature"] = 26

        df_30 = df[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (30°)",
                "Single mutant fitness (30°) stddev",
                "perturbation_type",
            ]
        ].copy()
        df_30["Temperature"] = 30

        # Rename the columns for fitness and stddev to be common for both dataframes
        df_26.rename(
            columns={
                "Single mutant fitness (26°)": "Single mutant fitness",
                "Single mutant fitness (26°) stddev": "Single mutant fitness stddev",
            },
            inplace=True,
        )

        df_30.rename(
            columns={
                "Single mutant fitness (30°)": "Single mutant fitness",
                "Single mutant fitness (30°) stddev": "Single mutant fitness stddev",
            },
            inplace=True,
        )

        # Concatenate the two dataframes
        combined_df = pd.concat([df_26, df_30], ignore_index=True)
        combined_df = combined_df.dropna()
        combined_df = combined_df.drop_duplicates()
        combined_df = combined_df.reset_index(drop=True)

        return combined_df

    @staticmethod
    def compute_reference_phenotype_std(df: pd.DataFrame):
        mean_stds = df.groupby("Temperature")["Single mutant fitness stddev"].mean()
        reference_phenotype_std_26 = mean_stds[26]
        reference_phenotype_std_30 = mean_stds[30]
        return reference_phenotype_std_26, reference_phenotype_std_30

    @staticmethod
    def create_experiment(row, reference_phenotype_std_26, reference_phenotype_std_30):
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )

        # Deal with different types of perturbations
        if "temperature_sensitive" in row["perturbation_type"]:
            genotype = InterferenceGenotype(
                perturbation=SgdTsAllelePerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                    strain_id=row["Strain ID"],
                )
            )
        elif "damp" in row["perturbation_type"]:
            genotype = InterferenceGenotype(
                perturbation=SgdDampPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                    strain_id=row["Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["perturbation_type"]:
            genotype = DeletionGenotype(
                perturbation=SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                    strain_id=row["Strain ID"],
                )
            )
        elif "NatMX_deletion" in row["perturbation_type"]:
            genotype = DeletionGenotype(
                perturbation=SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                    strain_id=row["Strain ID"],
                )
            )
        elif "suppression_allele" in row["perturbation_type"]:
            genotype = SuppressorGenotype(
                perturbation=SgdSuppressorAllelePerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                    strain_id=row["Strain ID"],
                )
            )

        environment = BaseEnvironment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(value=row["Temperature"]),
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        smf_key = "Single mutant fitness"
        smf_std_key = "Single mutant fitness"
        phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=row[smf_key],
            fitness_std=row[smf_std_key],
        )

        if row["Temperature"] == 26:
            reference_phenotype_std = reference_phenotype_std_26
        elif row["Temperature"] == 30:
            reference_phenotype_std = reference_phenotype_std_30
        reference_phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=1.0,
            fitness_std=reference_phenotype_std,
        )

        reference = FitnessExperimentReference(
            reference_genome=reference_genome,
            reference_environment=reference_environment,
            reference_phenotype=reference_phenotype,
        )

        experiment = FitnessExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )
        return experiment, reference

    # New method to save preprocess configuration to a JSON file
    def save_preprocess_config(self, preprocess):
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(preprocess, f)

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

    def get(self, idx):
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            # Deserialize the data and return it directly
            deserialized_data = pickle.loads(serialized_data)

            return deserialized_data

    @staticmethod
    def extract_systematic_gene_name(genotype):
        gene_name = genotype.perturbation.systematic_gene_name
        return gene_name

    def compute_gene_set(self):
        gene_set = GeneSet()
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            print("Computing gene set...")
            for key, value in tqdm(cursor):
                deserialized_data = pickle.loads(value)
                experiment = deserialized_data["experiment"]

                extracted_gene_name = self.extract_systematic_gene_name(
                    experiment.genotype
                )
                gene_set.add(extracted_gene_name)

        self.close_lmdb()
        return gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
            with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        elif self._gene_set is None:
            raise ValueError(
                "gene_set not written during process. "
                "Please call compute_gene_set in process."
            )
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


class DmfCostanzo2016Dataset(Dataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/torchcell/dmf_costanzo2016",
        subset_n: int = None,
        preprocess: dict | None = None,
        skip_process_file_exist: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
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
        self.env = None
        super().__init__(root, transform, pre_transform)
        # This was here before - not sure if it has something to do with gpu
        # self.env = None

    @property
    def skip_process_file_exist(self):
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> list[str]:
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
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
        # remove any excess files not needed
        os.remove(osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx"))

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
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def df(self):
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    def process(self):
        os.makedirs(self.preprocess_dir, exist_ok=True)
        self._length = None
        # Initialize an empty DataFrame to hold all raw data
        df = pd.DataFrame()

        # Read and concatenate all raw files
        print("Reading and Concatenating Raw Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)

            # Reading data using Pandas; limit rows for demonstration
            df_temp = pd.read_csv(file_path, sep="\t")

            # Concatenating data frames
            df = pd.concat([df, df_temp], ignore_index=True)
        # Functions for data filtering... duplicates selection,
        df = self.preprocess_raw(df, self.preprocess)
        self.save_preprocess_config(self.preprocess)

        # Subset
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)

        # Save preprocssed df - mainly for quick stats
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        print("Processing DMF Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference = self.create_experiment(
                    row,
                    reference_phenotype_std_26=self.reference_phenotype_std_26,
                    reference_phenotype_std_30=self.reference_phenotype_std_30,
                )

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {"experiment": experiment, "reference": reference}
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()
        self.gene_set = self.compute_gene_set()

    @staticmethod
    def create_experiment(row, reference_phenotype_std_26, reference_phenotype_std_30):
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )
        # genotype
        genotype = []
        # Query
        if "temperature_sensitive" in row["query_perturbation_type"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=SgdTsAllelePerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                        strain_id=row["Query Strain ID"],
                    )
                )
            )
        elif "damp" in row["query_perturbation_type"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=SgdDampPerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                        strain_id=row["Query Strain ID"],
                    )
                )
            )
        elif "KanMX_deletion" in row["query_perturbation_type"]:
            genotype.append(
                DeletionGenotype(
                    perturbation=SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                        strain_id=row["Query Strain ID"],
                    )
                )
            )

        elif "NatMX_deletion" in row["query_perturbation_type"]:
            genotype.append(
                DeletionGenotype(
                    perturbation=SgaNatMxDeletionPerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                        strain_id=row["Query Strain ID"],
                    )
                )
            )

        elif "suppression_allele" in row["query_perturbation_type"]:
            genotype.append(
                SuppressorGenotype(
                    perturbation=SgdSuppressorAllelePerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                        strain_id=row["Query Strain ID"],
                    )
                )
            )

        # Array
        if "temperature_sensitive" in row["array_perturbation_type"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=SgdTsAllelePerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array Strain ID"],
                    )
                )
            )
        elif "damp" in row["array_perturbation_type"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=SgdDampPerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array Strain ID"],
                    )
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            genotype.append(
                DeletionGenotype(
                    perturbation=SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array Strain ID"],
                    )
                )
            )

        elif "NatMX_deletion" in row["array_perturbation_type"]:
            genotype.append(
                DeletionGenotype(
                    perturbation=SgaNatMxDeletionPerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array Strain ID"],
                    )
                )
            )

        elif "suppression_allele" in row["array_perturbation_type"]:
            genotype.append(
                SuppressorGenotype(
                    perturbation=SgdSuppressorAllelePerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array Strain ID"],
                    )
                )
            )

        # genotype
        environment = BaseEnvironment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(value=row["Temperature"]),
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        dmf_key = "Double mutant fitness"
        dmf_std_key = "Double mutant fitness standard deviation"
        phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=row[dmf_key],
            fitness_std=row[dmf_std_key],
        )

        if row["Temperature"] == 26:
            reference_phenotype_std = reference_phenotype_std_26
        elif row["Temperature"] == 30:
            reference_phenotype_std = reference_phenotype_std_30
        reference_phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=1.0,
            fitness_std=reference_phenotype_std,
        )

        reference = FitnessExperimentReference(
            reference_genome=reference_genome,
            reference_environment=reference_environment,
            reference_phenotype=reference_phenotype,
        )

        experiment = FitnessExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )
        return experiment, reference

    def preprocess_raw(self, df: pd.DataFrame, preprocess: dict | None = None):
        print("Preprocess on raw data...")

        # Function to extract gene name
        def extract_systematic_name(x):
            return x.apply(lambda y: y.split("_")[0])

        # Extract gene names
        df["Query Systematic Name"] = extract_systematic_name(df["Query Strain ID"])
        df["Array Systematic Name"] = extract_systematic_name(df["Array Strain ID"])
        Temperature = df["Arraytype/Temp"].str.extract("(\d+)").astype(int)
        df["Temperature"] = Temperature
        df["query_perturbation_type"] = df["Query Strain ID"].apply(
            lambda x: "damp"
            if "damp" in x
            else "temperature_sensitive"
            if "tsa" in x or "tsq" in x
            else "KanMX_deletion"
            if "dma" in x
            else "NatMX_deletion"
            if "sn" in x  # or "S" in x or "A_S" in x
            else "suppression_allele"
            if "S" in x
            else "unknown"
        )
        df["array_perturbation_type"] = df["Array Strain ID"].apply(
            lambda x: "damp"
            if "damp" in x
            else "temperature_sensitive"
            if "tsa" in x or "tsq" in x
            else "KanMX_deletion"
            if "dma" in x
            else "NatMX_deletion"
            if "sn" in x  # or "S" in x or "A_S" in x
            else "suppression_allele"
            if "S" in x
            else "unknown"
        )
        means = df.groupby("Temperature")[
            "Double mutant fitness standard deviation"
        ].mean()
        # TODO remove TS_ALLELE_PROBLEMATIC

        # Extracting means for specific temperatures
        self.reference_phenotype_std_26 = means.get(26, None)
        self.reference_phenotype_std_30 = means.get(30, None)

        # Assuming df is your DataFrame
        def create_combined_systematic_name(row):
            names = sorted([row["Query Systematic Name"], row["Array Systematic Name"]])
            return "_".join(names)

        def create_combined_allele_name(row):
            names = sorted([row["Query allele name"], row["Array allele name"]])
            return "_".join(names)

        # TODO delete if not needed
        # df["combined_systematic_name"] = df.apply(
        #     create_combined_systematic_name, axis=1
        # )

        # df["combined_allele_name"] = df.apply(create_combined_allele_name, axis=1)

        return df

    # New method to save preprocess configuration to a JSON file
    def save_preprocess_config(self, preprocess):
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(preprocess, f)

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

    def get(self, idx):
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            # Deserialize the data and return it directly
            deserialized_data = pickle.loads(serialized_data)

            return deserialized_data

    @staticmethod
    def extract_systematic_gene_names(genotypes):
        gene_names = []
        for genotype in genotypes:
            if hasattr(genotype, "perturbation") and hasattr(
                genotype.perturbation, "systematic_gene_name"
            ):
                gene_name = genotype.perturbation.systematic_gene_name
                gene_names.append(gene_name)
        return gene_names

    def compute_gene_set(self):
        gene_set = GeneSet()
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            print("Computing gene set...")
            for key, value in tqdm(cursor):
                deserialized_data = pickle.loads(value)
                experiment = deserialized_data["experiment"]

                extracted_gene_names = self.extract_systematic_gene_names(
                    experiment.genotype
                )
                for gene_name in extracted_gene_names:
                    gene_set.add(gene_name)

        self.close_lmdb()
        return gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
            with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        elif self._gene_set is None:
            raise ValueError(
                "gene_set not written during process. "
                "Please call compute_gene_set in process."
            )
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


if __name__ == "__main__":
    dataset = DmfCostanzo2016Dataset(subset_n=None, preprocess=None)
    dataset[0]
    print(len(dataset))
    # print(json.dumps(dataset[0].model_dump(), indent=4))
    # print(dataset.reference_index)
    # # print(len(dataset.reference_index))
    # # print(dataset.reference_index[0])
    # serialized_data = dataset[0]["experiment"].model_dump()
    # print(dataset[0]["experiment"])
    # print(FitnessExperiment(**serialized_data))
    ######
    # Single mutant fitness
    # dataset = SmfCostanzo2016Dataset()
    # print(dataset)
    # print(len(dataset))
    # print(dataset[100]['experiment'])
    # serialized_data = dataset[100]['experiment'].model_dump()
    # print(FitnessExperiment.model_validate(serialized_data))
    print("done")
