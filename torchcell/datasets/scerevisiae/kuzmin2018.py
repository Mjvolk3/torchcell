# torchcell/datasets/scerevisiae/kuzmin2018.py
# [[torchcell.datasets.scerevisiae.kuzmin2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2018.py
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2018.py

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
import requests

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

# import polars as pl
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
    BaseGenotype,
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
    AllelePerturbation,
    SgdAllelePerturbation,
)
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DmfKuzmin2018Dataset(Dataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmf_kuzmin2018",
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

    @property
    def skip_process_file_exist(self):
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> str:
        return "aao1729_data_s1.tsv"

    @property
    def processed_file_names(self) -> list[str]:
        return "data.lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

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
        print("Processing...")

        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df, self.preprocess)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        print("Processing Dmf Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference = self.create_experiment(
                    row, reference_phenotype_std=self.reference_phenotype_std
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
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]
        # Select doubles only
        df = df[df["Combined mutant type"] == "digenic"].copy()

        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        df["query_perturbation_type"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            lambda x: "temperature_sensitive"
            if "tsa" in x
            else "KanMX_deletion"
            if "dma" in x
            else "unknown"
        )
        self.reference_phenotype_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()
        return df

    @staticmethod
    def create_experiment(row, reference_phenotype_std):
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )
        # genotype
        genotype = []
        # Query...
        if "KanMX_deletion" in row["query_perturbation_type"]:
            genotype.append(
                DeletionGenotype(
                    perturbation=SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Query systematic name no ho"],
                        perturbed_gene_name=row["Query allele name no ho"],
                        strain_id=row["Query strain ID"],
                    )
                )
            )
        elif "allele" in row["query_perturbation_type"]:
            genotype.append(
                BaseGenotype(
                    perturbation=SgdAllelePerturbation(
                        systematic_gene_name=row["Query systematic name no ho"],
                        perturbed_gene_name=row["Query allele name no ho"],
                        strain_id=row["Query strain ID"],
                    )
                )
            )

        # Array - only array has ts
        if "temperature_sensitive" in row["array_perturbation_type"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=SgdTsAllelePerturbation(
                        systematic_gene_name=row["Array systematic name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array strain ID"],
                    )
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            genotype.append(
                DeletionGenotype(
                    perturbation=SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Array systematic name"],
                        perturbed_gene_name=row["Array allele name"],
                        strain_id=row["Array strain ID"],
                    )
                )
            )

        # genotype
        environment = BaseEnvironment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        dmf_key = "Combined mutant fitness"
        dmf_std_key = "Combined mutant fitness standard deviation"
        phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=row[dmf_key],
            fitness_std=row[dmf_std_key],
        )

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
    dataset = DmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    print()
