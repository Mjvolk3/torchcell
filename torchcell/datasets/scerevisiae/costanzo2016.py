# torchcell/datasets/scerevisiae_/costanzo2016
# [[torchcell.datasets.scerevisiae.costanzo2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/costanzo2016
# Test file: tests/torchcell/datasets/scerevisiae/test_costanzo2016.py

import json
import logging
import os
import os.path as osp
import pickle
import shutil
import zipfile
from collections.abc import Callable
import numpy as np
import torch
import lmdb
import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm
from torchcell.dataset import Dataset, compute_experiment_reference_index
from torchcell.data import ExperimentReferenceIndex
from torchcell.datamodels import (
    BaseEnvironment,
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaDampPerturbation,
    SgaSuppressorAllelePerturbation,
    SgaTsAllelePerturbation,
    Temperature,
)
from torchcell.sequence import GeneSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
        self._experiment_reference_index = None
        super().__init__(root, transform, pre_transform)

    @property
    def skip_process_file_exist(self):
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> str:
        return "strain_ids_and_single_mutant_fitness.xlsx"

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

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
            osp.join(self.processed_dir, "lmdb"),
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
        (reference_phenotype_std_26, reference_phenotype_std_30) = (
            self.compute_reference_phenotype_std(df)
        )

        # Save preprocssed df - mainly for quick stats
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing SMF Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
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
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()
        self.gene_set = self.compute_gene_set()
        # This will cache the experiment_reference_index
        # When I comment out this line I don't get 'Error: the cannot pickle 'Environment' object'
        self.experiment_reference_index

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        df["Strain_ID_suffix"] = df["Strain ID"].str.split("_", expand=True)[1]

        # Determine perturbation type based on Strain_ID_suffix
        df["perturbation_type"] = df["Strain_ID_suffix"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x  # or "S" in x or "A_S" in x
                            else "suppression_allele" if "S" in x else "unknown"
                        )
                    )
                )
            )
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
            genotype = Genotype(
                perturbations=[
                    SgaTsAllelePerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "damp" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaDampPerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "KanMX_deletion" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaKanMxDeletionPerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "NatMX_deletion" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaNatMxDeletionPerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )
        elif "suppression_allele" in row["perturbation_type"]:
            genotype = Genotype(
                perturbations=[
                    SgaSuppressorAllelePerturbation(
                        systematic_gene_name=row["Systematic gene name"],
                        perturbed_gene_name=row["Allele/Gene name"],
                        strain_id=row["Strain ID"],
                    )
                ]
            )

        environment = BaseEnvironment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(value=row["Temperature"]),
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        smf_key = "Single mutant fitness"
        smf_std_key = "Single mutant fitness stddev"
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

        # Handling boolean index arrays or numpy arrays
        if isinstance(idx, (list, np.ndarray)):
            if isinstance(idx, list):
                idx = np.array(idx)
            if idx.dtype == np.bool_:
                idx = np.where(idx)[0]

            # If idx is a list/array of indices, return a list of data objects
            return [self.get_single_item(i) for i in idx]
        else:
            # Single item retrieval
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data
            # CHECK Doesn't work bc cannot pickle in data loader
            # experiment = self.experiment_class(**(deserialized_data["experiment"]))
            # reference = self.reference_class(**(deserialized_data["reference"]))
            # return {"experiment": experiment, "reference": reference}
            # CHECK

    @staticmethod
    def extract_systematic_gene_names(genotype):
        gene_names = []
        for perturbation in genotype.get("perturbations"):
            gene_name = perturbation.get("systematic_gene_name")
            gene_names.append(gene_name)
        return gene_names

    def compute_gene_set(self):
        gene_set = GeneSet()
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            log.info("Computing gene set...")
            for key, value in tqdm(cursor):
                deserialized_data = pickle.loads(value)
                experiment = deserialized_data["experiment"]

                extracted_gene_names = self.extract_systematic_gene_names(
                    experiment["genotype"]
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

    @property
    def experiment_reference_index(self):
        index_file_path = osp.join(
            self.preprocess_dir, "experiment_reference_index.json"
        )

        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                data = json.load(file)
                # Assuming ReferenceIndex can be constructed from a list of dictionaries
                self._experiment_reference_index = [
                    ExperimentReferenceIndex(**item) for item in data
                ]
        elif self._experiment_reference_index is None:
            self._experiment_reference_index = compute_experiment_reference_index(self)
            with open(index_file_path, "w") as file:
                # Convert each ExperimentReferenceIndex object to dict and save the list of dicts
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index],
                    file,
                    indent=4,
                )

        self.close_lmdb()
        return self._experiment_reference_index

    @property
    def experiment_class(self):
        return FitnessExperiment

    @property
    def reference_class(self):
        return FitnessExperimentReference

    def transform_item(self, item):
        experiment_data = item["experiment"]
        reference_data = item["reference"]
        experiment = self.experiment_class(**experiment_data)
        reference = self.reference_class(**reference_data)
        return {"experiment": experiment, "reference": reference}

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
        self._experiment_reference_index = None
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
        return "lmdb"

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
            osp.join(self.processed_dir, "lmdb"),
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
        log.info("Reading and Concatenating Raw Files...")
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

        log.info("Processing DMF Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
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
        # This will cache the experiment_reference_index
        self.experiment_reference_index

    @staticmethod
    def create_experiment(row, reference_phenotype_std_26, reference_phenotype_std_30):
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )
        # genotype
        perturbations = []
        # Query
        if "temperature_sensitive" in row["query_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "damp" in row["query_perturbation_type"]:
            perturbations.append(
                SgaDampPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["query_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )

        elif "NatMX_deletion" in row["query_perturbation_type"]:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )
        elif "suppression_allele" in row["query_perturbation_type"]:
            perturbations.append(
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name=row["Query Systematic Name"],
                    perturbed_gene_name=row["Query allele name"],
                    strain_id=row["Query Strain ID"],
                )
            )

        # Array
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "damp" in row["array_perturbation_type"]:
            perturbations.append(
                SgaDampPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )

        elif "NatMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaNatMxDeletionPerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )

        elif "suppression_allele" in row["array_perturbation_type"]:
            perturbations.append(
                SgaSuppressorAllelePerturbation(
                    systematic_gene_name=row["Array Systematic Name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array Strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
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
        log.info("Preprocess on raw data...")

        # Function to extract gene name
        def extract_systematic_name(x):
            return x.apply(lambda y: y.split("_")[0])

        # Extract gene names
        df["Query Systematic Name"] = extract_systematic_name(df["Query Strain ID"])
        df["Array Systematic Name"] = extract_systematic_name(df["Array Strain ID"])
        Temperature = df["Arraytype/Temp"].str.extract("(\d+)").astype(int)
        df["Temperature"] = Temperature
        df["query_perturbation_type"] = df["Query Strain ID"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x  # or "S" in x or "A_S" in x
                            else "suppression_allele" if "S" in x else "unknown"
                        )
                    )
                )
            )
        )
        df["array_perturbation_type"] = df["Array Strain ID"].apply(
            lambda x: (
                "damp"
                if "damp" in x
                else (
                    "temperature_sensitive"
                    if "tsa" in x or "tsq" in x
                    else (
                        "KanMX_deletion"
                        if "dma" in x
                        else (
                            "NatMX_deletion"
                            if "sn" in x  # or "S" in x or "A_S" in x
                            else "suppression_allele" if "S" in x else "unknown"
                        )
                    )
                )
            )
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

        # Handling boolean index arrays or numpy arrays
        if isinstance(idx, (list, np.ndarray)):
            if isinstance(idx, list):
                idx = np.array(idx)
            if idx.dtype == np.bool_:
                idx = np.where(idx)[0]

            # If idx is a list/array of indices, return a list of data objects
            return [self.get_single_item(i) for i in idx]
        else:
            # Single item retrieval
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data

    @staticmethod
    def extract_systematic_gene_names(genotype):
        gene_names = []
        for perturbation in genotype.perturbations:
            if hasattr(perturbation, "systematic_gene_name"):
                gene_name = perturbation.systematic_gene_name
                gene_names.append(gene_name)
        return gene_names

    def compute_gene_set(self):
        gene_set = GeneSet()
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            log.info("Computing gene set...")
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

    @property
    def experiment_reference_index(self):
        index_file_path = osp.join(
            self.preprocess_dir, "experiment_reference_index.json"
        )

        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                data = json.load(file)
                # Assuming ReferenceIndex can be constructed from a list of dictionaries
                self._experiment_reference_index = [
                    ExperimentReferenceIndex(**item) for item in data
                ]
        elif self._experiment_reference_index is None:
            self._experiment_reference_index = compute_experiment_reference_index(self)
            with open(index_file_path, "w") as file:
                # Convert each ExperimentReferenceIndex object to dict and save the list of dicts
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index],
                    file,
                    indent=4,
                )

        self.close_lmdb()
        return self._experiment_reference_index

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


import lmdb
import os.path as osp
import pickle
from multiprocessing import Process, Queue
from torch.utils.data import Dataset
import time
from typing import Generator
import queue  # This is correct
from typing import Iterable


class CustomDataLoader:
    def __init__(self, dataset, batch_size: int, num_workers: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.load_queue = Queue()
        self.data_queue = Queue(maxsize=num_workers)
        self.workers = []

        # Calculate how many batches are needed
        self.total_batches = (len(dataset) + batch_size - 1) // batch_size

        for _ in range(num_workers):
            worker = Process(
                target=self.worker_function,
                args=(self.load_queue, self.data_queue, self.dataset, self.batch_size),
            )
            worker.start()
            self.workers.append(worker)

        # Initially, load the first few batches depending on the number of workers
        for i in range(min(self.total_batches, num_workers)):
            self.load_queue.put(i)  # Signal with unique batch index

    @staticmethod
    def worker_function(load_queue: Queue, data_queue: Queue, dataset, batch_size: int):
        while True:
            batch_index = load_queue.get()  # Get unique batch index
            if batch_index is None:  # If None signal received, terminate the worker
                break

            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = [dataset[i] for i in range(start_idx, end_idx)]
            data_queue.put(batch)

    def __iter__(self) -> Iterable:
        self.batch_index = 0  # Reset batch index for new iteration
        return self

    def __next__(self):
        if self.batch_index >= self.total_batches:
            raise StopIteration

        batch = self.data_queue.get()

        # Prepare next batch if there are more batches to process
        if self.batch_index + len(self.workers) < self.total_batches:
            self.load_queue.put(self.batch_index + len(self.workers))

        self.batch_index += 1
        return batch

    def close(self):
        # Send termination signal to each worker
        for _ in range(len(self.workers)):
            self.load_queue.put(None)
        for worker in self.workers:
            worker.join()  # Wait for all workers to finish

# Usage example remains the same


if __name__ == "__main__":
    # dataset = DmfCostanzo2016Dataset(
    #     root="data/torchcell/dmf_costanzo2016_subset_n_1000",
    #     subset_n=1000,
    #     preprocess=None,
    # )
    # dataset.experiment_reference_index
    # dataset[0]
    # serialized_data = dataset[0]["experiment"].model_dump()
    # new_instance = FitnessExperiment.model_validate(serialized_data)
    # print(new_instance == dataset[0]['experiment'])
    # Usage example
    # print(len(dataset))
    # print(json.dumps(dataset[0].model_dump(), indent=4))
    # print(dataset.reference_index)
    # # print(len(dataset.reference_index))
    # # print(dataset.reference_index[0])
    # serialized_data = dataset[0]["experiment"].model_dump()
    # print(dataset[0]["experiment"])
    # print(FitnessExperiment(**serialized_data))
    ######
    # Single mutant fitness
    dataset = SmfCostanzo2016Dataset()
    print(len(dataset))
    # print(dataset[100])
    # serialized_data = dataset[100]["experiment"].model_dump()
    # new_instance = FitnessExperiment.model_validate(serialized_data)
    # print(new_instance == serialized_data)
    from torch_geometric.loader import DataLoader

    data_loader = CustomDataLoader(dataset, batch_size=10, num_workers=2)

    # Fetch and print the first 3 batches
    for i, batch in enumerate(data_loader):
        # batch_transformed = list(map(dataset.transform_item, batch))
        print(batch[0])
        print("---")
        if i == 3:
            break
    # Clean up worker processes
    data_loader.close()
    print("completed")
