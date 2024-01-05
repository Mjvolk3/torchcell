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
import msgpack

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


@define
class NeoSmfCostanzo2016Dataset:
    root: str = field(default="data/torchcell/smf_costanzo2016")
    url: str = field(
        repr=False,
        default="https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
    )
    raw_dir: str = field(init=False, repr=False)
    processed_dir: str = field(init=False, repr=False)
    data: list[BaseExperiment] = field(init=False, repr=False, factory=list)
    reference: list[FitnessExperimentReference] = field(
        init=False, repr=False, factory=list
    )
    reference_index: ReferenceIndex = field(init=False, repr=False)
    reference_phenotype_std_30 = field(init=False, repr=False)
    reference_phenotype_std_26 = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        if osp.exists(osp.join(self.processed_dir, "dataset.json")):
            self.load()
        else:
            self._download()
            self._extract()
            self._cleanup_after_extract()
            self.data, self.reference = self._process_excel()
            self.data, self.reference = self._remove_duplicates()
            self.reference_index = self.get_reference_index()
            self.save()

    # write a get item method to return a single experiment
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _download(self):
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            download_url(self.url, self.raw_dir)

    def save(self):
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        save_path = osp.join(self.processed_dir, "dataset.json")
        # Create a dictionary to store the serialized data
        serialized_data = {
            "data": [experiment.model_dump() for experiment in self.data],
            "reference": [ref.model_dump() for ref in self.reference],
            "reference_index": [
                ref_idx.model_dump() for ref_idx in self.reference_index.data
            ],
        }
        with open(save_path, "w") as file:
            json.dump(serialized_data, file, indent=4)

    def load(self):
        load_path = osp.join(self.processed_dir, "dataset.json")
        if not osp.exists(load_path):
            raise FileNotFoundError("Saved dataset not found.")

        with open(load_path, "r") as file:
            serialized_data = json.load(file)

        # Deserialize the data back into the appropriate classes
        self.data = [
            FitnessExperiment.model_validate(exp) for exp in serialized_data["data"]
        ]
        self.reference = [
            FitnessExperimentReference.model_validate(ref)
            for ref in serialized_data["reference"]
        ]
        self.reference_index = ReferenceIndex(
            data=[
                ExperimentReferenceIndex.model_validate(ref_idx)
                for ref_idx in serialized_data["reference_index"]
            ]
        )

    def _extract(self):
        zip_path = osp.join(
            self.raw_dir,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

    def _cleanup_after_extract(self):
        # We are only keeping the smf data for this dataset
        extracted_folder = osp.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        xlsx_file = osp.join(
            extracted_folder, "strain_ids_and_single_mutant_fitness.xlsx"
        )
        if osp.exists(xlsx_file):
            shutil.move(xlsx_file, self.raw_dir)
        if osp.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        zip_path = osp.join(
            self.raw_dir,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            os.remove(zip_path)

    def _process_excel(self):
        """
        Process the Excel file and convert each row to Experiment instances for 26°C and 30°C separately.
        """
        xlsx_path = osp.join(self.raw_dir, "strain_ids_and_single_mutant_fitness.xlsx")
        df = pd.read_excel(xlsx_path)
        # Process the DataFrame to average rows with 'tsa' or 'tsq'
        df = self._average_tsa_tsq(df)
        # This is an approximate since I cannot find the exact value in the paper
        df["Strain_ID_suffix"] = df["Strain ID"].str.split("_", expand=True)[1]

        # Filter out rows where 'Strain_ID_Part2' contains 'ts' or 'damp'
        filter_condition = ~df["Strain_ID_suffix"].str.contains("ts|damp", na=False)
        df_filtered = df[filter_condition]

        self.reference_phenotype_std_26 = (
            df_filtered["Single mutant fitness (26°) stddev"]
        ).mean()
        self.reference_phenotype_std_30 = (
            df_filtered["Single mutant fitness (30°) stddev"]
        ).mean()
        # Process data for 26°C and 30°C
        df_26 = df_filtered[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (26°)",
                "Single mutant fitness (26°) stddev",
            ]
        ].dropna()
        self._process_temperature_data(df_26, 26)

        df_30 = df_filtered[
            [
                "Strain ID",
                "Systematic gene name",
                "Allele/Gene name",
                "Single mutant fitness (30°)",
                "Single mutant fitness (30°) stddev",
            ]
        ].dropna()

        # This is modifying self.data, and self.reference
        self._process_temperature_data(df_30, 30)

        return self.data, self.reference

    def get_reference_index(self):
        # Serialize references for comparability using model_dump
        serialized_references = [
            json.dumps(ref.model_dump(), sort_keys=True) for ref in self.reference
        ]

        # Identify unique references and their indices
        unique_refs = {}
        for idx, ref_json in enumerate(serialized_references):
            if ref_json not in unique_refs:
                unique_refs[ref_json] = {
                    "indices": [],
                    "model": self.reference[idx],  # Store the Pydantic model
                }
            unique_refs[ref_json]["indices"].append(idx)

        # Create ExperimentReferenceIndex instances
        reference_indices = []
        for ref_info in unique_refs.values():
            bool_array = [i in ref_info["indices"] for i in range(len(self.data))]
            reference_indices.append(
                ExperimentReferenceIndex(reference=ref_info["model"], index=bool_array)
            )

        # Return ReferenceIndex instance
        return ReferenceIndex(data=reference_indices)

    def _average_tsa_tsq(self, df):
        """
        Replace 'tsa' and 'tsq' with 'ts' in the Strain ID and average duplicates.
        """
        # Replace 'tsa' and 'tsq' with 'ts' in Strain ID
        df["Strain ID"] = df["Strain ID"].str.replace("_ts[qa]\d*", "_ts", regex=True)

        # Columns to average
        columns_to_average = [
            "Single mutant fitness (26°)",
            "Single mutant fitness (26°) stddev",
            "Single mutant fitness (30°)",
            "Single mutant fitness (30°) stddev",
        ]

        # Averaging duplicates
        df_avg = (
            df.groupby(["Strain ID", "Systematic gene name", "Allele/Gene name"])[
                columns_to_average
            ]
            .mean()
            .reset_index()
        )

        # Merging averaged values back into the original DataFrame
        df_non_avg = df.drop(columns_to_average, axis=1).drop_duplicates(
            ["Strain ID", "Systematic gene name", "Allele/Gene name"]
        )
        df = pd.merge(
            df_non_avg,
            df_avg,
            on=["Strain ID", "Systematic gene name", "Allele/Gene name"],
        )

        return df

    def _process_temperature_data(self, df, temperature):
        """
        Process DataFrame for a specific temperature and add entries to the dataset.
        """
        for _, row in df.iterrows():
            experiment, ref = self.create_experiment(row, temperature)
            self.data.append(experiment)
            self.reference.append(ref)

    def create_experiment(self, row, temperature):
        """
        Create an Experiment instance from a row of the Excel spreadsheet for a given temperature.
        """
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )

        # Deal with different types of perturbations
        if "ts" in row["Strain ID"]:
            genotype = InterferenceGenotype(
                perturbation=DampPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                )
            )
        elif "damp" in row["Strain ID"]:
            genotype = InterferenceGenotype(
                perturbation=DampPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                )
            )
        else:
            genotype = DeletionGenotype(
                perturbation=DeletionPerturbation(
                    systematic_gene_name=row["Systematic gene name"],
                    perturbed_gene_name=row["Allele/Gene name"],
                )
            )
        environment = BaseEnvironment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(scalar=temperature),
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        smf_key = f"Single mutant fitness ({temperature}°)"
        smf_std_key = f"Single mutant fitness ({temperature}°) stddev"
        phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            fitness=row[smf_key],
            fitness_std=row[smf_std_key],
        )

        if temperature == 26:
            reference_phenotype_std = self.reference_phenotype_std_26
        elif temperature == 30:
            reference_phenotype_std = self.reference_phenotype_std_30
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

    def _remove_duplicates(self) -> list[BaseExperiment]:
        """
        Remove duplicate BaseExperiment instances from self.data.
        All fields of the object must match for it to be considered a duplicate.
        """
        unique_data = []
        seen = set()

        for experiment, reference in zip(self.data, self.reference):
            # Serialize the experiment object to a dictionary
            experiment_dict = experiment.model_dump()
            reference_dict = reference.model_dump()

            combined_dict = {**experiment_dict, **reference_dict}
            # Convert dictionary to a JSON string for comparability
            combined_json = json.dumps(combined_dict, sort_keys=True)

            if combined_json not in seen:
                seen.add(combined_json)
                unique_data.append((experiment, reference))

        self.data = [experiment for experiment, reference in unique_data]
        self.reference = [reference for experiment, reference in unique_data]

        return self.data, self.reference

    def df(self) -> pd.DataFrame:
        """
        Create a DataFrame from the list of BaseExperiment instances.
        Each instance is a row in the DataFrame.
        """
        rows = []
        for experiment in self.data:
            # Flatten the structure of each BaseExperiment instance
            row = {
                "species": experiment.experiment_reference_state.reference_genome.species,
                "strain": experiment.experiment_reference_state.reference_genome.strain,
                "media_name": experiment.environment.media.name,
                "media_state": experiment.environment.media.state,
                "temperature": experiment.environment.temperature.scalar,
                "genotype": experiment.genotype.perturbation.systematic_gene_name,
                "perturbed_gene_name": experiment.genotype.perturbation.perturbed_gene_name,
                "fitness": experiment.phenotype.fitness,
                "fitness_std": experiment.phenotype.fitness_std,
                # Add other fields as needed
            }
            rows.append(row)

        return pd.DataFrame(rows)


###########


@define
class NeoDmfCostanzo2016Dataset:
    root: str = field(default="data/torchcell/dmf_costanzo2016")
    url: str = field(
        repr=False,
        default="https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
    )
    raw_dir: str = field(init=False, repr=False)
    processed_dir: str = field(init=False, repr=False)
    data: list[BaseExperiment] = field(init=False, repr=False, factory=list)
    reference: list[FitnessExperimentReference] = field(
        init=False, repr=False, factory=list
    )
    reference_index: ReferenceIndex = field(init=False, repr=False)
    reference_phenotype_std_30 = field(init=False, repr=False)
    reference_phenotype_std_26 = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.raw_dir = osp.join(self.root, "raw")
        self.processed_dir = osp.join(self.root, "processed")
        if osp.exists(osp.join(self.processed_dir, "dataset.msgpack")):
            self.load()
        else:
            self._download()
            self._extract()
            self._cleanup_after_extract()
            ##
            self.data, self.reference = self._process_raw()
            self.data, self.reference = self._remove_duplicates()
            self.reference_index = self.get_reference_index()
            self.save()

    # write a get item method to return a single experiment
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _download(self):
        if not osp.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
            download_url(self.url, self.raw_dir)

    def save(self):
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        save_path = osp.join(self.processed_dir, "dataset.msgpack")
        # Create a dictionary to store the serialized data
        serialized_data = {
            "data": [experiment.model_dump() for experiment in self.data],
            "reference": [ref.model_dump() for ref in self.reference],
            "reference_index": [
                ref_idx.model_dump() for ref_idx in self.reference_index.data
            ],
        }
        with open(save_path, "wb") as file:
            file.write(msgpack.packb(serialized_data, use_bin_type=True))

    def load(self):
        logging.info("Loading Dataset")
        load_path = osp.join(self.processed_dir, "dataset.msgpack")
        if not osp.exists(load_path):
            raise FileNotFoundError("Saved dataset not found.")

        with open(load_path, "rb") as file:
            serialized_data = msgpack.unpackb(file.read(), raw=False)

        # Deserialize 'data' back into the appropriate classes
        logging.info("Deserializing data")
        self.data = []
        for exp in tqdm(serialized_data["data"], desc="Loading Data"):
            self.data.append(FitnessExperiment.model_validate(exp))

        # Deserialize 'reference' back into the appropriate classes
        logging.info("Deserializing reference")
        self.reference = []
        for ref in tqdm(serialized_data["reference"], desc="Loading Reference"):
            self.reference.append(FitnessExperimentReference.model_validate(ref))

        # Deserialize 'reference_index' back into the appropriate classes
        logging.info("Deserializing reference index")
        reference_index_data = []
        for ref_idx in tqdm(
            serialized_data["reference_index"], desc="Loading Reference Index"
        ):
            reference_index_data.append(
                ExperimentReferenceIndex.model_validate(ref_idx)
            )

        self.reference_index = ReferenceIndex(data=reference_index_data)

    def _extract(self):
        zip_path = osp.join(
            self.raw_dir,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

    def _cleanup_after_extract(self):
        # Define the folder where files are extracted
        extracted_folder = osp.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )

        # Check if the extracted folder exists
        if osp.exists(extracted_folder):
            for filename in os.listdir(extracted_folder):
                file_path = osp.join(extracted_folder, filename)
                # Check if the file is a TXT file
                if filename.endswith(".txt"):
                    # Move TXT file to the raw_dir
                    shutil.move(file_path, self.raw_dir)
                else:
                    # Remove non-TXT files
                    os.remove(file_path)

            # Remove the now-empty extracted folder
            os.rmdir(extracted_folder)

        # Remove the original ZIP file after extraction
        zip_path = osp.join(
            self.raw_dir,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            os.remove(zip_path)

    # def _process_raw(self):
    #     # for file in self.raw_dir load the txt file as a dataframe and process it
    #     df = self._combine_data_tables()
    #     df = self._process_data_table(df)
    #     log.info("Processing Experiments")
    #     for _, row in tqdm(df.iterrows()):
    #         experiment, ref = self.create_experiment(row)
    #         self.data.append(experiment)
    #         self.reference.append(ref)
    #     return self.data, self.reference

    def _process_raw(self):
        df = self._combine_data_tables()
        df = self._process_data_table(df)
        log.info("Processing Experiments")

        # Prepare the rows as a list of tuples
        rows = df.to_dict(orient="records")

        # Use multiprocessing Pool to process each row in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            partial_func = functools.partial(
                self.create_experiment,
                reference_phenotype_std_26=self.reference_phenotype_std_26,
                reference_phenotype_std_30=self.reference_phenotype_std_30,
            )

            # Using imap_unordered for potentially faster processing and progress monitoring
            results = list(tqdm(pool.imap_unordered(partial_func, rows), total=len(rows)))

        # Extend the data and reference lists with the results
        for experiment, ref in results:
            self.data.append(experiment)
            self.reference.append(ref)

        return self.data, self.reference

    def _combine_data_tables(self) -> pd.DataFrame:
        df = pd.DataFrame()
        log.info("Combining data tables")
        for file_name in os.listdir(self.raw_dir):
            df_temp = pd.read_csv(osp.join(self.raw_dir, file_name), sep="\t")
            df = pd.concat([df, df_temp])
        # return df.head(int(1e7))
        return df

    def _process_data_table(self, df) -> pd.DataFrame:
        log.info("Processing data table")
        df_processed = df.copy()
        query_systematic_name = df["Query Strain ID"].str.split("_", expand=True)[0]
        array_systematic_name = df["Array Strain ID"].str.split("_", expand=True)[0]
        Temperature = df["Arraytype/Temp"].str.extract("(\d+)").astype(int)
        df_processed["Query Systematic Name"] = query_systematic_name
        df_processed["Array Systematic Name"] = array_systematic_name
        df_processed["Temperature"] = Temperature
        df_processed["Query Strain ID"] = df["Query Strain ID"].str.replace(
            "_ts[qa]\d*", "_ts", regex=True
        )
        df_processed["Array Strain ID"] = df["Array Strain ID"].str.replace(
            "_ts[qa]\d*", "_ts", regex=True
        )
        self._compute_phenotype_std(df_processed)
        return df_processed

    def _compute_phenotype_std(self, df):
        mean_stds = (
            df[["Double mutant fitness standard deviation", "Temperature"]]
            .groupby("Temperature")
            .mean()
        )
        self.reference_phenotype_std_26 = mean_stds.loc[26][
            "Double mutant fitness standard deviation"
        ]
        self.reference_phenotype_std_30 = mean_stds.loc[30][
            "Double mutant fitness standard deviation"
        ]

    # def _process_genotype(self, row, ):
    #     # Deal with different types of perturbations
    #     genotype = []
    #     # Query
    #     if "ts" in row["Query Strain ID"]:
    #         genotype.append(
    #             InterferenceGenotype(
    #                 perturbation=DampPerturbation(
    #                     systematic_gene_name=row["Query Systematic Name"],
    #                     perturbed_gene_name=row["Query allele name"],
    #                 )
    #             )
    #         )
    #     elif "damp" in row["Query Strain ID"]:
    #         genotype.append(
    #             InterferenceGenotype(
    #                 perturbation=DampPerturbation(
    #                     systematic_gene_name=row["Query Systematic Name"],
    #                     perturbed_gene_name=row["Query allele name"],
    #                 )
    #             )
    #         )
    #     else:
    #         genotype.append(
    #             DeletionGenotype(
    #                 perturbation=DeletionPerturbation(
    #                     systematic_gene_name=row["Query Systematic Name"],
    #                     perturbed_gene_name=row["Query allele name"],
    #                 )
    #             )
    #             # Array
    #         )
    #     if "ts" in row["Array Strain ID"]:
    #         genotype.append(
    #             InterferenceGenotype(
    #                 perturbation=DampPerturbation(
    #                     systematic_gene_name=row["Array Systematic Name"],
    #                     perturbed_gene_name=row["Array allele name"],
    #                 )
    #             )
    #         )
    #     elif "damp" in row["Array Strain ID"]:
    #         genotype.append(
    #             InterferenceGenotype(
    #                 perturbation=DampPerturbation(
    #                     systematic_gene_name=row["Array Systematic Name"],
    #                     perturbed_gene_name=row["Array allele name"],
    #                 )
    #             )
    #         )
    #     else:
    #         genotype.append(
    #             DeletionGenotype(
    #                 perturbation=DeletionPerturbation(
    #                     systematic_gene_name=row["Array Systematic Name"],
    #                     perturbed_gene_name=row["Array allele name"],
    #                 )
    #             )
    #         )

    #     return genotype

    @staticmethod
    def create_experiment(row, reference_phenotype_std_26, reference_phenotype_std_30):
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )

        # genotype
        genotype = []
        # Query
        if "ts" in row["Query Strain ID"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=DampPerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                    )
                )
            )
        elif "damp" in row["Query Strain ID"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=DampPerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                    )
                )
            )
        else:
            genotype.append(
                DeletionGenotype(
                    perturbation=DeletionPerturbation(
                        systematic_gene_name=row["Query Systematic Name"],
                        perturbed_gene_name=row["Query allele name"],
                    )
                )
                # Array
            )
        if "ts" in row["Array Strain ID"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=DampPerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                    )
                )
            )
        elif "damp" in row["Array Strain ID"]:
            genotype.append(
                InterferenceGenotype(
                    perturbation=DampPerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                    )
                )
            )
        else:
            genotype.append(
                DeletionGenotype(
                    perturbation=DeletionPerturbation(
                        systematic_gene_name=row["Array Systematic Name"],
                        perturbed_gene_name=row["Array allele name"],
                    )
                )
            )

        # genotype

        environment = BaseEnvironment(
            media=Media(name="YEPD", state="solid"),
            temperature=Temperature(scalar=row["Temperature"]),
        )
        reference_environment = environment.model_copy()
        # Phenotype based on temperature
        smf_key = "Double mutant fitness"
        smf_std_key = "Double mutant fitness standard deviation"
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

    # def create_experiment(self, row):
    #     # Common attributes for both temperatures
    #     reference_genome = ReferenceGenome(
    #         species="saccharomyces Cerevisiae", strain="s288c"
    #     )

    #     genotype = self._process_genotype(row)

    #     environment = BaseEnvironment(
    #         media=Media(name="YEPD", state="solid"),
    #         temperature=Temperature(scalar=row["Temperature"]),
    #     )
    #     reference_environment = environment.model_copy()
    #     # Phenotype based on temperature
    #     smf_key = "Double mutant fitness"
    #     smf_std_key = "Double mutant fitness standard deviation"
    #     phenotype = FitnessPhenotype(
    #         graph_level="global",
    #         label="smf",
    #         label_error="smf_std",
    #         fitness=row[smf_key],
    #         fitness_std=row[smf_std_key],
    #     )

    #     if row["Temperature"] == 26:
    #         reference_phenotype_std = self.reference_phenotype_std_26
    #     elif row["Temperature"] == 30:
    #         reference_phenotype_std = self.reference_phenotype_std_30
    #     reference_phenotype = FitnessPhenotype(
    #         graph_level="global",
    #         label="smf",
    #         label_error="smf_std",
    #         fitness=1.0,
    #         fitness_std=reference_phenotype_std,
    #     )

    #     reference = FitnessExperimentReference(
    #         reference_genome=reference_genome,
    #         reference_environment=reference_environment,
    #         reference_phenotype=reference_phenotype,
    #     )

    #     experiment = FitnessExperiment(
    #         genotype=genotype, environment=environment, phenotype=phenotype
    #     )
    #     return experiment, reference

    def get_reference_index(self):
        # Serialize references for comparability using model_dump
        log.info("Serializing references")
        serialized_references = []
        for ref in tqdm(self.reference):
            serialized_reference = json.dumps(ref.model_dump(), sort_keys=True)
            serialized_references.append(serialized_reference)

        # Identify unique references and their indices
        log.info("Identifying unique references and their indices")
        unique_refs = {}
        for idx, ref_json in tqdm(enumerate(serialized_references)):
            if ref_json not in unique_refs:
                unique_refs[ref_json] = {
                    "indices": [],
                    "model": self.reference[idx],  # Store the Pydantic model
                }
            unique_refs[ref_json]["indices"].append(idx)

        # Create ExperimentReferenceIndex instances
        log.info("Creating experiment reference index")
        reference_indices = []
        for ref_info in tqdm(unique_refs.values()):
            # Convert indices list to a set for efficient lookup
            indices_set = set(ref_info["indices"])
            bool_array = [i in indices_set for i in range(len(self.data))]
            reference_indices.append(
                ExperimentReferenceIndex(reference=ref_info["model"], index=bool_array)
            )

        # Return ReferenceIndex instance
        return ReferenceIndex(data=reference_indices)

    def _remove_duplicates(self) -> list[BaseExperiment]:
        """
        Remove duplicate BaseExperiment instances from self.data.
        All fields of the object must match for it to be considered a duplicate.
        """
        unique_data = []
        seen = set()

        log.info("Removing duplicates")
        for experiment, reference in tqdm(zip(self.data, self.reference)):
            # Serialize the experiment object to a dictionary
            experiment_dict = experiment.model_dump()
            reference_dict = reference.model_dump()

            combined_dict = {**experiment_dict, **reference_dict}
            # Convert dictionary to a JSON string for comparability
            combined_json = json.dumps(combined_dict, sort_keys=True)

            if combined_json not in seen:
                seen.add(combined_json)
                unique_data.append((experiment, reference))

        self.data = [experiment for experiment, reference in unique_data]
        self.reference = [reference for experiment, reference in unique_data]

        return self.data, self.reference


if __name__ == "__main__":
    dataset = NeoDmfCostanzo2016Dataset()
    print(len(dataset))
    print(json.dumps(dataset[0].model_dump(), indent=4))
    # print(dataset.reference_index)
    # print(len(dataset.reference_index))
    # print(dataset.reference_index[0])
    print("done")
