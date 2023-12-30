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

import h5py
import lmdb
import matplotlib.pyplot as plt
import numpy as np
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
        summed_indices = sum([boolean_value for exp_ref_index in v for boolean_value in exp_ref_index.index])

        if summed_indices != len(v[0].index):
            raise ValueError("Sum of indices must equal the number of experiments")
        return v


@define
class NeoSmfCostanzo2016Dataset:
    root: str = field(default="data/neo4j/smf_costanzo2016")
    url: str = field(
        repr=False,
        default="https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
    )
    raw: str = field(init=False, repr=False)
    data: list[BaseExperiment] = field(init=False, repr=False, factory=list)
    reference: list[FitnessExperimentReference] = field(
        init=False, repr=False, factory=list
    )
    reference_index: ReferenceIndex = field(init=False, repr=False)
    reference_phenotype_std_30 = field(init=False, repr=False)
    reference_phenotype_std_26 = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.raw = osp.join(self.root, "raw")
        self._download()
        self._extract()
        self._cleanup_after_extract()
        self.data, self.reference = self._process_excel()
        self.data, self.reference = self._remove_duplicates()
        self.reference_index = self.get_reference_index()

    # write a get item method to return a single experiment
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _download(self):
        if not osp.exists(self.raw):
            os.makedirs(self.raw)
            download_url(self.url, self.raw)

    def _extract(self):
        zip_path = osp.join(
            self.raw,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw)

    def _cleanup_after_extract(self):
        # We are only keeping the smf data for this dataset
        extracted_folder = osp.join(
            self.raw,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        xlsx_file = osp.join(
            extracted_folder, "strain_ids_and_single_mutant_fitness.xlsx"
        )
        if osp.exists(xlsx_file):
            shutil.move(xlsx_file, self.raw)
        if osp.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
        zip_path = osp.join(
            self.raw,
            "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
        )
        if osp.exists(zip_path):
            os.remove(zip_path)

    def _process_excel(self):
        """
        Process the Excel file and convert each row to Experiment instances for 26°C and 30°C separately.
        """
        xlsx_path = osp.join(self.raw, "strain_ids_and_single_mutant_fitness.xlsx")
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


if __name__ == "__main__":
    dataset = NeoSmfCostanzo2016Dataset()
    print(len(dataset))
    print(json.dumps(dataset[0].model_dump(), indent=4))
    # print(dataset.reference_index)
    # print(len(dataset.reference_index))
    # print(dataset.reference_index[0])
    # print()
    for data in dataset:
        if data.genotype.perturbation.systematic_gene_name == "YIL154C":
            print()
