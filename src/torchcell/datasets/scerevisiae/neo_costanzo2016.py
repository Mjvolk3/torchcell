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
import polars as pl
import torch
from attrs import define, field
from polars import DataFrame, col
from pydantic import Field, validator
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
    Experiment,
    GeneId,
    GenePerturbation,
    GenePerturbationType,
    Media,
    ModelStrict,
    ReferenceGenome,
    Temperature,
)
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet

log = logging.getLogger(__name__)


# pydantic models
class SmfCostanzo2016Perturbation(GenePerturbation, ModelStrict):
    id_full: str
    allele_name: str


class SmfCostanzo2016Genotype(ModelStrict):
    perturbation: SmfCostanzo2016Perturbation | list[SmfCostanzo2016Perturbation]


class SmfCostanzo2016Phenotype(BasePhenotype, ModelStrict):
    smf: float
    smf_std: float


class SmfCostanzo2016Experiment(Experiment):
    reference_genome: ReferenceGenome
    reference_environment: BaseEnvironment
    reference_phenotype: SmfCostanzo2016Phenotype | None
    genotype: SmfCostanzo2016Genotype
    environment: BaseEnvironment
    phenotype: SmfCostanzo2016Phenotype


class SmfCostanzo2016Experiment(ModelStrict):
    reference_genome: ReferenceGenome
    reference_environment: BaseEnvironment
    reference_phenotype: SmfCostanzo2016Phenotype | None
    genotype: SmfCostanzo2016Genotype
    environment: BaseEnvironment
    phenotype: SmfCostanzo2016Phenotype


#
@define
class NeoSmfCostanzo2016Dataset:
    root: str = field(default="data/neo4j/smf_costanzo2016")
    url: str = field(
        repr=False,
        default="https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip",
    )
    raw: str = field(init=False, repr=False)
    data: list[SmfCostanzo2016Experiment] = field(init=False, repr=False, default=[])
    reference_phenotype_std = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.raw = osp.join(self.root, "raw")
        self._download()
        self._extract()
        self._cleanup_after_extract()
        self.data = self._process_excel()

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

    def _process_temperature_data(self, df, temperature):
        """
        Process DataFrame for a specific temperature and add entries to the dataset.
        """
        for _, row in df.iterrows():
            experiment = self.create_experiment(row, temperature)
            self.data.append(experiment)

    def _process_excel(self):
        """
        Process the Excel file and convert each row to Experiment instances for 26°C and 30°C separately.
        """
        xlsx_path = osp.join(self.raw, "strain_ids_and_single_mutant_fitness.xlsx")
        df = pd.read_excel(xlsx_path)
        # This is an approximate since I cannot find the exact value in the paper
        df["Strain_ID_suffix"] = df["Strain ID"].str.split("_", expand=True)[1]

        # Filter out rows where 'Strain_ID_Part2' contains 'ts' or 'damp'
        filter_condition = ~df["Strain_ID_suffix"].str.contains("ts|damp", na=False)
        df_filtered = df[filter_condition]

        self.reference_phenotype_std = pd.concat(
            [
                df_filtered["Single mutant fitness (26°) stddev"],
                df_filtered["Single mutant fitness (30°) stddev"],
            ]
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

        return self.data

    def create_experiment(self, row, temperature):
        """
        Create an Experiment instance from a row of the Excel spreadsheet for a given temperature.
        """
        # Common attributes for both temperatures
        reference_genome = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="s288c"
        )
        gene_id = GeneId(id=row["Systematic gene name"])
        gene_perturbation_type = GenePerturbationType(perturbation="Deletion")
        smf_costanzo_perturbation = SmfCostanzo2016Perturbation(
            id_full=row["Strain ID"],
            allele_name=row["Allele/Gene name"],
            gene_id=gene_id,
            gene_perturbation_type=gene_perturbation_type,
        )
        genotype = SmfCostanzo2016Genotype(perturbation=smf_costanzo_perturbation)
        media = Media(name="YPED", state="solid")
        environment = BaseEnvironment(
            media=media, temperature=Temperature(Celsius=temperature)
        )

        # Phenotype based on temperature
        smf_key = f"Single mutant fitness ({temperature}°)"
        smf_std_key = f"Single mutant fitness ({temperature}°) stddev"
        phenotype = SmfCostanzo2016Phenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            smf=row[smf_key],
            smf_std=row[smf_std_key],
        )
        reference_phenotype = SmfCostanzo2016Phenotype(
            graph_level="global",
            label="smf",
            label_error="smf_std",
            smf=1.0,
            smf_std=self.reference_phenotype_std,
        )

        # Create Experiment instance
        experiment = SmfCostanzo2016Experiment(
            reference_genome=reference_genome,
            reference_environment=environment.copy(),
            reference_phenotype=reference_phenotype,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

        return experiment


if __name__ == "__main__":
    dataset = NeoSmfCostanzo2016Dataset()
    print(dataset.data[0].json(indent=4))
    print()
