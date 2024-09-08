# torchcell/datasets/scerevisiae/kuzmin2020
# [[torchcell.datasets.scerevisiae.kuzmin2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2020
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2020.py

import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import zipfile
from collections.abc import Callable
import urllib.request
import requests
import ssl
import lmdb
import numpy as np
import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm
from torchcell.data import ExperimentReferenceIndex
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaAllelePerturbation,
    SgaTsAllelePerturbation,
    SgaDampPerturbation,
    Temperature,
    Experiment,
    ExperimentReference,
    GeneInteractionPhenotype,
    GeneInteractionExperimentReference,
    GeneInteractionExperiment,
    Publication,
)
from torchcell.sequence import GeneSet
from torchcell.data import ExperimentDataset, post_process
from torchcell.datasets.dataset_registry import register_dataset
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@register_dataset
class SmfKuzmin2020Dataset(ExperimentDataset):
    # original that doesn't work. Think Science blocks.
    # url = https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/smf_kuzmin2020",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> Experiment:
        return FitnessExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aaz5667-Table-S5.xlsx"

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_excel(osp.join(self.raw_dir, self.raw_file_names), skiprows=1)
        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Smf Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(row)

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)
        env.close()

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Mutant type"] == "Single mutant"]
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        if "delta" in row["Allele1"]:
            perturbation = SgaKanMxDeletionPerturbation(
                systematic_gene_name=row["ORF1"],
                perturbed_gene_name=row["Gene1"],
                strain_id=row["Query Strain ID"],
            )
        else:
            perturbation = SgaAllelePerturbation(
                systematic_gene_name=row["ORF1"],
                perturbed_gene_name=row["Gene1"],
                strain_id=row["Query Strain ID"],
            )

        genotype = Genotype(perturbations=[perturbation])

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_statistic="std",
            fitness=row["Fitness"],
            fitness_std=row["St.dev."],
        )

        phenotype_reference = FitnessPhenotype(
            graph_level="global",
            label="smf",
            label_statistic="std",
            fitness=1.0,
            fitness_std=None,
        )

        reference = FitnessExperimentReference(
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )

        publication = Publication(
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class DmfKuzmin2020Dataset(ExperimentDataset):
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmf_kuzmin2020",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> Experiment:
        return FitnessExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        return [
            "aaz5667-Table-S1.xlsx",
            "aaz5667-Table-S3.xlsx",
            "aaz5667-Table-S5.xlsx",
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df_s1 = pd.read_excel(
            osp.join(self.raw_dir, self.raw_file_names[0]), skiprows=1
        )
        df_s3 = pd.read_excel(
            osp.join(self.raw_dir, self.raw_file_names[1]), skiprows=1
        )
        df_s5 = pd.read_excel(
            osp.join(self.raw_dir, self.raw_file_names[2]), skiprows=1
        )

        df = self.preprocess_raw(df_s1, df_s3, df_s5)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(row)

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(
        self, df_s1: pd.DataFrame, df_s3: pd.DataFrame, df_s5: pd.DataFrame
    ) -> pd.DataFrame:
        # Combine S1 and S3, filtering for digenic interactions
        df_combined = pd.concat([df_s1, df_s3])
        df_combined = df_combined[df_combined["Combined mutant type"] == "digenic"]

        # Process S5 to get double mutant data
        df_s5_double = df_s5[df_s5["Mutant type"] == "Double mutant"]

        # Merge combined data with S5 to get standard deviations
        df = pd.merge(
            df_combined,
            df_s5_double[["Query Strain ID", "Fitness", "St.dev."]],
            left_on="Query strain ID",
            right_on="Query Strain ID",
            how="left",
        )

        # Verify fitness values match
        # TODO make assertion. 
        mask = (df["Double/triple mutant fitness"] - df["Fitness"]).abs() > 1e-6
        if mask.any():
            log.warning(f"Fitness mismatch found for {mask.sum()} rows")

        # Use S5 fitness and std where available, fallback to S1/S3 data
        df["fitness"] = df["Fitness"].fillna(df["Double/triple mutant fitness"])
        df["fitness_std"] = df["St.dev."].fillna(
            df["Double/triple mutant fitness standard deviation"]
        )

        # Clean up gene names
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)

        return df

    @staticmethod
    def create_experiment(row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        query_genes = row["Query allele name"].split("+")
        array_gene = row["Array allele name"]


        # TODO none of these are systematic gene names.
        for gene in query_genes + [array_gene]:
            if "delta" in gene:
                perturbation = SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene.split("_")[0],
                    perturbed_gene_name=gene.split("_")[0],
                    strain_id=row["Query strain ID"],
                )
            else:
                perturbation = SgaAllelePerturbation(
                    systematic_gene_name=gene.split("-")[0],
                    perturbed_gene_name=gene.split("-")[0],
                    strain_id=row["Query strain ID"],
                )
            perturbations.append(perturbation)

        genotype = Genotype(perturbations=perturbations)

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(
            graph_level="global",
            label="dmf",
            label_statistic="std",
            fitness=row["fitness"],
            fitness_std=row["fitness_std"],
        )

        phenotype_reference = FitnessPhenotype(
            graph_level="global",
            label="dmf",
            label_statistic="std",
            fitness=1.0,
            fitness_std=None,
        )

        reference = FitnessExperimentReference(
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )

        publication = Publication(
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class TmfKuzmin2020Dataset(ExperimentDataset):
    # original that doesn't work. Think Science blocks.
    # url = https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmf_kuzmin2020",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> Experiment:
        return FitnessExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aaz5667-Table-S1.xlsx"

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_excel(osp.join(self.raw_dir, self.raw_file_names))
        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Tmf Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(row)

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement preprocessing specific to TmfKuzmin2020Dataset
        df = df[df["Combined mutant type"] == "trigenic"]
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        return df

    @staticmethod
    def create_experiment(row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        for i in range(1, 4):  # For trigenic mutants
            gene_name = (
                row[f"Query allele name_{i}"] if i < 3 else row["Array allele name"]
            )
            systematic_name = (
                row[f"Query systematic name_{i}"]
                if i < 3
                else row["Array systematic name"]
            )
            strain_id = row["Query strain ID"] if i < 3 else row["Array strain ID"]

            if "delta" in gene_name:
                perturbation = SgaKanMxDeletionPerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            elif "ts" in gene_name:
                perturbation = SgaTsAllelePerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            else:
                perturbation = SgaAllelePerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            perturbations.append(perturbation)

        genotype = Genotype(perturbations=perturbations)

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(
            graph_level="global",
            label="tmf",
            label_statistic="tmf_std",
            fitness=row["Double/triple mutant fitness"],
            fitness_std=row["Double/triple mutant fitness standard deviation"],
        )

        phenotype_reference = FitnessPhenotype(
            graph_level="global",
            label="tmf",
            label_statistic="tmf_std",
            fitness=1.0,
            fitness_std=None,
        )

        reference = FitnessExperimentReference(
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )

        publication = Publication(
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class DmiKuzmin2020Dataset(ExperimentDataset):
    # original that doesn't work. Think Science blocks.
    # url = https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmi_kuzmin2020",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> ExperimentReference:
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aaz5667-Table-S1.xlsx"

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_excel(osp.join(self.raw_dir, self.raw_file_names))
        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmi Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(row)

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement preprocessing specific to DmiKuzmin2020Dataset
        df = df[df["Combined mutant type"] == "digenic"]
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        return df

    @staticmethod
    def create_experiment(row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        for i in range(1, 3):  # For digenic mutants
            gene_name = row["Query allele name"] if i == 1 else row["Array allele name"]
            systematic_name = (
                row["Query systematic name"] if i == 1 else row["Array systematic name"]
            )
            strain_id = row["Query strain ID"] if i == 1 else row["Array strain ID"]

            if "delta" in gene_name:
                perturbation = SgaKanMxDeletionPerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            elif "ts" in gene_name:
                perturbation = SgaTsAllelePerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            else:
                perturbation = SgaAllelePerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            perturbations.append(perturbation)

        genotype = Genotype(perturbations=perturbations)

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            graph_level="edge",
            label="dmi",
            label_statistic="p_value",
            interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            p_value=row["P-value"],
        )

        phenotype_reference = GeneInteractionPhenotype(
            graph_level="edge",
            label="dmi",
            label_statistic="p_value",
            interaction=0.0,
            p_value=None,
        )

        reference = GeneInteractionExperimentReference(
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )

        publication = Publication(
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class TmiKuzmin2020Dataset(ExperimentDataset):
    # original that doesn't work. Think Science blocks.
    # url = https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmi_kuzmin2020",
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> ExperimentReference:
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aaz5667-Table-S1.xlsx"

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_excel(osp.join(self.raw_dir, self.raw_file_names))
        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Tmi Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(row)

                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement preprocessing specific to TmiKuzmin2020Dataset
        df = df[df["Combined mutant type"] == "trigenic"]
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        return df

    @staticmethod
    def create_experiment(row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        for i in range(1, 4):  # For trigenic mutants
            gene_name = (
                row[f"Query allele name_{i}"] if i < 3 else row["Array allele name"]
            )
            systematic_name = (
                row[f"Query systematic name_{i}"]
                if i < 3
                else row["Array systematic name"]
            )
            strain_id = row["Query strain ID"] if i < 3 else row["Array strain ID"]

            if "delta" in gene_name:
                perturbation = SgaKanMxDeletionPerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            elif "ts" in gene_name:
                perturbation = SgaTsAllelePerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            else:
                perturbation = SgaAllelePerturbation(
                    systematic_gene_name=systematic_name,
                    perturbed_gene_name=gene_name,
                    strain_id=strain_id,
                )
            perturbations.append(perturbation)

        genotype = Genotype(perturbations=perturbations)

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            graph_level="edge",
            label="tmi",
            label_statistic="p_value",
            interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            p_value=row["P-value"],
        )

        phenotype_reference = GeneInteractionPhenotype(
            graph_level="edge",
            label="tmi",
            label_statistic="p_value",
            interaction=0.0,
            p_value=None,
        )

        reference = GeneInteractionExperimentReference(
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            genotype=genotype, environment=environment, phenotype=phenotype
        )

        publication = Publication(
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    # Test the datasets
    datasets = [
        # SmfKuzmin2020Dataset(),
        # DmfKuzmin2020Dataset(),
        # TmfKuzmin2020Dataset(),
        # DmiKuzmin2020Dataset(),
        # TmiKuzmin2020Dataset(),
    ]

    for dataset in datasets:
        print(f"Testing {dataset.__class__.__name__}:")
        print(f"Length: {len(dataset)}")
        print(f"First item: {dataset[0]}")
        print("\n")
