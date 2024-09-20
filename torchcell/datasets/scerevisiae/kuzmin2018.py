# torchcell/datasets/scerevisiae/kuzmin2018.py
# [[torchcell.datasets.scerevisiae.kuzmin2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2018.py
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2018.py
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import zipfile
from collections.abc import Callable
import lmdb
import numpy as np
import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaAllelePerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    Experiment,
    ExperimentReference,
    GeneInteractionPhenotype,
    GeneInteractionExperimentReference,
    GeneInteractionExperiment,
    Publication,
)
from torchcell.data import ExperimentDataset, post_process
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Fitness
@register_dataset
class SmfKuzmin2018Dataset(ExperimentDataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/smf_kuzmin2018",
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
        return "aao1729_data_s1.tsv"

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
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                # Serialize the Pydantic objects
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
            lambda x: (
                "temperature_sensitive"
                if "tsa" in x
                else "KanMX_deletion" if "dma" in x else "unknown"
            )
        )
        self.phenotype_reference_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()

        # array single mutants
        unique_array_allele_names = df["Array allele name"].drop_duplicates()
        df_array = df[
            df["Array allele name"].isin(unique_array_allele_names)
        ].drop_duplicates(subset=["Array allele name"])
        df_array["smf_type"] = "array_smf"
        # query single mutants, trigenic is not smf
        digenic_df = df[df["Combined mutant type"] == "digenic"]

        # Get unique 'Query allele name' and find first instances
        unique_query_allele_names = digenic_df[
            "Query allele name no ho"
        ].drop_duplicates()
        df_query = digenic_df[
            digenic_df["Query allele name no ho"].isin(unique_query_allele_names)
        ].drop_duplicates(subset=["Query allele name"])
        df_query["smf_type"] = "query_smf"
        df = pd.concat([df_array, df_query], axis=0)
        df = df.reset_index(drop=True)
        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = (
            df[~df["Query single/double mutant fitness"].isna()]
            .copy()
            .reset_index(drop=True)
        )
        return df

    @staticmethod
    def create_experiment(dataset_name, row, phenotype_reference_std):
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="S288C"
        )
        # genotype
        if row["smf_type"] == "query_smf":
            # Query
            if "KanMX_deletion" in row["query_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaKanMxDeletionPerturbation(
                            systematic_gene_name=row["Query systematic name no ho"],
                            perturbed_gene_name=row["Query allele name no ho"],
                            strain_id=row["Query strain ID"],
                        )
                    ]
                )

            elif "allele" in row["query_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaAllelePerturbation(
                            systematic_gene_name=row["Query systematic name no ho"],
                            perturbed_gene_name=row["Query allele name no ho"],
                            strain_id=row["Query strain ID"],
                        )
                    ]
                )

        elif row["smf_type"] == "array_smf":
            # Array
            if "KanMX_deletion" in row["array_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaKanMxDeletionPerturbation(
                            systematic_gene_name=row["Array systematic name"],
                            perturbed_gene_name=row["Array allele name"],
                            strain_id=row["Array strain ID"],
                        )
                    ]
                )

            elif "allele" in row["array_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaAllelePerturbation(
                            systematic_gene_name=row["Array systematic name"],
                            perturbed_gene_name=row["Array allele name"],
                            strain_id=row["Array strain ID"],
                        )
                    ]
                )

            # Only array has ts
            elif "temperature_sensitive" in row["array_perturbation_type"]:
                genotype = Genotype(
                    perturbations=[
                        SgaTsAllelePerturbation(
                            systematic_gene_name=row["Array systematic name"],
                            perturbed_gene_name=row["Array allele name"],
                            strain_id=row["Array strain ID"],
                        )
                    ]
                )

        # genotype
        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()
        # Phenotype
        if row["smf_type"] == "query_smf":
            smf_key = "Query single/double mutant fitness"
        elif row["smf_type"] == "array_smf":
            smf_key = "Array single mutant fitness"

        # We have no reported std for single mutants
        # Could use mean of all stds, would be a conservative estimate
        # Using nan for now
        phenotype = FitnessPhenotype(fitness=row[smf_key], std=None)

        phenotype_reference = FitnessPhenotype(fitness=1.0, std=phenotype_reference_std)

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class DmfKuzmin2018Dataset(ExperimentDataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmf_kuzmin2018",
        subset_n: int = None,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> Experiment:
        return FitnessExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aao1729_data_s1.tsv"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                # Serialize the Pydantic objects
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

        df["query_perturbation_type_no_ho"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            lambda x: (
                "temperature_sensitive"
                if "tsa" in x
                else "KanMX_deletion" if "dma" in x else "unknown"
            )
        )
        self.phenotype_reference_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()
        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(dataset_name, row, phenotype_reference_std):
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="S288C"
        )
        # genotype
        perturbations = []
        # Query...
        if "KanMX_deletion" in row["query_perturbation_type_no_ho"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_no_ho"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"],
                    strain_id=row["Query strain ID"],
                )
            )

        # Array - only array has ts
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 2, "Genotype must have 2 perturbations."
        # genotype
        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()
        # Phenotype
        if row["Combined mutant type"] == "digenic":
            dmf_key = "Combined mutant fitness"
            dmf_std_key = "Combined mutant fitness standard deviation"
            fitness_std = row[dmf_std_key]
        elif row["Combined mutant type"] == "trigenic":
            dmf_key = "Query single/double mutant fitness"
            # std of these fitnesses not reported
            fitness_std = np.nan
        phenotype = FitnessPhenotype(fitness=row[dmf_key], std=fitness_std)

        phenotype_reference = FitnessPhenotype(fitness=1.0, std=phenotype_reference_std)

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class TmfKuzmin2018Dataset(ExperimentDataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmf_kuzmin2018",
        subset_n: int = None,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> Experiment:
        return FitnessExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aao1729_data_s1.tsv"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                # Serialize the Pydantic objects
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
        df = df[df["Combined mutant type"] == "trigenic"].copy()

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
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            lambda x: (
                "temperature_sensitive"
                if "tsa" in x
                else "KanMX_deletion" if "dma" in x else "unknown"
            )
        )
        self.phenotype_reference_std = df[
            "Combined mutant fitness standard deviation"
        ].mean()
        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(dataset_name, row, phenotype_reference_std):
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="S288C"
        )
        # genotype
        perturbations = []
        # Query
        # Query 1
        if "KanMX_deletion" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Query 2
        if "KanMX_deletion" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Array - only array has ts
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 3, "Genotype must have 3 perturbations."
        # genotype
        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()
        # Phenotype based on temperature
        tmf_key = "Combined mutant fitness"
        tmf_std_key = "Combined mutant fitness standard deviation"
        phenotype = FitnessPhenotype(fitness=row[tmf_key], std=row[tmf_std_key])

        phenotype_reference = FitnessPhenotype(fitness=1.0, std=phenotype_reference_std)

        reference = FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


# Interactions
@register_dataset
class DmiKuzmin2018Dataset(ExperimentDataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmi_kuzmin2018",
        subset_n: int = None,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> ExperimentReference:
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aao1729_data_s1.tsv"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmi Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

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

        df["query_perturbation_type_no_ho"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            lambda x: (
                "temperature_sensitive"
                if "tsa" in x
                else "KanMX_deletion" if "dma" in x else "unknown"
            )
        )

        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="S288C"
        )

        perturbations = []
        # Query...
        if "KanMX_deletion" in row["query_perturbation_type_no_ho"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_no_ho"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"],
                    strain_id=row["Query strain ID"],
                )
            )

        # Array - only array has ts
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 2, "Genotype must have 2 perturbations."

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            p_value=row["P-value"],
        )

        # By definition in paper interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(gene_interaction=0.0, p_value=None)

        reference = GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class TmiKuzmin2018Dataset(ExperimentDataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmi_kuzmin2018",
        subset_n: int = None,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        **kwargs,
    ):
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> ExperimentReference:
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> ExperimentReference:
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> str:
        return "aao1729_data_s1.tsv"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Tmi Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

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
        # Select triples only
        df = df[df["Combined mutant type"] == "trigenic"].copy()

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
        df["query_perturbation_type_1"] = df["Query allele name_1"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["query_perturbation_type_2"] = df["Query allele name_2"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            lambda x: (
                "temperature_sensitive"
                if "tsa" in x
                else "KanMX_deletion" if "dma" in x else "unknown"
            )
        )

        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="saccharomyces Cerevisiae", strain="S288C"
        )

        perturbations = []
        # Query 1
        if "KanMX_deletion" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_1"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Query 2
        if "KanMX_deletion" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        elif "allele" in row["query_perturbation_type_2"]:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"],
                    strain_id=row["Query strain ID"],
                )
            )
        # Array
        if "temperature_sensitive" in row["array_perturbation_type"]:
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        elif "KanMX_deletion" in row["array_perturbation_type"]:
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"],
                    strain_id=row["Array strain ID"],
                )
            )
        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 3, "Genotype must have 3 perturbations."

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            p_value=row["P-value"],
        )

        # By definition in paper interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(gene_interaction=0.0, p_value=None)

        reference = GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )

        experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    # Fitness
    print("Fitness")
    dataset = SmfKuzmin2018Dataset()
    print(dataset[0])
    print(len(dataset))
    dataset = DmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    dataset = TmfKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    print()
    print("Interactions")
    # Interactions
    dataset = DmiKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
    dataset = TmiKuzmin2018Dataset()
    dataset[0]
    print(len(dataset))
