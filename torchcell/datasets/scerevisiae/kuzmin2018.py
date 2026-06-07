# torchcell/datasets/scerevisiae/kuzmin2018.py
# [[torchcell.datasets.scerevisiae.kuzmin2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2018.py
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2018.py
import hashlib
import json
import logging
import math
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

# ============================================================================
# Sample Size Metadata - Extracted from Kuzmin et al. 2018
# ============================================================================

# Query strain single mutant fitness measurements
# Quote: "Each high-density array was screened in triplicate for a total of
#         6 replicates."
# Source: SI-kuzminSystematicAnalysisComplex2018.mmd, Line 60
# Verified: SI-kuzminSystematicAnalysisComplex2018.pdf, Page 4,
#           Section "Query fitness estimation"
# Date extracted: 2026-01-26
# Note: 6 replicates = number of independent screens per query strain
#       (3 in triplicate × 2 control strains = 6 total replicates)
N_SAMPLES_QUERY_SMF = 6  # Number of replicate screens per query strain

# Wild-type control screens for variance estimation
# Quote: "Estimates of the variance of array single mutant fitness used in the
#         calculation of interaction p-values, were obtained by screening a
#         wild-type control query strain, Y13096, against the diagnostic array
#         (n=91)."
# Source: SI-kuzminSystematicAnalysisComplex2018.mmd, Line 64
# Date extracted: 2026-01-26
# Note: Array fitness values themselves were imported from Costanzo 2016 (reference 7)
#       and not re-measured. This n=91 is specifically for variance estimation
#       used in p-value calculations, not for per-strain fitness measurements.
N_SAMPLES_WT_VARIANCE = 91  # WT control screens for variance estimation

# Double and triple mutant fitness measurements
# Quote: "Every double mutant query strain was screened alongside its two
#         single mutant control strains in two independent replicates."
# Source: SI-kuzminSystematicAnalysisComplex2018.mmd, Line 76
# Verified: SI-kuzminSystematicAnalysisComplex2018.pdf, Page 4,
#           Section "Triple mutant synthetic genetic array (SGA) analysis"
# Date extracted: 2026-01-26
N_SAMPLES_DOUBLE_MUTANT = 2
N_SAMPLES_TRIPLE_MUTANT = 2

# Wild-type reference measurements
# For WT reference phenotypes, use the variance estimation n_samples
# Reference fitness = 1.0 is derived from wild-type control measurements
N_SAMPLES_WT_REFERENCE = N_SAMPLES_WT_VARIANCE  # 91


# Fitness
@register_dataset
class SmfKuzmin2018Dataset(ExperimentDataset):
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

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
            species="Saccharomyces cerevisiae", strain="S288C"
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
            n_samples = N_SAMPLES_QUERY_SMF
        elif row["smf_type"] == "array_smf":
            smf_key = "Array single mutant fitness"
            # Array fitness imported from Costanzo 2016 - not measured in this study
            # No individual std reported, so n_samples = None
            n_samples = None
        else:
            # Fallback for unexpected smf_type
            smf_key = "Query single/double mutant fitness"
            n_samples = None

        # SMF std not reported in Kuzmin2018 data
        fitness_std_val = None
        # Compute SE using Option 2 policy:
        # - None if std is None (missing data)
        # - NaN if n=1 (undefined)
        # - std/sqrt(n) if n≥2 and std available
        if fitness_std_val is None:
            fitness_se_val = None
        elif n_samples is None or n_samples == 1:
            fitness_se_val = math.nan
        else:
            fitness_se_val = fitness_std_val / math.sqrt(n_samples)

        phenotype = FitnessPhenotype(
            fitness=row[smf_key],
            fitness_std=fitness_std_val,
            fitness_se=fitness_se_val,
            n_samples=n_samples,
        )

        # Reference phenotype uses WT array measurements
        n_samples_ref = N_SAMPLES_WT_REFERENCE
        if (
            phenotype_reference_std is not None
            and n_samples_ref is not None
            and n_samples_ref > 1
        ):
            phenotype_reference_se = phenotype_reference_std / math.sqrt(n_samples_ref)
        elif n_samples_ref == 1:
            phenotype_reference_se = math.nan
        else:
            phenotype_reference_se = None

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            fitness_se=phenotype_reference_se,
            n_samples=n_samples_ref,
        )

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
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

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
            species="Saccharomyces cerevisiae", strain="S288C"
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
            fitness_std_val = row[dmf_std_key]
            n_samples = N_SAMPLES_DOUBLE_MUTANT
        elif row["Combined mutant type"] == "trigenic":
            dmf_key = "Query single/double mutant fitness"
            # std of these fitnesses not reported
            fitness_std_val = np.nan
            n_samples = N_SAMPLES_TRIPLE_MUTANT
        else:
            # Fallback for unexpected types
            dmf_key = "Combined mutant fitness"
            fitness_std_val = None
            n_samples = None

        # Compute SE using Option 2 policy
        if fitness_std_val is None or (isinstance(fitness_std_val, float) and np.isnan(fitness_std_val)):
            fitness_se_val = None
        elif n_samples is None or n_samples == 1:
            fitness_se_val = math.nan
        elif n_samples >= 2:
            fitness_se_val = fitness_std_val / math.sqrt(n_samples)
        else:
            fitness_se_val = None

        phenotype = FitnessPhenotype(
            fitness=row[dmf_key],
            fitness_std=fitness_std_val if not (isinstance(fitness_std_val, float) and np.isnan(fitness_std_val)) else None,
            fitness_se=fitness_se_val,
            n_samples=n_samples,
        )

        # Reference phenotype
        n_samples_ref = N_SAMPLES_WT_REFERENCE
        if (
            phenotype_reference_std is not None
            and n_samples_ref is not None
            and n_samples_ref > 1
        ):
            phenotype_reference_se = phenotype_reference_std / math.sqrt(n_samples_ref)
        elif n_samples_ref == 1:
            phenotype_reference_se = math.nan
        else:
            phenotype_reference_se = None

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            fitness_se=phenotype_reference_se,
            n_samples=n_samples_ref,
        )

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
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

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
            species="Saccharomyces cerevisiae", strain="S288C"
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
        fitness_std_val = row[tmf_std_key]
        n_samples = N_SAMPLES_TRIPLE_MUTANT

        # Compute SE using Option 2 policy
        if fitness_std_val is None or (isinstance(fitness_std_val, float) and np.isnan(fitness_std_val)):
            fitness_se_val = None
        elif n_samples == 1:
            fitness_se_val = math.nan
        elif n_samples >= 2:
            fitness_se_val = fitness_std_val / math.sqrt(n_samples)
        else:
            fitness_se_val = None

        phenotype = FitnessPhenotype(
            fitness=row[tmf_key],
            fitness_std=fitness_std_val if not (isinstance(fitness_std_val, float) and np.isnan(fitness_std_val)) else None,
            fitness_se=fitness_se_val,
            n_samples=n_samples,
        )

        # Reference phenotype
        n_samples_ref = N_SAMPLES_WT_REFERENCE
        if (
            phenotype_reference_std is not None
            and n_samples_ref is not None
            and n_samples_ref > 1
        ):
            phenotype_reference_se = phenotype_reference_std / math.sqrt(n_samples_ref)
        elif n_samples_ref == 1:
            phenotype_reference_se = math.nan
        else:
            phenotype_reference_se = None

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            fitness_se=phenotype_reference_se,
            n_samples=n_samples_ref,
        )

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
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

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
            species="Saccharomyces cerevisiae", strain="S288C"
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
            gene_interaction_p_value=row["P-value"],
        )

        # By definition in paper interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(gene_interaction=0.0, gene_interaction_p_value=None)

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
    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

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
            species="Saccharomyces cerevisiae", strain="S288C"
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
            gene_interaction_p_value=row["P-value"],
        )

        # By definition in paper interaction would be 0.
        phenotype_reference = GeneInteractionPhenotype(gene_interaction=0.0, gene_interaction_p_value=None)

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
    print("="*60)
    print("FITNESS DATASETS")
    print("="*60)

    print("\n1. SmfKuzmin2018Dataset")
    print("-"*60)
    dataset = SmfKuzmin2018Dataset()
    print(f"Length: {len(dataset)}")
    item = dataset[0]
    print(f"First item phenotype:")
    print(f"  fitness: {item['experiment']['phenotype']['fitness']}")
    print(f"  fitness_std: {item['experiment']['phenotype']['fitness_std']}")
    print(f"  fitness_se: {item['experiment']['phenotype']['fitness_se']}")
    print(f"  n_samples: {item['experiment']['phenotype']['n_samples']}")

    print("\n2. DmfKuzmin2018Dataset")
    print("-"*60)
    dataset = DmfKuzmin2018Dataset()
    print(f"Length: {len(dataset)}")
    item = dataset[0]
    print(f"First item phenotype:")
    print(f"  fitness: {item['experiment']['phenotype']['fitness']}")
    print(f"  fitness_std: {item['experiment']['phenotype']['fitness_std']}")
    print(f"  fitness_se: {item['experiment']['phenotype']['fitness_se']}")
    print(f"  n_samples: {item['experiment']['phenotype']['n_samples']}")

    print("\n3. TmfKuzmin2018Dataset")
    print("-"*60)
    dataset = TmfKuzmin2018Dataset()
    print(f"Length: {len(dataset)}")
    item = dataset[0]
    print(f"First item phenotype:")
    print(f"  fitness: {item['experiment']['phenotype']['fitness']}")
    print(f"  fitness_std: {item['experiment']['phenotype']['fitness_std']}")
    print(f"  fitness_se: {item['experiment']['phenotype']['fitness_se']}")
    print(f"  n_samples: {item['experiment']['phenotype']['n_samples']}")

    print("\n" + "="*60)
    print("INTERACTION DATASETS")
    print("="*60)

    print("\n4. DmiKuzmin2018Dataset")
    print("-"*60)
    dataset = DmiKuzmin2018Dataset()
    print(f"Length: {len(dataset)}")
    item = dataset[0]
    print(f"First item phenotype:")
    print(f"  gene_interaction: {item['experiment']['phenotype']['gene_interaction']}")
    print(f"  gene_interaction_p_value: {item['experiment']['phenotype']['gene_interaction_p_value']}")

    print("\n5. TmiKuzmin2018Dataset")
    print("-"*60)
    dataset = TmiKuzmin2018Dataset()
    print(f"Length: {len(dataset)}")
    item = dataset[0]
    print(f"First item phenotype:")
    print(f"  gene_interaction: {item['experiment']['phenotype']['gene_interaction']}")
    print(f"  gene_interaction_p_value: {item['experiment']['phenotype']['gene_interaction_p_value']}")

    print("\n" + "="*60)
    print("✓ All datasets loaded successfully!")
    print("="*60)
