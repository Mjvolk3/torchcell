# torchcell/datasets/scerevisiae/kuzmin2020
# [[torchcell.datasets.scerevisiae.kuzmin2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2020
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2020.py

import logging
import os
import os.path as osp
import pickle
import zipfile
from collections.abc import Callable
import lmdb
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

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Mutant type"] == "Single mutant"]
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(dataset_name, row):
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
        assert len(genotype) == 1, "Genotype must have 1 perturbation."

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(fitness=row["Fitness"], std=row["St.dev."])

        phenotype_reference = FitnessPhenotype(fitness=1.0, std=None)

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
        mask = (df["Double/triple mutant fitness"] - df["Fitness"]).abs() > 1e-6
        if mask.any():
            log.warning(f"Fitness mismatch found for {mask.sum()} rows")

        # Use S5 fitness and std where available, fallback to S1/S3 data
        df["fitness"] = df["Fitness"].fillna(df["Double/triple mutant fitness"])
        df["fitness_std"] = df["St.dev."].fillna(
            df["Double/triple mutant fitness standard deviation"]
        )

        # Split Query strain ID and Query allele name
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)

        # Extract systematic names
        df["Query systematic name_1"] = df["Query strain ID_1"].str.split(
            "_", expand=True
        )[0]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]

        # Create 'no ho' versions
        df["Query allele name no ho"] = (
            df["Query allele name"].str.replace("hoΔ", "").str.replace("+", "")
        )
        df["Query systematic name"] = df["Query strain ID"].str.split("_", expand=True)[
            0
        ]
        df["Query systematic name no ho"] = (
            df["Query systematic name"].str.replace("YDL227C", "").str.replace("+", "")
        )

        # Determine perturbation types (using Δ before replacement)
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

        # Calculate phenotype reference std
        self.phenotype_reference_std = df["fitness_std"].mean()

        # Clean up gene names (after determining perturbation types)
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)

        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []

        # Process query perturbation (excluding ho)
        query_systematic_name = row["Query systematic name no ho"]
        query_allele_name = row["Query allele name no ho"]
        query_perturbation_type = row["query_perturbation_type_no_ho"]

        if query_perturbation_type == "KanMX_deletion":
            perturbation = SgaKanMxDeletionPerturbation(
                systematic_gene_name=query_systematic_name,
                perturbed_gene_name=query_allele_name.split("_")[0],
                strain_id=row["Query strain ID"],
            )
        else:  # allele
            perturbation = SgaAllelePerturbation(
                systematic_gene_name=query_systematic_name,
                perturbed_gene_name=query_allele_name.split("_")[0],
                strain_id=row["Query strain ID"],
            )
        perturbations.append(perturbation)

        # Process array perturbation
        array_systematic_name = row["Array systematic name"]
        array_allele_name = row["Array allele name"]
        array_perturbation_type = row["array_perturbation_type"]

        if array_perturbation_type == "KanMX_deletion":
            perturbation = SgaKanMxDeletionPerturbation(
                systematic_gene_name=array_systematic_name,
                perturbed_gene_name=array_allele_name.split("_")[0],
                strain_id=row["Array strain ID"],
            )
        elif array_perturbation_type == "temperature_sensitive":
            perturbation = SgaTsAllelePerturbation(
                systematic_gene_name=array_systematic_name,
                perturbed_gene_name=array_allele_name.split("_")[0],
                strain_id=row["Array strain ID"],
            )
        else:  # unknown or other types
            perturbation = SgaAllelePerturbation(
                systematic_gene_name=array_systematic_name,
                perturbed_gene_name=array_allele_name.split("_")[0],
                strain_id=row["Array strain ID"],
            )
        perturbations.append(perturbation)

        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 2, "Genotype must have 2 perturbations."

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(fitness=row["fitness"], std=row["fitness_std"])

        phenotype_reference = FitnessPhenotype(fitness=1.0, std=None)

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
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class TmfKuzmin2020Dataset(ExperimentDataset):
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmf_kuzmin2020",
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
    def raw_file_names(self) -> list[str]:
        return ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]

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

        df = self.preprocess_raw(df_s1, df_s3)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Tmf Files...")

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
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

    def preprocess_raw(self, df_s1: pd.DataFrame, df_s3: pd.DataFrame) -> pd.DataFrame:
        # Combine S1 and S3, filtering for trigenic interactions
        df = pd.concat([df_s1, df_s3])
        df = df[df["Combined mutant type"] == "trigenic"].copy()

        # Use the provided fitness and standard deviation
        df["fitness"] = df["Double/triple mutant fitness"]
        df["fitness_std"] = df["Double/triple mutant fitness standard deviation"]

        # Split Query strain ID and Query allele name
        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)

        # Extract systematic names
        df["Query systematic name_1"] = df["Query strain ID_1"].str.split(
            "_", expand=True
        )[0]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]

        # Determine perturbation types
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

        # Calculate phenotype reference std
        self.phenotype_reference_std = df["fitness_std"].mean()

        # Clean up gene names
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)

        return df.reset_index(drop=True)

    @staticmethod
    def create_experiment(dataset_name, row, phenotype_reference_std):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        # Query 1
        if row["query_perturbation_type_1"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"].split("_")[0],
                    strain_id=row["Query strain ID_1"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"].split("_")[0],
                    strain_id=row["Query strain ID_1"],
                )
            )

        # Query 2
        if row["query_perturbation_type_2"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"].split("_")[0],
                    strain_id=row["Query strain ID_2"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"].split("_")[0],
                    strain_id=row["Query strain ID_2"],
                )
            )

        # Array
        if row["array_perturbation_type"] == "temperature_sensitive":
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )
        elif row["array_perturbation_type"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )

        genotype = Genotype(perturbations=perturbations)
        assert len(genotype) == 3, "Genotype must have 3 perturbations."

        environment = Environment(
            media=Media(name="YEPD", state="solid"), temperature=Temperature(value=30)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(fitness=row["fitness"], std=row["fitness_std"])

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
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class DmiKuzmin2020Dataset(ExperimentDataset):
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmi_kuzmin2020",
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
    def raw_file_names(self) -> list[str]:
        return ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]

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

        df = self.preprocess_raw(df_s1, df_s3)
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

    def preprocess_raw(self, df_s1: pd.DataFrame, df_s3: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df_s1, df_s3])
        df = df[df["Combined mutant type"] == "digenic"].copy()

        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"].str.split(
            "_", expand=True
        )[0]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
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

        df["query_perturbation_type_no_ho"] = df["Query allele name no ho"].apply(
            lambda x: "KanMX_deletion" if "Δ" in x else "allele"
        )
        df["array_perturbation_type"] = df["Array strain ID"].apply(
            lambda x: (
                "temperature_sensitive"
                if "tsa" in x
                else "KanMX_deletion" if "dma" in x else "unknown"
            )
        )

        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        
        return df.reset_index(drop=True)

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        # Query
        if row["query_perturbation_type_no_ho"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"].split("_")[0],
                    strain_id=row["Query strain ID"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name no ho"],
                    perturbed_gene_name=row["Query allele name no ho"].split("_")[0],
                    strain_id=row["Query strain ID"],
                )
            )

        # Array
        if row["array_perturbation_type"] == "temperature_sensitive":
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )
        elif row["array_perturbation_type"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
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
            interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            p_value=row["P-value"],
        )

        phenotype_reference = GeneInteractionPhenotype(interaction=0.0, p_value=None)

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
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class TmiKuzmin2020Dataset(ExperimentDataset):
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmi_kuzmin2020",
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
    def raw_file_names(self) -> list[str]:
        return ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]

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

        df = self.preprocess_raw(df_s1, df_s3)
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

    def preprocess_raw(self, df_s1: pd.DataFrame, df_s3: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df_s1, df_s3])
        df = df[df["Combined mutant type"] == "trigenic"].copy()

        df[["Query strain ID_1", "Query strain ID_2"]] = df[
            "Query strain ID"
        ].str.split("+", expand=True)
        df[["Query allele name_1", "Query allele name_2"]] = df[
            "Query allele name"
        ].str.split("+", expand=True)
        df["Query systematic name_1"] = df["Query strain ID_1"].str.split(
            "_", expand=True
        )[0]
        df["Query systematic name_2"] = df["Query strain ID_2"].str.split(
            "_", expand=True
        )[0]
        df["Array systematic name"] = df["Array strain ID"].str.split("_", expand=True)[
            0
        ]

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

        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        
        return df.reset_index(drop=True)

    @staticmethod
    def create_experiment(dataset_name, row):
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="s288c"
        )

        perturbations = []
        # Query 1
        if row["query_perturbation_type_1"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"].split("_")[0],
                    strain_id=row["Query strain ID_1"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_1"],
                    perturbed_gene_name=row["Query allele name_1"].split("_")[0],
                    strain_id=row["Query strain ID_1"],
                )
            )

        # Query 2
        if row["query_perturbation_type_2"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"].split("_")[0],
                    strain_id=row["Query strain ID_2"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Query systematic name_2"],
                    perturbed_gene_name=row["Query allele name_2"].split("_")[0],
                    strain_id=row["Query strain ID_2"],
                )
            )

        # Array
        if row["array_perturbation_type"] == "temperature_sensitive":
            perturbations.append(
                SgaTsAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )
        elif row["array_perturbation_type"] == "KanMX_deletion":
            perturbations.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
                    strain_id=row["Array strain ID"],
                )
            )
        else:
            perturbations.append(
                SgaAllelePerturbation(
                    systematic_gene_name=row["Array systematic name"],
                    perturbed_gene_name=row["Array allele name"].split("_")[0],
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
            interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            p_value=row["P-value"],
        )

        phenotype_reference = GeneInteractionPhenotype(interaction=0.0, p_value=None)

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
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


if __name__ == "__main__":
    # Test the datasets
    datasets = [
        SmfKuzmin2020Dataset(),
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
