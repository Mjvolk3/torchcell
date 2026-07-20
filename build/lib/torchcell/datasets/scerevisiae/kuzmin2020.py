# torchcell/datasets/scerevisiae/kuzmin2020
# [[torchcell.datasets.scerevisiae.kuzmin2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2020
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2020.py
"""Kuzmin 2020 S. cerevisiae fitness and gene-interaction experiment datasets."""

import logging
import os
import os.path as osp
import zipfile
from collections.abc import Callable
from typing import Any

import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.media import SGA_TM_SELECTION
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    GenePerturbationType,
    Genotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaAllelePerturbation,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Sample size for double/triple mutant fitness measurements. The reported fitness
# "St.dev." / "Double/triple mutant fitness standard deviation" is a SAMPLE SD
# over colony replicates (Boone-lab SGA pipeline; Baryshnikova 2010 ref, Eq. 14
# colony s.d.) -> sample_sd over colonies, SE = SD/sqrt(n) auto-derived.
# n = 4, matching Kuzmin 2018: the per-record colony count is not in the data;
# 4 is the conservative Baryshnikova per-screen base and the empirical central
# estimate (Kuzmin-2018 p-value back-solve favours n=4; see kuzmin2018.py), and
# is consistent with Costanzo 2016 DMF.
N_SAMPLES_COMBINED_MUTANT = 4


def _combined_mutant_uncertainty(std_val: Any) -> dict[str, Any]:
    """Ontology fields for a double/triple-mutant-fitness sample SD over colonies.

    Returns empty uncertainty when the SD is unreported (None/NaN); otherwise
    fitness_se is auto-derived by FitnessPhenotype as SD/sqrt(n_samples).
    """
    if pd.isna(std_val):
        return {"fitness_uncertainty": None, "fitness_uncertainty_type": None}
    return {
        "fitness_uncertainty": std_val,
        "fitness_uncertainty_type": UncertaintyType.sample_sd,
        "n_samples": N_SAMPLES_COMBINED_MUTANT,
        "sample_unit": SampleUnit.colony,
    }


@register_dataset
class SmfKuzmin2020Dataset(ExperimentDataset):
    """Single-mutant fitness experiments from Kuzmin et al. 2020."""

    # original that doesn't work. Think Science blocks.
    # url = https://www.science.org/doi/suppl/10.1126/science.aaz5667/suppl_file/aaz5667-tables_s1_to_s13.zip
    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/smf_kuzmin2020",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset at the given root, triggering download/process."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the raw supplementary Excel file name."""
        return "aaz5667-Table-S5.xlsx"

    @property
    def processed_file_names(self) -> str:
        """Return the processed output name (the LMDB directory)."""
        return "lmdb"

    def download(self) -> None:
        """Download and extract the supplementary zip, then remove the archive."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Read, preprocess, and serialize experiments into the LMDB store."""
        df = pd.read_excel(osp.join(self.raw_dir, self.raw_file_names), skiprows=1)
        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Smf Files...")

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
        env.close()
        interned_env.close()

    def preprocess_raw(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]  # dataset-specific signature
        """Filter to single mutants and clean allele names for processing."""
        df = df[df["Mutant type"] == "Single mutant"]
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.dropna(subset=["Fitness"])
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbation: GenePerturbationType
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(
            fitness=row["Fitness"],
            fitness_std=row["St.dev."],
            **_combined_mutant_uncertainty(row["St.dev."]),
        )

        phenotype_reference = FitnessPhenotype(fitness=1.0, fitness_std=None)

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
    """Double-mutant fitness experiments from Kuzmin et al. 2020."""

    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmf_kuzmin2020",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset at the given root, triggering download/process."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw supplementary Excel file names."""
        return [
            "aaz5667-Table-S1.xlsx",
            "aaz5667-Table-S3.xlsx",
            "aaz5667-Table-S5.xlsx",
        ]

    @property
    def processed_file_names(self) -> str:
        """Return the processed output name (the LMDB directory)."""
        return "lmdb"

    def download(self) -> None:
        """Download and extract the supplementary zip, then remove the archive."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Read, preprocess, and serialize experiments into the LMDB store."""
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

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(  # type: ignore[override]  # dataset-specific signature
        self, df_s1: pd.DataFrame, df_s3: pd.DataFrame, df_s5: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the supplementary tables and assemble double-mutant fitness rows."""
        # Combine S1 and S3, filtering for digenic interactions
        df_combined = pd.concat([df_s1, df_s3])
        df_combined = df_combined[df_combined["Combined mutant type"] == "digenic"]

        # Process S5 to get double mutant data
        df_s5_double = df_s5[df_s5["Mutant type"] == "Double mutant"]
        df_s5_double = df_s5_double.dropna(subset=["Fitness"])

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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
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
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []

        # Process query perturbation (excluding ho)
        query_systematic_name = row["Query systematic name no ho"]
        query_allele_name = row["Query allele name no ho"]
        query_perturbation_type = row["query_perturbation_type_no_ho"]

        perturbation: GenePerturbationType
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(
            fitness=row["fitness"],
            fitness_std=row["fitness_std"],
            **_combined_mutant_uncertainty(row["fitness_std"]),
        )

        phenotype_reference = FitnessPhenotype(fitness=1.0, fitness_std=None)

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
    """Triple-mutant fitness experiments from Kuzmin et al. 2020."""

    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmf_kuzmin2020",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to subset_n records."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw supplementary Excel file names."""
        return ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed output name (the LMDB directory)."""
        return "lmdb"

    def download(self) -> None:
        """Download and extract the supplementary zip, then remove the archive."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Read, preprocess, and serialize experiments into the LMDB store."""
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

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row, phenotype_reference_std=self.phenotype_reference_std
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(  # type: ignore[override]  # dataset-specific signature
        self, df_s1: pd.DataFrame, df_s3: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the supplementary tables and assemble triple-mutant fitness rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
            )
        )

        # Calculate phenotype reference std
        self.phenotype_reference_std = df["fitness_std"].mean()

        # Clean up gene names
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)

        return df.reset_index(drop=True)

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        phenotype = FitnessPhenotype(
            fitness=row["fitness"], fitness_std=row["fitness_std"]
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0, fitness_std=phenotype_reference_std
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
            pubmed_id="32586993",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/32586993/",
            doi="10.1126/science.aaz5667",
            doi_url="https://www.science.org/doi/10.1126/science.aaz5667",
        )

        return experiment, reference, publication


@register_dataset
class DmiKuzmin2020Dataset(ExperimentDataset):
    """Digenic interaction experiments from Kuzmin et al. 2020."""

    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmi_kuzmin2020",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to subset_n records."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw supplementary Excel file names."""
        return ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed output name (the LMDB directory)."""
        return "lmdb"

    def download(self) -> None:
        """Download and extract the supplementary zip, then remove the archive."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Read, preprocess, and serialize experiments into the LMDB store."""
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

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(  # type: ignore[override]  # dataset-specific signature
        self, df_s1: pd.DataFrame, df_s3: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the supplementary tables and assemble interaction rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
            )
        )

        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)

        return df.reset_index(drop=True)

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects from one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        # Digenic interaction: relates two gene nodes -> an edge (not a hyperedge,
        # which is >=3 genes). GeneInteractionPhenotype defaults to "hyperedge"
        # (the trigenic case), so digenic loaders override it explicitly.
        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            gene_interaction_p_value=row["P-value"],
            graph_level="edge",
        )

        phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None, graph_level="edge"
        )

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
    """Trigenic interaction experiments from Kuzmin et al. 2020."""

    url = "https://uofi.box.com/shared/static/464ogx5kpafav7i3zv7gesb9lcn0dm94.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmi_kuzmin2020",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to subset_n records."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw supplementary Excel file names."""
        return ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]

    @property
    def processed_file_names(self) -> str:
        """Return the processed output name (the LMDB directory)."""
        return "lmdb"

    def download(self) -> None:
        """Download and extract the supplementary zip, then remove the archive."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Read, preprocess, and serialize experiments into the LMDB store."""
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

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))

        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference, publication = self.create_experiment(
                    self.name, row
                )

                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )

        env.close()
        interned_env.close()

    def preprocess_raw(  # type: ignore[override]  # dataset-specific signature
        self, df_s1: pd.DataFrame, df_s3: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge the supplementary tables and assemble interaction rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
            )
        )

        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)

        return df.reset_index(drop=True)

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects from one row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            gene_interaction_p_value=row["P-value"],
        )

        phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=0.0, gene_interaction_p_value=None
        )

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


def main() -> None:
    """Build and print summaries of the Kuzmin 2020 datasets for testing."""
    # Test the datasets
    datasets = [
        # SmfKuzmin2020Dataset(),
        # DmfKuzmin2020Dataset(),
        # TmfKuzmin2020Dataset(),
        # DmiKuzmin2020Dataset(),
        TmiKuzmin2020Dataset()
    ]

    for dataset in datasets:
        print(f"Testing {dataset.__class__.__name__}:")
        print(f"Length: {len(dataset)}")
        print(f"First item: {dataset[0]}")
        print("\n")


if __name__ == "__main__":
    main()
