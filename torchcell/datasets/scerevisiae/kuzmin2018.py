# torchcell/datasets/scerevisiae/kuzmin2018.py
# [[torchcell.datasets.scerevisiae.kuzmin2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/kuzmin2018.py
# Test file: tests/torchcell/datasets/scerevisiae/test_kuzmin2018.py
"""Kuzmin 2018 fitness and genetic-interaction datasets (single/double/triple mutants)."""

import logging
import os
import os.path as osp
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
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


# Sample size for combined (double/triple) mutant fitness measurements.
# The reported "Combined mutant fitness standard deviation" (Additional Data S1
# col. 12) is a SAMPLE SD over colony replicates. Kuzmin's SI (si1.md line 59)
# does not restate its type and defers to Baryshnikova 2010 (ref 8), whose Eq. 14
# defines it as the colony s.d. sigma_Iij over N_ij colonies -> we record it as
# sample_sd over colonies (SE = SD/sqrt(n), auto-derived).
#
# n = 4. The exact per-record colony count is NOT in the released data (12 cols,
# no count/SE column), so it is fixed by three converging lines:
#  (1) Empirical back-solve against the reported P-value (the "other provided
#      statistic"): the single-term normal model 2*Phi(-|eps|/(sd/sqrt(n))) over
#      410k digenic records matches the reported P-value median (0.358) best at
#      n=4 (0.377); n=8 overshoots badly (0.211). Spearman(p_pred,p_reported)=0.985
#      confirms this SD column drives the p-value ranking. Exact recovery is
#      precluded by the unpublished pooled-background term in the Baryshnikova
#      p-value, so this fixes the central estimate, not an exact per-record n.
#  (2) Conservative lower-end of the Baryshnikova range (typically 4/screen, 4-8).
#  (3) Consistency with Costanzo 2016 DMF (also n=4).
# NB: the "12-24 colony measurements" in si1.md line 59 is the QUERY fitness
# (col 9, bootstrap), a DIFFERENT column whose std the loader does not store.
N_SAMPLES_COMBINED_MUTANT = 4


def _combined_mutant_uncertainty(std_val: Any) -> dict[str, Any]:
    """Ontology fields for a combined-mutant-fitness sample SD over colonies.

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


# Fitness
@register_dataset
class SmfKuzmin2018Dataset(ExperimentDataset):
    """Single-mutant fitness experiments from Kuzmin 2018."""

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/smf_kuzmin2018",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset at ``root`` and trigger download/processing."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment model class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the name of the raw Kuzmin 2018 data file."""
        return "aao1729_data_s1.tsv"

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed LMDB directory."""
        return "lmdb"  # type: ignore[return-value]  # single processed dir as str, base/PyG accept str

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
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

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
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
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from a data row."""
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()
        # Phenotype
        if row["smf_type"] == "query_smf":
            smf_key = "Query single/double mutant fitness"
        elif row["smf_type"] == "array_smf":
            smf_key = "Array single mutant fitness"

        # No reported std for single mutants (SMF value comes from Costanzo/query
        # estimates); leave uncertainty unset.
        phenotype = FitnessPhenotype(fitness=row[smf_key], fitness_std=None)

        # Reference noise = mean combined-mutant SD -> sample_sd over colonies.
        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            **_combined_mutant_uncertainty(phenotype_reference_std),
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
    """Double-mutant fitness experiments from Kuzmin 2018."""

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmf_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment model class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the name of the raw Kuzmin 2018 data file."""
        return "aao1729_data_s1.tsv"

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
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

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
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
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from a data row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )
        # genotype
        perturbations: list[GenePerturbationType] = []
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
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
        phenotype = FitnessPhenotype(
            fitness=row[dmf_key],
            fitness_std=fitness_std,
            **_combined_mutant_uncertainty(fitness_std),
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            **_combined_mutant_uncertainty(phenotype_reference_std),
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
    """Triple-mutant fitness experiments from Kuzmin 2018."""

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmf_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment model class for this dataset."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the name of the raw Kuzmin 2018 data file."""
        return "aao1729_data_s1.tsv"

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        log.info("Processing Dmf Files...")

        # Initialize LMDB environment
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

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
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
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series, phenotype_reference_std: float
    ) -> tuple[FitnessExperiment, FitnessExperimentReference, Publication]:
        """Build the experiment, reference, and publication objects from a data row."""
        # Common attributes for both temperatures
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )
        # genotype
        perturbations: list[GenePerturbationType] = []
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()
        # Phenotype based on temperature
        tmf_key = "Combined mutant fitness"
        tmf_std_key = "Combined mutant fitness standard deviation"
        tmf_std = row[tmf_std_key]
        phenotype = FitnessPhenotype(
            fitness=row[tmf_key],
            fitness_std=tmf_std,
            **_combined_mutant_uncertainty(tmf_std),
        )

        phenotype_reference = FitnessPhenotype(
            fitness=1.0,
            fitness_std=phenotype_reference_std,
            **_combined_mutant_uncertainty(phenotype_reference_std),
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
    """Double-mutant genetic-interaction experiments from Kuzmin 2018."""

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/dmi_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[GeneInteractionExperiment]:
        """Return the experiment model class for this dataset."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the name of the raw Kuzmin 2018 data file."""
        return "aao1729_data_s1.tsv"

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
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

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
            )
        )

        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects from a data row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []
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

        # By definition in paper interaction would be 0.
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
            pubmed_id="29674565",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29674565/",
            doi="10.1126/science.aao1729",
            doi_url="https://www.science.org/doi/10.1126/science.aao1729",
        )

        return experiment, reference, publication


@register_dataset
class TmiKuzmin2018Dataset(ExperimentDataset):
    """Triple-mutant genetic-interaction experiments from Kuzmin 2018."""

    url = "https://raw.githubusercontent.com/Mjvolk3/torchcell/main/data/host/kuzmin2018/aao1729_data_s1.zip"

    def __init__(
        self,
        root: str = "data/torchcell/tmi_kuzmin2018",
        subset_n: int | None = None,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset, optionally subsetting to ``subset_n`` rows."""
        self.subset_n = subset_n
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[GeneInteractionExperiment]:
        """Return the experiment model class for this dataset."""
        return GeneInteractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference model class for this dataset."""
        return GeneInteractionExperimentReference

    @property
    def raw_file_names(self) -> str:  # type: ignore[override]  # single raw file as str, base is list[str]
        """Return the name of the raw Kuzmin 2018 data file."""
        return "aao1729_data_s1.tsv"

    def download(self) -> None:
        """Download the Kuzmin 2018 archive and extract it into the raw directory."""
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

    @post_process
    def process(self) -> None:
        """Preprocess the raw table and write experiment records to the LMDB store."""
        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names), sep="\t")

        df = self.preprocess_raw(df)
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

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Clean and reshape the raw Kuzmin table into per-genotype rows."""
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
                else "KanMX_deletion"
                if "dma" in x
                else "unknown"
            )
        )

        # replace delta symbol for neo4j import
        df = df.replace("'", "_prime", regex=True)
        df = df.replace("Δ", "_delta", regex=True)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def create_experiment(  # type: ignore[override]  # dataset-specific signature
        dataset_name: str, row: pd.Series
    ) -> tuple[
        GeneInteractionExperiment, GeneInteractionExperimentReference, Publication
    ]:
        """Build the experiment, reference, and publication objects from a data row."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )

        perturbations: list[GenePerturbationType] = []
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
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        )
        environment_reference = environment.model_copy()

        phenotype = GeneInteractionPhenotype(
            gene_interaction=row["Adjusted genetic interaction score (epsilon or tau)"],
            gene_interaction_p_value=row["P-value"],
        )

        # By definition in paper interaction would be 0.
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
